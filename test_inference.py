import os
import json
import torch
import numpy as np
import time
import argparse
from tqdm import tqdm
from transformers import AutoModel, AutoConfig
from sacrebleu.metrics import BLEU
from accelerate import Accelerator
from accelerate.utils import gather_object, broadcast_object_list
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HFDataset
import whisper
from multiprocessing import Manager

# =====================================================================
# 1. DATASETS
# =====================================================================

class AudioFeatureDataset(Dataset):
    """Edge: extract embeddings from raw audio"""
    def __init__(self, data, audio_root):
        self.data = data
        self.audio_root = audio_root

    def __len__(self):
        return len(self.data)

    def _prepare_audio(self, audio_input):
        if isinstance(audio_input, str):
            return whisper.load_audio(audio_input)
        if isinstance(audio_input, dict):
            audio_array = audio_input["array"]
            sr = audio_input["sampling_rate"]
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            return audio_array.astype(np.float32)
        raise ValueError("Input should be a file path or HF Audio dict")

    def __getitem__(self, i):
        item = self.data[i]
        audio_key = item["audio"]
        audio_path = os.path.join(self.audio_root, audio_key) if isinstance(audio_key, str) else audio_key

        audio = self._prepare_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=128)

        prompt = f"<|{item.get('src', '')}|><|{item.get('tgt', '')}|>"

        return {
            "mel": mel,
            "prompt": prompt,
            "id": item.get("id"),
            "audio": item.get("audio"),
            "src": item.get("src"),
            "tgt": item.get("tgt"),
            "asr": item.get("asr", ""),
            "s2tt": item.get("s2tt", "")
        }


class EmbeddingsInferenceDataset(Dataset):
    """Cloud: load pre-computed embeddings for LLM decoding"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        prompt = f"<|{item.get('src', '')}|><|{item.get('tgt', '')}|>"
        embeds = torch.tensor(item["adapter_embeds"], dtype=torch.bfloat16)

        return {
            "adapter_embeds": embeds,
            "prompt": prompt,
            "id": item.get("id"),
            "audio": item.get("audio"),
            "src": item.get("src"),
            "tgt": item.get("tgt"),
            "asr": item.get("asr", ""),
            "s2tt": item.get("s2tt", "")
        }

# =====================================================================
# 2. MAIN PIPELINE
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ESRT Inference Pipeline")
    parser.add_argument("--mllm_path", type=str, default="yxdu/ESRT-4B")
    parser.add_argument("--jsonl_path", type=str, default="./fleurs_eng_test/srt_test_eng.jsonl")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--edge_batch_size", type=int, default=1)
    parser.add_argument("--cloud_batch_size", type=int, default=256)
    parser.add_argument("--cloud_beam_search", type=int, default=1)
    parser.add_argument("--cloud_use_vllm", action="store_true", default=False)
    parser.add_argument("--edge_cache_path", type=str, default=None)
    parser.add_argument("--src_langs", type=str, nargs="+", default=["eng"])
    parser.add_argument("--tgt_langs", type=str, nargs="+", default=["ara", "cmn", "deu", "fra", "jpn"])
    parser.add_argument("--max_new_tokens", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()

    mllm_path = args.mllm_path
    jsonl_path = args.jsonl_path
    edge_batch_size = args.edge_batch_size
    cloud_batch_size = args.cloud_batch_size
    cloud_beam_search = args.cloud_beam_search
    cloud_use_vllm = args.cloud_use_vllm
    max_new_tokens = args.max_new_tokens
    src_langs = args.src_langs
    tgt_langs = args.tgt_langs

    audio_root = os.path.dirname(os.path.abspath(jsonl_path))
    output_path = args.output_path or f"{mllm_path.split('/')[-1]}_results.jsonl"

    edge_cache_path = args.edge_cache_path

    # Check if edge cache exists, broadcast to all GPUs
    cache_exists = os.path.exists(edge_cache_path)
    cache_status = [cache_exists]
    broadcast_object_list(cache_status)
    cache_exists = cache_status[0]

    # =====================================================================
    # EDGE: extract audio features and save to cache
    # =====================================================================
    if not cache_exists:
        if accelerator.is_main_process:
            print(f"\n>>> [Edge] No cache found, starting cache build...")
            print(f">>> Cache path: {edge_cache_path}")

        # Load and deduplicate JSONL data
        dataset_raw_edge = []
        if accelerator.is_main_process:
            unique_audio_data = []
            seen_audio = set()
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    if not (item.get("src") in src_langs and item.get("tgt") in tgt_langs):
                        continue
                    audio_key = item.get("audio")
                    if audio_key not in seen_audio:
                        unique_audio_data.append(item)
                        seen_audio.add(audio_key)
            dataset_raw_edge = unique_audio_data
            print(f">>> Loaded {len(dataset_raw_edge)} unique audio items")

        dataset_list = [dataset_raw_edge]
        broadcast_object_list(dataset_list)
        dataset_raw_edge = dataset_list[0]

        # Prepare dataloader
        edge_loader = DataLoader(
            AudioFeatureDataset(dataset_raw_edge, audio_root),
            batch_size=edge_batch_size,
            shuffle=False,
            num_workers=8
        )

        # Load encoder model for feature extraction
        if accelerator.is_main_process:
            print(f">>> Loading encoder model...")
        edge_model = AutoModel.from_pretrained(
            mllm_path,
            use_vllm=False,
            encoder_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).eval()

        edge_model, edge_loader = accelerator.prepare(edge_model, edge_loader)

        local_edge_results = []

        # Extract features
        with torch.inference_mode():
            for batch in tqdm(edge_loader, desc="Edge cache build", disable=not accelerator.is_local_main_process):
                mels = batch["mel"]
                model_engine = edge_model.module if hasattr(edge_model, "module") else edge_model
                adapter_embeds = model_engine.translate_encode(mels)

                batch_size_curr = mels.size(0)
                for j in range(batch_size_curr):
                    embed = adapter_embeds[j]
                    if torch.is_tensor(embed):
                        embed = embed.detach().cpu().half().numpy()

                    local_edge_results.append({
                        "audio": batch["audio"][j],
                        "adapter_embeds": embed
                    })

        # Gather results and save to HF Dataset
        accelerator.wait_for_everyone()
        all_edge_data = gather_object(local_edge_results)

        if accelerator.is_main_process:
            all_edge_data = all_edge_data[:len(dataset_raw_edge)]

            # Deduplicate
            unique_data_dict = {}
            for item in all_edge_data:
                audio_key = item["audio"]
                if audio_key not in unique_data_dict:
                    unique_data_dict[audio_key] = {
                        "audio": audio_key,
                        "adapter_embeds": item["adapter_embeds"]
                    }
            final_edge_data = list(unique_data_dict.values())

            def gen():
                for item in final_edge_data:
                    yield item

            print(f">>> Writing HuggingFace Dataset to disk...")
            hf_dataset = HFDataset.from_generator(gen, writer_batch_size=1000)
            hf_dataset.save_to_disk(edge_cache_path)
            print(f">>> Edge cache saved: {edge_cache_path}\n")

        # Free GPU memory before loading cloud model
        del edge_model
        del edge_loader
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    else:
        if accelerator.is_main_process:
            print(f"\n>>> [Edge] Cache found: {edge_cache_path}, skipping extraction.")

    # =====================================================================
    # CLOUD: load embeddings and run LLM decoding
    # =====================================================================
    if accelerator.is_main_process:
        print(f"\n>>> [Cloud] Loading pre-computed cache...")

    dataset_raw_cloud = []
    if accelerator.is_main_process:
        hf_dataset = HFDataset.load_from_disk(edge_cache_path)
        embeds_dict = {item["audio"]: item["adapter_embeds"] for item in hf_dataset}

        print(f">>> Loaded {len(embeds_dict)} tensors from cache")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                audio_key = item.get("audio")

                if (item.get("src") in src_langs and
                    item.get("tgt") in tgt_langs and
                    audio_key in embeds_dict):

                    item["adapter_embeds"] = embeds_dict[audio_key]
                    dataset_raw_cloud.append(item)

        print(f">>> Total items ready for decoding: {len(dataset_raw_cloud)}")

    dataset_list = [dataset_raw_cloud]
    broadcast_object_list(dataset_list)
    dataset_raw_cloud = dataset_list[0]

    cloud_loader = DataLoader(
        EmbeddingsInferenceDataset(dataset_raw_cloud),
        batch_size=cloud_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # Load cloud decoding model
    if accelerator.is_main_process:
        print(f">>> Loading cloud model...")

    cloud_model = AutoModel.from_pretrained(
        mllm_path,
        use_vllm=cloud_use_vllm,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).eval()

    cloud_model, cloud_loader = accelerator.prepare(cloud_model, cloud_loader)

    bleu = BLEU(tokenize="flores200")
    local_cloud_results = []

    # Distributed inference
    accelerator.wait_for_everyone()
    loop_start_time = time.perf_counter()
    first_batch_time = 0

    with torch.inference_mode():
        for step, batch in enumerate(tqdm(cloud_loader, desc=f"GPU {accelerator.process_index}", disable=not accelerator.is_local_main_process)):
            total_batches = len(cloud_loader)
            embeds = batch["adapter_embeds"]
            prompts = batch["prompt"]

            model_engine = cloud_model.module if hasattr(cloud_model, "module") else cloud_model
            batch_start_time = time.perf_counter()

            outs = model_engine.translate_batch_embeds(
                cloud_beam_search,
                embeds,
                prompts,
                max_new_tokens=max_new_tokens,
                use_vllm=cloud_use_vllm
            )

            if step == 0:
                first_batch_time = time.perf_counter() - batch_start_time

            batch_size_curr = embeds.size(0)
            for j in range(batch_size_curr):
                current_prompt = prompts[j]
                out = outs[j]

                res_parts = out.split(current_prompt)
                if len(res_parts) != 2:
                    res_parts = out.rsplit('>', 1)

                asr_r = res_parts[0].strip() if len(res_parts) == 2 else out.strip()
                s2tt_r = res_parts[1].strip() if len(res_parts) == 2 else out.strip()

                local_cloud_results.append({
                    "id": batch["id"][j],
                    "audio": batch["audio"][j],
                    "prompt": batch["prompt"][j],
                    "src": batch["src"][j],
                    "tgt": batch["tgt"][j],
                    "asr": batch["asr"][j],
                    "s2tt": batch["s2tt"][j],
                    "asr_r": asr_r,
                    "s2tt_r": s2tt_r,
                    "response": out
                })

            if accelerator.is_local_main_process:
                last_idx = -1

                print(f"\n{'='*40} BATCH CHECK (Last Item) {'='*40}")
                print(f"Progress    : Batch {step + 1}/{total_batches} | Processed {(step + 1) * cloud_batch_size} samples")
                print(f"Last Item ID: {batch['id'][last_idx]}")
                print(f"Audio       : {batch['audio'][last_idx]}")
                print(f"Prompt      : {batch['prompt'][last_idx]}")
                print(f"Src Lang    : {batch['src'][last_idx]}")
                print(f"Tgt Lang    : {batch['tgt'][last_idx]}")
                print(f"{'-'*40} Ground Truth {'-'*40}")
                print(f"ASR GT      : {batch['asr'][last_idx]}")
                print(f"S2TT GT     : {batch['s2tt'][last_idx]}")
                print(f"{'-'*40} Model Outputs {'-'*40}")
                print(f"Raw Response: {out}")
                print(f"ASR Result  : {asr_r}")
                print(f"S2TT Result : {s2tt_r}")
                print(f"{'='*105}\n")

                

    # Gather results from all GPUs
    accelerator.wait_for_everyone()
    loop_end_time = time.perf_counter()
    all_cloud_data = gather_object(local_cloud_results)

    # Post-processing and metrics (main process only)
    if accelerator.is_main_process:
        total_loop_time = loop_end_time - loop_start_time

        if total_batches > 1:
            rest_batches_time = total_loop_time - first_batch_time
            avg_batch_time = rest_batches_time / (total_batches - 1)
            vllm_startup_time = max(0, first_batch_time - avg_batch_time)
        else:
            vllm_startup_time = 0
            avg_batch_time = total_loop_time

        effective_loop_time = total_loop_time - vllm_startup_time

        # Deduplicate results
        unique_results_dict = {str(d["id"]): d for d in all_cloud_data}
        unique_data = list(unique_results_dict.values())

        # Throughput
        total_samples = len(unique_data)
        samples_per_second = total_samples / effective_loop_time if effective_loop_time > 0 else 0

        print(f"\n{'='*30} Timing & Throughput {'='*30}")
        print(f"Total samples (dedup):   {total_samples}")
        print(f"Batches per GPU:         {total_batches}")
        print(f"Total loop time:         {total_loop_time:.2f}s")
        print(f"vLLM startup overhead:   {vllm_startup_time:.2f}s")
        print(f"Pure inference time:     {effective_loop_time:.2f}s")
        print(f"Avg batch time:          {avg_batch_time:.2f}s")
        print(f"Throughput:              {samples_per_second:.2f} samples/s")
        print(f"{'='*83}\n")

        print(f"Raw results: {len(all_cloud_data)}")
        print(f"Dedup results: {len(unique_data)}")

        # Write output file
        with open(output_path, "w", encoding="utf-8") as f:
            for d in unique_data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        # Compute BLEU scores
        res_map = {}
        for d in unique_data:
            pair = (d["src"], d["tgt"])
            res_map.setdefault(pair, [[], []])
            res_map[pair][0].append(d["s2tt_r"])
            res_map[pair][1].append(d["s2tt"])

        header = f"\n{'Pair':<12} | {'spBLEU':<6} | {'Count'}\n{'-'*32}"
        print(header)

        scores = []
        for (s, t), (hyps, refs) in res_map.items():
            score = bleu.corpus_score(hyps, [refs]).score
            print(f"{s}->{t:<7} | {score:<6.2f} | {len(hyps)}")
            scores.append(score)

        if scores:
            avg_score = sum(scores) / len(scores)
            total_cnt = sum(len(h) for h, _ in res_map.values())
            print(f"{'-'*32}\n{'Average':<12} | {avg_score:<6.2f} | {total_cnt}")

if __name__ == '__main__':
    main()
