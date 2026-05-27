import os
import json
import torch
import numpy as np
import time 
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
# 1. DATASETS 定义
# =====================================================================

class AudioFeatureDataset(Dataset):
    """阶段一：用于从原始音频提取 Embeddings 的 Dataset"""
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
        raise ValueError("输入应为文件路径或 HF Audio 字典")

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
    """阶段二：直接读取预计算特征进行大模型解码的 Dataset"""
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
# 2. MAIN 流水线
# =====================================================================

def main():
    accelerator = Accelerator()
    device = accelerator.device
    
    m_path = "yxdu/ESRT-4B"

    jsonl_path = "./fleurs_eng_test/srt_test_eng.jsonl"
    
    audio_root = os.path.dirname(os.path.abspath(jsonl_path))
    output_jsonl = f"{m_path.split('/')[-1]}_results.jsonl"
    
    batch_size_stage1 = 1  
    batch_size_stage2 = 256
    beam_search = 1 # 仅在use_vllm=False时生效，vLLM模式下固定为贪心解码
    use_vllm = True


    # 语言集配置
    langs_45 = ['ara', 'azj', 'bul', 'ben', 'cat', 'ces', 'dan', 'deu', 'ell', 'eng',
                'spa', 'fas', 'fin', 'fra', 'heb', 'hin', 'hrv', 'hun', 'ind', 'ita',
                'jpn', 'kaz', 'khm', 'kor', 'lao', 'msa', 'mya', 'nob', 'nld', 'pol',
                'por', 'ron', 'rus', 'slk', 'slv', 'swe', 'tam', 'tha', 'tgl', 'tur',
                'urd', 'uzb', 'vie', 'yue', 'cmn'] # 45语言全覆盖
    langs_5 = [
                "ara", "cmn", "deu", "fra", "jpn" # 快速测试用的5种语言对
            ]
    langs_1 = ['eng']
    
    src_langs = langs_1
    tgt_langs = langs_5
    langnum = len(src_langs)

    # -----------------------------------------------------------------
    # 动态命名与检查
    # -----------------------------------------------------------------
    # 读取模型 config 以获取 llm_dim 和 query_len (假设存在于 config 中)
    config = AutoConfig.from_pretrained(m_path, trust_remote_code=True)
    hidden = getattr(config, "llm_dim")
    query_len = getattr(config, "query_len") # 如果 config 里无此字段，请修改为模型对应的默认默认值
    
    hf_dataset_path = f"./fleurs_eng_qformer_80_768"

    # 判断特征缓存是否存在
    cache_exists = os.path.exists(hf_dataset_path)
    # 分布式环境下广播状态，确保所有卡步调一致
    cache_status = [cache_exists]
    broadcast_object_list(cache_status)
    cache_exists = cache_status[0]

    # =====================================================================
    # STAGE 1: 如果特征不存在，启动特征提取与保存
    # =====================================================================
    if not cache_exists:
        if accelerator.is_main_process:
            print(f"\n>>> [Stage 1] 未检测到缓存，启动特征提取流水线...")
            print(f">>> 目标保存路径: {hf_dataset_path}")

        # 1. 载入原始 JSONL 数据并去重
        dataset_raw_stage1 = []
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
            dataset_raw_stage1 = unique_audio_data
            print(f">>> 原始文件读取完成，去重后唯一音频数: {len(dataset_raw_stage1)}")

        dataset_list = [dataset_raw_stage1]
        broadcast_object_list(dataset_list)
        dataset_raw_stage1 = dataset_list[0]

        # 2. 准备 Dataloader
        stage1_loader = DataLoader(
            AudioFeatureDataset(dataset_raw_stage1, audio_root), 
            batch_size=batch_size_stage1, 
            shuffle=False,
            num_workers=8
        )

        # 3. 加载 12B 模型用于提取特征 (translate_encode)
        if accelerator.is_main_process:
            print(f">>> 正在加载模型用于特征提取...")
        model_stage1 = AutoModel.from_pretrained(
            m_path, 
            use_vllm=False,
            encoder_only=True,  # 只加载编码器部分
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).eval()

        model_stage1, stage1_loader = accelerator.prepare(model_stage1, stage1_loader)
        
        local_stage1_results = []

        # 4. 提取循环
        with torch.inference_mode():
            for batch in tqdm(stage1_loader, desc="特征提取", disable=not accelerator.is_local_main_process):
                mels = batch["mel"]
                prompts = batch["prompt"]
                
                model_engine = model_stage1.module if hasattr(model_stage1, "module") else model_stage1
                adapter_embeds = model_engine.translate_encode(beam_search, mels, prompts, max_new_tokens=200, use_vllm=True)

                batch_size_curr = mels.size(0)
                for j in range(batch_size_curr):
                    embed = adapter_embeds[j]
                    if torch.is_tensor(embed):
                        embed = embed.detach().cpu().half().numpy()

                    local_stage1_results.append({
                        "audio": batch["audio"][j],
                        "adapter_embeds": embed 
                    })

        # 5. 聚合与保存到本地 HF Dataset
        accelerator.wait_for_everyone()
        all_stage1_data = gather_object(local_stage1_results)

        if accelerator.is_main_process:
            all_stage1_data = all_stage1_data[:len(dataset_raw_stage1)]
            
            # 去重
            unique_data_dict = {}
            for item in all_stage1_data:
                audio_key = item["audio"]
                if audio_key not in unique_data_dict:
                    unique_data_dict[audio_key] = {
                        "audio": audio_key,
                        "adapter_embeds": item["adapter_embeds"]
                    }
            final_stage1_data = list(unique_data_dict.values())

            def gen():
                for item in final_stage1_data:
                    yield item

            print(f">>> 正在通过 Generator 写入 HuggingFace Dataset 到磁盘...")
            hf_dataset = HFDataset.from_generator(gen, writer_batch_size=1000)
            hf_dataset.save_to_disk(hf_dataset_path)
            print(f">>> Cache缓存构建成功: {hf_dataset_path}\n")

        # 6. 【核心显存清理】释放 Stage 1 模型，防止 Stage 2 加载时 OOM
        del model_stage1
        del stage1_loader
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    else:
        if accelerator.is_main_process:
            print(f"\n>>> [Stage 1] 检测到已有特征缓存: {hf_dataset_path}，跳过提取，直接读取。")

    # =====================================================================
    # STAGE 2: 加载特征，启动大模型分布式解码推理
    # =====================================================================
    if accelerator.is_main_process:
        print(f"\n>>> [Stage 2] 开始载入预计算特征...")
    
    dataset_raw_stage2 = []
    if accelerator.is_main_process:
        hf_dataset = HFDataset.load_from_disk(hf_dataset_path)
        embeds_dict = {item["audio"]: item["adapter_embeds"] for item in hf_dataset}
        
        print(f">>> Embeddings 缓存加载完毕，共 {len(embeds_dict)} 条...")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                audio_key = item.get("audio")
                
                # 与原始代码2的过滤逻辑保持一致
                if (item.get("src") in src_langs and 
                    item.get("tgt") in tgt_langs and 
                    audio_key in embeds_dict):
                    
                    item["adapter_embeds"] = embeds_dict[audio_key]
                    dataset_raw_stage2.append(item)
                    
        print(f">>> 数据合并完成，最终待解码总数据: {len(dataset_raw_stage2)}")

    dataset_list = [dataset_raw_stage2]
    broadcast_object_list(dataset_list)
    dataset_raw_stage2 = dataset_list[0]

    stage2_loader = DataLoader(
        EmbeddingsInferenceDataset(dataset_raw_stage2), 
        batch_size=batch_size_stage2, 
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # 载入 12B 解码模型
    if accelerator.is_main_process:
        print(f">>> 正在加载解码模型...")
        
    model_stage2 = AutoModel.from_pretrained(
        m_path, 
        use_vllm=use_vllm,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).eval()
    
    model_stage2, stage2_loader = accelerator.prepare(model_stage2, stage2_loader)

    bleu = BLEU(tokenize="flores200")
    local_stage2_results = []

    # 分布式推理
    accelerator.wait_for_everyone() 
    loop_start_time = time.perf_counter() 
    first_batch_time = 0                  

    with torch.inference_mode():
        for step, batch in enumerate(tqdm(stage2_loader, desc=f"GPU {accelerator.process_index}", disable=not accelerator.is_local_main_process)):
            embeds = batch["adapter_embeds"] 
            prompts = batch["prompt"]        

            model_engine = model_stage2.module if hasattr(model_stage2, "module") else model_stage2
            batch_start_time = time.perf_counter() 
            
            outs = model_engine.translate_batch_embeds(
                beam_search, 
                embeds, 
                prompts, 
                max_new_tokens=200, 
                use_vllm=use_vllm
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

                local_stage2_results.append({
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
                # 获取当前 batch 的最后一个样本的索引，确保数据完全对应
                last_idx = -1 
                
                print(f"\n{'='*40} BATCH CHECK (Last Item) {'='*40}")
                print(f"ID          : {batch['id'][last_idx]}")
                print(f"Audio       : {batch['audio'][last_idx]}")
                print(f"Prompt      : {batch['prompt'][last_idx]}")
                print(f"Src Lang    : {batch['src'][last_idx]}")
                print(f"Tgt Lang    : {batch['tgt'][last_idx]}")
                print(f"{'-'*40} Ground Truth {'-'*40}")
                print(f"ASR GT      : {batch['asr'][last_idx]}")
                print(f"S2TT GT     : {batch['s2tt'][last_idx]}")
                print(f"{'-'*40} Model Outputs {'-'*40}")
                print(f"Raw Response: {out}")         # 此时变量保持为循环最后一次的值
                print(f"ASR Result  : {asr_r}")       # 此时变量保持为循环最后一次的值
                print(f"S2TT Result : {s2tt_r}")      # 此时变量保持为循环最后一次的值
                print(f"{'='*105}\n")

    # 聚合所有卡的结果
    accelerator.wait_for_everyone() 
    loop_end_time = time.perf_counter() 
    all_stage2_data = gather_object(local_stage2_results)

    # 最终处理与计算指标（主进程）
    if accelerator.is_main_process:
        total_batches = len(stage2_loader)
        total_loop_time = loop_end_time - loop_start_time
        
        if total_batches > 1:
            rest_batches_time = total_loop_time - first_batch_time
            avg_batch_time = rest_batches_time / (total_batches - 1)
            vllm_startup_time = max(0, first_batch_time - avg_batch_time)
        else:
            vllm_startup_time = 0
            avg_batch_time = total_loop_time
            
        effective_loop_time = total_loop_time - vllm_startup_time

        # 去重
        unique_results_dict = {str(d["id"]): d for d in all_stage2_data}
        unique_data = list(unique_results_dict.values())
        
        # 计算每秒处理样本数 (Throughput)
        total_samples = len(unique_data)
        samples_per_second = total_samples / effective_loop_time if effective_loop_time > 0 else 0

        print(f"\n{'='*30} 耗时与吞吐统计 (Timing & Throughput) {'='*30}")
        print(f"处理总样本数 (去重后):    {total_samples} 条")
        print(f"单个 GPU 运行 Batch 数:  {total_batches}")
        print(f"For 循环总计物理耗时:    {total_loop_time:.2f} 秒")
        print(f"预估 vLLM 引擎启动耗时:  {vllm_startup_time:.2f} 秒")
        print(f"扣除启动后纯推理耗时:    {effective_loop_time:.2f} 秒")
        print(f"预估单个 Batch 平均耗时: {avg_batch_time:.2f} 秒")
        print(f"平均每秒处理样本数量:    {samples_per_second:.2f} samples/s (基于纯推理时间)")
        print(f"{'='*83}\n")

        print(f"汇总后原始数据量: {len(all_stage2_data)}")
        print(f"去重后最终数据量: {len(unique_data)}")

        # 写入结果文件
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for d in unique_data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        # 计算 BLEU
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