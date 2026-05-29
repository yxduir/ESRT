# ESRT

## Bandwidth-Efficient and Privacy-Preserving Edge-Cloud Many-to-Many Speech Translation.

ESRT supports many-to-many speech-to-text translation across **45 languages** (45 × 44 directions). It uses an edge-cloud split inference architecture to protect voice privacy and reduce bandwidth by transmitting only compressed acoustic features instead of raw audio.

[![arXiv](https://img.shields.io/badge/arXiv-2605.28642-b31b1b.svg)](https://arxiv.org/abs/2605.28642)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow "https://huggingface.co/yxdu")](https://huggingface.co/yxdu/ESRT-4B)

## Timeline

- **2026-05-29** — macOS CPU support added
- **2026-05-28** — ESRT-4B has been released on [Hugging Face](https://huggingface.co/yxdu/ESRT-4B) with GPU support.


## Setup

```bash
git clone https://github.com/yxduir/ESRT
cd ESRT
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt 

# uv pip install -r requirements_mac.txt
```
> **Note**: The GPU setup includes `vllm`. macOS uses a CPU backend with `transformers`.


## Test Data

```bash
hf download --repo-type dataset yxdu/fleurs_eng_test --local-dir ./fleurs_eng_test
```

## Inference

Two-stage inference: edge side and cloud side.

```bash
#Offline for Quick Testing

bash run_test.sh 
#bash run_test_mac.sh 

#Online deployment guide coming soon.
```
> **Note**: The GPU only supports 'bf16' inference..


## Training

Training code will be open-sourced in a future release. Validated on:

- **GPU**: NVIDIA A100 80GB × 8
- **NPU**: Huawei Ascend 910C 64GB × 8

## Supported Languages

| Family        | Languages                                                                                                                                                                                                         |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Afro-Asiatic  | Arabic, Hebrew                                                                                                                                                                                                    |
| Austroasiatic | Khmer, Vietnamese                                                                                                                                                                                                 |
| Austronesian  | Indonesian, Malay, Tagalog                                                                                                                                                                                        |
| Dravidian     | Tamil                                                                                                                                                                                                             |
| Indo-European | Bengali, Bulgarian, Catalan, Czech, Danish, Dutch, English, French, German, Greek, Hindi, Croatian, Italian, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Spanish, Swedish, Urdu |
| Japonic       | Japanese                                                                                                                                                                                                          |
| Koreanic      | Korean                                                                                                                                                                                                            |
| Kra–Dai      | Lao, Thai                                                                                                                                                                                                         |
| Sino-Tibetan  | Chinese, Burmese, Cantonese                                                                                                                                                                                       |
| Turkic        | Azerbaijani, Kazakh, Turkish, Uzbek                                                                                                                                                                               |
| Uralic        | Finnish, Hungarian                                                                                                                                                                                                |

## Citation

```bibtex
@misc{du2026bandwidthefficientprivacypreservingedgecloudmanytomany,
      title={Bandwidth-Efficient and Privacy-Preserving Edge-Cloud Many-to-Many Speech Translation}, 
      author={Yexing Du and Kaiyuan Liu and Youcheng Pan and Bo Yang and Ming Liu and Bing Qin and Yang Xiang},
      year={2026},
      eprint={2605.28642},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.28642}, 
}
```
