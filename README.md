# ESRT

## Bandwidth-Efficient and Privacy-Preserving Edge-Cloud Many-to-Many Speech Translation.

ESRT supports many-to-many speech-to-text translation across **45 languages** (45 × 44 directions). It uses an edge-cloud split inference architecture to protect voice privacy and reduce bandwidth by transmitting only compressed acoustic features instead of raw audio.

[![arXiv](https://img.shields.io/badge/arXiv-2503.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2503.xxxxx)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow "https://huggingface.co/yxdu")](https://huggingface.co/yxdu)

## Setup

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Test Data

```bash
git clone https://huggingface.co/datasets/yxdu/fleurs_eng_test ./fleurs_eng_test
```

## Inference

```bash
python test_inference.py
```

Online deployment guide coming soon.

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

```
