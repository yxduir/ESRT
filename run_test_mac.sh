#!/bin/bash

#LANGUAGES
ALL_LANGS_LIST="ara azj bul ben cat ces dan deu ell eng \
                spa fas fin fra heb hin hrv hun ind ita \
                jpn kaz khm kor lao msa mya nob nld pol \
                por ron rus slk slv swe tam tha tgl tur \
                urd uzb vie yue cmn"

DEMO_LANGS_LIST="cmn fra ara"

MLLM_PATH="yxdu/ESRT-4B"
JSONL_PATH="./fleurs_eng_test/srt_test_eng.jsonl"
OUTPUT_PATH="fleurs_eng_test_eng_3langs.jsonl"

MODEL_NAME=$(basename "$MLLM_PATH")
DATA_NAME=$(basename "$JSONL_PATH" .jsonl)
EDGE_CACHE_PATH="./cache_${MODEL_NAME}_${DATA_NAME}"
EDGE_BATCH_SIZE=1
CLOUD_BATCH_SIZE=16 # lower batch size to avoid OOM
CLOUD_BEAM_SEARCH=1
SRC_LANGS="eng"
TGT_LANGS=$DEMO_LANGS_LIST
MAX_NEW_TOKENS=200

python ./test_inference.py \
    --mllm_path $MLLM_PATH \
    --jsonl_path $JSONL_PATH \
    --output_path $OUTPUT_PATH \
    --edge_batch_size $EDGE_BATCH_SIZE \
    --edge_cache_path $EDGE_CACHE_PATH \
    --cloud_batch_size $CLOUD_BATCH_SIZE \
    --cloud_beam_search $CLOUD_BEAM_SEARCH \
    --src_langs $SRC_LANGS \
    --tgt_langs $TGT_LANGS \
    --max_new_tokens $MAX_NEW_TOKENS
