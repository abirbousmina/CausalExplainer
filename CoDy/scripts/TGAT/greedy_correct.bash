#!/usr/bin/env bash

# Greedy correct evaluation

#------ Change these parameters ----------

DATASET="uci_messages"
MODEL="TGAT"
ROOT_DIR="rootdir"
TIMEOUT="60"

#------------------------------------------

# These parameters are set automatically, according to the
SAMPLER="all"
PROCESSED_DATA_DIR="$ROOT_DIR/resources/datasets/processed"
RESULTS_DIR="$ROOT_DIR/resources/results"
MODEL_PATH="$ROOT_DIR/resources/models/$DATASET/$MODEL-$DATASET.pth"
EVAL_RESULTS_DIR="$RESULTS_DIR/$DATASET"
RESULTS_SAVE_DIR="$EVAL_RESULTS_DIR/greedy"
EXPLAINED_IDS_PATH="$EVAL_RESULTS_DIR/evaluation_event_ids_tgat.npy"

python "$ROOT_DIR/scripts/evaluate_cf_explainer.py" -d "$PROCESSED_DATA_DIR/$DATASET" --cuda --model "$MODEL_PATH" --explainer greedy --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR" --dynamic --predict_for_each_sample --sample_size 10 --candidates_size 64 --sampler "$SAMPLER" --sampler_model_path "$SAMPLER_MODEL_PATH" --max_time "$TIMEOUT" --optimize --type "TGAT"
