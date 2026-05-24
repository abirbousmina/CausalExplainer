#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source "$SCRIPT_DIR/common.bash"

evaluate_explainer() {
  TGNN_PATH="$PARENT_DIR/resources/models/$2/$1-$2.pth"
  EVAL_RESULTS_DIR="$RESULTS_DIR/$2"
  RESULTS_SAVE_DIR="$EVAL_RESULTS_DIR/$3"
  EXPLAINED_IDS_PATH="$EVAL_RESULTS_DIR/$1_evaluation_event_ids_wrong_prediction_only.npy"
  if [ ! -d "$EVAL_RESULTS_DIR" ]; then
    mkdir -p "$EVAL_RESULTS_DIR"
    echo "Created new directory for the evaluation results for the $2 dataset at $MODEL_PATH"
  fi
  mkdir -p "$RESULTS_SAVE_DIR"

  echo "Starting evaluation for dataset $2 with explainer $3"
  case $3 in
  pg_explainer)
    PG_EXP_MODEL_PATH="$PARENT_DIR/resources/models/$2/pg_explainer/$1_final.pth"
    if [ "$5" = "--bipartite" ];then
      python "$SCRIPT_DIR/evaluate_factual_explainer.py" -d "$PROCESSED_DATA_DIR/$2" --bipartite --cuda --explainer_model_path "$PG_EXP_MODEL_PATH" --model "$TGNN_PATH" --type "$1" --candidates_size 30 --explainer pg_explainer --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR/results_$1_$2_$3_wrong_only.csv" --wrong_predictions_only --max_time "$4"
    else
      python "$SCRIPT_DIR/evaluate_factual_explainer.py" -d "$PROCESSED_DATA_DIR/$2" --cuda --explainer_model_path "$PG_EXP_MODEL_PATH" --model "$TGNN_PATH" --type "$1" --candidates_size 30 --explainer pg_explainer --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR/results_$1_$2_$3_wrong_only.csv" --wrong_predictions_only --max_time "$4"
    fi
    ;;
  tgnnexplainer)
    PG_EXP_MODEL_PATH="$PARENT_DIR/resources/models/$2/pg_explainer/$1_final.pth"
    if [ "$5" = "--bipartite" ];then
      python "$SCRIPT_DIR/evaluate_factual_explainer.py" -d "$PROCESSED_DATA_DIR/$2" --bipartite --cuda --explainer_model_path "$PG_EXP_MODEL_PATH" --model "$TGNN_PATH" --type "$1" --candidates_size 30 --explainer t_gnnexplainer --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR/results_$1_$2_$3_wrong_only.csv" --rollout 500 --mcts_save_dir "$RESULTS_SAVE_DIR/" --wrong_predictions_only --max_time "$4"
    else
      python "$SCRIPT_DIR/evaluate_factual_explainer.py" -d "$PROCESSED_DATA_DIR/$2" --cuda --explainer_model_path "$PG_EXP_MODEL_PATH" --model "$TGNN_PATH" --type "$1" --candidates_size 30 --explainer t_gnnexplainer --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR/results_$1_$2_$3_wrong_only.csv" --rollout 500 --mcts_save_dir "$RESULTS_SAVE_DIR/" --wrong_predictions_only --max_time "$4"
    fi
    ;;
  greedy)
    echo "Selected sampler $4"
    if [ "$6" = "--bipartite" ];then
      python "$SCRIPT_DIR/evaluate_cf_explainer.py" -d "$PROCESSED_DATA_DIR/$2" --bipartite --cuda --model "$TGNN_PATH" --type "$1" --explainer greedy --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR" --dynamic --sample_size 10 --candidates_size 64 --sampler "$4" --wrong_predictions_only --max_time "$5" --optimize
    else
      python "$SCRIPT_DIR/evaluate_cf_explainer.py" -d "$PROCESSED_DATA_DIR/$2" --cuda --model "$TGNN_PATH" --type "$1" --explainer greedy --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR" --dynamic --sample_size 10 --candidates_size 64 --sampler "$4" --wrong_predictions_only --max_time "$5" --optimize
    fi
    ;;
  cody)
    echo "Selected sampler $4"
    if [ "$6" = "--bipartite" ];then
      python "$SCRIPT_DIR/evaluate_cf_explainer.py" -d "$PROCESSED_DATA_DIR/$2" --bipartite --cuda --model "$TGNN_PATH" --type "$1" --explainer cody --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR" --dynamic --candidates_size 64 --sampler "$4" --wrong_predictions_only --max_time "$5" --max_steps 300 --optimize
    else
      python "$SCRIPT_DIR/evaluate_cf_explainer.py" -d "$PROCESSED_DATA_DIR/$2" --cuda --model "$TGNN_PATH" --type "$1" --explainer cody --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR" --dynamic --candidates_size 64 --sampler "$4" --wrong_predictions_only --max_time "$5" --max_steps 300 --optimize
    fi
    ;;
  *)
    show_help
    ;;
  esac
}


show_help() {
  echo -e "
Evaluation script

Usage: bash $SCRIPT_DIR/evaluate_wrong_predictions_only.bash ${RED}MODEL-TYPE DATASET-NAME EXPLAINER-NAME [SAMPLER-NAME] --bipartite${NC}

For the ${RED}MODEL-TYPE${NC} parameter provide the name of any of the model.
Possible values: ${CYAN}[${MODEL_TYPES[*]}]${NC}
For the ${RED}DATASET-NAME${NC} parameter provide the name of any of the preprocessed datasets.
Possible values: ${CYAN}[${DATASET_NAMES[*]}]${NC}
For the ${RED}EXPLAINER-NAME${NC} parameter provide the name of any of the possible explainers.
Possible values: ${CYAN}[${EXPLAINER_TYPES[*]}]${NC}
Optional: For the ${RED}SAMPLER-NAME${NC} parameter provide the name of any of the possible samplers (Counterfactual Explainers only).
Possible values: ${CYAN}[${SAMPLER_TYPES[*]}]${NC}

Provide the ${RED}--bipartite${NC} flag if the dataset is bipartite
"
exit 1
}

if [ $# -lt 2 ]; then
  show_help
else
  test_exists "$1" "${MODEL_TYPES[@]}"
  test_exists "$2" "${DATASET_NAMES[@]}"
  test_exists "$3" "${EXPLAINER_TYPES[@]}"
  time="600" # 10 Hours default value
  if value_in_array "$3" "${COUNTERFACTUAL_EXPLAINER_TYPES[@]}"; then
    if [ $# -gt 3 ]; then
      time="$5"
      echo "Concluding evaluation after maximum time of $time minutes"
    fi
    test_exists "$4" "${ALL_SAMPLER_TYPES[@]}"
    echo "Evaluating explainer $3 with sampler $4 on dataset $2"
    evaluate_explainer "$1" "$2" "$3" "$4" "$time" "$6"
  elif value_in_array "$3" "${FACTUAL_EXPLAINER_TYPES[@]}"; then
    if [ $# -gt 2 ]; then
      time="$4"
      echo "Concluding evaluation after maximum time of $time minutes"
    fi
    echo "Evaluating explainer $3 on dataset $2"
    evaluate_explainer "$1" "$2" "$3" "$4" "$5"
  else
    show_help
  fi
fi
