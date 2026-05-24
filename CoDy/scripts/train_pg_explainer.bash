#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source "$SCRIPT_DIR/common.bash"

train_pg_explainer() {
  TGNN_PATH="$PARENT_DIR/resources/models/$2/$1-$2.pth"
  MODEL_PATH="$PARENT_DIR/resources/models/$2/pg_explainer"
  if [ ! -d "$MODEL_PATH" ]; then
    mkdir -p "$MODEL_PATH"
    echo "Created new directory for the final PGExplainer model and model checkpoints for the $2 dataset at $MODEL_PATH"
  fi

  if [ "$3" = "--bipartite" ]; then
    echo "Training PGExplainer model for the $2 dataset (bipartite)..."
    python "$SCRIPT_DIR/train_pgexplainer.py" -d "$PROCESSED_DATA_DIR/$2" --bipartite --cuda --model_path "$MODEL_PATH" --epochs 100 --model $TGNN_PATH --type "$1" --candidates_size 30
  else
    echo "Training PGExplainer model for the $2 dataset..."
    python "$SCRIPT_DIR/train_pgexplainer.py" -d "$PROCESSED_DATA_DIR/$2" --cuda --model_path "$MODEL_PATH" --epochs 100 --model $TGNN_PATH --type "$1" --candidates_size 30
  fi
}


show_help() {
  echo -e "
PGExplainer training script

Usage: bash $SCRIPT_DIR/train_pg_explainer.bash ${RED}MODEL-TYPE${NC} ${RED}DATASET-NAME${NC} ${RED}--bipartite${NC}

For the ${RED}MODEL-TYPE${NC} parameter provide the name of any of the model.
Possible values: ${CYAN}[${MODEL_TYPES[*]}]${NC}

For the ${RED}DATASET-NAME${NC} parameter provide the name of any of the preprocessed datasets.
Possible values: ${CYAN}[${DATASET_NAMES[*]}]${NC}
Provide the ${RED}--bipartite${NC} flag if the dataset is bipartite
"
exit 1
}

if [ $# -eq 0 ]; then
  show_help
else
  test_exists "$1" "${MODEL_TYPES[@]}"
  test_exists "$2" "${DATASET_NAMES[@]}"
  train_pg_explainer "$1" "$2" "$3"
fi
