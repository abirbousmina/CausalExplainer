#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source "$SCRIPT_DIR/common.bash"

train_tgn() {
  MODEL_PATH="$PARENT_DIR/resources/models/$2"
  if [ ! -d "$MODEL_PATH" ]; then
    mkdir -p "$MODEL_PATH"
    echo "Created new directory for the final model and model checkpoints for the $2 dataset at $MODEL_PATH"
  fi
  if [ "$3" = "--bipartite" ]; then
    echo "Training $1 model for the $2 dataset (bipartite)..."
    python "$SCRIPT_DIR/train_tgnn.py" -d "$PROCESSED_DATA_DIR/$2" --bipartite --cuda --model_path "$MODEL_PATH/" -e 30 --type "$1"
  else
    echo "Training $1 model for the $2 dataset..."
    python "$SCRIPT_DIR/train_tgnn.py" -d "$PROCESSED_DATA_DIR/$2" --cuda --model_path "$MODEL_PATH/" -e 30 --type "$1"
  fi
}


show_help() {
  echo -e "
Train TGN Model script

Usage: bash $SCRIPT_DIR/train_tgn_model.bash ${RED}MODEL-TYPE${NC} ${RED}DATASET-NAME${NC} ${RED}--bipartite${NC}

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
  train_tgn "$1" "$2" "$3"
fi
