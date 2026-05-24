#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source "$SCRIPT_DIR/common.bash"

show_help() {
  echo -e "
Evaluation script

Usage: bash $SCRIPT_DIR/add_counterfactual_information.bash ${RED}MODEL-TYPE RESULTS-PATH DATASET-NAME${NC}

For the ${RED}MODEL-TYPE${NC} parameter provide the name of any of the model.
Possible values: ${CYAN}[${MODEL_TYPES[*]}]${NC}
For the ${RED}RESULTS-PATH${NC} parameter provide the path to the results file of a TGNNExplainer evaluation you want to test.
For the ${RED}DATASET-NAME${NC} parameter provide the name of any of the preprocessed datasets.
Possible values: ${CYAN}[${DATASET_NAMES[*]}]${NC}

Provide the ${RED}--bipartite${NC} flag as last argument if the dataset is bipartite
"
exit 1
}

if [ $# -lt 2 ]; then
  show_help
else
  test_exists "$1" "${MODEL_TYPES[@]}"
  test_exists "$2" "${DATASET_NAMES[@]}"
  TGNN_PATH="$PARENT_DIR/resources/models/$3/$1-$3.pth"
  if [ "$4" = "--bipartite" ];then
    python "$SCRIPT_DIR/evaluate_factual_subgraphs.py" -d "$PROCESSED_DATA_DIR/$3" --bipartite --cuda --model "$TGNN_PATH" --type "$1" --results "$2"
  else
    python "$SCRIPT_DIR/evaluate_factual_subgraphs.py" -d "$PROCESSED_DATA_DIR/$3" --cuda --model "$TGNN_PATH" --type "$1" --results "$2"
  fi
fi
