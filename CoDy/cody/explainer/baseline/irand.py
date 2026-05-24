from typing import Optional

import numpy as np

from cody.constants import COL_ID, EXPLAINED_EVENT_MEMORY_LABEL
from cody.explainer.base import Explainer, CounterFactualExample
from cody.implementations.connector import TGNNWrapper


class IRandExplainer(Explainer):

    def __init__(self, tgnn_wrapper: TGNNWrapper, candidates_size: int = 75, verbose: bool = False,
                 approximate_predictions: bool = True, max_iterations: Optional[int] = None):
        super().__init__(tgnn_wrapper, candidates_size=candidates_size, verbose=verbose,
                         approximate_predictions=approximate_predictions)
        if max_iterations is None:
            max_iterations = candidates_size
        self.max_iterations = max_iterations

    def explain(self, explained_event_id: int) -> CounterFactualExample:
        subgraph = self.subgraph_generator.get_fixed_size_k_hop_temporal_subgraph(num_hops=self.num_hops,
                                                                                  base_event_id=explained_event_id,
                                                                                  size=self.candidates_size)
        min_event_id = subgraph[COL_ID].min() - 1  # One less since we do not want to simulate the minimal event

        self.tgnn.set_evaluation_mode(True)
        self.tgnn.reset_model()
        original_prediction = self.calculate_original_score(explained_event_id, min_event_id)

        best_cf_example = CounterFactualExample(explained_event_id, original_prediction,  original_prediction, False,
                                                 np.array([]), np.array([]))

        for i in range(1, self.max_iterations + 1):
            subgraph_sample = subgraph[COL_ID].sample(n=i, replace=False).to_numpy()
            prediction = self.calculate_subgraph_prediction(candidate_events=subgraph[COL_ID].to_numpy(),
                                                            cf_example_events=subgraph_sample[:i - 1],
                                                            explained_event_id=explained_event_id,
                                                            candidate_event_id=subgraph_sample[i - 1],
                                                            original_prediction=original_prediction,
                                                            memory_label=EXPLAINED_EVENT_MEMORY_LABEL)
            if prediction * original_prediction < 0:
                best_cf_example = CounterFactualExample(explained_event_id, original_prediction, prediction, True,
                                                        subgraph_sample, np.array([None] * i))
                break

            # Fallback to best identified explanation that is not counterfactual
            if original_prediction > 0:
                if ((original_prediction - best_cf_example.counterfactual_prediction) <
                        (original_prediction - prediction)):
                    best_cf_example = CounterFactualExample(explained_event_id, original_prediction, prediction, False,
                                                        subgraph_sample, np.array([None] * i))
            else:
                if ((original_prediction - best_cf_example.counterfactual_prediction) >
                        (original_prediction - prediction)):
                    best_cf_example = CounterFactualExample(explained_event_id, original_prediction, prediction, False,
                                                            subgraph_sample, np.array([None] * i))

        self.tgnn.remove_memory_backup(EXPLAINED_EVENT_MEMORY_LABEL)
        self.tgnn.reset_model()
        return best_cf_example
