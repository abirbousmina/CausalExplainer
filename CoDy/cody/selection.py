from dataclasses import dataclass
from typing import List

import torch
import numpy as np
import pandas as pd
from cody.constants import COL_ID, COL_SUBGRAPH_DISTANCE, COL_TIMESTAMP
from cody.embedding import Embedding


def create_embedding_model(emb: Embedding, model_path: str = None, device: str = 'cpu'):
    embedding_model = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(emb.single_dimension, 32),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(32, 32)
    )
    if model_path is not None:
        embedding_model.load_state_dict(torch.load(model_path))
    embedding_model.to(torch.device(device))
    return embedding_model


def filter_subgraph(base_event_id: int, excluded_events: np.ndarray, subgraph: pd.DataFrame,
                    known_cf_examples: List[np.ndarray] | None = None) -> pd.DataFrame:
    excluded_events = np.concatenate((excluded_events, np.array([base_event_id])))
    filtered_subgraph = subgraph[~subgraph[COL_ID].isin(excluded_events)]
    # Make sure that events that would lead to an already known cf example are not sampled as candidates
    further_events_to_exclude = []
    if known_cf_examples is not None:
        for cf_example in known_cf_examples:
            total_occurrences = np.sum(np.isin(excluded_events, cf_example))
            if total_occurrences >= len(cf_example) - 1:
                already_excluded_events = excluded_events[np.isin(excluded_events, cf_example)]
                event_to_exclude = cf_example[~np.isin(cf_example, already_excluded_events)][0]
                further_events_to_exclude.append(event_to_exclude)
    return filtered_subgraph[~filtered_subgraph[COL_ID].isin(further_events_to_exclude)]


class SelectionPolicy:

    def __init__(self, subgraph: pd.DataFrame):
        assert len(subgraph) > 0
        self.subgraph = subgraph

    def sample(self, base_event_id: int, excluded_events: np.ndarray, size: int,
               known_cf_examples: List[np.ndarray] | None = None) -> np.ndarray:
        ranked_subgraph = self.rank_subgraph(base_event_id, excluded_events, known_cf_examples)
        if len(ranked_subgraph) < size:
            return ranked_subgraph
        return ranked_subgraph[:size]

    def rank_subgraph(self, base_event_id: int, excluded_events: np.ndarray,
                      known_cf_examples: List[np.ndarray] | None = None):
        raise NotImplementedError


class RandomSelectionPolicy(SelectionPolicy):

    def rank_subgraph(self, base_event_id: int, excluded_events: np.ndarray,
                      known_cf_examples: List[np.ndarray] | None = None):
        filtered_subgraph = filter_subgraph(base_event_id, excluded_events, self.subgraph, known_cf_examples)
        return filtered_subgraph.sample(frac=1)[COL_ID].to_numpy()


class TemporalSelectionPolicy(SelectionPolicy):

    def rank_subgraph(self, base_event_id: int, excluded_events: np.ndarray,
                      known_cf_examples: List[np.ndarray] | None = None):
        filtered_subgraph = filter_subgraph(base_event_id, excluded_events, self.subgraph, known_cf_examples)
        return filtered_subgraph[COL_ID].to_numpy()[::-1]


class SpatioTemporalSelectionPolicy(SelectionPolicy):

    def rank_subgraph(self, base_event_id: int, excluded_events: np.ndarray,
                      known_cf_examples: List[np.ndarray] | None = None):
        filtered_subgraph = filter_subgraph(base_event_id, excluded_events, self.subgraph, known_cf_examples)
        sorted_subgraph = filtered_subgraph.sort_values(by=[COL_SUBGRAPH_DISTANCE, COL_TIMESTAMP],
                                                        ascending=[True, False])
        return sorted_subgraph[COL_ID].to_numpy()

class LocalEventImpactSelectionPolicy(SelectionPolicy):

    def __init__(self, subgraph: pd.DataFrame):
        super().__init__(subgraph)
        self.subgraph['weight'] = 0

    def set_event_weight(self, event_id: int, weight: float):
        self.subgraph.loc[self.subgraph[COL_ID] == event_id, 'weight'] = weight

    def rank_subgraph(self, base_event_id: int, excluded_events: np.ndarray,
                      known_cf_examples: List[np.ndarray] | None = None):
        filtered_subgraph = filter_subgraph(base_event_id, excluded_events, self.subgraph, known_cf_examples)
        sorted_subgraph = filtered_subgraph.sort_values(by='weight', ascending=False)
        return sorted_subgraph[COL_ID].to_numpy()
