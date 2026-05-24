import numpy as np
import torch
from cody.implementations.tgn import TGNWrapper, to_data_object
from cody.utils import ProgressBar
from TGN.model.tgn import TGN
from cody.data import BatchData, ContinuousTimeDynamicGraphDataset
from submodules.tgn.TGN.utils.utils import get_neighbor_finder


class TGATWrapper(TGNWrapper):

    def __init__(self, model: TGN, dataset: ContinuousTimeDynamicGraphDataset, num_hops: int, model_name: str,
                 device: str = 'cpu', n_neighbors: int = 20, batch_size: int = 32, checkpoint_path: str = None):
        super().__init__(model=model, dataset=dataset, num_hops=num_hops, model_name=model_name, device=device,
                         n_neighbors=n_neighbors, batch_size=batch_size, checkpoint_path=checkpoint_path,
                         use_memory=False)

    def initialize(self, event_id: int, show_progress: bool = False, memory_label: str = None):
        pass

    def post_batch_cleanup(self):
        pass

    def rollout_until_event(self, event_id: int = None, batch_data: BatchData = None,
                            progress_bar: ProgressBar = None, event_ids_to_rollout: np.ndarray = None) -> None:
        self.latest_event_id = event_id + 1

    def compute_edge_probabilities_for_subgraph(self, event_id, edges_to_drop: np.ndarray,
                                                result_as_logit: bool = False,
                                                event_ids_to_rollout: np.ndarray = None) -> (
    torch.Tensor, torch.Tensor):
        # Insert a new neighborhood finder so that the model does not consider dropped edges
        original_ngh_finder = self.model.neighbor_finder
        self.model.set_neighbor_finder(get_neighbor_finder(to_data_object(self.dataset, edges_to_drop=edges_to_drop),
                                                           uniform=False))

        source_node, target_node, timestamp, edge_id = self.extract_event_information(event_ids=event_id)
        probabilities = self.compute_edge_probabilities(source_node, target_node, timestamp, edge_id,
                                                        result_as_logit=result_as_logit, perform_memory_update=False)
        # Reinsert the original neighborhood finder so that the model can be used as usual
        self.model.set_neighbor_finder(original_ngh_finder)
        return probabilities

    def get_memory(self):
        return None

    def detach_memory(self):
        pass

    def restore_memory(self, memory_backup, event_id):
        pass

    def reset_model(self):
        self.reset_latest_event_id()
