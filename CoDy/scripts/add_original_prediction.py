import json
import math

from tqdm import tqdm

from cody.data import TrainTestDatasetParameters
from common import add_dataset_arguments, add_wrapper_model_arguments, parse_args, create_tgnn_wrapper_from_args, \
    create_dataset_from_args
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Explainer Evaluation')
    add_dataset_arguments(parser)
    add_wrapper_model_arguments(parser)
    parser.add_argument('--results', required=True, type=str,
                        help='Path to store the predictions to.')

    args = parse_args(parser)

    dataset = create_dataset_from_args(args, TrainTestDatasetParameters(0.2, 0.6, 0.8, 500,
                                                                        500, 500))

    tgn_wrapper = create_tgnn_wrapper_from_args(args, dataset)

    tgn_wrapper.reset_model()
    tgn_wrapper.set_evaluation_mode(True)

    batch_size = tgn_wrapper.batch_size
    number_of_instances = len(dataset.source_node_ids)

    predictions = []

    num_batches = math.ceil(number_of_instances / batch_size)

    for batch_id in tqdm(range(32), total=num_batches):
        start_id = batch_id * batch_size
        end_id = min(number_of_instances, start_id + batch_size)

        batch_data = dataset.get_batch_data(start_id, end_id)

        res, _ = tgn_wrapper.compute_edge_probabilities(source_nodes=batch_data.source_node_ids,
                                                        target_nodes=batch_data.target_node_ids,
                                                        edge_timestamps=batch_data.timestamps,
                                                        edge_ids=batch_data.edge_ids,
                                                        result_as_logit=True)

        new_res = res.cpu().detach().numpy().tolist()

        predictions.extend(new_res)

    json.dump(predictions, open(args.results, 'w+'))
