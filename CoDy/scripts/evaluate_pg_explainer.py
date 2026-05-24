import argparse
import os.path

import numpy as np
import pandas as pd

from cody.data import TrainTestDatasetParameters
from cody.utils import ProgressBar
from common import (add_dataset_arguments, add_wrapper_model_arguments, create_dataset_from_args,
                    create_tgnn_wrapper_from_args, parse_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Explainer Evaluation')
    add_dataset_arguments(parser)
    add_wrapper_model_arguments(parser)
    parser.add_argument('--results', required=True, type=str,
                        help='Path to the file containing the results of the factual explainer evaluation')

    args = parse_args(parser)

    dataset = create_dataset_from_args(args, TrainTestDatasetParameters(0.2, 0.6, 0.8, 500,
                                                                        500, 500))

    tgn_wrapper = create_tgnn_wrapper_from_args(args, dataset)
    tgn_wrapper.set_evaluation_mode(True)

    if not os.path.exists(args.results):
        raise FileNotFoundError('Failed to locate the file containing the results')

    if args.results.endswith('parquet'):
        raw_filepath = args.results.rstrip('parquet')
        results_df = pd.read_parquet(args.results)
    else:
        raw_filepath = args.results.rstrip('csv')
        results_df = pd.read_csv(args.results)
        results_df = results_df.iloc[:, 1:]

    scores_at_6 = []
    scores_at_12 = []
    scores_at_18 = []

    expl_events_only_at_6 = []
    expl_events_only_at_12 = []
    expl_events_only_at_18 = []
    progress_bar = ProgressBar(len(results_df), 'Adding information to results')
    for index, row in results_df.iterrows():
        explained_event_id = row['explained_event_id']

        event_ids = row['event_ids']

        exp_at_6 = np.array(event_ids[:6], dtype=int)
        exp_at_12 = np.array(event_ids[:12], dtype=int)
        exp_at_18 = np.array(event_ids[:18], dtype=int)

        score_at_6, _ = tgn_wrapper.compute_edge_probabilities_for_subgraph(explained_event_id,
                                                                       edges_to_drop=exp_at_6,
                                                                       result_as_logit=True)
        tgn_wrapper.reset_model()
        scores_at_6.append(score_at_6.detach().cpu().item())

        score_at_12, _ = tgn_wrapper.compute_edge_probabilities_for_subgraph(explained_event_id,
                                                                         edges_to_drop=exp_at_12,
                                                                         result_as_logit=True)
        tgn_wrapper.reset_model()
        scores_at_12.append(score_at_12.detach().cpu().item())

        score_at_18, _ = tgn_wrapper.compute_edge_probabilities_for_subgraph(explained_event_id,
                                                                         edges_to_drop=exp_at_18,
                                                                         result_as_logit=True)
        tgn_wrapper.reset_model()
        scores_at_18.append(score_at_18.detach().cpu().item())


        edges_to_drop = dataset.edge_ids[~np.isin(dataset.edge_ids,
                                                  np.concatenate([exp_at_6, np.array([explained_event_id])]))]
        prediction_at_6, _ = tgn_wrapper.compute_edge_probabilities_for_subgraph(explained_event_id, edges_to_drop, True,
                                                                     exp_at_6)
        tgn_wrapper.reset_model()
        expl_events_only_at_6.append(prediction_at_6.detach().cpu().item())


        edges_to_drop = dataset.edge_ids[~np.isin(dataset.edge_ids,
                                                  np.concatenate([exp_at_12, np.array([explained_event_id])]))]
        prediction_at_12, _ = tgn_wrapper.compute_edge_probabilities_for_subgraph(explained_event_id, edges_to_drop,
                                                                                 True,
                                                                                 exp_at_12)
        tgn_wrapper.reset_model()
        expl_events_only_at_12.append(prediction_at_12.detach().cpu().item())


        edges_to_drop = dataset.edge_ids[~np.isin(dataset.edge_ids,
                                                  np.concatenate([exp_at_18, np.array([explained_event_id])]))]
        prediction_at_18, _ = tgn_wrapper.compute_edge_probabilities_for_subgraph(explained_event_id, edges_to_drop,
                                                                                 True,
                                                                                 exp_at_18)
        tgn_wrapper.reset_model()
        expl_events_only_at_18.append(prediction_at_18.detach().cpu().item())
        progress_bar.next()

        if int(index) % 5 == 0:
            results_df['counterfactual_prediction_at_6'] = scores_at_6 + [None] * (len(results_df) - len(scores_at_6))
            results_df['counterfactual_prediction_at_12'] = scores_at_12 + [None] * (len(results_df) - len(scores_at_6))
            results_df['counterfactual_prediction_at_18'] = scores_at_18 + [None] * (len(results_df) - len(scores_at_6))
            results_df['prediction_explanation_events_only_at_6'] = expl_events_only_at_6 + [None] * (len(results_df) - len(scores_at_6))
            results_df['prediction_explanation_events_only_at_12'] = expl_events_only_at_12 + [None] * (len(results_df) - len(scores_at_6))
            results_df['prediction_explanation_events_only_at_18'] = expl_events_only_at_18 + [None] * (len(results_df) - len(scores_at_6))
            results_df.to_csv(f'{raw_filepath}csv')
            try:
                results_df.to_parquet(f'{raw_filepath}parquet')
                print(f'Intermediate results saved to {raw_filepath}parquet')
            except ModuleNotFoundError:
                print('Could not save to parquet format. Install pyarrow if you want to export to parquet format')



    progress_bar.close()
    results_df['counterfactual_prediction_at_6'] = scores_at_6
    results_df['counterfactual_prediction_at_12'] = scores_at_12
    results_df['counterfactual_prediction_at_18'] = scores_at_18
    results_df['prediction_explanation_events_only_at_6'] = expl_events_only_at_6
    results_df['prediction_explanation_events_only_at_12'] = expl_events_only_at_12
    results_df['prediction_explanation_events_only_at_18'] = expl_events_only_at_18

    results_df.to_csv(f'{raw_filepath}csv')
    print(f'Results saved to {raw_filepath}csv')
    try:
        results_df.to_parquet(f'{raw_filepath}parquet')
        print(f'Results saved to {raw_filepath}parquet')
    except ModuleNotFoundError:
        print('Could not save to parquet format. Install pyarrow if you want to export to parquet format')
