import pandas as pd
import numpy as np
import os
import argparse
import json
from metrics import diversity, human, congruity, toxicity, model_base, entailment_diversity, individuation, exaggeration

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='persona_experiment')
    parser.add_argument('--metrics', type=str, default=None, nargs='*',
        help = 'additional metrics to compute. valid arguments: "diversity", "human", "toxicity", "model_base", "entailment_diversity", "individuation", "exaggeration"')
    parser.add_argument('--persona-filepath', type=str, default='src/personas_and_topics.json', 
        help='Where to look up personas?')
    args = parser.parse_args()
    return(args)
    
def get_agreement(group, cols):
    # group: group from df.groupby operation
    # cols: list of evaluation columns to aggregate over
    # returns total agreement for each eval column
    to_return = {}
    for col in cols:
        num = (group[col] == 'Agree').sum()
        denom = group[col].count() - (group[col] == 'Error').sum()
        col_id = col.split(' ')[-1]
        col_name = f'{col_id} steerability'
        to_return[col_name] = num/denom
    return(pd.Series(to_return))

if __name__ == "__main__":
    args = parse_args()
    filepath = f'persona-tracking-results/{args.exp_name}.csv'
    exp_data = pd.read_csv(f'persona-tracking-data/{args.exp_name}.csv')
    personas = json.load(open(args.persona_filepath, 'r'))

    # get persona agreement
    if not os.path.exists(filepath):
        eval_cols = [col for col in exp_data.columns if 'evals' in col]
        grouped_stats = exp_data.groupby(['model', 'persona', 'topic', 'first'])
        grouped_stats = grouped_stats.apply(lambda x: get_agreement(x, eval_cols)).reset_index()
        grouped_stats.to_csv(filepath, index=False)

    # compute metrics
    grouped_stats = pd.read_csv(filepath)
    for metric in args.metrics:
        metric_function = globals().get(metric, None)
        assert metric_function is not None and callable(metric_function), 'invalid metric name'
        grouped_stats[metric] = metric_function(exp_data, grouped_stats, args.exp_name, personas)

    grouped_stats.to_csv(filepath, index=False)