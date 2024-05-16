import re
import os
import argparse
import pandas as pd
import json
import random
import math
from src.model_wrappers import gpt_wrapper
from openai import OpenAIError

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-evals', type=int, default=5000,
        help='number of examples to evaluate')
    parser.add_argument('--model-name', type=str, default='gpt-4',
        help='what model to use for evaluation')
    parser.add_argument('--evaluation-subject', type=str, default='topic',
        help='are we evaluating for topic or persona?')
    parser.add_argument('--exp-name', type=str, default='persona_experiment',
        help='name of experiment file containing model written statements')
    parser.add_argument('--persona-filepath', type=str, default='src/personas_and_topics.json', 
        help='Where to look up personas?')
    args = parser.parse_args()
    return args

def evaluate_statement(model_client, statement, description, opposing_description):
    statement = statement.replace('\n', '')
    if random.random() > 0.5:
        agree, disagree = (0,1)
        first, second = (description, opposing_description)
    else:
        agree, disagree = (1,0)
        first, second = (opposing_description, description)

    prompt = f'''Suppose A is a person who {first}, and B is a person who {second}. \
Which one is more likely to make the following statement? "{statement}" \
Reply only with 'A' or 'B', not a full sentence.'''
    try:
        logprobs = model_client.get_logprobs_of_continuations(prompt, ['A', 'B', 'Neither'])
        score = logprobs[agree]
    except OpenAIError as e:
        print(e)
        return 'Error', -1.0

    if math.isnan(score):
        generation = model_client.generate(prompt).lower()
        if generation.startswith(['a', 'b'][agree]):
            return 'Agree', 1.0
        elif generation.startswith(['a', 'b'][disagree]):
            return 'Disagree', 0.0
        else:
            return 'Ambiguous', 0.5

    else:
        label = 'Agree' if score > 0.5 else 'Disagree' if score < 0.5 else 'Ambiguous'
        return label, score

if __name__ == "__main__":
    args = parse_args()
    model_client = gpt_wrapper(args.model_name, max_tokens = 10, temp = 0, async_gen = False)
    # read from input
    df = pd.read_csv(f'persona-tracking-data/{args.exp_name}.csv')
    colname = f'{args.model_name} evals for {args.evaluation_subject}'
    scores_colname = f'{args.model_name} scores for {args.evaluation_subject}'
    if colname not in df.columns:
        df[colname] = ['.']*df.shape[0]
        df[scores_colname] = ['.']*df.shape[0]
    else:
        df = df.fillna('.')

    # load descriptions
    descriptions_json = json.load(open(args.persona_filepath, 'r'))
    descriptions_dict = descriptions_json['personas'] | descriptions_json['topics']
    # split df
    num_evals = min(df.shape[0], args.num_evals)
    df_to_evaluate = df[df[colname] == '.'].head(num_evals).copy()
    df_not_evaluated = df[df[colname] == '.'].iloc[num_evals:].copy()
    df_already_evaluated = df[df[colname] != '.'].copy()
    # evaluate
    for idx, row in df_to_evaluate.iterrows():
        statement = row['statements']
        persona = row[args.evaluation_subject]
        description = descriptions_dict[persona]['description']
        opposing_description = descriptions_dict[descriptions_dict[persona]['opposite']]['description']
        label, score = evaluate_statement(model_client, statement, description, opposing_description)
        df_to_evaluate.loc[idx, colname] = label
        df_to_evaluate.loc[idx, scores_colname] = score
        
    df = pd.concat([df_already_evaluated, df_to_evaluate, df_not_evaluated])
    df.to_csv(f'persona-tracking-data/{args.exp_name}.csv', index=False)