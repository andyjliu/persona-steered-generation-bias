from src.model_wrappers import vllm_wrapper, gpt_wrapper
from src.generate_stances import convert_to_binary_preferences
import argparse
import json
import pandas as pd
import numpy as np
import ast
import string

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True,
        help = 'model name')
    parser.add_argument('--filepath', type=str, default='src/personas_and_topics.json',
        help = 'filepath to save model views in')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    personas_and_topics = json.load(open(args.filepath, 'r'))
    topics = personas_and_topics['topics']
    info_dfs = {}
    
    # load client
    if 'gpt' in args.model_name.lower():
        client = gpt_wrapper(model = args.model_name, temp = 1)
    else:
        client = vllm_wrapper(model = args.model_name, max_tokens = 1, temp=1)

    for topic in topics.keys():
        survey_id = topics[topic]['survey_id']
        try:
            info_df = info_dfs[survey_id]
        except KeyError:
            info_df = pd.read_csv(f'opinions_qa/data/human_resp/American_Trends_Panel_W{survey_id}/info.csv')
            info_dfs[survey_id] = info_df
        row = info_df[info_df['key'] == topics[topic]['id']]
        prompt = f"Question: {row['question'].iloc[0]}\n"
        continuations = []

        for letter_idx, response in enumerate(ast.literal_eval(row['references'].iloc[0])):
            letter = string.ascii_uppercase[letter_idx]
            continuations.append(letter)
            prompt += f'{letter}. {response}\n'
        prompt += 'Answer: '
        logprobs = client.get_logprobs_of_continuations(prompt=prompt, continuations=continuations)
        
        opt = ast.literal_eval(row['option_mapping'].iloc[0])
        opt_values = {v: logprobs[i] for i, v in enumerate(list(opt.values())[:-1])}
        sort_key = ast.literal_eval(row['option_ordinal'].iloc[0])

        preferences = convert_to_binary_preferences(opt_values, opt, sort_key)
        lower_response, higher_response = preferences.keys()
        lower_value, higher_value = preferences.values()

        if topics[topic]['ordinal'] == 'lower':
            try:
                topics[topic]['model_base_response'][args.model_name] = lower_value
            except KeyError:
                topics[topic]['model_base_response'] = {args.model_name : lower_value}
        else:
            try:
                topics[topic]['model_base_response'][args.model_name] = higher_value
            except KeyError:
                topics[topic]['model_base_response'] = {args.model_name : higher_value}

    json.dump(personas_and_topics, open(args.filepath, 'w'), indent=2)