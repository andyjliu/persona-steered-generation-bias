import pandas as pd
import numpy as np
import ast
import math
import json
import re
import argparse
from src.model_wrappers import gpt_wrapper

def get_unique_id(d, s):
    if s not in d.keys():
        return(s)
    else:
        count = 1
        while s in d.keys():
            s = f'{s}{count}'
            count += 1
        return(s)

def convert_to_binary_preferences(response_dict, ordinal_dict, sort_key):

    def get_total_agreement(response_dict, ordinal_dict, values_list):
        return(sum([response_dict[ordinal_dict[v]] for v in values_list]))

    values = list(ordinal_dict.keys())
    values.remove(99.0) # refusal
    sort_key, values = (list(t) for t in zip(*sorted(zip(sort_key, values))))
    if len(values) % 2 == 0:
        below_threshold = values[:len(values)//2]
        above_threshold = values[len(values)//2:] 
    else:
        below_threshold = values[:len(values)//2]
        above_threshold = values[len(values)//2 + 1:]

    below_agreement = get_total_agreement(response_dict, ordinal_dict, below_threshold)
    above_agreement = get_total_agreement(response_dict, ordinal_dict, above_threshold)
    return({ordinal_dict[values[0]] : (below_agreement)/(below_agreement+above_agreement),
            ordinal_dict[values[-1]] : (above_agreement)/(below_agreement+above_agreement)})

def get_persona_agreement_rates(info_df, response_df, idx, key, lower_response, higher_response, personas):
    rates = {lower_response:{}, higher_response:{}}
    persona_list = ['base'] + list(personas.keys())
    for persona in persona_list:
        if persona == 'base':
            persona_df = df
        else:
            field = personas[persona]['field']
            values = personas[persona]['values']
            persona_df = df[df[field].isin(values)]

        opt = ast.literal_eval(info_df['option_mapping'][idx])
        sort_key = ast.literal_eval(info_df['option_ordinal'][idx])
        question = info_df['question'][idx]

        value_counts = dict(df[key].value_counts(normalize=True))
        persona_value_counts = dict(persona_df[key].value_counts(normalize=True))
        for k in persona_value_counts:
            value_counts[k] = persona_value_counts[k]

        opt_values = {k : value_counts[k] for k in opt.values()}
        preferences = convert_to_binary_preferences(opt_values, opt, sort_key)
        rates[lower_response][persona] = preferences[lower_response]
        rates[higher_response][persona] = preferences[higher_response]
    return(rates)

def get_stances(info_df, response_df, personas, survey_id):
    gpt_short = gpt_wrapper('gpt-4-0613', temp=1, max_tokens=5, async_gen=False, stop_sequences=['.', '<|endoftext|>'])
    gpt_long = gpt_wrapper('gpt-4-0613', temp=0, max_tokens=50, async_gen=False, stop_sequences=['.', '<|endoftext|>'])
    topics = {}

    # default persona
    for idx, key in enumerate(info_df['key']):
        try:
            opt = ast.literal_eval(info_df['option_mapping'][idx])
            sort_key = ast.literal_eval(info_df['option_ordinal'][idx])
            question = info_df['question'][idx]

            value_counts = response_df[key].value_counts(normalize=True)
            opt_values = {k : value_counts[k] for k in opt.values()}

            preferences = convert_to_binary_preferences(opt_values, opt, sort_key)
            lower_response, higher_response = preferences.keys()
            lower_value, higher_value = preferences.values()

            pattern = r'[\. ]'
            lower_shortname = re.sub(pattern, '', gpt_short.generate(f'A descriptive identifier of two words of someone who would answer {lower_response} to the question "{question}" is')).lower()
            higher_shortname = re.sub(pattern, '', gpt_short.generate(f'A descriptive identifier of two words of someone who would answer {higher_response} to the question "{question}" is')).lower()
            lower_shortname = get_unique_id(topics, re.sub('[^a-zA-Z]+', '', lower_shortname))
            higher_shortname = get_unique_id(topics, re.sub('[^a-zA-Z]+', '', higher_shortname))
               
            lower_description = gpt_long.generate(f'Remain as close to the original wording as possible. A person who responds {lower_response} to the question "{question}" is a person who')
            higher_description = gpt_long.generate(f'Remain as close to the original wording as possible. A person who responds {higher_response} to the question "{question}" is a person who')
            rates = get_persona_agreement_rates(info_df, response_df, idx, key, lower_response, higher_response, personas)
            topics[lower_shortname] = {'id':key, 'question':question, 'response':lower_response, 'ordinal':'lower', 'description':lower_description, 'rate':rates[lower_response], 'opposite':higher_shortname, 'survey_id':survey_id}
            topics[higher_shortname] = {'id':key, 'question':question, 'response':higher_response, 'ordinal':'higher', 'description':higher_description, 'rate':rates[higher_response], 'opposite':lower_shortname, 'survey_id':survey_id}
        
        except KeyError:
            pass

    return(topics)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-filepath', type=str, default='src/personas.json',
        help = 'name of json file to add stances to')
    parser.add_argument('--output-filepath', type=str, default='src/personas.json',
        help = 'name of json file to save persona and stance data')
    parser.add_argument('--survey-id', type=str, default=29,
        help = 'ID of ATP Wave to use')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    personas = json.load(open(args.input_filepath, 'r'))
    survey_id = args.survey_id
    assert survey_id in ['26', '27', '29', '32', '34', '36', '41', '42', '43', '45', '49', '50', '54', '82', '92'], 'invalid form id'
    
    personas = json.load(open(args.input_filepath, 'r'))['personas']
    
    df = pd.read_csv(f'opinions_qa/data/human_resp/American_Trends_Panel_W{survey_id}/responses.csv')
    info_df = pd.read_csv(f'opinions_qa/data/human_resp/American_Trends_Panel_W{survey_id}/info.csv')
    topics = get_stances(info_df, df, personas, survey_id)
    personas_and_topics = {'personas':personas, 'topics':topics}

    with open(args.output_filepath, 'w') as f:
        json.dump(personas_and_topics, f)