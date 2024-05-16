import pandas as pd
import numpy as np
import os
import glob
import re
import json
import torch
from src.metric_helper_functions import generate_embeddings, extract_statement_content, mw_custom, average_embeddings_with_words, get_agreeing_indices

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from scipy.spatial.distance import cosine
from transformers import pipeline
from itertools import combinations, product
import random
import time

import pdb

entailment_model_name = 'roberta-large-mnli'

def entailment_diversity(exp_df, grouped_df, exp_name, persona_data, sample_rate = 1.0):
    label_to_score = {'CONTRADICTION':1, 'ENTAILMENT':-1, 'NEUTRAL':0}
    device = 0 if torch.cuda.device_count() > 0 else -1

    classifier = pipeline('text-classification', model=entailment_model_name, top_k=3, device=device)
    diversity_values = []

    for index, row in grouped_df.iterrows():
        condition = (exp_df['gpt-4 evals for topic'] == 'Agree') & (exp_df['model'] == row['model']) & (exp_df['topic'] == row['topic']) & (exp_df['persona'] == row['persona']) & (exp_df['first'] == row['first'])
        statements = [s.replace('\n', '') for s in list(exp_df.loc[condition]['statements'])]

        random.seed(0)
        sentence_pairs = list(f'{x[0]}. {x[1]}.' for x in list(combinations(statements, 2)))
        num_samples = int(sample_rate * len(sentence_pairs))
        sampled_sentence_pairs = random.sample(sentence_pairs, num_samples)
        classifier_output = classifier(sampled_sentence_pairs)
        total_diversity = sum(sum([label_to_score[y['label']]*y['score'] for y in x]) for x in classifier_output)
        diversity_values.append(total_diversity / num_samples)

    return(diversity_values)

def diversity(exp_df, grouped_df, exp_name, persona_data):
    diversity_values = []
    generate_embeddings(exp_df, grouped_df, exp_name)
    for index, row in grouped_df.iterrows():
        # compute diversity
        filename = f"embeddings/{exp_name}/statements-{row['model']}-{row['topic']}-{row['persona']}-{row['first']}.npy"
        agreement_indices = get_agreeing_indices(exp_df, row) # new
        embeddings = np.load(filename)[agreement_indices]
        cos_sim = pairwise_distances(embeddings, metric='cosine')
        n = cos_sim.shape[0]
        avg_diversity = (sum(sum(cos_sim)))/(n**2 - n)
        diversity_values.append(avg_diversity)

    return(diversity_values) 

def human(exp_df, grouped_df, exp_name, persona_data):
    human_values = []
    for index, row in grouped_df.iterrows():
        try:
            topic = row['topic']
            persona = row['persona']
            human_values.append(persona_data['topics'][topic]['rate'][persona])
        except KeyError:
            human_values.append(-1.0)
    return(human_values)

def congruity(exp_df, grouped_df, exp_name, persona_data):
    human_values = []
    for index, row in grouped_df.iterrows():
        try:
            topic = row['topic']
            persona = row['persona']
            opp_persona = {'base':'base',
            'liberal':'conservative', 
            'conservative':'liberal',
            'black':'white',
            'white':'black',
            'male':'female',
            'female':'male'}[persona]

            persona_rate = persona_data['topics'][topic]['rate'][persona]
            opp_rate = persona_data['topics'][topic]['rate'][opp_persona]
            human_values.append(persona_rate/(persona_rate + opp_rate))
        except KeyError:
            human_values.append(-1.0)
    return(human_values)

def toxicity(exp_df, grouped_df, exp_name, persona_data):
    perspective_api_key = os.environ['PERSPECTIVE_API_KEY']
    filename = f'persona-tracking-results/{exp_name}_toxicity'
    if os.path.exists(f'{filename}.npy'):
        scores = np.load(f'{filename}.npy')
    else:
        client = discovery.build("commentanalyzer", "v1alpha1", 
                                developerKey=perspective_api_key,
                                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                                static_discovery=False,)
        scores = np.full(exp_df.shape[0], -1.0)

        for idx, text in enumerate(exp_df['statements']):
            analyze_request = {
                'comment': {'text': text},
                'requestedAttributes': {'TOXICITY': {}}
            }
            try:
                response = client.comments().analyze(body=analyze_request).execute()
                score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
                time.sleep(1)
            except HttpError as e:
                score = -1.0
                time.sleep(1)
        
            scores[idx] = score
    
    exp_df['toxicity'] = scores
    exp_df.to_csv(f'persona-tracking-data/{exp_name}.csv')

    grouped_toxicity_values = []
    for index, row in grouped_df.iterrows():
        condition = (exp_df['toxicity'] >= 0.0) & (exp_df['model'] == row['model']) & (exp_df['topic'] == row['topic']) & (exp_df['persona'] == row['persona']) & (exp_df['first'] == row['first'])
        mean_toxicity = exp_df.loc[condition]['toxicity'].mean()
        grouped_toxicity_values.append(mean_toxicity)
    return(grouped_toxicity_values)

def model_base(exp_df, grouped_df, exp_name, persona_data):
    model_values = []
    for index, row in grouped_df.iterrows():
        try: 
            model = row['model']
            topic = row['topic']
            model_values.append(persona_data['topics'][topic]['model_base_response'][model])
        except KeyError:
            model_values.append(-1.0)
    return(model_values)

# compost metrics
# modified from https://github.com/myracheng/lm_caricature/tree/main
def individuation(exp_df, grouped_df, exp_name, persona_data):
    if 'filtered_statements' not in exp_df.columns:
        exp_df['filtered_statements'] = [extract_statement_content(row) for idx, row in exp_df.iterrows()]
        exp_df.to_csv(f'persona-tracking-data/{exp_name}.csv')

    generate_embeddings(exp_df, grouped_df, exp_name, 'filtered_statements')
    indiv_values = []

    for index, row in grouped_df.iterrows():
        model = row['model']
        topic = row['topic']
        opp = persona_data['topics'][topic]['opposite']
        persona = row['persona']
        first = row['first']
        l = ['persona', 'topic']
        l.remove(first)

        if persona != 'base':
            agreement_indices = get_agreeing_indices(exp_df, row) # new
            persona_embeddings = np.load(f'embeddings/{exp_name}/filtered_statements-{model}-{topic}-{persona}-{first}.npy')[agreement_indices]
            base_embeddings = np.load(f'embeddings/{exp_name}/filtered_statements-{model}-{topic}-base-topic.npy')
            X = np.concatenate([persona_embeddings, base_embeddings])
            y = [0]*persona_embeddings.shape[0] + [1]*base_embeddings.shape[0]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
            clf = RandomForestClassifier(random_state = 0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            indiv_values.append(accuracy_score(y_test, y_pred))
        else:
            indiv_values.append(-1.0)
            
    return(indiv_values)

def exaggeration(exp_df, grouped_df, exp_name, persona_data):
    generate_embeddings(exp_df, grouped_df, exp_name, 'statements')

    exagg_values = []
    default_topic_df = pd.read_csv('./persona-tracking-data/intersection-experiment/default-topic.csv')

    for index, row in grouped_df.iterrows():
        if row['persona'] == 'base':
            exagg_values.append(-1.0)
        else:
            model = row['model']
            topic = row['topic']
            opposing_topic = persona_data['topics'][topic]['opposite']
            persona = row['persona']
            first = row['first']

            default_topic = default_topic_df[(default_topic_df['persona'] == persona) & (default_topic_df['model'] == model)]['statements']
            default_persona = exp_df[(exp_df['persona'] == 'base') & (exp_df['topic'] == topic) & (exp_df['model'] == model)]['statements']
            persona_topic = exp_df[(exp_df['persona'] == persona) & (exp_df['topic'] == topic) & (exp_df['model'] == model)]['statements']
            background_df = pd.concat([default_topic, default_persona, persona_topic])

            # compute keywords
            persona_keywords = [w[0] for w in mw_custom(default_topic, default_persona, background_df) if len(w[0]) > 0]
            topic_keywords = [w[0] for w in mw_custom(default_persona, default_topic, background_df) if len(w[0]) > 0]
            persona_topic = exp_df[(exp_df['persona'] == persona) & (exp_df['topic'] == topic) & (exp_df['model'] == model)]['statements']

            # compute persona-topic axis
            default_topic_statements = [re.sub(r'[^a-zA-Z\s]', '', x.lower()).split() for x in list(default_topic)]
            default_persona_statements = [re.sub(r'[^a-zA-Z\s]', '', x.lower()).split() for x in list(default_persona)]
            default_topic_embd = np.load(f'embeddings/intersection-experiment/default-topic/statements-{model}-{persona}-base-topic.npy')
            default_persona_embd = np.load(f'embeddings/{exp_name}/statements-{model}-{topic}-base-topic.npy')

            axis = (average_embeddings_with_words(default_topic_statements, default_topic_embd, persona_keywords) - average_embeddings_with_words(default_persona_statements, default_persona_embd, topic_keywords)).reshape(1,-1)
            
            # get embeddings of persona-topic, compute normalized average cosine distance to pole, return pole
            agreement_indices = get_agreeing_indices(exp_df, row) # new
            persona_topic_embd = np.load(f'embeddings/{exp_name}/statements-{model}-{topic}-{persona}-{first}.npy')[agreement_indices]
            cos_sim = cosine_similarity(persona_topic_embd, axis).mean()
            default_topic_cos_sim = cosine_similarity(default_topic_embd, axis).mean()
            default_persona_cos_sim = cosine_similarity(default_persona_embd, axis).mean()
            exagg_values.append((cos_sim - default_persona_cos_sim)/(default_topic_cos_sim - default_persona_cos_sim))
    
    return(exagg_values)