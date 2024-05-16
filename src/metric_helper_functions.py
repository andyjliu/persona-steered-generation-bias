import os
import numpy as np
import math
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import torch
import re
import pdb

def generate_embeddings(exp_df, grouped_df, exp_name, colname = 'statements'):
    embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    def statements_to_embeddings(list_of_statements, filename, model):
        if not os.path.exists(filename):
            embeddings = model.encode(list_of_statements, convert_to_numpy = True)
            np.save(filename, embeddings)

    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    sst_model = SentenceTransformer(embedding_model_name, device=device)
    diversity_values = []
    try:
        os.mkdir(f"embeddings/{exp_name}")
    except OSError:
        pass

    for index, row in grouped_df.iterrows():
        # generate embeddings if needed
        condition = (exp_df['model'] == row['model']) & (exp_df['topic'] == row['topic']) & (exp_df['persona'] == row['persona']) & (exp_df['first'] == row['first'])
        statements = list(exp_df.loc[condition][colname])
        filename = f"embeddings/{exp_name}/{colname}-{row['model']}-{row['topic']}-{row['persona']}-{row['first']}.npy"
        statements_to_embeddings(statements, filename, sst_model)

def extract_statement_content(row):
    pattern = '[^a-zA-Z\s,]'
    statement = re.sub(pattern, '', row['statements'].lower())

    if statement.count(',') > 0:
        if (statement.split(',')[0].startswith('as a')) or (statement.split(',')[0].startswith('as someone')):
            start_ind = 1
            for idx, phrase in enumerate(statement.split(',')):
                if phrase.startswith(' i '):
                    start_ind = idx
                    break
            return(','.join(row['statements'].split(',')[start_ind:]).lstrip())
    return(row['statements'])

def get_agreeing_indices(exp_df, row):
    condition = (exp_df['model'] == row['model']) & (exp_df['topic'] == row['topic']) & (exp_df['persona'] == row['persona']) & (exp_df['first'] == row['first'])
    statements = exp_df.loc[condition]
    statements = statements.reset_index()
    indices = statements.index[statements['gpt-4 evals for topic'] == 'Agree'].tolist()
    return(indices)



# CoMPosT helper functions
def average_embeddings_with_words(statements, embeddings, words):
    word_embeddings = np.zeros((len(words), embeddings.shape[1]))
    assert len(statements) == embeddings.shape[0], f"size mismatch between f{len(statements)} and {embeddings.shape}"
    for word_id, word in enumerate(words):
        inds = [idx for idx, statement in enumerate(statements) if word in statement]
        word_embeddings[word_id,:] = embeddings[inds].mean(axis=0)
    return(np.nanmean(word_embeddings, axis=0))

# from https://github.com/myracheng/lm_caricature/tree/main
def mw_custom(df1,df2,df0,thr=1.96):
    grams={}
    delt = get_log_odds(df1, df2 ,df0,False) #first one is the positive-valued one
    # print(delt)

    c1 = []
    c2 = []
    for k,v in delt.items():
        if v > thr:
            c1.append([k,v])
        elif v < -thr:
            c2.append([k,v])

    # added: if not enough signifiant keywords, add most significant
    if len(c1) < 5:
        c1 = ([k,delt[k]] for k in sorted(delt, key=delt.get, reverse=True)[:5]) 

    return(c1)

def get_log_odds(df1, df2, df0,verbose=False,lower=True):
    """Monroe et al. Fightin' Words method to identify top words in df1 and df2
    against df0 as the background corpus"""
    if lower:
        counts1 = defaultdict(int,[[i,j] for i,j in df1.str.lower().str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        counts2 = defaultdict(int,[[i,j] for i,j in df2.str.lower().str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        prior = defaultdict(int,[[i,j] for i,j in df0.str.lower().str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
    else:
        counts1 = defaultdict(int,[[i,j] for i,j in df1.str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        counts2 = defaultdict(int,[[i,j] for i,j in df2.str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        prior = defaultdict(int,[[i,j] for i,j in df0.str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])

    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    for word in prior.keys():
        prior[word] = int(prior[word] + 0.5)

    for word in counts2.keys():
        counts1[word] = int(counts1[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    for word in counts1.keys():
        counts2[word] = int(counts2[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    n1 = sum(counts1.values())
    n2 = sum(counts2.values())
    nprior = sum(prior.values())

    for word in prior.keys():
        if prior[word] > 0:
            l1 = float(counts1[word] + prior[word]) / (( n1 + nprior ) - (counts1[word] + prior[word]))
            l2 = float(counts2[word] + prior[word]) / (( n2 + nprior ) - (counts2[word] + prior[word]))
            sigmasquared[word] =  1/(float(counts1[word]) + float(prior[word])) + 1/(float(counts2[word]) + float(prior[word]))
            sigma[word] =  math.sqrt(sigmasquared[word])
            delta[word] = ( math.log(l1) - math.log(l2) ) / sigma[word]

    if verbose:
        for word in sorted(delta, key=delta.get)[:10]:
            print("%s, %.3f" % (word, delta[word]))

        for word in sorted(delta, key=delta.get,reverse=True)[:10]:
            print("%s, %.3f" % (word, delta[word]))
    return delta