import numpy as np
import pandas as pd
import torch
import pickle
import random
import csv
import os
import json as js

from datasets import Dataset, Audio
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

name = 'princeton-nlp/sup-simcse-bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name).to(DEVICE)

def read_json(json_path):
    with open(json_path) as f:
        data = js.load(f)

    return data

def load_csv(path):
    data = pd.read_csv(path, sep=',', header=0)
    return data

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def cal_sim_des_signal_with_users(sid, uid_list, des_emb_list, tag):
    des_len = len(des_emb_list)
    cosine_vib = np.zeros([des_len, des_len])
    for i in range(des_len):
        for j in range(des_len):
            cosine_sim_i_j = 1 - cosine(des_emb_list[i], des_emb_list[j])
            cosine_vib[i][j] = round(cosine_sim_i_j, 2)

    matrix = list(cosine_vib)
    with open('../output/similarity/signal/matrix_{}_{}.csv'.format(sid,tag), 'w') as f:
        for i, row in enumerate(matrix):
            uid = uid_list[i]
            row = ','.join([str(j) for j in list(row)])
            row = str(uid) + ',' + row
            f.write(row+'\n')

def read_data(json_path):
    data = read_json(json_path)
    for sid in data.keys():
        emo_des_list, sen_des_list, ass_des_list = [], [], []
        user_des = data[sid]
        uid_list = []
        uid_set= set(list(user_des.keys()))
        uid_set.remove('vibviz')
        for uid in uid_set:
            print('sid:{},uid:{}'.format(sid, uid))
            emo_des = user_des[uid]['free_text_emotional']
            sen_des = user_des[uid]['free_text_sensory']
            ass_des = user_des[uid]['free_text_association']

            emo_des_list.append(emo_des)
            sen_des_list.append(sen_des)
            ass_des_list.append(ass_des)

            uid_list.append(uid)

        inputs_emo = tokenizer(emo_des_list, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        inputs_sen = tokenizer(sen_des_list, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        inputs_ass = tokenizer(ass_des_list, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
                embeddings_emo = model(**inputs_emo, output_hidden_states=False, return_dict=True).pooler_output
                embeddings_sen = model(**inputs_sen, output_hidden_states=False, return_dict=True).pooler_output
                embeddings_ass = model(**inputs_ass, output_hidden_states=False, return_dict=True).pooler_output

        # cal_sim_des_signal(vibid, embeddings_emo.cpu(), tag='emo')
        # cal_sim_des_signal(vibid, embeddings_sen.cpu(), tag='sen')
        # cal_sim_des_signal(vibid, embeddings_ass.cpu(), tag='ass')

        cal_sim_des_signal_with_users(sid, uid_list, embeddings_emo.cpu(), tag='emo')
        cal_sim_des_signal_with_users(sid, uid_list, embeddings_sen.cpu(), tag='sen')
        cal_sim_des_signal_with_users(sid, uid_list, embeddings_ass.cpu(), tag='ass')

json_path=r'../output/signal_map_11.16.json'

read_data(json_path)