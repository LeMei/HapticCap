import json as js
import os
import pandas as pd
import numpy as np
import torch
import csv

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

def to_json(data,path):
    with open(path, 'w') as f:
        js.dump(data,f,indent = 6)

def cal_sim_des_signal(uid, vid_list, des_emb_list, tag):
    des_len = len(des_emb_list)
    print('des_len:{}'.format(des_len))
    cosine_vib = np.zeros([des_len, des_len])
    for i in range(des_len):
        for j in range(des_len):
            cosine_sim_i_j = 1 - cosine(des_emb_list[i], des_emb_list[j])
            cosine_vib[i][j] = round(cosine_sim_i_j, 2)

    matrix = list(cosine_vib)
    with open('../output/similarity/user/filtered/matrix_{}_{}.csv'.format(uid, tag), 'w') as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow(row)

def filter_out_low_sim_user(matrix_path, bar=0.5):
    files = os.listdir(matrix_path)
    remain_sid = []
    sid_list = []
    remain_file_list = []
    for file in files:
        file_path = os.path.join(matrix_path, file)
        if file.endswith('csv'):
            sid = file.split('_')[1]
            category = file.split('_')[-1].split('.')[0]
            sid_list.append(sid)
            # print('vibid:{}'.format(vibid))
            data = load_csv(file_path).values
            sim_data = data[:,1:]
            sum_ele = np.sum(sim_data,axis=1) / len(sim_data)
            for i, avg_user in enumerate(sum_ele):
                uid = str(int(data[i,0]))
                print('sid:{}, uid:{}, avg_sim:{}'.format(sid, uid, avg_user))
                if avg_user >= 0.5:
                    remain_sid.append((sid, category, uid))
                    if file not in remain_file_list:
                        remain_file_list.append(file)
                    ### we also need to recoder the user_id
        
    print('length of original sid:{}, original des:{}, remaining sid:{}, des:{}'\
          .format(len(set(sid_list)),len(files),len(set(remain_sid)), len(remain_sid)))
    
    return remain_file_list,remain_sid

def read_user_des_with_filter(matrix_path, group_user_path):
    data = read_json(group_user_path)
    remain_file_list,remain_info = filter_out_low_sim_user(matrix_path)
    # print(remain_file_list)
    # print(remain_info)
    for uid in data.keys():
        user_item = data[uid]
        sen_des_list, emo_des_list, ass_des_list = [], [], []
        for cid in user_item.keys():
            category_item = user_item[cid]
            # print('uid:{},cid:{}'.format(uid,cid))
            vid_list = []
        
            if cid == 'sen':
                for vid in category_item.keys():
                    info = (vid,cid, uid)
                    # print('vid:{}'.format(vid))
                    indictor = 'matrix_{}_{}.csv'.format(vid,cid)
      
                    if info in remain_info and indictor in remain_file_list:
                        des = category_item[vid].strip()
                        sen_des_list.append(des)
            elif cid == 'emo':
                for vid in category_item.keys():
                    info = (vid,cid, uid)
                    indictor = 'matrix_{}_{}.csv'.format(vid,cid)    
                    if info in remain_info and indictor in remain_file_list:
                        des = category_item[vid].strip()
                        emo_des_list.append(des)
            else:
                for vid in category_item.keys():
                    info = (vid,cid,uid)
                    indictor = 'matrix_{}_{}.csv'.format(vid,cid)        
                    if info in remain_info and indictor in remain_file_list:
                        des = category_item[vid].strip()
                        ass_des_list.append(des)
        print('len_sen:{}, len_emo:{}, len_ass:{}'.format(len(sen_des_list), len(emo_des_list), len(ass_des_list)))
        if len(sen_des_list) != 0:
            inputs_sen = tokenizer(sen_des_list, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                embeddings_sen = model(**inputs_sen, output_hidden_states=False, return_dict=True).pooler_output
            cal_sim_des_signal(uid, vid_list, embeddings_sen.cpu(), tag='sen')

        if len(emo_des_list) != 0:
            inputs_emo = tokenizer(emo_des_list, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                embeddings_emo = model(**inputs_emo, output_hidden_states=False, return_dict=True).pooler_output
            cal_sim_des_signal(uid, vid_list, embeddings_emo.cpu(), tag='emo')

        if len(ass_des_list) != 0:
            inputs_ass = tokenizer(ass_des_list, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                embeddings_ass = model(**inputs_ass, output_hidden_states=False, return_dict=True).pooler_output
            cal_sim_des_signal(uid, vid_list, embeddings_ass.cpu(), tag='ass')

def read_user_des_without_filter(group_user_path):
    data = read_json(group_user_path)
    for uid in data.keys():
        user_item = data[uid]
        sen_des_list, emo_des_list, ass_des_list = [], [], []
        for cid in user_item.keys():
            category_item = user_item[cid]
            print('uid:{},cid:{}'.format(uid,cid))
            vid_list = []
        
            if cid == 'sen':
                for vid in category_item.keys(): 
                    des = category_item[vid].strip()
                    sen_des_list.append(des)
            elif cid == 'emo':
                for vid in category_item.keys():
                    des = category_item[vid].strip()
                    emo_des_list.append(des)
                        
            else:
                for vid in category_item.keys():
                    des = category_item[vid].strip()
                    ass_des_list.append(des)
                        
        print('len_sen:{}, len_emo:{}, len_ass:{}'.format(len(sen_des_list), len(emo_des_list), len(ass_des_list)))
        if len(sen_des_list) != 0:
            inputs_sen = tokenizer(sen_des_list, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            inputs_emo = tokenizer(emo_des_list, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            inputs_ass = tokenizer(ass_des_list, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                embeddings_sen = model(**inputs_sen, output_hidden_states=False, return_dict=True).pooler_output
                embeddings_emo = model(**inputs_emo, output_hidden_states=False, return_dict=True).pooler_output
                embeddings_ass = model(**inputs_ass, output_hidden_states=False, return_dict=True).pooler_output
            
            cal_sim_des_signal(uid, vid_list, embeddings_sen.cpu(), tag='sen')
            cal_sim_des_signal(uid, vid_list, embeddings_emo.cpu(), tag='emo')
            cal_sim_des_signal(uid, vid_list, embeddings_ass.cpu(), tag='ass')

json_path=r'../output/group_user_11.16.json'
matrix_path = r'../output/similarity/signal/'

read_user_des_with_filter(matrix_path, json_path)

# json_path=r'../output/group_user_11.16.json'

# read_user_des_without_filter(json_path)