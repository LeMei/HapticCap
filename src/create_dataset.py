import sys
import os
import re
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from subprocess import check_call
from datasets import Dataset, Audio

import torch
import torch.nn as nn
import json as js
from utils.util import load_pickle, to_pickle, load_csv, load_json

class Haptics:
    def __init__(self, config):
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        self.train = load_pickle(DATA_PATH + '/train_data.pkl')
        self.dev = load_pickle(DATA_PATH + '/valid_data.pkl')
        self.test = load_pickle(DATA_PATH + '/test_data.pkl')
        self.multi = True
        self.pretrained_emb, self.word2id = None, None


    def get_data(self, mode):
        if mode == "train":
            return self.train, self.word2id, None
        elif mode == "valid":
            return self.dev, self.word2id, None
        elif mode == "test":
            return self.test, self.word2id, None
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()

class Collections:
    def __init__(self, config):
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        if not config.set_audio_model:
            self.train = load_pickle(DATA_PATH + '/aug8_seq_train_feature.pkl')
            self.dev = load_pickle(DATA_PATH + '/aug8_seq_valid_feature.pkl')
            self.test = load_pickle(DATA_PATH + '/aug8_seq_test_feature.pkl')
        
            self.multi = True
            self.pretrained_emb, self.word2id = None, None
        
        else:
            if config.set_filter and config.set_supervised and config.set_user:
                if config.train_lyer:
                    self.train = load_pickle(DATA_PATH+'/lyer/lyer_filter_raw_train_user_supervised.pkl')
                    self.valid = load_pickle(DATA_PATH+'/lyer/lyer_filter_raw_valid_user_supervised.pkl')
                    self.test = load_pickle(DATA_PATH+'/lyer/lyer_filter_raw_test_user_supervised.pkl')
                elif config.category_name and not config.full_data:
                    category = config.category_name
                    self.train = load_pickle(DATA_PATH+'/filter_raw_train_{}_user_supervised.pkl'.format(category))
                    self.valid = load_pickle(DATA_PATH+'/filter_raw_valid_{}_user_supervised.pkl'.format(category))
                    self.test = load_pickle(DATA_PATH+'/filter_raw_test_{}_user_supervised.pkl'.format(category))
                else:
                    ### FULL DATA AND FULL Category For Training
                    if config.category_name:
                        category = config.category_name
                        # self.train = load_pickle(DATA_PATH+'/full_data/filter_raw_train_{}_user_supervised.pkl'.format(category))
                        # self.valid = load_pickle(DATA_PATH+'/full_data/filter_raw_valid_{}_user_supervised.pkl'.format(category))
                        # self.test = load_pickle(DATA_PATH+'/full_data/filter_raw_test_{}_user_supervised.pkl'.format(category))
                        # updated
                        # self.train = load_pickle(DATA_PATH+'/full_data/filter_raw_train_{}_user_supervised_10_2.pkl'.format(category))
                        # self.valid = load_pickle(DATA_PATH+'/full_data/filter_raw_valid_{}_user_supervised_10_2.pkl'.format(category))
                        # self.test = load_pickle(DATA_PATH+'/full_data/filter_raw_test_{}_user_supervised_10_2.pkl'.format(category))

                        # 11.16 update
                        self.train = load_pickle(DATA_PATH+'/haptic_11.16/model_data/filter_raw_train_{}_11.16.pkl'.format(category))
                        self.valid = load_pickle(DATA_PATH+'/haptic_11.16/model_data/filter_raw_valid_{}_11.16.pkl'.format(category))
                        self.test = load_pickle(DATA_PATH+'/haptic_11.16/model_data/filter_raw_test_{}_11.16.pkl'.format(category))
                    else:
                        # self.train = load_pickle(DATA_PATH+'/full_data/filter_raw_train_user_supervised.pkl')
                        # self.valid = load_pickle(DATA_PATH+'/full_data/filter_raw_valid_user_supervised.pkl')
                        # self.test = load_pickle(DATA_PATH+'/full_data/filter_raw_test_user_supervised.pkl')
                        ### update 
                        # self.train = load_pickle(DATA_PATH+'/full_data/filter_raw_train_user_supervised_10_2.pkl')
                        # self.valid = load_pickle(DATA_PATH+'/full_data/filter_raw_valid_user_supervised_10_2.pkl')
                        # self.test = load_pickle(DATA_PATH+'/full_data/filter_raw_test_user_supervised_10_2.pkl')
                        ### 11.16 
                        self.train = load_pickle(DATA_PATH+'/haptic_11.16/model_data/filter_raw_train_11_16.pkl')
                        self.valid = load_pickle(DATA_PATH+'/haptic_11.16/model_data/filter_raw_valid_11_16.pkl')
                        self.test = load_pickle(DATA_PATH+'/haptic_11.16/model_data/filter_raw_test_11_16.pkl')

            elif config.set_filter and config.set_supervised:
                self.train = load_pickle(DATA_PATH+'/filter_raw_train_collection_supervised.pkl')
                self.valid = load_pickle(DATA_PATH+'/filter_raw_valid_collection_supervised.pkl')
                self.test = load_pickle(DATA_PATH+'/filter_raw_test_collection_supervised.pkl')
            elif config.set_filter:
                self.train = load_pickle(DATA_PATH+'/filter_raw_train_collection.pkl')
                self.valid = load_pickle(DATA_PATH+'/filter_raw_valid_collection.pkl')
                self.test = load_pickle(DATA_PATH+'/filter_raw_test_collection.pkl')
            else:
                print('-----------------error-------------------')
            # self.train = load_pickle(DATA_PATH+'/raw_train_collection.pkl')
            # self.valid = load_pickle(DATA_PATH+'/raw_valid_collection.pkl')
            # self.test = load_pickle(DATA_PATH+'/raw_test_collection.pkl')

            # except:
            #         raw_audio_path = r'./aug_viblib'
            #         raw_text_path = r'./VRcontrollers'
            #         csv_path = r'./patterns.csv'
            #         to_dir = r'./'
            #         data_dict = {"audio":[],"text":[], "category":[],"vibid":[]}

            #         def construct_mapping(csv_path):
            #             data = load_csv(csv_path)
            #             mapping = {}
            #             for index, row in data.iterrows():
            #                 id = row['id']
            #                 filename = row['filename']
            #                 vibviz_id = row['vibviz-id']
            #                 mapping[str(id)] = (vibviz_id, filename)
            #             return mapping


            #         files = os.listdir(raw_text_path)
            #         mapping = construct_mapping(csv_path)

            #         for file in files:
            #             file_path = os.path.join(raw_text_path, file)
            #             print('file_path:{}'.format(file_path))

            #             if os.path.isfile(file_path):
            #                 if file_path.endswith('json'):
            #                     data = load_json(file_path)

            #             values = data['FREE_TEXT']['value']
            #             for value in values:
            #                 signal_index = value['signal_index']
            #                 category = value['category']
            #                 text = value['data']

            #                 (vibviz_id, filename) = mapping[str(signal_index)]
            #                 print(vibviz_id)

            #                 raw_audio_file = os.path.join(raw_audio_path, '{}.wav'.format(vibviz_id))
            #                 ### each audio can be augmentated more than 8 versions

            #                 data_dict['audio'].append(raw_audio_file)
            #                 data_dict['text'].append(text.strip())
            #                 data_dict['category'].append(category)
            #                 data_dict['vibid'].append(vibviz_id)

            #                 for i in range(0, 8):
            #                     aug_audio_path = os.path.join(raw_audio_path, '{}_aug{}.wav'.format(vibviz_id,str(i)))
            #                     data_dict['audio'].append(aug_audio_path)
            #                     data_dict['text'].append(text.strip())
            #                     data_dict['category'].append(category)
            #                     data_dict['vibid'].append('{}_aug{}'.format(vibviz_id, str(i)))
                    
            #         data_dataset = Dataset.from_dict(data_dict).cast_column("audio", Audio())


            #         data_size = len(data_dataset)
            #         print('data_size:{}'.format(data_size))
            #         n_train = data_size//10 * 7
            #         n_valid = data_size//10 + n_train

            #         train_data, valid_data, test_data = [], [], []
            #         for i in range(data_size):
            #             audio = data_dataset[i]["audio"]["array"]
            #             text = data_dataset[i]["text"]
            #             category = data_dataset[i]["category"]
            #             vibid = data_dataset[i]['vibid']
            #             feature = (vibid, text, audio, category)
            #             if i < n_train:
            #                 train_data.append(feature)
            #             elif i < n_valid:
            #                 valid_data.append(feature)
            #             else:
            #                 test_data.append(feature)

            #         to_pickle(train_data, to_dir+'/raw_train_collection.pkl')
            #         to_pickle(valid_data, to_dir+'/raw_valid_collection.pkl')
            #         to_pickle(test_data, to_dir+'/raw_test_collection.pkl')

            #         self.train = load_pickle(DATA_PATH+'/raw_train_collection.pkl')
            #         self.valid = load_pickle(DATA_PATH+'/raw_valid_collection.pkl')
            #         self.test = load_pickle(DATA_PATH+'/raw_test_collection.pkl')


    def get_data(self, mode):
        if mode == "train":
            return self.train
        elif mode == "valid":
            return self.valid
        elif mode == "test":
            return self.test
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()

class GenData:
    def __init__(self, config):
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        if config.set_positive_vote:
            ###vote == 1 remaining
            print('vote==1 without match task')
            self.train = load_pickle(DATA_PATH+'/data/gen_raw_train_user_supervised.pkl')
            self.valid = load_pickle(DATA_PATH+'/data/gen_raw_valid_user_supervised.pkl')
            self.test = load_pickle(DATA_PATH+'/data/gen_raw_test_user_supervised.pkl')
        elif not config.set_match_task:
            ###vote == 1 or -1 without match task
            print('vote==1or-1 without match task')
            self.train = load_pickle(DATA_PATH+'/data/gen_full_raw_train_user_supervised_with_haptic_audio.pkl')
            self.valid = load_pickle(DATA_PATH+'/data/gen_full_raw_valid_user_supervised_with_haptic_audio.pkl')
            self.test = load_pickle(DATA_PATH+'/data/gen_full_raw_test_user_supervised_with_haptic_audio.pkl')
        else:
            ###vote == 1 or -1 with match task
            print('vote==1or-1 with match task')
            self.train = load_pickle(DATA_PATH+'/data/gen_full_raw_train_user_supervised_with_haptic_audio.pkl')
            self.valid = load_pickle(DATA_PATH+'/data/gen_full_raw_valid_user_supervised_with_haptic_audio.pkl')
            self.test = load_pickle(DATA_PATH+'/data/gen_full_raw_test_user_supervised_with_haptic_audio.pkl')
            
    def get_data(self, mode):
        if mode == "train":
            return self.train
        elif mode == "valid":
            return self.valid
        elif mode == "test":
            return self.test
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()


class AllData:
    def __init__(self, config):
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        if config.category_name:
            category = config.category_name
            # updated
            self.train = load_pickle(DATA_PATH+'/full_data/filter_raw_train_{}_user_supervised_10_2.pkl'.format(category))
            self.valid = load_pickle(DATA_PATH+'/full_data/filter_raw_valid_{}_user_supervised_10_2.pkl'.format(category))
            self.test = load_pickle(DATA_PATH+'/full_data/filter_raw_test_{}_user_supervised_10_2.pkl'.format(category))
        else:
            # self.train = load_pickle(DATA_PATH+'/full_data/filter_raw_train_user_supervised.pkl')
            # self.valid = load_pickle(DATA_PATH+'/full_data/filter_raw_valid_user_supervised.pkl')
            # self.test = load_pickle(DATA_PATH+'/full_data/filter_raw_test_user_supervised.pkl')
            ### update 
            self.train = load_pickle(DATA_PATH+'/two_datasets_raw_train_user_supervised_with_haptic_audio.pkl')
            self.valid = load_pickle(DATA_PATH+'/two_datasets_raw_valid_user_supervised_with_haptic_audio.pkl')
            self.test = load_pickle(DATA_PATH+'/two_datasets_raw_test_user_supervised_with_haptic_audio.pkl')
            
    def get_data(self, mode):
        if mode == "train":
            return self.train
        elif mode == "valid":
            return self.valid
        elif mode == "test":
            return self.test
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()



