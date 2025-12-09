# -*- encoding:utf-8 -*-
import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer
import random
from create_dataset import Collections, Haptics, GenData, AllData
from config import DEVICE,category_dict
import warnings
import json as js
from transformers import T5Tokenizer, AutoProcessor, AutoTokenizer
# if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
#     warnings.warn("PyTorch version 1.7.1 or higher is recommended")

# _tokenizer = _Tokenizer()
# t5_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
ast_audio_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
wav2vec_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
encodec_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

class HapticsDataset(Dataset):
    def __init__(self, config, args):
        self.config = config
        ## Fetch dataset
        if "haptics" in str(config.data_dir).lower():
            dataset = Haptics(config)
            self.multi = True
        elif 'collection' in str(config.data_dir).lower():
            dataset = Collections(config)
        elif 'hapticgen' in str(config.data_dir).lower():
            dataset = GenData(config)
        elif 'alldata' in str(config.data_dir).lower():
            dataset = AllData(config)
        else:
            print("Dataset not defined correctly")
            exit()

        if args.text_encoder_name == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
        elif args.text_encoder_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        elif args.text_encoder_name == 'llama':
            access_token = ''
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",token=access_token)
        elif args.text_encoder_name == 'mistral':
            access_token = ''
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",token=access_token)
        else:
            print('invalid text encoder name')

        self.data = dataset.get_data(config.mode)
        self.len = len(self.data)

    @property
    def ta_dim(self):
        t_dim = 768
        if self.config.dataset == 'haptics':
            return t_dim, 768
        else:
            return t_dim, 200

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def get_loader(args, config, truncate=False, prompt_dict = None, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = HapticsDataset(config,args)

    print('mode:{}'.format(config.mode))
    config.data_len = len(dataset)
    config.ta_dim = dataset.ta_dim

    if config.mode == 'train':
        args.n_train = len(dataset)
    elif config.mode == 'valid':
        args.n_valid = len(dataset)
    elif config.mode == 'test':
        args.n_test = len(dataset)

    dataset_name = args.dataset
    tokenizer = dataset.tokenizer

    def pad_sequence(sequences, target_len=-1, batch_first=False, padding_value=0.0):
        if target_len < 0:
            max_size = sequences[0].size()
            trailing_dims = max_size[1:]
        else:
            max_size = target_len
            trailing_dims = sequences[0].size()[1:]

        max_len = max([s.size(0) for s in sequences])
        if batch_first:
            out_dims = (len(sequences), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(sequences)) + trailing_dims

        out_tensor = sequences[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            # use index notation to prevent duplicate references to the tensor
            if batch_first:
                out_tensor[i, :length, ...] = tensor
            else:
                out_tensor[:length, i, ...] = tensor
        return out_tensor

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length

        a_lens = []
        labels = []
        ids = []
        texts = []
        audio = []
        for sample in batch:
            print('args.dataset:{}'.format(args.dataset))
            if args.dataset == 'haptics':
                vibviz_id, uid, raw_sentence, sentence_embedding, \
                audio_embedding, tags, tags_desc, compound_score = sample
                ###audio_embedding [(468,),]
                if len(sample[4]) > 1:
                    a_lens.append(torch.IntTensor([len(sample[4])]))
                    labels.append(sample[5])
                    ids.append(sample[0])
                    texts.append(tags_desc)
                    # texts.append(raw_sentence)
                    audio.append(audio_embedding)
            elif args.dataset == 'collection':
                vibviz_id, text, audio_feature, category = sample
                # print(len(audio_feature))
                a_lens.append(torch.IntTensor([len(audio_feature)]))
                labels.append(category_dict[category])
                ids.append(vibviz_id)
                texts.append(text)
                audio.append(audio_feature)
            elif args.dataset == 'gendata':
                vibviz_id, text, raw_audio, category, label = sample
                # print(len(audio_feature))
                categories.append(category_dict[str(category)])
                ids.append(vibviz_id)
                texts.append(text)
                audio_batch.append(raw_audio)
                labels.append(label)
            else:
                print('invalid dataset:{}'.format(args.dataset))

        alens = torch.cat(a_lens)
        audio = pad_sequence([torch.FloatTensor(sample) for sample in audio], target_len=alens.max().item())
        audio = torch.FloatTensor(audio)

        # print('labels:', labels)
        # label_ids = torch.cat([torch.IntTensor(label) for label in labels])
        # print(label_ids)
        label_ids = torch.LongTensor(labels)
        if isinstance(texts, str):
            texts = [texts]

        ###T5 model
        task_prefix = "sst2 sentence: "
        encoding = tokenizer(
            [task_prefix + sequence for sequence in texts],
            return_tensors="pt", padding=True
        )
        # print(label_batch)
        # T5 model things are batch_first
        t5_input_id = encoding.input_ids
        t5_att_mask = encoding.attention_mask

        return t5_input_id, t5_att_mask, audio,alens,label_ids
    
    def collate_fn_with_pre(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length

        a_lens = []
        categories = []
        labels = []
        ids = []
        texts = []
        audio_batch = []
        for sample in batch:
            # vibviz_id, uid, raw_sentence, sentence_embedding, \
            # audio_embedding, tags, tags_desc, compound_score = sample
            if args.dataset == 'haptics':
                vibviz_id, uid, raw_sentence, sentence_embedding, \
                audio_embedding, tags, tags_desc, compound_score = sample
                ###audio_embedding [(468,),]
                if len(sample[4]) > 1:
                    a_lens.append(torch.IntTensor([len(sample[4])]))
                    labels.append(sample[5])
                    ids.append(sample[0])
                    texts.append(tags_desc)
                    # texts.append(raw_sentence)
                    audio_batch.append(audio_embedding)
            elif args.dataset in ['collection', 'alldata']:
                vibviz_id, text, raw_audio, category, label = sample
                # print(len(audio_feature))
                categories.append(category_dict[category])
                ids.append(vibviz_id)
                texts.append(text)
                audio_batch.append(raw_audio)
                labels.append(label)
            elif args.dataset == 'gendata':
                vibviz_id, text, raw_audio, category, label = sample
                # print(len(audio_feature))
                ### category means the vote score
                categories.append(category_dict[str(category)])
                ids.append(vibviz_id)
                texts.append(text)
                audio_batch.append(raw_audio)
                labels.append(label)
            else:
                print('invalid dataset:{}'.format(args.dataset))

        if args.dataset == 'haptics':
            alens = torch.cat(a_lens)
            audio = pad_sequence([torch.FloatTensor(sample) for sample in audio_batch], target_len=alens.max().item())
            audio_inputs = torch.FloatTensor(audio)

        elif args.dataset in ['collection', 'alldata']:
            if args.acoustic_encoder_name == 'ast':
                audio_inputs = ast_audio_processor(audio_batch, sampling_rate=args.sampling_rate, return_tensors="pt")
            elif args.acoustic_encoder_name == 'wav2vec':
                audio_inputs = wav2vec_processor(audio_batch, sampling_rate=args.sampling_rate, padding=True, return_tensors="pt")
            elif args.acoustic_encoder_name == 'encodec':
                audio_inputs = encodec_processor(audio_batch, sampling_rate=24000, return_tensors="pt")
            else:
                print('invalid acoustic enocdr')
        elif args.dataset == 'gendata':
            if args.acoustic_encoder_name == 'ast':
                audio_inputs = ast_audio_processor(audio_batch, sampling_rate=args.sampling_rate, return_tensors="pt")
            elif args.acoustic_encoder_name == 'wav2vec':
                audio_inputs = wav2vec_processor(audio_batch, sampling_rate=args.sampling_rate, padding=True, return_tensors="pt")
            elif args.acoustic_encoder_name == 'encodec':
                audio_inputs = encodec_processor(audio_batch, sampling_rate=args.sampling_rate, return_tensors="pt")
            else:
                print('invalid acoustic enocdr')


        category_ids = torch.LongTensor(categories)
        label_ids = torch.LongTensor(labels)
        if isinstance(texts, str):
            texts = [texts]

        if args.text_encoder_name == 't5':
            ###T5 model
            task_prefix = ""
            encoding = tokenizer(
                [task_prefix + sequence for sequence in texts],
                return_tensors="pt", padding=True
            )
            # print(label_batch)
            # T5 model things are batch_first
            t5_input_id = encoding.input_ids
            t5_att_mask = encoding.attention_mask

            # breakpoint()
            return t5_input_id, t5_att_mask, audio_inputs,category_ids,label_ids
        elif args.text_encoder_name == 'bert':
            input_text = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            return input_text, None, audio_inputs, category_ids, label_ids
        elif args.text_encoder_name in ['llama','mistral']:
            tokenizer.pad_token = tokenizer.eos_token
            input_text = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

            return input_text, None, audio_inputs, category_ids, label_ids

        
    if not args.set_model:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn)
    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_with_pre)
    
    return data_loader