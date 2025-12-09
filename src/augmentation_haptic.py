import pandas as pd
import os
import json as js
import pickle
from nlpaug.augmenter.word import SynonymAug
import nlpaug.augmenter.word as naw
from googletrans import Translator

aug_rep = naw.SynonymAug(aug_src='wordnet')
aug_del = naw.RandomWordAug(action="delete")
translator = Translator()

def load_json(path):
    with open(path, 'rb') as f:
        data = js.load(f)
    return data

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_csv(path):
    data = pd.read_csv(path, sep=',', header=0)
    return data

def construct_mapping(csv_path):
    data = load_csv(csv_path)
    mapping = {}
    for index, row in data.iterrows():
        id = row['id']
        filename = row['filename']
        vibviz_id = row['vibviz-id']
        mapping[str(id)] = (vibviz_id, filename)
    return mapping

def recursive_listdir(path,csv_path,aug_haptic_path,to_dir):

    data_dict = []
    files = os.listdir(path)
    aug_features = load_pickle(aug_haptic_path)
    # print(aug_features.keys())
    mapping = construct_mapping(csv_path)
    for file in files:
        file_path = os.path.join(path, file)
        print('file_path:{}'.format(file_path))

        if os.path.isfile(file_path):
           if file_path.endswith('json'):
              data = load_json(file_path)
        else:
            exit()


        values = data['FREE_TEXT']['value']
        for value in values:
            signal_index = value['signal_index']
            category = value['category']
            text = value['data']

            aug_text_list = augmentated_text(text)[0]

            (vibviz_id, filename) = mapping[str(signal_index)]
            # print(vibviz_id)
            ###augmented 8
            aug_hap_feature_list = aug_features[vibviz_id][0][0]
            ###augmented 4
            # aug_hap_feature_list = aug_features[vibviz_id][0][0]

            # print('aug_hap_feature_list:{}'.format(aug_hap_feature_list))

            print('aug_haptic_len:{},aug_text_len:{}'.\
                  format(len(aug_hap_feature_list),len(aug_text_list)))

            for aug_hap_feature in aug_hap_feature_list:
                for aug_text in aug_text_list:
                    # breakpoint()
                    print('text:{}, aug_text:{},aug_hap_feature.shape:{}'
                          .format(text, aug_text, aug_hap_feature[0].shape))
                    feature = (vibviz_id, aug_text, aug_hap_feature, category)
                    data_dict.append(feature)

    split_unit = len(data_dict)//10
    print('data length:{},split_unit:{}'.format(len(data_dict),split_unit))
    
    train_data =data_dict[:7*split_unit]
    valid_data = data_dict[7*split_unit:8*split_unit]
    test_data = data_dict[8*split_unit:]
    to_pickle(train_data, to_dir.format('train'))
    to_pickle(test_data, to_dir.format('test'))
    to_pickle(valid_data, to_dir.format('valid'))


def augmentated_text(sentence):

    aug_sentence_list = []
    def augmenteated(sentence,aug):
        # synonyms
        if aug == 0:
            augmented_sentence = aug_rep.augment(sentence)
        elif aug == 1:
            # random delete
            augmented_sentence = aug_del.augment(sentence)
        elif aug == 2:
            # translated
            translated = translator.translate(sentence, src='en', dest='ch').text
            augmented_sentence = translator.translate(translated, src='ch', dest='en').text
        
        return augmented_sentence[0]
    
    sent0 = augmenteated(sentence, aug=0)
    sent1 = augmenteated(sentence, aug=1)
    # sent2 = augmenteated(sentence, aug=2)
    # print('sen:{},sent0:{},sent1:{}'.format(sentence, sent0, sent1))
    aug_sentence_list.append([sentence, sent0,sent1])
    return aug_sentence_list


data_dir = r'../data/collection/VRcontrollers/'
csv_path = r'../data/collection/patterns.csv'
audio_path = r'../data/collection/aug8_seq_audio_feature.pkl'
to_dir = r'../data/collection/aug8_seq_{}_feature.pkl'

recursive_listdir(data_dir, csv_path, audio_path, to_dir)






    