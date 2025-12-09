import pandas as pd
import os
import json as js
import pickle


def load_json(path):
    with open(path, 'rb') as f:
        data = js.load(f)
    return data

def to_json(path, data):
    with open(path, 'w') as f:
        js.dump(data,f,indent = 6)

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
        signal_index = row['study_id']
        filename = row['viblib_name']
        vibviz_id = filename.split('.')[0]
        mapping[str(signal_index)] = (vibviz_id, filename)
    return mapping

def recursive_listdir(path,csv_path):

    files = os.listdir(path)
    signal_dict = {}
    mapping = construct_mapping(csv_path)
    for user_id, file in enumerate(files):
        file_path = os.path.join(path, file)
        print('file_path:{}'.format(file_path))

        if os.path.isfile(file_path) and file_path.endswith('json'):
            data = load_json(file_path)
            values = data['FREE_TEXT']['value']
            user_name = file.split('.')[0]
            print('user_name:{}'.format(user_name))
            for i, value in enumerate(values):
                signal_index = value['signal_index']
                category = value['category']
                text = value['data']

                if str(signal_index) in mapping.keys():
                    (vibviz_id, _) = mapping[str(signal_index)]
                else:
                    vibviz_id = None

                if signal_index not in signal_dict.keys():
                    signal_dict[signal_index]={}
                    signal_dict[signal_index]['vibviz']=str(vibviz_id).strip()
                    if user_name not in signal_dict[signal_index].keys():
                        signal_dict[signal_index][user_name]={}
                        signal_dict[signal_index][user_name][category] = text
                else:
                    if user_name in signal_dict[signal_index]:
                        signal_dict[signal_index][user_name][category]=text
                    else:
                        signal_dict[signal_index][user_name] = {}
                        signal_dict[signal_index][user_name][category]=text
            

    for signal_index in signal_dict.keys():
        print('signal_index:{},user_num:{}'.format(signal_index, len(signal_dict[signal_index])))
        # for user in signal_dict[vibviz_id]:
        #     print(user)

    to_json(r'../output/signal_map_11.16.json', signal_dict)


def group_user(json_path):
    data = load_json(json_path)

    new_data = {}
    for sid in data.keys():
        sid_item = data[sid]
        vid = sid_item['vibviz']
        print('sid:{}'.format(sid))
        uid_set= set(list(sid_item.keys()))
        uid_set.remove('vibviz')
        for uid in uid_set:
            print('uid:{}'.format(uid))
            uid_item = sid_item[uid]
            for tag in uid_item.keys():
                if tag == 'free_text_sensory':
                    tag = 'sen'
                    des = uid_item['free_text_sensory']
                elif tag == 'free_text_emotional':
                    tag = 'emo'
                    des = uid_item['free_text_emotional']
                else:
                    tag = 'ass'
                    des = uid_item['free_text_association']

                if uid not in new_data:
                    new_data[uid] = {}
                    new_data[uid][tag] = {}

                    if sid not in new_data[uid][tag]:
                        new_data[uid][tag][sid] = des.strip()
                else:
                    if tag not in new_data[uid]:
                        new_data[uid][tag] = {}
                        new_data[uid][tag][sid] = des.strip()
                    else:
                        if sid not in new_data[uid][tag]:
                            new_data[uid][tag][sid] = des.strip()

    to_json(r'../output/group_user_11.16.json',new_data)



# data_dir = r'../json_file'
# csv_path = r'../mapping/vibviz_mapping.csv'
# recursive_listdir(data_dir, csv_path)

json_path = r'../output/signal_map_11.16.json'
group_user(json_path=json_path)

    




            