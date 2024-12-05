import sys
sys.path.append('../')

import os
import re
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from logparser import Spell, Drain

# get [log key, delta time] as input for deeplog
input_dir  = os.path.expanduser('./data/')
output_dir = './output/'  # The output directory of parsing results
log_file   = "openstack.log"  # The input log file name

log_structured_file = output_dir + log_file + "_structured.csv"
log_templates_file = output_dir + log_file + "_templates.csv"
log_sequence_file = output_dir + "openstack_sequence.csv"

def mapping():
    log_temp = pd.read_csv(log_templates_file)
    log_temp.sort_values(by = ["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {event: idx+1 for idx , event in enumerate(list(log_temp["EventId"])) }
    print(log_temp_dict)
    with open (output_dir + "openstack_log_templates.json", "w") as f:
        json.dump(log_temp_dict, f)


def parser(input_dir, output_dir, log_file, log_format, type='drain'):
    if type == 'spell':
        tau        = 0.5  # Message type threshold (default: 0.5)
        regex      = [
            "(/[-\w]+)+", #replace file path with *
            "(?<=blk_)[-\d]+" #replace block_id with *

        ]  # Regular expression list for optional preprocessing (default: [])

        parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex, keep_para=False)
        parser.parse(log_file)

    elif type == 'drain':
        #regex = [
        #    r"(?<=blk_)[-\d]+", # block_id
        #    r'\d+\.\d+\.\d+\.\d+',  # IP
        #    r"(/[-\w]+)+",  # file path
            #r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        #]
        regex = [
            #r"(?<=instance:\s)[^\]]*", # instance id
            r'\d+\.\d+\.\d+\.\d+',  # IP
            r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',  # any hexdecimal string
            r"(^[<=\"].*?[?=\"]$)",  # HTTP Request,
            r"(?<=status:\s)[\d]+",
            r"(?<=len:\s)[\d]+",
            r"(?<=time:\s)[.\d]+",
            r'[0-9a-f]{40}',
            r'[0-9a-f]{32}',
            #r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.5  # Similarity threshold
        depth = 5  # Depth of all leaf nodes


        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=False)
        parser.parse(log_file)


def _sampling(log_file, window='session'):
    assert window == 'session', "Only window=session is supported for OpenStack dataset."
    print("Loading", log_file)
    df = pd.read_csv(log_file, engine='c',
            na_filter=False, memory_map=True, dtype={'Date':object, "Time": object})

    with open(output_dir + "openstack_log_templates.json", "r") as f:
        event_num = json.load(f)
    df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))

    data_dict = defaultdict(list) #preserve insertion order of items
    for idx, row in tqdm(df.iterrows()):
        instance_list = re.findall(r'(\[instance:\s.*?\])', row['Content'])
        instance_set = set(instance_list)
        for instance in instance_set:
            data_dict[instance].append(row["EventId"])

    data_df = pd.DataFrame(list(data_dict.items()), columns=['InstanceId', 'EventSequence'])
    print(data_df.head())
    data_df['InstanceId'] = data_df['InstanceId'].str.replace('[instance: ','')
    data_df['InstanceId'] = data_df['InstanceId'].str.replace(']','')
    print(data_df.head())
    data_df.to_csv(log_sequence_file, index=None)
    
    print("openstack sampling done")

def openstack_sampling(log_file, window='session'):
    assert window == 'session', "Only window=session is supported for OpenStack dataset."
    print("Loading", log_file)
    df = pd.read_csv(log_file, engine='c',
            na_filter=False, memory_map=True, dtype={'Date':object, "Time": object})

    #print(df.head())
    
    with open(output_dir + "openstack_log_templates.json", "r") as f:
        event_num = json.load(f)
    
    #print(event_num)
    
    df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))

    #print(df.head())
    #df.to_csv(output_dir + "openstack_sequence_test.csv",index=None)
    
    data_dict = defaultdict(list) #preserve insertion order of items
    instance_list = []
    #instance_set = set()
    for idx, row in tqdm(df.iterrows()):
        instance_list.append(re.findall(r'(\[instance:\s.*?\])', row['Content']))
        #instance_set = set(instance_list)
        #for instance in instance_set:
            #data_dict[instance].append(row["EventId"])
    
    #print(instance_list)
    #for i, n in enumerate(instance_list):
    #    n = n.st
    instance_list_cleaned = []
    for line in instance_list:
        for ln in line:
            ln = ln.replace(r'[instance: ', '')
            ln = ln.replace(r']', '')
            if ln not in instance_list_cleaned:
                instance_list_cleaned.append(ln)

    
    #print(instance_list_cleaned)
    #instance_set = set(instance_list_cleaned)
    #print(instance_set)
    
    #for idx, row in tqdm(df.iterrows()):
    #    instance_list.append(re.findall(r'(\[instance:\s.*?\])', row['Content']))
    
        #data_dict[instance].append(row["EventId"])
            
    #print(data_dict)
    
    for instance in tqdm(instance_list_cleaned):
    #instance = '5b98ace2-4126-46c3-a43e-1e1d879f0a8f'
        df_select = df[df.Content.str.contains(instance)]
    #print(df_select)
        for idx, row in df_select.iterrows():
            data_dict[instance].append(row["EventId"])

    data_df = pd.DataFrame(list(data_dict.items()), columns=['InstanceId', 'EventSequence'])
    
    #print(data_df.head())
    #data_df['InstanceId'] = data_df['InstanceId'].str.replace('[instance: ','')
    #data_df['InstanceId'] = data_df['InstanceId'].str.replace(']','')
    #print(data_df.head())
    data_df.to_csv(log_sequence_file, index=None)
    
    print("openstack sampling done")

def generate_train_test(sequence_file, ratio=0.8):
    label_dict = {}
    label_file = os.path.join(input_dir, "anomaly_labels.csv")
    df = pd.read_csv(label_file)
    for _ , row in tqdm(df.iterrows()):
        label_dict[row["Instance"]] = 1 if row["Label"] == "Anomaly" else 0
    
    print(label_dict)

    seq = pd.read_csv(sequence_file)
    seq["Label"] = 0
    print(seq.head())
    seq["Label"] = seq["InstanceId"].apply(lambda x: label_dict.get(x)) #add label to the sequence of each blockid

    seq["Label"] = seq["Label"].replace(np.nan, 0)
    print(seq.head())
    normal_seq = seq[seq["Label"] == 0]["EventSequence"]
    normal_seq = normal_seq.sample(frac=1, random_state=20) # shuffle normal data

    abnormal_seq = seq[seq["Label"] == 1]["EventSequence"]
    normal_len, abnormal_len = len(normal_seq), len(abnormal_seq)
    train_len = int(normal_len * ratio)
    print("normal size {0}, abnormal size {1}, training size {2}".format(normal_len, abnormal_len, train_len))

    train = normal_seq.iloc[:train_len]
    test_normal = normal_seq.iloc[train_len:]
    test_abnormal = abnormal_seq

    df_to_file(train, output_dir + "train")
    df_to_file(test_normal, output_dir + "test_normal")
    df_to_file(test_abnormal, output_dir + "test_abnormal")
    print("generate train test data done")


def df_to_file(df, file_name):
    with open(file_name, 'w') as f:
        for _, row in df.items():
            f.write(' '.join([str(ele) for ele in eval(row)]))
            f.write('\n')

def generate_logformat_regex(logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        print(splitters)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        
        print(regex)
        regex = re.compile('^' + regex + '$')
        return headers, regex

if __name__ == "__main__":
    # 1. parse OpenStack log
    log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>' #openstack log format
    #generate_logformat_regex(log_format)
    parser(input_dir, output_dir, log_file, log_format, 'drain')
    mapping()
    #_sampling(log_structured_file)
    openstack_sampling(log_structured_file)
    generate_train_test(log_sequence_file)
