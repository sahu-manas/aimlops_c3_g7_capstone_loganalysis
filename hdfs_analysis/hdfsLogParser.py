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

import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../deeplog')

import logging


logging.basicConfig(level=logging.WARNING,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

import argparse
import torch

from logbert_pytorch.dataset import WordVocab
from logbert_pytorch import Predictor, Trainer
from logbert_pytorch.dataset.utils import seed_everything

options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
options["output_dir"] = "./output/"
options["model_dir"] = options["output_dir"] + "bert/"
options["model_path"] = options["model_dir"] + "best_bert.pth"
options["train_vocab"] = options["output_dir"] + "train"
options["vocab_path"] = options["output_dir"] + "vocab.pkl"  # pickle file

options["window_size"] = 128
options["adaptive_window"] = True
options["seq_len"] = 512
options["max_len"] = 512 # for position embedding
options["min_len"] = 10
options["mask_ratio"] = 0.65
# sample ratio
options["train_ratio"] = 1
options["valid_ratio"] = 0.1
options["test_ratio"] = 1

# features
options["is_logkey"] = True
options["is_time"] = False

options["hypersphere_loss"] = True
options["hypersphere_loss_test"] = True

options["scale"] = None # MinMaxScaler()
options["scale_path"] = options["model_dir"] + "scale.pkl"

# model
options["hidden"] = 256 # embedding size
options["layers"] = 4
options["attn_heads"] = 4

options["epochs"] = 20
options["n_epochs_stop"] = 10
options["batch_size"] = 32

options["corpus_lines"] = None
options["on_memory"] = True
options["num_workers"] = 5
options["lr"] = 1e-3
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.00
options["with_cuda"]= True
options["cuda_devices"] = None
options["log_freq"] = None

# predict
options["num_candidates"] = 6
options["gaussian_mean"] = 0
options["gaussian_std"] = 1

seed_everything(seed=1234)

if not os.path.exists(options['model_dir']):
    os.makedirs(options['model_dir'], exist_ok=True)

logger.info(f"device", options["device"])
#print("device", options["device"])
print("features logkey:{} time: {}\n".format(options["is_logkey"], options["is_time"]))
print("mask ratio", options["mask_ratio"])

# get [log key, delta time] as input for deeplog
input_dir  = os.path.expanduser('./data/')
output_dir = './output/'  # The output directory of parsing results

log_file   = "HDFS_Anomaly_1.txt"  # The input log file name
#log_file   = "HDFS_Normal_1.txt"  # The input log file name

log_structured_file = output_dir + log_file + "_structured.csv"
log_templates_file = output_dir + log_file + "_templates.csv"
log_sequence_file = output_dir + "hdfs_sequence.csv"

def mapping():
    log_temp = pd.read_csv(log_templates_file)
    log_temp.sort_values(by = ["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {event: idx+1 for idx , event in enumerate(list(log_temp["EventId"])) }
    print(log_temp_dict)
    with open (output_dir + "hdfs_log_templates.json", "w") as f:
        json.dump(log_temp_dict, f)


def parser(logs, log_format, type='drain'):
    if type == 'spell':
        tau        = 0.5  # Message type threshold (default: 0.5)
        regex      = [
            "(/[-\w]+)+", #replace file path with *
            "(?<=blk_)[-\d]+" #replace block_id with *

        ]  # Regular expression list for optional preprocessing (default: [])

        parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex, keep_para=False)
        parser.parseInMemory(log_file)

    elif type == 'drain':
        regex = [
            r"(?<=blk_)[-\d]+", # block_id
            r'\d+\.\d+\.\d+\.\d+',  # IP
            r"(/[-\w]+)+",  # file path
            #r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.5  # Similarity threshold
        depth = 5  # Depth of all leaf nodes


        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=False)
        df_log = parser.parseInMemory(logs)
        print(df_log.head())
        return df_log


def hdfs_sampling(df, window='session'):
    assert window == 'session', "Only window=session is supported for HDFS dataset."
    print("Loading", log_file)
    #df = pd.read_csv(log_file, engine='c',
    #        na_filter=False, memory_map=True, dtype={'Date':object, "Time": object})

    with open(output_dir + "hdfs_log_templates.json", "r") as f:
        event_num = json.load(f)
    df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))

    data_dict = defaultdict(list) #preserve insertion order of items
    for idx, row in tqdm(df.iterrows()):
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            data_dict[blk_Id].append(row["EventId"])

    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
    #data_df.to_csv(log_sequence_file, index=None)
    print("hdfs sampling done")
    print(data_df.head())
    return data_df


def generate_data(data_df, n=None, ratio=0.3):
    #blk_label_dict = {}
    #blk_label_file = os.path.join(input_dir, "anomaly_label.csv")
    #blk_df = pd.read_csv(blk_label_file)
    #for _ , row in tqdm(blk_df.iterrows()):
    #    blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    #seq = pd.read_csv(hdfs_sequence_file)
    #seq["Label"] = seq["BlockId"].apply(lambda x: blk_label_dict.get(x)) #add label to the sequence of each blockid

    #normal_seq = seq[seq["Label"] == 0]["EventSequence"]
    #normal_seq = normal_seq.sample(frac=1, random_state=20) # shuffle normal data

    #abnormal_seq = seq[seq["Label"] == 1]["EventSequence"]
    #normal_len, abnormal_len = len(normal_seq), len(abnormal_seq)
    #train_len = n if n else int(normal_len * ratio)
    #print("normal size {0}, abnormal size {1}, training size {2}".format(normal_len, abnormal_len, train_len))

    #train = normal_seq.iloc[:train_len]
    #test_normal = normal_seq.iloc[train_len:]
    #test_abnormal = abnormal_seq

    #df_to_file(train, output_dir + "train")
    #df_to_file(test_normal, output_dir + "test_normal")
    #df_to_file(test_abnormal, output_dir + "test_abnormal")
    seq = data_df["EventSequence"]
    print("generate data done")
    print(seq.head())
    return seq


def df_to_file(df, file_name):
    with open(file_name, 'w') as f:
        for _, row in df.items():
            f.write(' '.join([str(ele) for ele in eval(row)]))
            f.write('\n')


if __name__ == "__main__":
    
    #aparser = argparse.ArgumentParser()
    #subparsers = aparser.add_subparsers()

    #predict_parser = subparsers.add_parser('predict')
    #predict_parser.set_defaults(mode='predict')
    #predict_parser.add_argument("-m", "--mean", type=float, default=0)
    #predict_parser.add_argument("-s", "--std", type=float, default=1)
    
    # 1. parse HDFS log
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
    file = os.path.join(input_dir, log_file)
    log_messages = []
    linecount = 0
    cnt = 0
    with open(file, 'r') as fin:
        for line in fin.readlines():
            cnt += 1
            log_messages.append(line)
            linecount += 1
    
            
    print("Total size after encoding is", linecount, cnt)
    #logdf = pd.DataFrame(log_messages)
    #print(logdf.head())
    df = parser(log_messages, log_format, 'drain')
    #mapping()
    data_df = hdfs_sampling(df)
    data = generate_data(data_df, n=4855)
    
    #args = aparser.parse_args()
    #print("arguments", args)

    for line in data:
        print(line)
        res = ' '.join([str(s) for s in line])
        print(res)
        Predictor(options).predictInMemory(res)
