#!/usr/bin/env python  
#-*- coding:utf-8 _*-
"""
@author:watercow
@license: Apache Licence
@file: beforeStart.py
@site:
@software: PyCharm

This Part works for changing original data into
     the format used in train/test
"""
import os
import json
import jieba
from tqdm import tqdm
from collections import Counter

STOPWORDS = [':','：','、','\\','N','；',';','（','）','◆','\n',
             '[',']','【','】','＋',',']


def preprocess(args):
    if not os.path.exists(args.train_test_dir):
        os.makedirs(args.target_dir)

    prepro_each(args, "data/step1_data/data_train.json", 0.0, 1.0,  out_name='train')
    prepro_each(args, "data/step1_data/data_test.json", 0.0, 1.0, out_name='test')


def save(data, shared, out_name):
    data_path = os.path.join("data/train-test_data", "data_{}.json".format(out_name))
    shared_path = os.path.join("data/train-test_data", "shared_{}.json".format(out_name))

    json.dump(data, open(data_path, 'w', encoding='utf8'), ensure_ascii=False)
    json.dump(shared, open(shared_path, 'w', encoding='utf8'), ensure_ascii=False)
    return 0


def prepro_each(args, data_path, start_ratio, stop_ratio, out_name):
    # Choose tokenizer
    if args.tokenizer == 'jieba':
        word_tokenizer = jieba.cut
    else: # TODO: other tokenizers
        raise Exception()

    # Reading data.json
    source_path = data_path
    source_data = []
    with open(source_path, 'r', encoding='utf8') as f:
        for line in f:
            source_data.append(json.loads(line))

    # counter for words number & some vars
    word_counter = Counter()
    job_posting, resume, pair_id_list, label_list = [], [], [], []
    job_id_list, resume_id_list = [], []

    # start & end position
    start_ai = int(round(len(source_data) * start_ratio))
    stop_ai = int(round(len(source_data) * stop_ratio))

    # GET INFO & GENERATE return DICT
    for ai, content in enumerate(tqdm(source_data[start_ai:stop_ai])):

        jd_list = source_data[ai]['job_posting']
        resume_list = source_data[ai]['resume']
        job_id = source_data[ai]['job_id']
        resume_id = source_data[ai]['resume_id']
        pair_id = source_data[ai]['pair_id']
        label = source_data[ai]['label']

        jp = [] # job_posting context after word-seg
        rp = [] # resume_exp context after word-seg

        # job_posting
        for ji, job_ability in enumerate(jd_list):
            # word segment
            xi = list(word_tokenizer(job_ability))
            jp.append(xi)

            # word counter
            for word in xi:
                word_counter[word] += 1

        # resume_exp
        if isinstance(resume_list[0], list):
            resume_list = resume_list[0]

        for ri, exp in enumerate(resume_list):
            # word segment
            xi = list(word_tokenizer(exp))

            # word counter
            for word in xi:
                rp.append(word)
                word_counter[word] += 1

        job_posting.append(jp)
        resume.append([rp])
        job_id_list.append(job_id)
        resume_id_list.append(resume_id)
        pair_id_list.append(pair_id)
        label_list.append(label)

    # get word2vec dict
    word2vec_dict = {}

    data = {
        'job_posting': job_posting, # word-level
        'resume': resume,           # word-level
        'pair_id_list': pair_id_list,
        'label': label_list,
        'job_id_list': job_id_list,
        'resume_id_list': resume_id_list
    }

    shared = {
        'word_counter': word_counter,
        'word2vec': word2vec_dict
    }

    print('{}_data saving...'.format(out_name))
    save(data, shared, out_name)
    print('finised')
    return


# def preprocess_Infer(args):
#     if not os.path.exists(args.train_test_dir):
#         os.makedirs(args.target_dir)
#
#     prepro_Infer_each(args, "data/step1_data/exp_morethan_50/data_train.json", "train")
#     prepro_Infer_each(args, "data/step1_data/exp_morethan_50/data_test.json", "test")
#     prepro_Infer_each(args, "data/step1_data/exp_morethan_50/data_dev.json", "dev")


def prepro_Infer_each(args, data_path, out_name, exp_len = 50):
    # Choose tokenizer
    if args.tokenizer == 'jieba':
        word_tokenizer = jieba.cut
    else:  # TODO: other tokenizers
        raise Exception()

    # Reading data.json
    source_data = []
    with open(data_path, 'r', encoding='utf8') as f:
        for line in f:
            source_data.append(json.loads(line))

    jd_write_file = open("data/train-test_data/s1.{}".format(out_name), 'w', encoding='utf8')
    cv_write_file = open("data/train-test_data/s2.{}".format(out_name), 'w', encoding='utf8')
    label_write_file = open("data/train-test_data/labels.{}".format(out_name), 'w', encoding='utf8')

    # Generate and write
    for i, content in enumerate(tqdm(source_data[0:len(source_data)])):

        jd_list = source_data[i]['job_posting']
        resume_list = source_data[i]['resume']
        label = source_data[i]['label']

        # job posting
        jd_content = ""
        for job_req in jd_list:
            jd_content += job_req
        jd_line_list = list(word_tokenizer(jd_content))
        jd_line = ""
        for word in jd_line_list:
            if word not in STOPWORDS:
                jd_line += word + " "

        # resume exp
        if isinstance(resume_list[0], list):
            resume_list = resume_list[0]

        cv_content = ""
        for exp in resume_list:
            cv_content += exp
        cv_line_list = list(word_tokenizer(cv_content))
        cv_line = ""
        for word in cv_line_list:
            if word not in STOPWORDS:

                cv_line += word + " "

        # graph info
        # find R-R by jd

        # write
        if(len(cv_line.split(' ')) >= exp_len and len(jd_line.split(' ')) >= 15):
            jd_write_file.write(jd_line + '\n')
            cv_write_file.write(cv_line + '\n')
            label_write_file.write(str(label) + '\n')


def preprocess_Graph(args):
    if not os.path.exists(args.train_test_dir):
        os.makedirs(args.train_test_dir)

    prepro_Graph_each(args, "data/train_data/", "train")
    prepro_Graph_each(args, "data/test_data/", "test")
    prepro_Graph_each(args, "data/val_data/", "dev")


def prepro_Graph_each(args, data_path, out_name, exp_len=50, graph_num=5):
    # Choose tokenizer
    if args.tokenizer == 'jieba':
        word_tokenizer = jieba.cut
    else:  # TODO: other tokenizers
        raise Exception()

    # Reading data.json
    source_data = []
    with open(data_path+'data.json', 'r', encoding='utf8') as f:
        for line in f:
            source_data.append(json.loads(line))

    # Reading luqu.json
    with open(data_path+"graph_hired_user.json", 'r', encoding='utf8') as f:
        user_luqu_dict = json.load(f)
    with open(data_path+"graph_hired_jd.json", 'r', encoding='utf8') as f:
        jd_recruit_dict = json.load(f)

    # Reading nothired.json
    # with open("data/step1_data/exp_morethan_0_graph/graph_nothired_user.json", 'r', encoding='utf8') as f:
    #     user_nothired_dict = json.load(f)
    # with open("data/step1_data/exp_morethan_0_graph/graph_nothired_jd.json", 'r', encoding='utf8') as f:
    #     jd_nothired_dict = json.load(f)

    jd_write_file = open("data/train-test_data/s1.{}".format(out_name), 'w', encoding='utf8')
    cv_write_file = open("data/train-test_data/s2.{}".format(out_name), 'w', encoding='utf8')
    label_write_file = open("data/train-test_data/labels.{}".format(out_name), 'w', encoding='utf8')
    jd_graph_file = open("data/train-test_data/s1_graph.{}".format(out_name), 'w', encoding='utf8')
    cv_graph_file = open("data/train-test_data/s2_graph.{}".format(out_name), 'w', encoding='utf8')

    # Generate and write
    for i, content in tqdm(enumerate(source_data)):

        jd_list = source_data[i]['job_posting']
        resume_list = source_data[i]['resume']
        job_id = source_data[i]['job_id']
        resume_id = source_data[i]['resume_id']
        # pair_id = source_data[i]['pair_id']
        label = source_data[i]['label']

        # job posting
        jd_content = ""
        for job_req in jd_list:
            jd_content += job_req
        jd_line_list = list(word_tokenizer(jd_content))
        jd_line = " ".join(word for word in jd_line_list if word not in STOPWORDS)
        # resume exp
        if isinstance(resume_list[0], list):
            resume_list = resume_list[0]

        cv_content = ""
        for exp in resume_list:
            cv_content += exp
        cv_line_list = list(word_tokenizer(cv_content))
        cv_line = " ".join(word for word in cv_line_list if word not in STOPWORDS)
        # Graph cv (R-R)
        R_R_list = [resume_id]
        for user_item in jd_recruit_dict.get(job_id, []):
            if user_item != resume_id:
                R_R_list.append(user_item)
                # R_R_list.append(user_item)
        # R_R_line = resume_id + " "
        R_R_line = " ".join([str(r_) for r_ in R_R_list])

        # Graph jd (J-J)
        J_J_list = [job_id]
        for jd_item in user_luqu_dict.get(resume_id, []):
            if jd_item != job_id:
                # J_J_list.append(jd_item)
                J_J_list.append(jd_item)
        J_J_line = " ".join([str(j_) for j_ in J_J_list])
        # write file
        if (len(cv_line.split(' ')) >= exp_len and len(jd_line.split(' ')) >= 15):
           #and len(R_R_list) and len(J_J_list)):
            jd_write_file.write(jd_line + '\n')
            cv_write_file.write(cv_line + '\n')
            label_write_file.write(str(label) + '\n')
            jd_graph_file.write(J_J_line + '\n')
            cv_graph_file.write(R_R_line + '\n')

    jd_write_file.close()
    cv_write_file.close()
    label_write_file.close()
    jd_graph_file.close()
    cv_graph_file.close()


def split_train_test(train_ratio, test_ratio, val_ratio):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(args.table_action)
    df_sat0 = df[(df['satisfied'] == 0) & (df['delivered'] == 1)]
    df_sat1 = df[df['satisfied'] == 1]

    df_train_sat0, df_testval_sat0 = train_test_split(df_sat0, train_size=train_ratio, random_state=42)
    df_test_sat0, df_val_sat0 = train_test_split(df_testval_sat0, train_size=test_ratio/(test_ratio+val_ratio), random_state=42)

    df_train_sat1, df_testval_sat1 = train_test_split(df_sat1, train_size=train_ratio, random_state=42)
    df_test_sat1, df_val_sat1 = train_test_split(df_testval_sat1, train_size=test_ratio/(test_ratio+val_ratio), random_state=42)

    max_train_size = min(df_train_sat0.shape[0], df_train_sat1.shape[0])
    df_train = pd.concat([df_train_sat0.head(max_train_size), df_train_sat1.head(max_train_size)])
    df_train.to_csv(args.table_action.replace(".csv", "_train.csv"), index=False)

    max_test_size = min(df_test_sat0.shape[0], df_test_sat1.shape[0])
    df_test = pd.concat([df_test_sat0.head(max_test_size), df_test_sat1.head(max_test_size)])
    df_test.to_csv(args.table_action.replace(".csv", "_test.csv"), index=False)

    max_val_size = min(df_val_sat0.shape[0], df_val_sat1.shape[0])
    df_val = pd.concat([df_val_sat0.head(max_val_size), df_val_sat1.head(max_val_size)])
    df_val.to_csv(args.table_action.replace(".csv", "_val.csv"), index=False)


import argparse
from data.dataprocessor import AliDataProcessor, RealDataProcessor

if __name__ == '__main__':
    # 2. change data.json into train/test.json
    parser = argparse.ArgumentParser()
    parser.add_argument('--orignal_data_format', default='ali', choices=['ali', 'real'])
    parser.add_argument('--orignal_data_dir', default='Recruitment_round1_train_20190716')
    parser.add_argument('--tokenizer', default='jieba')
    parser.add_argument('--train_test_dir', default='data/train-test_data')
    parser.add_argument('--table_action', default='train-test_data\\table3_action')
    args = parser.parse_args()
    split_train_test(train_ratio=0.8, test_ratio=0.1, val_ratio=0.1)

    # Step 0. choose Processor according to the original data
    for split_ in ('train', 'test', 'val'):
        DataProcessor = AliDataProcessor(args.orignal_data_dir, split_=split_)
        # Step 1. change original data into data.json
        print('total user nums: ', len(DataProcessor.user_dict))
        print('total jd nums: ', len(DataProcessor.jd_dict))
        import os
        path = f"data/{split_}_data"
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)
        DataProcessor.generate_datajson(f'{path}/data.json', exp_len=0)
        DataProcessor.dump_json(f'{path}/graph_hired_jd.json', mode='graph_jd')
        DataProcessor.dump_json(f'{path}/graph_hired_user.json', mode='graph_user')
        DataProcessor.dump_json(f'{path}/graph_nothired_jd.json', mode='graph_nothired_jd')
        DataProcessor.dump_json(f'{path}/graph_nothired_user.json', mode='graph_nothired_user')
    DataProcessor.dump_json(f'data/user.json', mode='user')
    DataProcessor.dump_json(f'data/jd.json', mode='jd')
    preprocess_Graph(args)
