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

    prepro_Graph_each(args, "data/step1_data/exp_morethan_0_graph/data_dev.json", "dev")
    prepro_Graph_each(args, "data/step1_data/exp_morethan_0_graph/data_train.json", "train")
    prepro_Graph_each(args, "data/step1_data/exp_morethan_0_graph/data_test.json", "test")


def prepro_Graph_each(args, data_path, out_name, exp_len=50, graph_num=5):
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

    # Reading luqu.json
    with open("data/step1_data/exp_morethan_0_graph/graph_hired_user.json", 'r', encoding='utf8') as f:
        user_luqu_dict = json.load(f)
    with open("data/step1_data/exp_morethan_0_graph/graph_hired_jd.json", 'r', encoding='utf8') as f:
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

    # Reading index-table
    user_id2num, user_num2id, jd_id2num, jd_num2id = load_index_table(args)

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
        R_R_list = [user_id2num[resume_id]]
        if job_id in jd_recruit_dict:
            user_recruit_list = jd_recruit_dict[job_id]
        else:
            # print('jobID', job_id, 'does not exist.')
            user_recruit_list = []
        for user_item in user_recruit_list:
            if user_item != resume_id:
                R_R_list.append(user_id2num[user_item])
                # R_R_list.append(user_item)
        # R_R_line = resume_id + " "
        R_R_line = " ".join([str(r_) for r_ in R_R_list])

        # Graph jd (J-J)
        J_J_list = [jd_id2num[job_id]]
        if resume_id in user_luqu_dict:
            jd_luqu_list = user_luqu_dict[resume_id]
        else:
            print('CVID', resume_id)
            jd_luqu_list = []
        for jd_item in jd_luqu_list:
            if jd_item != job_id:
                # J_J_list.append(jd_item)
                J_J_list.append(jd_id2num[jd_item])
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


def index_table(config):
    user_id2num = {}
    user_num2id = set()
    jd_id2num = {}
    jd_num2id = set()
    import csv
    file_path = config.table_action
    f = open(file_path, 'r', encoding='utf8')
    next(f)
    csv_reader = csv.reader(f, delimiter=',', quotechar='"')
    for content_list in tqdm(csv_reader):
        user_id = content_list[0]
        jd_no = content_list[1]
        if user_id not in user_num2id:
            user_num2id.add(user_id)
            user_id2num[user_id] = len(user_num2id) - 1
        if jd_no not in jd_num2id:
            jd_num2id.add(jd_no)
            jd_id2num[jd_no] = len(jd_num2id) - 1

    print('User: ', len(user_num2id))
    print('JD: ', len(jd_num2id))

    f_save_uid2num = open(config.f_save_uid2num, 'w', encoding='utf8')
    json.dump(user_id2num, f_save_uid2num, ensure_ascii=False)
    f_save_unum2id = open(config.f_save_unum2id, 'w', encoding='utf8')
    json.dump(list(user_num2id), f_save_unum2id, ensure_ascii=False)
    f_save_jid2num = open(config.f_save_jid2num, 'w', encoding='utf8')
    json.dump(jd_id2num, f_save_jid2num, ensure_ascii=False)
    f_save_jnum2id = open(config.f_save_jnum2id, 'w', encoding='utf8')
    json.dump(list(jd_num2id), f_save_jnum2id, ensure_ascii=False)

    f_save_unum2id.close()
    f_save_uid2num.close()
    f_save_jid2num.close()
    f_save_jnum2id.close()
    return


def load_index_table(config):
    f_save_uid2num = open(config.f_save_uid2num, 'r', encoding='utf8')
    user_id2num = json.load(f_save_uid2num)
    f_save_unum2id = open(config.f_save_unum2id, 'r', encoding='utf8')
    user_num2id = json.load(f_save_unum2id)
    f_save_jid2num = open(config.f_save_jid2num, 'r', encoding='utf8')
    jd_id2num = json.load(f_save_jid2num)
    f_save_jnum2id = open(config.f_save_jnum2id, 'r', encoding='utf8')
    jd_num2id = json.load(f_save_jnum2id)

    f_save_unum2id.close()
    f_save_uid2num.close()
    f_save_jid2num.close()
    f_save_jnum2id.close()
    return user_id2num, user_num2id, jd_id2num, jd_num2id


def split_train_test(input_, train_ratio, test_ratio, val_ratio):
    assert train_ratio+test_ratio+val_ratio==1.0
    import random
    zeros = []
    ones = []
    # Open the text file and read all lines
    directory = os.path.dirname(input_)
    with open(input_, 'r') as f:
        for line in f.readlines():
            datum = json.loads(line)
            if datum['label'] == 0:
                zeros.append(line)
            elif datum['label'] == 1:
                ones.append(line)
    # Randomly split 0 and 1 data into train, test, val
    random.shuffle(zeros)
    random.shuffle(ones)
    max_size = min(len(zeros), len(ones))
    # Calculate the splits
    train_size = int(max_size * train_ratio)
    test_size = int(max_size * test_ratio)
    val_size = int(max_size * val_ratio)
    train_lines = zeros[:train_size]+ones[:train_size]
    test_lines = zeros[train_size:train_size + test_size]+ones[train_size:train_size + test_size]
    val_lines = zeros[train_size+test_size:train_size+test_size+val_size]+ones[train_size+test_size:train_size+test_size+val_size]
    random.shuffle(train_lines)
    random.shuffle(test_lines)
    random.shuffle(val_lines)
    with open(os.path.join(directory, 'data_train.json'), 'w') as f:
        f.writelines(train_lines)
        print('total training data: ', len(train_lines))
    with open(os.path.join(directory, 'data_test.json'), 'w') as f:
        f.writelines(test_lines)
        print('total test data: ', len(test_lines))
    with open(os.path.join(directory, 'data_dev.json'), 'w') as f:
        f.writelines(val_lines)
        print('total validation data: ', len(val_lines))

import argparse
from data.dataprocessor import AliDataProcessor, RealDataProcessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--orignal_data_format', default='ali', choices=['ali', 'real'])
    parser.add_argument('--orignal_data_dir', default='Recruitment_round1_train_20190716')
    parser.add_argument('--tokenizer', default='jieba')
    parser.add_argument('--train_test_dir', default='data/train-test_data')
    parser.add_argument('--table_action', default='train-test_data\\table3_action')
    parser.add_argument('--f_save_uid2num', default='data/step1_data/uid2num.json')
    parser.add_argument('--f_save_unum2id', default='data/step1_data/unum2id.json')
    parser.add_argument('--f_save_jid2num', default='data/step1_data/jid2num.json')
    parser.add_argument('--f_save_jnum2id', default='data/step1_data/jnum2id.json')
    args = parser.parse_args()

    # Step 0. choose Processor according to the original data
    if args.orignal_data_format == 'ali':
        DataProcessor = AliDataProcessor(args.orignal_data_dir)
    else:
        DataProcessor = RealDataProcessor(args.orignal_data_dir)

    # Step 1. change original data into data.json
    print('total user nums: ', len(DataProcessor.user_dict))
    print('total jd nums: ', len(DataProcessor.jd_dict))
    import os
    path = "data/step1_data/exp_morethan_0_graph"
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
    DataProcessor.generate_datajson('data/step1_data/exp_morethan_0_graph/data.json', exp_len=0)
    DataProcessor.dump_json('data/step1_data/exp_morethan_0_graph/user.json', mode='user')
    DataProcessor.dump_json('data/step1_data/exp_morethan_0_graph/jd.json', mode='jd')
    DataProcessor.dump_json('data/step1_data/exp_morethan_0_graph/graph_hired_jd.json', mode='graph_jd')
    DataProcessor.dump_json('data/step1_data/exp_morethan_0_graph/graph_hired_user.json', mode='graph_user')
    DataProcessor.dump_json('data/step1_data/exp_morethan_0_graph/graph_nothired_jd.json', mode='graph_nothired_jd')
    DataProcessor.dump_json('data/step1_data/exp_morethan_0_graph/graph_nothired_user.json', mode='graph_nothired_user')

    # 2. change data.json into train/test.json
    split_train_test('data/step1_data/exp_morethan_0_graph/data.json', train_ratio=0.8, test_ratio=0.1, val_ratio=0.1)

    index_table(args)
    preprocess_Graph(args)
