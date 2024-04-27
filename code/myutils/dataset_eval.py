import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config_speed import config
import torch
np.random.seed(1126)
def get_similar_dict(pos_pairs):
    pos_pairs = pos_pairs.sort_values(by=['label'], ascending=False)
    similar_dict = {}
    for pair in zip(pos_pairs['trialid_1'].to_numpy(), pos_pairs['trialid_2'].to_numpy()):
        ref_id, sim_id = pair[0], pair[1]
        if ref_id not in similar_dict:
            similar_dict[ref_id] = [sim_id]
        else:
            similar_dict[ref_id].append(sim_id)
    return similar_dict

def get_disimilar_dict(neg_pairs):
    neg_pairs = neg_pairs.sort_values(by=['label'])
    disimilar_dict = {}
    for pair in zip(neg_pairs['trialid_1'].to_numpy(), neg_pairs['trialid_2'].to_numpy()):
        ref_id, sim_id = pair[0], pair[1]
        if ref_id not in disimilar_dict:
            disimilar_dict[ref_id] = [sim_id]
        else:
            disimilar_dict[ref_id].append(sim_id)
    return disimilar_dict

def get_ref_list(pair_data):
    refs = np.unique(pair_data['trialid_1'].to_numpy())
    return refs.tolist()

def sample_batch(similar_dict, disimilar_dict, reference_id_list, max_num):
    count = 0
    batch_ref = []
    batch_poss = []
    batch_negs = []
    for new_id in reference_id_list:
        batch_ref.append(new_id)
        batch_poss.append(similar_dict[new_id][0:max_num])
        batch_negs.append(disimilar_dict[new_id][0:max_num])
        count += 1
    return (batch_ref, batch_poss, batch_negs)



class Mytrial(Dataset):
    def __init__(self, train_pairs_path=None, raw_data_path=None, is_test=False, ref_path=None, simi_path=None, disimi_path=None, small=False):
        if config.small:
            small = False
        self.is_test = is_test
        if train_pairs_path is not None:
            pair_data = pd.read_csv(train_pairs_path)
            self.ref_list = get_ref_list(pair_data)
            if config.used_date != '2022-06-15':
                pos_pairs = pair_data.loc[pair_data['label'] > 0]
                neg_pairs = pair_data.loc[pair_data['label'] < 0]
            else:
                pos_pairs = pair_data.loc[pair_data['label'] == 1]
                neg_pairs = pair_data.loc[pair_data['label'] == 0]
            self.similar_dict = get_similar_dict(pos_pairs)
            self.disimilar_dict = get_disimilar_dict(neg_pairs)
            for k in range(len(self.ref_list) - 1, -1, -1):
                if self.ref_list[k] not in self.similar_dict or self.ref_list[k] not in self.disimilar_dict or len(
                        self.similar_dict[self.ref_list[k]]) < 10 or len(self.disimilar_dict[self.ref_list[k]]) < 10:
                    del self.ref_list[k]
        else:
            self.ref_list = pickle.load(open(ref_path, 'rb'))
            self.similar_dict = pickle.load(open(simi_path, 'rb'))
            self.disimilar_dict = pickle.load(open(disimi_path, 'rb'))
        self.id2data = pickle.load(open(raw_data_path, 'rb'))
        if small:
            self.ref_list = self.ref_list[0:len(self.ref_list)//4]
        for k in range(len(self.ref_list)-1, -1, -1):
            if self.ref_list[k] not in self.similar_dict or self.ref_list[k] not in self.disimilar_dict or len(self.similar_dict[self.ref_list[k]]) < 10 or len(self.disimilar_dict[self.ref_list[k]]) < 10:
                del self.ref_list[k]
            #if is_test:
            #    if np.isnan(self.id2data[self.ref_list[k]]['data_field']['enrollment_rate']):
            #        del self.ref_list[k]
    def __len__(self):
        return len(self.ref_list)

    def id2raw_data(self, id):
        raw_table = self.id2data[id]
        return raw_table

    def __getitem__(self, idx):
        ref_id = self.ref_list[idx]
        if not self.is_test:
            simi_ids = np.random.choice(self.similar_dict[ref_id], config.pos_num, replace=True)
            disimi_ids = np.random.choice(self.disimilar_dict[ref_id], config.neg_num, replace=True)
        else:
            simi_ids = self.similar_dict[ref_id][0:200]
            config.pos_num = len(simi_ids)
            disimi_ids = self.disimilar_dict[ref_id][0:200]
            config.neg_num = len(disimi_ids)
        labels = [1 for one in simi_ids] + [0 for one in disimi_ids]
        labels = np.array(labels)
        index = np.arange(len(labels))
        np.random.shuffle(index)
        labels = labels[index]
        candidate_ids = np.concatenate([simi_ids, disimi_ids], 0)
        candidate_ids = candidate_ids[index]

        return [self.id2raw_data(ref_id)] + [self.id2raw_data(u) for u in candidate_ids], labels

class MytrialEval(Dataset):
    def __init__(self, train_pairs_path=None, raw_data_path=None, is_test=False, ref_path=None, simi_path=None, disimi_path=None, small=False):
        if config.small:
            small = False
        self.is_test = is_test
        if train_pairs_path is not None:
            pair_data = pd.read_csv(train_pairs_path)
            self.ref_list = get_ref_list(pair_data)
            if config.used_date != '2022-06-15':
                pos_pairs = pair_data.loc[pair_data['label'] > 0]
                neg_pairs = pair_data.loc[pair_data['label'] < 0]
            else:
                pos_pairs = pair_data.loc[pair_data['label'] == 1]
                neg_pairs = pair_data.loc[pair_data['label'] == 0]
            self.similar_dict = get_similar_dict(pos_pairs)
            self.disimilar_dict = get_disimilar_dict(neg_pairs)
            for k in range(len(self.ref_list) - 1, -1, -1):
                if self.ref_list[k] not in self.similar_dict or self.ref_list[k] not in self.disimilar_dict or len(
                        self.similar_dict[self.ref_list[k]]) < 10 or len(self.disimilar_dict[self.ref_list[k]]) < 10:
                    del self.ref_list[k]
        else:
            self.ref_list = pickle.load(open(ref_path, 'rb'))
            self.similar_dict = pickle.load(open(simi_path, 'rb'))
            self.disimilar_dict = pickle.load(open(disimi_path, 'rb'))
        self.id2data = pickle.load(open(raw_data_path, 'rb'))
        if small:
            self.ref_list = self.ref_list[0:len(self.ref_list)//4]
        for k in range(len(self.ref_list)-1, -1, -1):
            if self.ref_list[k] not in self.similar_dict or self.ref_list[k] not in self.disimilar_dict or len(self.similar_dict[self.ref_list[k]]) < 10 or len(self.disimilar_dict[self.ref_list[k]]) < 10:
                del self.ref_list[k]
            #if is_test:
            #    if np.isnan(self.id2data[self.ref_list[k]]['data_field']['enrollment_rate']):
            #        del self.ref_list[k]
    def __len__(self):
        return len(self.ref_list)

    def id2raw_data(self, id):
        raw_table = self.id2data[id]
        return raw_table

    def __getitem__(self, idx):
        ref_id = self.ref_list[idx]
        if not self.is_test:
            simi_ids = np.random.choice(self.similar_dict[ref_id], config.pos_num, replace=True)
            disimi_ids = np.random.choice(self.disimilar_dict[ref_id], config.neg_num, replace=True)
        else:
            simi_ids = self.similar_dict[ref_id][0:200]
            if ref_id in simi_ids:
                print('ref in simi')
            config.pos_num = len(simi_ids)
            disimi_ids = self.disimilar_dict[ref_id][0:200]
            config.neg_num = len(disimi_ids)
        labels = [1 for one in simi_ids] + [0 for one in disimi_ids]
        labels = np.array(labels)
        index = np.arange(len(labels))
        np.random.shuffle(index)
        labels = labels[index]
        candidate_ids = np.concatenate([simi_ids, disimi_ids], 0)
        candidate_ids = candidate_ids[index]

        return [self.id2raw_data(ref_id)] + [self.id2raw_data(u) for u in candidate_ids], labels, ref_id, candidate_ids

def pad_list(raw_list):
    new_list = []
    max_length = np.max([len(x) for x in raw_list])
    cut_length = min(max_length, config.cut_length)
    for oid, one_list in enumerate(raw_list):
        temp = '\n'.join(one_list)
        new_list.append(temp)
    return new_list



def collate_fn_stacked(data):
    batch_title = [x['text_field']['title'] for x in data]
    batch_objective = [x['text_field']['objective'] for x in data]
    batch_data = [[x['data_field']['enrollment_rate'],x['data_field']['site_count'],x['data_field']['country_count']] for x in data]
    batch_study_design = pad_list([x['list_term_field']['studydesign'] for x in data])
    batch_drugcls = pad_list([x['list_term_field']['durgclassification'] for x in data])
    batch_mechanism = pad_list([x['list_term_field']['mechanismsofaction'] for x in data])
    batch_inlcusion = pad_list([x['list_txt_field']['inclusioncriteria'] for x in data])
    batch_exclusion = pad_list([x['list_txt_field']['exclusioncriteria'] for x in data])
    return batch_title, batch_objective, batch_data, \
           batch_study_design, batch_drugcls, batch_mechanism, batch_inlcusion, batch_exclusion
myrange = { 'below': (-np.log(1.25), -np.log(1.5)), 'bad': (np.log(1.5), np.log(2.0)), 'problematic': (np.log(2.0), np.log(np.inf)),
          'median': (-np.log(1.25), np.log(1.25)), 'good': (np.log(1.25), np.log(1.5)), 'great': (np.log(1.5), np.log(2.0)), 'excellent': (np.log(2.0), np.log(np.inf))}
def collect_fn(data):
    temp = []
    labels = []
    enrolls = []
    mid_enrolls = []
    all_enrolls = []
    for data_one_search in data:
        temp += data_one_search[0]
        labels.append(data_one_search[1])
        tar_enrollment = max(data_one_search[0][0]['data_field']['enrollment_rate'], 0)
        candidates_enrollment = []
        for x, label in zip(data_one_search[0][1:], data_one_search[1]):
            if label == 1:
                candidates_enrollment.append(x['data_field']['enrollment_rate'])
        all_enrollment = [
            x['data_field']['enrollment_rate'] for x in
            data_one_search[0][1:]]
        all_enrolls.append(all_enrollment)
        mid = max(np.nanmedian(candidates_enrollment), 0)
        if np.isnan(mid):
            mid = 0
        if np.isnan(tar_enrollment):
            tar_enrollment = mid
        mid_enrolls.append(np.log(mid+1e-10))
        enrolls.append(np.log(tar_enrollment+1e-10))
    batched_parts = collate_fn_stacked(temp)
    labels = torch.LongTensor(labels).to(config.device)
    log_enrolls = torch.FloatTensor(enrolls).to(config.device)
    mid_enrolls = torch.FloatTensor(mid_enrolls).to(config.device)
    all_enrolls = torch.FloatTensor(all_enrolls).to(config.device)
    log_diffs = log_enrolls - mid_enrolls
    return batched_parts, labels, log_enrolls, mid_enrolls, log_diffs, all_enrolls+1e-10

def collect_fn_eval(data):
    temp = []
    labels = []
    enrolls = []
    mid_enrolls = []
    all_enrolls = []
    ref_ids = []
    candidate_ids = []
    for data_one_search in data:
        temp += data_one_search[0]
        labels.append(data_one_search[1])
        ref_ids.append(data_one_search[2])
        candidate_ids.append(data_one_search[3])
        tar_enrollment = max(data_one_search[0][0]['data_field']['enrollment_rate'], 0)
        candidates_enrollment = []
        for x, label in zip(data_one_search[0][1:], data_one_search[1]):
            if label == 1:
                candidates_enrollment.append(x['data_field']['enrollment_rate'])
        all_enrollment = [
            x['data_field']['enrollment_rate'] for x in
            data_one_search[0][1:]]
        all_enrolls.append(all_enrollment)
        mid = max(np.nanmedian(candidates_enrollment), 0)
        if np.isnan(mid):
            mid = 0
        if np.isnan(tar_enrollment):
            tar_enrollment = mid
        mid_enrolls.append(np.log(mid+1e-10))
        enrolls.append(np.log(tar_enrollment+1e-10))
    batched_parts = collate_fn_stacked(temp)
    labels = torch.LongTensor(labels).to(config.device)
    log_enrolls = torch.FloatTensor(enrolls).to(config.device)
    mid_enrolls = torch.FloatTensor(mid_enrolls).to(config.device)
    all_enrolls = torch.FloatTensor(all_enrolls).to(config.device)
    log_diffs = log_enrolls - mid_enrolls
    return batched_parts, labels, log_enrolls, mid_enrolls, log_diffs, all_enrolls+1e-10, ref_ids, candidate_ids

def collect_fn_noraml(data):
    ref = [x[0] for x in data]
    pos = [x[1] for x in data]
    neg = [x[2] for x in data]
    labels = [x[3] for x in data]
    return ref, pos, neg, labels
