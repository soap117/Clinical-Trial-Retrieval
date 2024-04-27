import pickle
import numpy as np
import torch
from myutils.dataset_eval import Mytrial, collect_fn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torchmetrics import RetrievalMAP, RetrievalMRR, RetrievalPrecision
from config_speed import config
from math import ceil
from myutils.models_speed_heavy import *
from myutils.modeling_trialbert_speed import BertModel
from transformers import BertConfig
def build():
    if config.encode == 'join':
        config.batch_size = 6
    train_dataset = torch.load('./example_train.bin')
    test_dataset = torch.load('./example_test.bin')
    val_dataset = torch.load('./example_val.bin')
    train_dataloader = DataLoader(train_dataset, config.batch_size, collate_fn=collect_fn)
    test_dataloader = DataLoader(test_dataset, 1, collate_fn=collect_fn)
    val_dataloader = DataLoader(val_dataset, 1, collate_fn=collect_fn)
    model = TrialTransformer(config).to(config.device)
    if config.multi_gpu:
        model = nn.DataParallel(model, device_ids=config.multi_gpu_list)
    model.train()
    LR = 1e-4
    BERT_LR = 2e-5
    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if 'bert' not in k]}
    bert_params = {'params': [v for k, v in params if 'bert' in k], 'lr': BERT_LR}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)
    loss_diff_func = nn.L1Loss()
    loss_dist_func = RankNetLoss()
    return train_dataloader, val_dataloader, test_dataloader, model, optimizer, loss_diff_func, loss_dist_func


def sub_process(inputs, enrolls, model):
    sub_size = 20
    if config.limitation == 'one2one':
        sub_size = 1
    sub_num = int(ceil((inputs[0].shape[1]-1)/sub_size))
    pred_enrolls = 0
    total_positive = 1e-10
    similarity_out = []
    for i in range(sub_num):
        sub_inputs = []
        enrolls_sub = enrolls[0:1, i*sub_size:(i+1)*sub_size]
        for input in inputs:
            if input is None:
                sub_inputs.append(None)
                continue
            new_input = torch.cat([input[0:1, 0:1], input[0:1, 1+i*sub_size:1+(i+1)*sub_size]], dim=1)
            sub_inputs.append(new_input)
        config.pos_num = sub_inputs[0].shape[1]-1
        config.neg_num = 0
        summary_out_sub, similarity_out_sub, hiddens = model(*sub_inputs)
        similarity_out.append(similarity_out_sub)
        sigmoid_rs = torch.sigmoid(similarity_out_sub)
    similarity_out = torch.cat(similarity_out, dim=1)
    return similarity_out


def test(test_dataloader, model, optimizer):
    config.cut_length = 30
    RMAP = RetrievalMAP()
    RMRR = RetrievalMRR()
    model.eval()
    rmaps = []
    rmrrs = []
    loss_diffs_count = []
    loss_dis_count = []
    RP_10 = RetrievalPrecision(k=10)
    RP_20 = RetrievalPrecision(k=20)
    rp_10 = []
    rp_20 = []
    with torch.no_grad():
        pred_enrolls = []
        grad_enrolls = []
        for step, (batch_input, labels, log_enrolls, mid_enrolls, log_diffs, enrolls_all) in tqdm(enumerate(test_dataloader)):
            inputs = create_inputs(batch_input, config.tokenizer, ['title', 'objective', 'data', 'studydesign', 'drugcls', 'mechanism', 'inclusion', 'exclusion'])
            similarity_out = sub_process(inputs, enrolls_all, model)
            preds = torch.sigmoid(similarity_out.squeeze(0))
            labels = labels.squeeze(0).unsqueeze(-1)
            indexs = torch.zeros_like(labels, dtype=torch.int64) + step
            rmap = RMAP(preds, labels, indexes=indexs)
            rmrr = RMRR(preds, labels, indexes=indexs)
            rp_10.append(RP_10(preds, labels, indexes=indexs).cpu().numpy())
            rp_20.append(RP_20(preds, labels, indexes=indexs).cpu().numpy())
            rmaps.append(rmap.cpu().numpy())
            rmrrs.append(rmrr.cpu().numpy())
    model.train()
    pos_num_next = np.random.randint(config.lower, config.upper)
    neg_num_next = np.random.randint(config.lower, config.upper)
    config.pos_num = pos_num_next
    config.neg_num = neg_num_next
    print('-----eval results----')
    rs = {'rmap': np.mean(rmaps), 'rmrr': np.mean(rmrrs), 'rp_10': np.mean(rp_10), 'rp_20': np.mean(rp_20)}
    print(rs)


    return rs

def train(train_dataloader, val_dataloader, test_dataloader, model, optimizer):
    best_score = -100
    count_worse = 0
    eval_rs_list = []
    train_loss = []
    #eval_loss = test(test_dataloader, model, optimizer)
    for epoch in range(config.epoch):
        avg_loss = []
        config.cut_length = 10
        print('Start Epoch:{}'.format(epoch))
        for step, (batch_input, labels, log_enrolls, mid_enrolls_gt, log_diffs_gt, enrolls_all) in tqdm(enumerate(train_dataloader)):
            inputs = create_inputs(batch_input, config.tokenizer, ['title', 'objective', 'data', 'studydesign', 'drugcls', 'mechanism', 'inclusion', 'exclusion'])
            summary_out, similarity_out, hiddens = model(*inputs)
            labels_positive = torch.where(labels > 0, torch.ones_like(labels), torch.zeros_like(labels))
            labels_negative = torch.where(labels == 0, torch.ones_like(labels), torch.zeros_like(labels))
            if config.limitation == 'worst':
                mean_simi_positive = torch.sum(similarity_out.squeeze(-1) * labels_positive, dim=-1) / torch.sum(
                    labels_positive, dim=-1)
                mean_simi_negative = torch.sum(similarity_out.squeeze(-1) * labels_negative, dim=-1) / torch.sum(
                    labels_negative, dim=-1)
                labels_dist = torch.ones_like(mean_simi_positive)
                loss_group_distance = loss_dist_func(mean_simi_positive, mean_simi_negative, labels_dist)

                mean_simi_positive = torch.topk(similarity_out.squeeze(-1)*labels_positive, 3, dim=-1, largest=False)[0].mean(dim=-1)
                mean_simi_negative = torch.topk(similarity_out.squeeze(-1)*labels_negative, 3, dim=-1, largest=True)[0].mean(dim=-1)
                labels_dist = torch.ones_like(mean_simi_positive)

                loss_group_distance_worset = loss_dist_func(mean_simi_positive, mean_simi_negative, labels_dist)
                loss_group_distance += 0.1*loss_group_distance_worset
            elif config.limitation == 'std':
                mean_simi_positive = torch.sum(similarity_out.squeeze(-1) * labels_positive, dim=-1) / torch.sum(
                    labels_positive, dim=-1)
                mean_simi_negative = torch.sum(similarity_out.squeeze(-1) * labels_negative, dim=-1) / torch.sum(
                    labels_negative, dim=-1)
                labels_dist = torch.ones_like(mean_simi_positive)
                loss_group_distance = loss_dist_func(mean_simi_positive, mean_simi_negative, labels_dist)
                loss_group_distance_worset = 0
                for bid in range(similarity_out.shape[0]):
                    positive_distance = similarity_out[bid].squeeze(-1)[labels_positive[bid].type(torch.bool)]
                    negative_distance = similarity_out[bid].squeeze(-1)[labels_negative[bid].type(torch.bool)]
                    loss_group_distance_worset += (positive_distance.std()**2+negative_distance.std()**2)/similarity_out.shape[0]
                loss_group_distance += 0.1 * loss_group_distance_worset
            elif config.limitation == 'one2one':
                mean_simi_positive = torch.sum(similarity_out.squeeze(-1) * labels_positive, dim=-1) / torch.sum(
                    labels_positive, dim=-1)
                mean_simi_negative = torch.sum(similarity_out.squeeze(-1) * labels_negative, dim=-1) / torch.sum(
                    labels_negative, dim=-1)
                labels_dist = torch.ones_like(mean_simi_positive)
                loss_group_distance = loss_dist_func(mean_simi_positive, mean_simi_negative, labels_dist)
                loss_group_distance_worset = torch.zeros([1])

            else:
                mean_simi_positive = torch.sum(similarity_out.squeeze(-1) * labels_positive, dim=-1) / torch.sum(
                    labels_positive, dim=-1)
                mean_simi_negative = torch.sum(similarity_out.squeeze(-1) * labels_negative, dim=-1) / torch.sum(
                    labels_negative, dim=-1)
                labels_dist = torch.ones_like(mean_simi_positive)
                loss_group_distance = loss_dist_func(mean_simi_positive, mean_simi_negative, labels_dist)
                loss_group_distance_worset = torch.zeros([1])

            loss = loss_group_distance
            loss_enroll = torch.zeros([1])
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            if step%100 == 0:
                print('Loss diff: %f, Loss group dis: %f Loss group dis worst: %f' %(loss_enroll.item(), loss_group_distance.item(), loss_group_distance_worset.item()))
                #print(torch.sigmoid(similarity_out[0]).squeeze().cpu().detach().numpy())
            pos_num_next = np.random.randint(config.lower, config.upper)
            neg_num_next = np.random.randint(config.lower, config.upper)
            config.pos_num = pos_num_next
            config.neg_num = neg_num_next
        avg_loss = np.mean(avg_loss)
        train_loss.append(avg_loss)
        print(train_loss)
        if config.limitation == 'one2one' and epoch < 14:
            continue
        eval_rs = test(val_dataloader, model, optimizer)
        eval_rs_list.append(eval_rs)
        if eval_rs['rmap'] > best_score:
            count_worse = 0
            best_rs = eval_rs
            best_score = eval_rs['rmap']
            state = {'model': model.state_dict(), 'epoch': epoch, 'eval_rs': best_rs}
            torch.save(state, './cache/best_{}_model_{}_{}_ts_{}_ec_{}_lm_{}.bin'.format(config.used_date, config.enroll_mode, config.small, config.token_simi, config.encode, config.limitation))
            if epoch == 0:
                best_state = torch.load('./cache/best_{}_model_{}_{}_ts_{}_ec_{}_lm_{}.bin'.format(config.used_date, config.enroll_mode, config.small, config.token_simi, config.encode, config.limitation))
                model.load_state_dict(best_state['model'])
        else:
            count_worse += 1
            if count_worse > 9:
                break

    best_state = torch.load('./cache/best_{}_model_{}_{}_ts_{}_ec_{}_lm_{}.bin'.format(config.used_date, config.enroll_mode, config.small, config.token_simi, config.encode, config.limitation))
    model.load_state_dict(best_state['model'])
    eval_rs_test = test(test_dataloader, model, optimizer)
    print('Best val result')
    print(best_rs)
    print('Best test result')
    print(eval_rs_test)
    eval_rs_list.append(eval_rs_test)
    pickle.dump(eval_rs_list, open('./cache/exp_{}_{}_results_{}_ts_{}_ec_{}_lm_{}.pkl'.format(config.used_date, config.enroll_mode, config.small, config.token_simi, config.encode, config.limitation), 'wb'))

if __name__ == '__main__':
    import sys
    if config.limitation == 'one2one':
        config.lower = 1
        config.upper = 2
    train_dataloader, val_dataloader, test_dataloader, model, optimizer, loss_diff_func, loss_dist_func = build()
    train(train_dataloader, val_dataloader, test_dataloader, model, optimizer)