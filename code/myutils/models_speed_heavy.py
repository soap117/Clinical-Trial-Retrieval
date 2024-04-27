import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertModel
from myutils.modeling_trialbert_speed import BertModel
from transformers import BertConfig, DistilBertConfig
from config_speed import config
from math import floor
tokentype2id= {'summary': 0, 'title': 1, 'objective': 2, 'studydesign': 3, 'drugcls':4, 'mechanism':5, 'inclusion':6, 'exclusion':7}

class RankNetLoss(nn.Module):
    def __init__(self):
        super(RankNetLoss, self).__init__()

        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, scores_pos, scores_neg, probs):
        """
        Ranknet loss: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
        """

        x = scores_pos - scores_neg
        loss = self.bce(x, probs)
        return loss

def get_category_position_ids(embeddings, token_type):
    token_types = torch.zeros(embeddings.shape[0:-1], dtype=torch.int64).to(config.device) + tokentype2id[token_type]
    length = embeddings.shape[2]
    token_positions = torch.arange(0, length, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
    token_positions = token_positions.expand(embeddings.shape[0:3])
    return token_positions, token_types

def get_inner_connection(section_lens):
    # first be [SUMMARY], second be [SIMILARITY]
    length = np.sum(section_lens)
    temp = torch.zeros([length, length], dtype=torch.int)
    temp[0, :] = 1
    temp[:, 0] = 1
    r_index = 0
    for s_id, section_len in enumerate(section_lens):
        temp[r_index:r_index+section_len, r_index:r_index+section_len] = 1
        r_index += section_len
    return temp
# [B, G, S, S]
def fill_inner_connection(inner_connection_shape, inner_connection):
    length = inner_connection_shape[1]*inner_connection_shape[2]
    temp = torch.zeros([length, length], dtype=torch.int)
    r_index = 0
    for i in range(inner_connection_shape[1]):
        temp[r_index:r_index+inner_connection_shape[2], r_index:r_index+inner_connection_shape[2]] = inner_connection
        r_index += inner_connection_shape[2]
    return temp
def fill_query_connection(section_lengths, inner_connection_shape, temp):
    query_anchor = 0
    trial_length = np.sum(section_lengths)
    # [SUMMARY]
    for s_id, section_length in enumerate(section_lengths):
        for i in range(inner_connection_shape[1]):
            temp[query_anchor+i*trial_length:query_anchor+i*trial_length+section_length, query_anchor:query_anchor+section_length] = 1
        query_anchor += section_length
    return temp.to(config.device)

class LongTextEncoder(nn.Module):
    def __init__(self, config):
        super(LongTextEncoder, self).__init__()
        dis_config = DistilBertConfig.from_pretrained(config.bert_path)
        dis_config.n_layers = 6
        self.bert = DistilBertModel(dis_config).from_pretrained(config.bert_path, output_hidden_states=True)
        self.align_layer = nn.Sequential(nn.Linear(768, config.emb_dim), nn.Tanh())
        self.drop_layer = nn.Dropout(0.15)

    def forward(self, batch_sections, mask):
        if config.cpu_test:
            temp = [torch.zeros([batch_sections.shape[0], batch_sections.shape[1], 768]).to(config.device) for x in range(7)]
            batch_embeddings = [torch.zeros([batch_sections.shape[0], batch_sections.shape[1], 768]).to(config.device), temp]
        else:
            batch_embeddings = self.bert(batch_sections, mask)
        x = batch_embeddings[0][:, 0, :]
        x = self.drop_layer(x)
        x = self.align_layer(x)
        x_words = batch_embeddings[1]
        return x, x_words

class ShortTermEncoder(nn.Module):
    def __init__(self, config, embeddings):
        super(ShortTermEncoder, self).__init__()
        self.embed = nn.Embedding(embeddings.word_embeddings.weight.shape[0], embeddings.word_embeddings.weight.shape[1], padding_idx=0)
        self.embed.weight.data[:,:] = embeddings.word_embeddings.weight.data[:,:]
        self.conv = nn.ModuleList()
        self.drop_layer = nn.Dropout(0.15)
        self.align_layer = nn.Sequential(nn.Linear(config.emb_dim, config.emb_dim), nn.Tanh())
        self.align_layer_2 = nn.Sequential(nn.Linear(config.word_emb_dim, config.emb_dim), nn.Tanh())
        filter_size = config.filter_size
        for l in range(6):
            if l == 0:
                tmp = nn.Conv1d(config.word_emb_dim, config.emb_dim, kernel_size=filter_size,
                                padding=int(floor(filter_size / 2)))
            else:
                tmp = nn.Conv1d(config.emb_dim, config.emb_dim, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            self.conv.add_module('baseconv_%d' %l, tmp)
            if l == 0:
                tmp = nn.MaxPool1d(3, 2)
            else:
                tmp = nn.MaxPool1d(2, 2)
            self.conv.add_module('pool_%d' %l, tmp)
            tmp = nn.ReLU()
            self.conv.add_module('ReLu_%d' % l, tmp)

    def forward(self, input_ids, is_special=False):
        x = self.embed(input_ids)
        if not is_special:
            x = x.transpose(1, 2)
            tmp = x
            for idx, md in enumerate(self.conv):
                tmp = md(tmp)
            x = tmp
            x, _ = torch.max(x, dim=2)
            x = self.drop_layer(x)
            x = self.align_layer(x)
        else:
            x = x[:,1,:]
            x = self.align_layer_2(x)
        return x

class CategoryEncoder(nn.Module):
    def __init__(self, config):
        super(CategoryEncoder, self).__init__()
        self.embedd = nn.Embedding(12, config.emb_dim)
    def forward(self, cat_ids):
        return self.embedd(cat_ids)

class ToeknSimiEncoder(nn.Module):
    def __init__(self, config):
        super(ToeknSimiEncoder, self).__init__()
        self.embedd = nn.Embedding(12, config.emb_dim)
        print('Using Semantic Dim {}'.format(config.embedding_compression_num))
        if config.embedding_compression_num > 0:
            self.word_align_layer = nn.Sequential(nn.Linear(768, config.embedding_compression_num), nn.LayerNorm(config.embedding_compression_num))
        self.k = 3
        self.n_layer = 7+config.embedding_compression_num
        self.conv = nn.ModuleList()
        filter_size = 3
        emb_dim = 256
        if 'Filter' in config.encode:
            self.importance_filter = nn.Linear(768, 1)
        self.drop_layer = nn.Dropout(0.15)
        self.align_layer = nn.Sequential(nn.Linear(config.emb_dim//4, config.emb_dim), nn.Tanh())
        for l in range(4):
            if l == 0:
                tmp = nn.Conv2d(self.n_layer, config.emb_dim//4, kernel_size=filter_size,
                                padding=int(floor(filter_size / 2)))
            else:
                tmp = nn.Conv2d(config.emb_dim//4, config.emb_dim//4, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            self.conv.add_module('baseconv_%d' %l, tmp)
            tmp = nn.MaxPool2d(2, 2)
            self.conv.add_module('pool_%d' %l, tmp)
            tmp = nn.ReLU()
            self.conv.add_module('ReLu_%d' % l, tmp)

    def forward(self, qtitle_token_embeddings, ctitle_token_embeddings, is_eval=False):
        # embedding enhanced
        if config.embedding_compression_num > 0:
            qc_map = self.word_align_layer(qtitle_token_embeddings[0]).unsqueeze(2)+self.word_align_layer(ctitle_token_embeddings[0]).unsqueeze(3)
            qc_map = qc_map.view(-1, qc_map.shape[2], qc_map.shape[3], qc_map.shape[4])
            # q[B,1,L,D] c[B,K,L,D] [B,L,D] [B, K*L, D] [B, K*L, L]
        simis = []

        B = qtitle_token_embeddings[0].shape[0]
        LQ = qtitle_token_embeddings[0].shape[2]
        LC = ctitle_token_embeddings[0].shape[2]
        LK = ctitle_token_embeddings[0].shape[1]
        if 'Filter' in config.encode:
            Q_filter = torch.sigmoid(self.importance_filter(qtitle_token_embeddings[0])).view(B, 1, LQ)
            C_filter = torch.sigmoid(self.importance_filter(ctitle_token_embeddings[0])).view(B, LC*LK, 1)
            CQ_filter = C_filter@Q_filter
        #[B, 1, LQ] [B, LC*LK, 1] -> [B, LC*LK, LQ]
        for a_emb, b_emb in zip(qtitle_token_embeddings, ctitle_token_embeddings):
            a_emb = a_emb.squeeze(1)
            b_emb = b_emb.reshape(b_emb.shape[0], -1, b_emb.shape[-1])
            a_denom = a_emb.norm(p=2, dim=2).reshape(B, 1, LQ).expand(B, LC*LK, LQ) + 1e-9  # avoid 0div
            b_denom = b_emb.norm(p=2, dim=2).reshape(B, LC*LK, 1).expand(B, LC*LK, LQ) + 1e-9  # avoid 0div
            perm = a_emb.permute(0, 2, 1)
            if 'Filter' in config.encode:
                sim = b_emb.bmm(perm)*CQ_filter
            else:
                sim = b_emb.bmm(perm)
            sim = sim / (a_denom * b_denom)
            # [B, K*L, L]
            sim = sim.view(B, LK, LC, LQ).view(B*LK, LC, LQ)
            simis.append(sim)
        # [B*K, 6, LC, LQ]
        if config.embedding_compression_num > 0:
            qc_map = qc_map.permute(0, 3, 1, 2)
        simis = torch.stack(simis, dim=1)
        if config.embedding_compression_num > 0:
            simis = torch.cat([simis, qc_map], dim=1)
        #print(simis.shape)
        if is_eval:
            simi_raw = simis
            simis.retain_grad()
        for idx, md in enumerate(self.conv):
            if idx == 0 and is_eval:
                features = simis
                features.retain_grad()
            simis = md(simis)
        x = torch.max(simis, dim=-1)[0].max(dim=-1)[0]
        x = self.align_layer(x)
        x = self.drop_layer(x)
        # [B*K, 256]
        x = x.view(B, LK, -1)
        if is_eval:
            return x, features, simi_raw
        return x


def add_special(titles):
    summary = []
    summary2 = []
    similarity = []
    similarity2 = []
    for title in titles:
        summary.append('[unused1]')
        summary2.append('[unused2]')
        similarity.append('[unused3]')
        similarity2.append('[unused4]')
    return summary, summary2, similarity, similarity2
def create_inputs(batch_input, tokenizer, category_seq):
    inputs = []
    summary, summary2, similarity, similarity2 = add_special(batch_input[0])
    summary_data = tokenizer(summary, padding=True, return_tensors='pt', truncation=True)
    summary_ids = summary_data['input_ids'].to(config.device)
    summary_mask = summary_data['attention_mask'].to(config.device)
    inputs.append(summary_ids.view(-1, config.pos_num + config.neg_num + 1, summary_ids.shape[-1]))
    inputs.append(summary_mask.view(-1, config.pos_num + config.neg_num + 1, summary_ids.shape[-1]))
    similarity_data = tokenizer(similarity, padding=True, return_tensors='pt', truncation=True)
    similarity_ids = similarity_data['input_ids'].to(config.device)
    similarity_mask = similarity_data['attention_mask'].to(config.device)
    inputs.append(similarity_ids.view(-1, config.pos_num + config.neg_num + 1, similarity_ids.shape[-1]))
    inputs.append(similarity_mask.view(-1, config.pos_num + config.neg_num + 1, similarity_ids.shape[-1]))
    for bid, (one_type_input, cat) in enumerate(zip(batch_input, category_seq)):
        if bid == 2:
            # skip data input
            batch_data = torch.log(torch.FloatTensor(one_type_input).nan_to_num(0)+1e-10).to(config.device)
            batch_data = batch_data.view(-1, config.pos_num + config.neg_num + 1, batch_data.shape[-1])
            batch_data[:, 0] = 0
            if 'WOdata' in config.encode:
                batch_data[:, :] = 0
            inputs.append(batch_data)
            inputs.append(None)
            continue
        batch_data = tokenizer(one_type_input, padding=True, return_tensors='pt', truncation=True)
        batch_ids = batch_data['input_ids'].to(config.device)
        batch_mask = batch_data['attention_mask'].to(config.device)
        if bid <= 2:
            inputs.append(batch_ids.view(-1, config.pos_num + config.neg_num + 1, batch_ids.shape[-1]))
            inputs.append(batch_mask.view(-1, config.pos_num + config.neg_num + 1, batch_ids.shape[-1]))
        else:
            # B*NUM*SEQ
            inputs.append(batch_ids.view(inputs[0].shape[0], config.pos_num + config.neg_num + 1, -1, batch_ids.shape[-1]))
            inputs.append(batch_mask.view(inputs[0].shape[0], config.pos_num + config.neg_num + 1, -1, batch_ids.shape[-1]))
    return inputs


class TrialTransformer(nn.Module):
    def __init__(self, config):
        super(TrialTransformer, self).__init__()
        if 'global' in config.token_simi:
            self.category_seq = ['title', 'objective', 'data', 'studydesign', 'drugcls', 'mechanism', 'inclusion', 'exclusion']
        else:
            self.category_seq = ['title', 'objective']
        self.model_long = LongTextEncoder(config)
        self.model_short = ShortTermEncoder(config, self.model_long.bert.embeddings)
        self.model_data = nn.Linear(3, config.emb_dim)
        self.model_cat = CategoryEncoder(config)
        self.model_token_simi = ToeknSimiEncoder(config)
        self.tokenizer = config.tokenizer
        transformer_config = BertConfig.from_pretrained('bert-base-uncased')
        transformer_config.hidden_size = config.emb_dim
        transformer_config.intermediate_size = 4 * config.emb_dim
        transformer_config.num_hidden_layers = 3
        transformer_config.num_attention_heads = 8
        transformer_config.max_position_embeddings = 512
        self.model_transformer = BertModel(transformer_config)
        if 'gate' in config.token_simi:
            self.simi_cls_out = nn.Linear(config.emb_dim, 1)
            self.simi_similarity_out = nn.Linear(config.emb_dim, 1)
            self.simi_global_out = nn.Linear(config.emb_dim, 1)
        elif config.token_simi == 'wordglobal':
            self.simi_global_out = nn.Linear(3*config.emb_dim, 1)
        elif config.token_simi == 'none':
            self.simi_global_out = nn.Linear(config.emb_dim, 1)
        else:
            self.simi_global_out = nn.Linear(config.emb_dim, 1)
        self.weight_out = nn.Linear(config.emb_dim, 1)
        self.cls_simi = nn.Sequential(nn.Linear(2*config.emb_dim, config.emb_dim), nn.Tanh())
        self.gate_layer_word = nn.Sequential(nn.Linear(4*config.emb_dim, config.emb_dim), nn.Tanh(), nn.Linear(config.emb_dim, 4))
        self.gate_layer_global = nn.Sequential(nn.Linear(config.emb_dim, config.emb_dim), nn.Tanh(),
                                             nn.Linear(config.emb_dim, 4))
        self.drop_layer = nn.Dropout(0.15)

    def apply_transformer(self, final_input_embs, final_type_ids, final_positions, is_eval):
        complete_embeddings = torch.cat(final_input_embs, dim=2)
        length_of_sections = [x.shape[2] for x in final_input_embs]
        connect_map = get_inner_connection(length_of_sections)
        connect_map = fill_inner_connection(complete_embeddings.shape[0:3], connect_map)
        connect_map = fill_query_connection(length_of_sections, complete_embeddings.shape[0:3], connect_map)
        # complete_embeddings_stacked = complete_embeddings.view(complete_embeddings.shape[0], -1,
        #                                                       complete_embeddings.shape[-1])
        connect_map = connect_map.unsqueeze(0).unsqueeze(0).to(complete_embeddings.device)

        # process categories
        final_type_ids = torch.cat(final_type_ids, dim=2)
        final_type_embeddings = self.model_cat(final_type_ids)
        # complete_categorys_stacked = final_type_embeddings.view(final_type_embeddings.shape[0], -1,
        #                                                       final_type_embeddings.shape[-1])
        # process token positions
        complete_position_ids = torch.cat(final_positions, dim=2)
        complete_position_ids_stacked = complete_position_ids.view(complete_position_ids.shape[0], -1).to(
            complete_embeddings.device)
        # final modeling
        if 'WOC' in config.encode:
            connect_map = None
        if not is_eval:
            (summary_embeddings), hiddens = self.model_transformer(
                inputs_embeds=complete_embeddings + final_type_embeddings,
                connection_mask=connect_map, position_ids=complete_position_ids_stacked)
        else:
            (summary_embeddings), hiddens = self.model_transformer(
                inputs_embeds=complete_embeddings + final_type_embeddings,
                connection_mask=connect_map, position_ids=complete_position_ids_stacked, output_attentions=True)

        return summary_embeddings, hiddens

    def forward(self, *batch_input, is_eval=False):
        batch_size = batch_input[0].shape[0]
        final_input_embs = []
        final_positions = []
        final_type_ids = []
        summary_embeddings = self.model_short(batch_input[0].view(-1, batch_input[0].shape[-1]), True)
        summary_embeddings = summary_embeddings.view(batch_size, config.pos_num + config.neg_num + 1, -1,
                                                     summary_embeddings.shape[-1])
        summary_token_positions, summary_token_types = get_category_position_ids(summary_embeddings,
                                                                                 'summary')
        final_input_embs.append(summary_embeddings)
        final_positions.append(summary_token_positions)
        final_type_ids.append(summary_token_types)

        similarity_embeddings_group = []
        similarity_embeddings_cls = []
        for bid, cat in enumerate(self.category_seq):
            if bid == 2:
                continue
            else:
                if 'global' in config.token_simi:
                    batch_embeddings = self.model_short(
                        batch_input[bid * 2 + 4].view(-1, batch_input[bid * 2 + 4].shape[-1]))
                    batch_embeddings = batch_embeddings.view(batch_size, config.pos_num + config.neg_num + 1, -1,
                                                             batch_embeddings.shape[-1])
                    batch_token_positions, batch_token_types = get_category_position_ids(batch_embeddings,
                                                                                      cat)
                    final_input_embs.append(batch_embeddings)
                    final_positions.append(batch_token_positions)
                    final_type_ids.append(batch_token_types)

                if cat in ['title', 'objective']:
                    batch_embeddings, batch_token_embeddings_hiddens = self.model_long(
                        batch_input[bid * 2 + 4].view(-1, batch_input[bid * 2 + 4].shape[-1]),
                        batch_input[bid * 2 + 5].view(-1, batch_input[bid * 2 + 5].shape[-1]))
                    batch_embeddings = batch_embeddings.view(batch_size, config.pos_num + config.neg_num + 1, -1,
                                                             batch_embeddings.shape[-1])
                    batch_token_embeddings = [x.view(batch_size, config.pos_num + config.neg_num + 1, -1,
                                                     x.shape[-1]) * batch_input[bid * 2 + 5].unsqueeze(-1) for x in
                                              batch_token_embeddings_hiddens]
                    q_title_embeddings = [x[:, 0:1, 1:] for x in batch_token_embeddings]
                    c_title_embeddings = [x[:, 1:, 1:] for x in batch_token_embeddings]
                    q_cls = batch_embeddings[:, 0:1, 0]
                    c_cls = batch_embeddings[:, 1:, 0]
                    q_cls = q_cls.expand_as(c_cls)
                    simi_info = torch.cat([q_cls-c_cls, q_cls*c_cls], dim=-1)
                    cls_simi_embeddings = self.cls_simi(simi_info)

                    similarity_embeddings_cls.append(q_cls)
                    similarity_embeddings_cls.append(c_cls)

                    #[4, 1, 256] [4, 10, 256] [4, 10, 512]
                    if is_eval:
                        similarity_embeddings, temp, temp_raw = self.model_token_simi(q_title_embeddings, c_title_embeddings, is_eval=is_eval)
                        if cat == 'title':
                            title_token_feature = temp
                            title_token_raw = temp_raw
                    else:
                        similarity_embeddings = self.model_token_simi(q_title_embeddings, c_title_embeddings,
                                                                            is_eval=is_eval)
                    similarity_embeddings_group.append(cls_simi_embeddings)
                    similarity_embeddings_group.append(similarity_embeddings)

        similarity_embeddings_cls = self.drop_layer(torch.cat(similarity_embeddings_cls, dim=2))
        gate_out_word = self.gate_layer_word(similarity_embeddings_cls)

        if 'global' in config.token_simi:
            summary_embeddings, hiddens = self.apply_transformer(final_input_embs, final_type_ids, final_positions, is_eval)
            gate_out_global = self.gate_layer_global(summary_embeddings)

        if config.token_simi == 'gate':
                similarity_out = torch.cat([self.simi_cls_out(similarity_embeddings_group[0]), self.simi_cls_out(similarity_embeddings_group[2]),
                                        self.simi_similarity_out(similarity_embeddings_group[1]), self.simi_similarity_out(similarity_embeddings_group[3])], dim=-1)
                gate = torch.softmax(gate_out_word, dim=-1)
                similarity_out = (gate*similarity_out).sum(dim=-1, keepdim=True)

        elif config.token_simi == 'contact':
            similarity_out = torch.concat(similarity_embeddings_group, dim=-1)
            similarity_out = self.simi_out(similarity_out)
        elif config.token_simi == 'contactword':
            similarity_out = torch.concat([similarity_embeddings_group[1],similarity_embeddings_group[3]], dim=-1)
            similarity_out = self.simi_out(similarity_out)
        elif config.token_simi == 'global':
            similarity_out_global = self.simi_global_out(summary_embeddings)
            similarity_out = similarity_out_global
        elif config.token_simi == 'gateword':
            similarity_out_title = self.simi_similarity_out(similarity_embeddings_group[1])
            similarity_out_objective = self.simi_similarity_out(similarity_embeddings_group[3])
            similarity = torch.cat([similarity_out_title, similarity_out_objective], dim=-1)
            gate = torch.softmax(gate_out_word[:, :, 0:2], dim=-1)
            similarity_out = (gate * similarity).sum(dim=-1, keepdim=True)
        elif config.token_simi == 'gatewordglobal':
            similarity_out_title = self.simi_similarity_out(similarity_embeddings_group[1])
            similarity_out_objective = self.simi_similarity_out(similarity_embeddings_group[3])
            similarity_out_global = self.simi_global_out(summary_embeddings)
            similarity = torch.cat([similarity_out_title, similarity_out_objective, similarity_out_global], dim=-1)
            gate = torch.softmax(gate_out_global[:, :, 0:3], dim=-1)
            similarity_out = (gate * similarity).sum(dim=-1, keepdim=True)
        elif config.token_simi == 'wordglobal':
            similarity_out = torch.concat([summary_embeddings, similarity_embeddings_group[1], similarity_embeddings_group[3]], dim=-1)
            similarity_out = self.simi_global_out(similarity_out)
        if is_eval:
            similarity_out_title.sum().backward()
            grad_map = title_token_feature.grad[:, ]
            torch.cuda.empty_cache()
            hiddens = (hiddens, grad_map, title_token_raw)
            return similarity_out_global, similarity_out, hiddens
        else:
            return None, similarity_out, None






