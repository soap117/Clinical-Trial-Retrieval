from transformers import AutoTokenizer
import torch
class Config():
    def __init__(self):
        self.cut_length = 10
        self.emb_dim = 256
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.word_emb_dim = 768
        self.filter_size = 2
        self.used_date = '2022-06-15'
        self.epoch = 30
        self.encode = 'usualWOdataSkip'
        self.batch_size = 1
        self.GTT_layer_n = 6
        self.query_attention = False
        #self.multi_gpu = False
        self.multi_gpu = True if torch.cuda.device_count()>1 else False
        self.pos_num = 5
        self.neg_num = 5
        self.lower = 5
        self.upper = 10
        self.small = True
        self.not_full = True
        self.multi_gpu_list = [0, 1]
        self.sample_n = 5
        self.bert_path = 'distilbert-base-uncased'
        self.enroll_mode = 'pure'
        self.token_simi = 'gate'
        self.tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased")
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[unused1]', '[unused2]', '[unused3]', '[unused4]']})
        self.tokenizer.model_max_length = 128
        self.limitation = 'std'

config = Config()