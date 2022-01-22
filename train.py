import json, time 
import numpy as np 
from tqdm import tqdm 

from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertModel, BertTokenizer
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import os
import unicodedata
from pyhanlp import *

#from reasoner import *
from conopt import *
from gcn import *
from graphModule import *
from einops import rearrange
from config import args
from biaffine import *
from crf import CRF
from torch_multi_head_attention import MultiHeadAttention
#from apnb import *
import statistics
from axial_attention import AxialAttention
from recall import *
from torch import optim
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#BERT_PATH = "./chinese_roberta_wwm_ext_pytorch"
BERT_PATH = "./chinese_roberta_wwm_ext_pytorch"
maxlen = 256 ####256 
'''
def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in tqdm(f):
            l = json.loads(l)
            d = {'text': l['text'], 'triple_list': []}
            for spo in l['triple_list']:
                for k, v in spo['object'].items():
                    d['triple_list'].append(
                        (spo['subject'], spo['predicate'] + '_' + k, v)
                    )
            D.append(d)
    return D 
'''
def load_data(filename):
    D = []
    
    data = json.load(open(filename))
    for item in data:
        d = {'text': item['text'], 'triple_list': []}
        for sub_item in item['triple_list']:
            d['triple_list'].append(
                        (sub_item[0], sub_item[1], sub_item[2])
                    )
        D.append(d)
    
    return D 


# 加载数据集
train_data = load_data('./data/CMED/train_triples.json') 
valid_data = load_data('./data/CMED/dev_triples.json') 
print ("验证集大小：", len(valid_data))

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

train_data_new = []   # 创建新的训练集，把结束位置超过250的文本去除，可见并没有去除多少
for data in tqdm(train_data):
    #print (data)
    flag = 1
    for s, p, o in data['triple_list']:
        s_begin = search(s, data['text'])
        o_begin = search(o, data['text'])
        if s_begin == -1 or o_begin == -1 or s_begin + len(s) > 250 or o_begin + len(o) > 250:
            flag = 0
            break 
    if flag == 1:
        train_data_new.append(data)
print(len(train_data_new))

# 读取schema
'''
with open('RE/data/schema.json', encoding='utf-8') as f:
    id2predicate, predicate2id, n = {}, {}, 0
    predicate2type = {}
    for l in f:
        l = json.loads(l)
        predicate2type[l['predicate']] = (l['subject_type'], l['object_type'])
        for k, _ in sorted(l['object_type'].items()):
            key = l['predicate'] + '_' + k
            id2predicate[n] = key
            predicate2id[key] = n
            n += 1
print(len(predicate2id))
'''

with open('./data/CMED/rel2id.json', encoding='utf-8') as f:
    #id2predicate, predicate2id, n = {}, {}, 0
    l = json.load(f)
    id2predicate = l[0] 
    predicate2id = l[1]  
print(len(predicate2id))  


class OurTokenizer(BertTokenizer):
    def tokenize(self, text):
        R = []
        for c in text:
            if c in self.vocab:
                R.append(c)
            elif self._is_whitespace(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R 
    
    def _is_whitespace(self, char):
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

# 初始化分词器
tokenizer = OurTokenizer(vocab_file="./chinese_roberta_wwm_ext_pytorch/vocab.txt")   

######依存句法树+分词
def seg_pos(text):
    head, seg_word, Dep_rel, str_le = [], [], [], []
    #tree = HanLP.parseDependency(text)
    parser = JClass('com.hankcs.hanlp.dependency.nnparser.NeuralNetworkDependencyParser')()
    parser.enableDeprelTranslator(False)
    tree = parser.parse(text)    
    for word in tree.iterator():  # 通过dir()可以查看sentence的方法
        head.append(word.HEAD.ID)
        for i in word.LEMMA.split():
            str_le.append(i)
        seg_word.append(word.LEMMA)
        Dep_rel.append(word.DEPREL)
    return head,seg_word,Dep_rel,str_le
 
def out_list_word(seg_word):
    temp = ""
    for word in seg_word:
        temp += " " + word
        text_out = temp.lstrip(" ") 
    return text_out 

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab.keys() else 0 for t in tokens]
    return ids

def vocab_json():
    vocab_out = json.load(open("./Geo_dic.json"))
    return vocab_out

def dep_json():
    dep_out = json.load(open("./dep.json"))
    return dep_out
    
class TorchDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        t = self.data[i]
        #print ('t!!!',t) ######{'text': '齐志江，男，汉族，中共党员，大学学历', 'triple_list': [('齐志江', '民族', '汉族')]}
        word_input, center_word = [], []
        head_in, seg_word, Dep_rel, str_le = seg_pos(t['text'])
        
        head = [x for x in head_in]
        for j in head:
            if j == 0:
                center_word.append('Root')
            else:
                center_word.append(str_le[j-1])
        center_word = map_to_ids(center_word, vocab_json()) #中心词
        Dep_rel = map_to_ids(Dep_rel, dep_json()) #依存关系
        text1 = out_list_word(seg_word) 
        text1_list = text1.split() 
        seg_word_out = map_to_ids(text1_list, vocab_json()) #词
        
        head = np.array(head)
        
        text = out_list_word(seg_word) 
        text_list = text.split() 
        #print (text_list)
        for item in text_list:
            length = len(item)
            word = [item] * length
            word_input.extend(word)
        if len(word_input) > 256:
            word_input = word_input[:256]
        word_input = ["[CLS]"] + word_input + ["[SEP]"]
        #print ('train_word_input:',word_input)
        tokens_words = map_to_ids(word_input, vocab_json()) ######词向量
        word_seg_ids = [0] * len(tokens_words) 
        
        
        x = tokenizer.tokenize(t['text'])
        #print (x)
        x = ["[CLS]"] + x + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(x)
        seg_ids = [0] * len(token_ids) 
        assert len(token_ids) == len(t['text'])+2
        spoes = {}
        for s, p, o in t['triple_list']:
            s = tokenizer.tokenize(s)
            s = tokenizer.convert_tokens_to_ids(s)
            p = predicate2id[p]
            o = tokenizer.tokenize(o)
            o = tokenizer.convert_tokens_to_ids(o)
            s_idx = search(s, token_ids)
            o_idx = search(o, token_ids)
    
            if s_idx != -1 and o_idx != -1:
                s = (s_idx, s_idx + len(s) - 1)
                o = (o_idx, o_idx + len(o) - 1, p)  # 同时预测o和p
                if s not in spoes:
                    spoes[s] = []
                spoes[s].append(o)
        #print(spoes) {(2, 5): [(13, 15, 31), (19, 21, 38), (29, 31, 45)]}

        if spoes:
            sub_labels = np.zeros((len(token_ids), 2))
            #print (sub_labels)
            for s in spoes:
                #print (s) #(2, 5)
                #print (sub_labels)
                #print(s[0]) 
                sub_labels[s[0], 0] = 1 
                sub_labels[s[1], 1] = 1
            # 随机选一个subject
            start, end = np.array(list(spoes.keys())).T
            start = np.random.choice(start)
            #print (start)
            end = sorted(end[end >= start])[0]
            sub_ids = (start, end)
            obj_labels = np.zeros((len(token_ids), len(predicate2id), 2))
            for o in spoes.get(sub_ids, []):
                #print (o)
                obj_labels[o[0], o[2], 0] = 1 
                obj_labels[o[1], o[2], 1] = 1 
        
        token_ids = self.sequence_padding(token_ids, maxlen=maxlen)
        seg_ids = self.sequence_padding(seg_ids, maxlen=maxlen)
        sub_labels = self.sequence_padding(sub_labels, maxlen=maxlen, padding=np.zeros(2))
        sub_ids = np.array(sub_ids)
        obj_labels = self.sequence_padding(obj_labels, maxlen=maxlen,
                                           padding=np.zeros((len(predicate2id), 2)))
        
        head_out = torch.LongTensor(self.sequence_padding(head, maxlen=maxlen))
        tokens_words_out = torch.LongTensor(self.sequence_padding(tokens_words, maxlen=maxlen))
        word_seg_ids = self.sequence_padding(word_seg_ids, maxlen=maxlen)
        masks_out = torch.eq(tokens_words_out, 0)
        
        center_word = torch.LongTensor(self.sequence_padding(center_word, maxlen=maxlen))
        Dep_rel = torch.LongTensor(self.sequence_padding(Dep_rel, maxlen=maxlen))
        seg_word_out = torch.LongTensor(self.sequence_padding(seg_word_out, maxlen=maxlen))
        mask_word_out = torch.eq(seg_word_out, 0)
        #print (mask_word_out)
        
        return (torch.LongTensor(token_ids), torch.LongTensor(seg_ids), torch.LongTensor(sub_ids),  
               torch.LongTensor(sub_labels), torch.LongTensor(obj_labels), tokens_words_out, masks_out, head_out,
               center_word, Dep_rel, seg_word_out, mask_word_out, torch.LongTensor(word_seg_ids))
 
    def __len__(self):
        data_len = len(self.data)
        return data_len
    
    def sequence_padding(self, x, maxlen, padding=0):
        output = np.concatenate([x, [padding]*(maxlen-len(x))]) if len(x)<maxlen else np.array(x[:maxlen])
        return output 

train_dataset = TorchDataset(train_data_new)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch1, shuffle=True,drop_last = True)
# for i, x in enumerate(train_loader):
#     print([_.shape for _ in x])
#     if i == 10:
#         break
class GRUnet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        """
        vocab_size: 词典长度，也就是嵌入矩阵的行数
        embedding_dim: 词向量的维度，也就是嵌入矩阵的列数，也是W的列数，也是输入GRU的x_t的维度
        hidden_dim: GRU神经元的个数，也就是W的行数
        layer_dim: GRU的层数
        output_dim: 隐藏层输出的维度
        """
        super(GRUnet, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # GRU + 全连接
        self.gru = nn.GRU(embedding_dim, hidden_dim, layer_dim,
                         batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # x : [bacth, time_step, vocab_size]
        embeds = self.embedding(x)
        #print(embeds.shape)
        # embeds : [batch, time_step, embedding_dim]
        r_out, h_n = self.gru(embeds, None)
        #print (r_out.shape)
        # r_out : [batch, time_step, hidden_dim]
        #out = self.fc1(r_out[:, -1, :])
        out = self.fc1(r_out)
        # out : [batch, time_step, output_dim]
        return out

class GCN(nn.Module):
    def __init__(self, hidden_size=768):
        super(GCN, self).__init__()
        self.hidden_size = hidden_size
        #self.fc = nn.Linear(self.hidden_size, self.hidden_size // 2)

    def forward(self, x, adj, is_relu=True):
        out = x

        # Make permutations for matrix multiplication
        # Assuming batch_first = False
        #print (out.shape)
        #out = out.permute(1, 0, 2) # to: batch, seq_len, hidden
        #adj = adj.permute(2, 0, 1) # to: batch, seq_len, seq_len

        out = torch.bmm(adj, out) #.permute(1, 0, 2) # to: seq_len, batch, hidden

        if is_relu == True:
            out = F.relu(out)

        return out
        
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)   # [bs, maxlen, 1]
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Attention2(nn.Module):
    """
    1.输入 [batch_size,time_step,hidden_dim] -> Linear、Tanh
    2.[batch_size,time_step,hidden_dim] -> transpose
    3.[batch_size,hidden_dim,time_step] -> Softmax
    4.[batch_size,hidden_dim,time_step] -> mean
    5.[batch_size,time_step] -> unsqueeze
    5.[batch_size,1,time_step] -> expand
    6.[batch_size,hidden_dim,time_step] -> transpose
    7.[batch_size,time_step,hidden_dim]
    """

    def __init__(self, hidden_dim):
        super(Attention2, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, features, mean=True):
        batch_size, time_step, hidden_dim = features.size()
        #weight = nn.Tanh()(self.dense(features))
        weight = nn.ReLU()(self.dense(features))

        # mask给负无穷使得权重为0
        mask_idx = torch.sign(torch.abs(features).sum(dim=-1))
        mask_idx = mask_idx.unsqueeze(-1).expand(batch_size, time_step, hidden_dim)
        paddings = torch.ones_like(mask_idx) * (-2 ** 32 + 1)
        weight = torch.where(torch.eq(mask_idx, 1), weight, paddings)

        weight = weight.transpose(2, 1)
        #weight = nn.Softmax(dim=2)(weight)
        #weight = nn.Sigmoid(weight)
        if mean:
            weight = weight.mean(dim=1)
            weight = weight.unsqueeze(1)
            weight = weight.expand(batch_size, hidden_dim, time_step)
        weight = weight.transpose(2, 1)
        features_attention = weight * features

        return features_attention

class KeyValueMemoryNetwork(nn.Module):
    def __init__(self, vocab_size, feature_vocab_size, emb_size):
        super(KeyValueMemoryNetwork, self).__init__()
        self.key_embedding = nn.Embedding(vocab_size, emb_size, padding_idx = 0)
        self.value_embedding = nn.Embedding(feature_vocab_size, emb_size, padding_idx = 0)
        self.scale = np.power(emb_size, 0.5)

    def forward(self, key_embed, value_embed, hidden, mask_matrix):

        #key_embed = self.key_embedding(key_seq)
        #print (key_embed.shape)
        #value_embed = self.value_embedding(value_seq)
        #print (value_embed.shape)
        #hidden = self.key_embedding(hidden)
        u = torch.bmm(hidden.float(), key_embed.transpose(1, 2))
        u = u / self.scale
        exp_u = torch.exp(u)
        #print ('exp_u',exp_u.shape)
        delta_exp_u = torch.mul(exp_u.float(), mask_matrix.float())
        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)
        p = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)
        #print ('exp_u',p.shape)(9,256,256)
        #embedding_val = value_embed.permute(3, 0, 1, 2)
        o = torch.mul(p.float(), value_embed.float())
        #print (o.shape)
        #o = o.permute(1, 2, 3, 0)
        #o = torch.sum(o, 2)

        #aspect_len = (o != 0).sum(dim=1)
        #o = o.float().sum(dim=1)
        #avg_o = torch.div(o, aspect_len)
        return o#avg_o.type_as(hidden)

def entity_average(hidden_output, e_mask):
    """
    Average the entity hidden state vectors (H_i ~ H_j)
    :param hidden_output: [batch_size, j-i+1, dim]
    :param e_mask: [batch_size, max_seq_len]
    e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
    :return: [batch_size, dim]
    """
    e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
    length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

    # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
    sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
    avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
    return avg_vector

class WordCharLSTM(nn.Module):
    def __init__(
            self,
            num_word_embeddings,
            num_tags,
            word_embeddings,
            num_char_embeddings,
            word_lstm,
            char_lstm,
            char_padding_idx=0,
            train_word_embeddings=False):
        super(WordCharLSTM, self).__init__()

        self.char_embeddings = nn.Embedding(
            num_embeddings=num_char_embeddings,
            embedding_dim=CHAR_EMBEDDING_DIM,
            padding_idx=char_padding_idx)

        self.word_embeddings = nn.Embedding(
            num_embeddings=num_word_embeddings,
            embedding_dim=WORD_EMBEDDING_DIM,
            padding_idx=PAD_IDX,
            _weight=word_embeddings)

        if word_embeddings:
            self.word_embeddings.weight.requires_grad = train_word_embeddings

        self.char_lstm = char_lstm
        self.embedding_dropout = nn.Dropout(0.3)
        self.word_lstm = word_lstm
        self.output_dropout = nn.Dropout(0.3)
        self.out = nn.Linear(WORD_LSTM_HIDDEN_SIZE, num_tags)

        nn.init.xavier_uniform_(self.out.weight)
        for name, param in self.word_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

        for name, param in self.char_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    # TODO : maybe other initialization methods?
    def init_hidden(self, batch_size):  # initialize hidden states
        h = zeros(WORD_LSTM_NUM_LAYERS * WORD_LSTM_NUM_DIRS,
                  batch_size,
                  WORD_LSTM_HIDDEN_SIZE // WORD_LSTM_NUM_DIRS)  # hidden states
        c = zeros(WORD_LSTM_NUM_LAYERS * WORD_LSTM_NUM_DIRS,
                  batch_size,
                  WORD_LSTM_HIDDEN_SIZE // WORD_LSTM_NUM_DIRS)  # cell states
        return (h, c)

    def forward(self, word_x, mask, char_x):
        char_output = self._char_forward(char_x)
        batch_size = word_x.size(0)
        max_seq_len = word_x.size(1)
        char_output = char_output.reshape(batch_size, max_seq_len, -1)  # last dimension is for char lstm hidden size

        word_x = self.word_embeddings(word_x)
        word_x = torch.cat([word_x, char_output], -1)
        word_x = self.embedding_dropout(word_x)

        initial_hidden = self.init_hidden(batch_size)  # batch size is first
        word_x = nn.utils.rnn.pack_padded_sequence(word_x, mask.sum(1).int(), batch_first=True)
        output, hidden = self.word_lstm(word_x, initial_hidden)

        output, recovered_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.output_dropout(output)
        output = self.out(output)  # batch x seq_len x num_tags
        output *= mask.unsqueeze(-1)  # mask - batch x seq_len -> batch x seq_len x 1
        return output

    def _char_forward(self, x):
        word_lengths = x.gt(0).sum(1)  # actual word lengths
        sorted_padded, order = _sort(x, word_lengths)
        embedded = self.char_embeddings(sorted_padded)

        word_lengths_copy = word_lengths.clone()
        word_lengths_copy[word_lengths == 0] = 1
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, word_lengths_copy[order], True)
        packed_output, _ = self.char_lstm(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, True)

        _, reverse_sort_order = torch.sort(order, dim=0)
        output = output[reverse_sort_order]

        indices_of_lasts = (word_lengths_copy - 1).unsqueeze(1).expand(-1, output.shape[2]).unsqueeze(1)
        output = output.gather(1, indices_of_lasts).squeeze()
        output[word_lengths == 0] = 0
        return output


def init_hidden(batch_size):  # initialize hidden states
        h = zeros(1 * 2,
                  batch_size,
                  768 // 2)  # hidden states
        c = zeros(1 * 2,
                  batch_size,
                  768 // 2)  # cell states
        return (h, c) 
        
def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() 
    
class REModel(nn.Module):
    def __init__(self):
        super(REModel, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        self.linear = nn.Linear(768, 768)
        #self.linear1 = nn.Linear(768, 768)
        self.relu = nn.ReLU()
        self.sub_output = nn.Linear(768, 2)
        #self.suopand =  nn.Linear(1024, 768)
        #self.cat_output = nn.Linear(1024, 768)
        self.obj_output = nn.Linear(768, len(predicate2id)*2)
        
        self.sub_pos_emb = nn.Embedding(256, 768)   # subject位置embedding
        self.layernorm = BertLayerNorm(768, eps=1e-12)
        #self.GCN_model = GCNClassifier(opt, emb_matrix=None)
        
        #self.GRU = GRUnet(13714, 768, 1024, 6, 768)
        #self.CRF_S = CRF_S(768, 16, if_bias=True)
        #self.LSTM_CRF = LSTM_CRF(23922, 16, 768, 768, 1, 0.5, large_CRF=True)
        '''
        self.char_lstm = nn.LSTM(
        input_size=100,
        hidden_size=768 // 2,
        num_layers=1,
        bias=True,
        batch_first=True,
        bidirectional=True)
        '''
        self.word_lstm = nn.LSTM(
        input_size=768,
        hidden_size=768 // 2,
        num_layers=1,
        bias=True,
        batch_first=True,
        bidirectional=True)
        
        #self.output_dropout = nn.Dropout(0.3)
        
        #self.biaffine = BiaffineTagger(768,2)
        
        #self.GCN = GCN(hidden_size=768)
        #self.attention2 = Attention2(hidden_dim=768)
        self.gcu1 = GraphConv2(batch = args.batch1, h=[16,32,64,128,256], w=[16,32,64,128,256], d=[768,512], V=[2,4,8,16],outfeatures=[64,32])
        #self.gcu2 = GraphConv2(batch = args.batch2, h=[16,32,64,128,256], w=[16,32,64,128,256], d=[768,512], V=[2,4,8,32],outfeatures=[256,128])
        self.cov = nn.Conv2d(832, 768 ,1)
        #self.GCN_model = GCNClassifier(opt, emb_matrix = None)
        #self.emb = nn.Embedding(13714, 768)
        self.out = nn.Linear(768, 2)
        #self.emb1 = nn.Embedding(37, 384)
        #self.keyvalue = KeyValueMemoryNetwork(13714, 13714, 768)
        #self.apnb = APNB(in_channels=768, out_channels=768, key_channels=256, value_channels=256,dropout=0.05, sizes=([1]))
        #self.embedding_dropout = nn.Dropout(0.3)
        #self.crf = CRF(target_size=768, average_batch=True)
        self.axialatt = AxialAttention(
                     dim = 768,               # embedding dimension
                     dim_index = 1,         # where is the embedding dimension
                     dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                     heads = 1,             # number of heads for multi-head attention
                     num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
                     sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
                    )


    def forward(self, token_ids, seg_ids, inputs, sub_ids=None):
        out, _ = self.bert(token_ids, token_type_ids=seg_ids, 
                           output_all_encoded_layers=False)   # [batch_size, maxlen, size]       
        #print ('out.shape',token_ids.shape)
        batch_size = out.size(0)
        #print (out.shape)
        #print (out.shape)                   
        #logits, pooling_output = self.GCN_model(inputs)
        #print('logits.shape',logits.shape)
        #print('pooling_output',pooling_output.shape)
        #print (inputs[8])
        '''
        word_re_embed, __ = self.bert(inputs[0], token_type_ids=inputs[7], 
                           output_all_encoded_layers=False)#self.emb(inputs[0])
        '''
        #mask = inputs[0].data.gt(0).float()
        #word_x = torch.cat([out, word_re_embed], -1)
        #print (word_x.shape)
        #mask = inputs[0].data.gt(1).float()
        #mask = mask.sum(1)
        #mask = mask.cpu().int()
        #mask = mask.cpu().numpy()
        #mask.data.tolist()
        #mask.astype(np.int64)
        #print (mask.shape)
        #word_x = self.embedding_dropout(word_x)
        initial_hidden = init_hidden(batch_size)
        #print(word_x.shape)
        #word_x = nn.utils.rnn.pack_padded_sequence(word_x, mask, batch_first=True, enforce_sorted=False)
        output, hidden = self.word_lstm(out, initial_hidden)
        output = torch.reshape(output,(-1, 16, 16, 768))
        output = output.permute(0,3,1,2)
        output = self.axialatt(output)
       # print ('111:',output.shape)
        #output, recovered_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        #print (output.shape)
        #utput = self.output_dropout(output)
        #print (output.shape)
        #output = self.out(output)
        #print (output.shape)
        #普通注意力
        #output = self.attention2(output)
        #output = self.axialatt(output)
        b1, c1, h1 , w1 = output.shape
        output = rearrange(output, 'b1 c1 h1 w1 -> b1 c1 (h1 w1)')
        output = output.permute(0,2,1)
        #print (output.shape)
        #out = self.biaffine(out+word_re_embed)
        #out = self.expand(out)
        #print (out.shape) 
        #out = self.cat_output(out)
        #print (out.shape) 
        '''
        word_re_embed = torch.reshape(word_re_embed,(-1, 16, 16, 768))
        #print ('out1:',out1.shape)
        word_re_embed = word_re_embed.permute(0,3,1,2)
        word_re_embed = self.apnb(word_re_embed)
        #print (word_re_embed.shape)
        b0, c0, h0, w0 = word_re_embed.shape
        word_re_embed = rearrange(word_re_embed, 'b0 c0 h0 w0 -> b0 c0 (h0 w0)')
        word_re_embed = word_re_embed.permute(0,2,1)
        #print (word_re_embed.shape)
        #out = self.attention2(out+word_re_embed)
        '''
        '''
        biaffine_out = out.matmul(self.biaffine_weight).matmul(word_re_embed.permute(0,2,1))
        # shape(biaffine_out) = batch_size, seq_len, seq_len
        #concat_out = self.concat_weight(torch.cat([out, word_re_embed], dim=-1))
        #concat_out = self.concat_weight(out+word_re_embed)
        #print (concat_out.shape)
        # shape(concat_out) = batch_size, seq_len, 1
        #concat_output = (biaffine_out + concat_out)
        #print (output.shape)
        # shape(output) = batch_size, seq_len, seq_len
        gcn_out = self.GCN(out, biaffine_out, is_relu=True)
        '''
        sub_preds = self.sub_output(output)   # [batch_size, maxlen, 2]
        sub_preds = torch.sigmoid(sub_preds) 
        # sub_preds = sub_preds ** 2 

        if sub_ids is None:
            return sub_preds
        
        #print(sub_ids)
        #print(sub_ids[:, :1])
        # 融入subject特征信息
        sub_pos_start = self.sub_pos_emb(sub_ids[:, :1]) #取主实体首位置
        sub_pos_end = self.sub_pos_emb(sub_ids[:, 1:])   # [batch_size, 1, size] #取主实体尾位置
        
        #print(sub_pos_start)

        sub_id1 = sub_ids[:, :1].unsqueeze(-1).repeat(1, 1, out.shape[-1])     # subject开始的位置id 重复字编码次数
        #print (sub_id1)
        sub_id2 = sub_ids[:, 1:].unsqueeze(-1).repeat(1, 1, out.shape[-1])     # [batch_size, 1, size]
        sub_start = torch.gather(out, 1, sub_id1)   #按照sub_id1位置索引去找bert编码后的值，在列维度进行索引
        #print(sub_start.shape)
        sub_end = torch.gather(out, 1, sub_id2)   # [batch_size, 1, size]
        
        sub_start = sub_pos_start + sub_start #位置编码向量+bert字编码向量
        sub_end = sub_pos_end + sub_end
        out1 = out + sub_start + sub_end
        
        #word_re_embed = word_re_embed.permute(1,0,2,3)
        #print (word_re_embed.shape)
        #print ('out1.shape',out1.shape)
        out1 = torch.reshape(out1,(-1, 16, 16, 768))
        #print ('out1:',out1.shape)
        out1 = out1.permute(0,3,1,2)
        #print (out.shape)
        if out1.shape[0] == args.batch1:
            out1  = self.gcu1(out1)
            #word_re_embed,_ = self.LSTM_CRF(inputs[0],hidden=None,t = True)
        else:
            out1  = GraphConv2(batch = out1.shape[0], h=[16,32,64,128,256], w=[16,32,64,128,256], d=[768,512], V=[2,4,8,16],outfeatures=[64,32])(out1)
            #word_re_embed,_ =  LSTM_CRF1(23922, 16, 768, 768, 1, 0.5, large_CRF=True, t = out1.shape[0]).to(DEVICE)(inputs[0],hidden=None)
        #print ('out1_',out1.shape)

        out1 = self.cov(out1)
        #out1 = self.apnb(out1)
        #out = out.permute(0,2,3,1)
        #print (out.shape)
        b, c, h , w = out1.shape
        out1 = rearrange(out1, 'b c h w -> b c (h w)')
        out1 = out1.permute(0,2,1)
        #out1 = out1 + sub_start + sub_end
       
        #out1 = torch.cat((out1,pooling_output),dim=1)
        out1 = self.layernorm(out1)
        out1 = F.dropout(out1, p=0.5, training=self.training)
        
        output = self.relu(self.linear(out1))  
        output = F.dropout(output, p=0.4, training=self.training)
        output = self.obj_output(output)  # [batch_size, maxlen, 2*plen]
        ######
        #logits_output = torch.unsqueeze(logits, dim = 1)
        #final_output = logits_output + output
        output = torch.sigmoid(output)
        # output = output ** 2
        
        obj_preds = output.view(-1, output.shape[1], len(predicate2id), 2)
        return sub_preds, obj_preds


net = REModel().to(DEVICE)
print(DEVICE)


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

class ValidDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        t = self.data[i]
        word_input, center_word = [],[]
        #print (t['triple_list'])
        if len(t['text']) > 254:
            t['text'] = t['text'][:254]
        x = tokenizer.tokenize(t['text'])
        x = ["[CLS]"] + x + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(x)
        
        seg_ids = [0] * len(token_ids) 
        assert len(token_ids) == len(t['text'])+2
        
        head_in, seg_word, Dep_rel, str_le = seg_pos(t['text'])
        head = [x for x in head_in]
        for j in head:
            if j == 0:
                center_word.append('Root')
            else:
                center_word.append(str_le[j-1])
        center_word = map_to_ids(center_word, vocab_json()) #中心词
        Dep_rel = map_to_ids(Dep_rel, dep_json()) #依存关系
        text1 = out_list_word(seg_word) 
        text1_list = text1.split() 
        seg_word_out = map_to_ids(text1_list, vocab_json()) #词
        word_seg_ids = [0] * len(seg_word_out)
        
        head = np.array(head)
        
        text = out_list_word(seg_word) 
        text_list = text.split()
        for item in text_list:
            length = len(item)
            word = [item] * length
            word_input.extend(word)  
        if len(word_input) > 254:
            word_input = word_input[:254]
        word_input = ["[CLS]"] + word_input + ["[SEP]"]    
        # add
        #print("text_list",word_input)
        
        tokens_words = map_to_ids(word_input, vocab_json()) ######词向量
        
        token_ids = torch.LongTensor(self.sequence_padding(token_ids, maxlen=maxlen))
        seg_ids = torch.LongTensor(self.sequence_padding(seg_ids, maxlen=maxlen))
        
        head_out = torch.LongTensor(self.sequence_padding(head, maxlen=maxlen))
        tokens_words_out = torch.LongTensor(self.sequence_padding(tokens_words, maxlen=maxlen))
        word_seg_ids = torch.LongTensor(self.sequence_padding(word_seg_ids, maxlen=maxlen))
        masks_out = torch.eq(tokens_words_out, 0)

        center_word = torch.LongTensor(self.sequence_padding(center_word, maxlen=maxlen))
        Dep_rel = torch.LongTensor(self.sequence_padding(Dep_rel, maxlen=maxlen))
        seg_word_out = torch.LongTensor(self.sequence_padding(seg_word_out, maxlen=maxlen))
        mask_word_out = torch.eq(seg_word_out, 0)
        
        #tri = t['triple_list']
        #print('tri',tri)
        '''
        return {'token_ids':token_ids,
                'seg_ids':seg_ids,
                'text':t['text'],
                'triple_list':t['triple_list']}
        '''
        #return token_ids, seg_ids, list(t['text']), list(t['triple_list'])
        return token_ids, seg_ids, t, tokens_words_out, masks_out, head_out, center_word, Dep_rel, seg_word_out, mask_word_out, word_seg_ids
        
    def __len__(self):
        data_len = len(self.data)
        return data_len
    
    def sequence_padding(self, x, maxlen, padding=0):
        output = np.concatenate([x, [padding]*(maxlen-len(x))]) if len(x)<maxlen else np.array(x[:maxlen])
        return output 
    
valid_dataset = ValidDataset(valid_data)

valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch2, shuffle=False, drop_last = True)
    
def extract_spoes(data, model, device):
    '''
    """抽取三元组"""
    if len(text) > 254:
        text = text[:254]
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    assert len(token_ids) == len(text) + 2 
    seg_ids = [0] * len(token_ids) 
    '''
    #print (data['text'][0])
    #print (data['text'])
    #token_ids = data['token_ids']
    token_ids = data[0]
    
    #seg_ids = data['seg_ids']
    seg_ids = data[1]
    inputs = data[3].to(device), data[4].to(device), data[5].to(device),data[6].to(device), data[7].to(device), data[8].to(device),data[9].to(device),data[10].to(device)
    sub_preds = model(token_ids.to(device), 
                      seg_ids.to(device), inputs)
    sub_preds = sub_preds.detach().cpu().numpy()  # [1, maxlen, 2]
    # print(sub_preds[0,])
    start = np.where(sub_preds[0, :, 0] > 0.5)[0]
    end = np.where(sub_preds[0, :, 1] > 0.5)[0]
    # print(start, end)
    tmp_print = []
    subjects = []
    for i in start: 
        j = end[end>=i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
            tmp_print.append(data[2][i-1: j])

    if subjects:
        spoes = []
        #print (len(subjects)) 只有2
        token_ids = np.repeat(token_ids, len(subjects), 0)   # [len_subjects, seqlen]
        #print(token_ids.shape)
        seg_ids = np.repeat(seg_ids, len(subjects), 0)
        subjects = np.array(subjects)   # [len_subjects, 2]
        # 传入subject 抽取object和predicate
        _, object_preds = model(token_ids.to(device), 
                            seg_ids.to(device), 
                            inputs,
                            torch.LongTensor(subjects).to(device))
        object_preds = object_preds.detach().cpu().numpy()
        #         print(object_preds.shape)
        for sub, obj_pred in zip(subjects, object_preds):
            # obj_pred [maxlen, 55, 2]
            start = np.where(obj_pred[:, :, 0] > 0.5)
            end = np.where(obj_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append(
                            ((sub[0]-1, sub[1]-1), predicate1, (_start-1, _end-1))
                        )
                        break
        #print (spoes)
        return [(data[2][s[0]:s[1]+1], id2predicate[str(p)], data[2][o[0]:o[1]+1]) for s, p, o in spoes]
    else:
        return []
       
def evaluate(valid_data, valid_load, model, device):
    """评估函数，计算f1、precision、recall
    """
    F1 = []
    P = []
    Re = []
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open("./data/CMED/dev_pred.json", 'w', encoding='utf-8')
    pbar = tqdm()
    #for d in data:
    #with torch.no_grad:
    #print (type(valid_load))
    #return
    for idx, data in tqdm(enumerate(valid_load)):   
        #print (valid_data[idx]['text'])
        #print (data)
        input = data[0], data[1], valid_data[idx]['text'], data[3], data[4], data[5], data[6], data[7], data[8],data[9],data[10]
        #input = data[0], data[1], valid_data[idx]['text'], valid_data[idx]['triple_list']
        R = extract_spoes(input, model, device)
        #print ('R:',R)
        T = valid_data[idx]['triple_list']
        '''
        tri = data[3]
        #tri = tuple(tri)
        T = []
        for tris in tri:
            temp = tuple()
            for i in tris:
                temp += i
            T.append(temp)
        '''
        #print ('tri:',tri)
        #print ('tri:',temp_tri)
        R = set(R)
        #print ('R',R)
        T = set(T)
        
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        F1.append(f1)
        P.append(precision)
        Re.append(recall)
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        
        if f1 > 0.5: 
        
          s = json.dumps({
              'text': valid_data[idx]['text'],
              'triple_list': list(T),
              'triple_list_pred': list(R),
              'new': list(R - T),
              'lack': list(T - R),
          }, ensure_ascii=False, indent=4)
          f.write(s + '\n')
    pbar.close()
    f.close()
    #return f1, precision, recall
    return statistics.mean(F1), statistics.mean(P), statistics.mean(Re)
'''        
def evaluate(data, model, device):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open("/home/jason/EXP/NLP/triple_test/data/CMED/dev_pred.json", 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:    
        R = extract_spoes(d['text'], model, device)
        
        T = d['triple_list']
        #print (T)
        R = set(R)
        #print ('R',R)
        T = set(T)
        
        #T = set()
        #for item in T1:
        #  for i in item:
        #    T.add(i)
        
        #print ('T',T)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        
        if f1 > 0.5: 
        
          s = json.dumps({
              'text': d['text'],
              'triple_list': list(T),
              'triple_list_pred': list(R),
              'new': list(R - T),
              'lack': list(T - R),
          }, ensure_ascii=False, indent=4)
          f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall
'''


def train(model, train_loader, epoches, device):
    #model.train()
    for _ in range(epoches):
        print('epoch: ', _ + 1)
        start = time.time()
        train_loss_sum = 0.0
        if (_+1) <= 100:
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)#原1e-5
        elif (_+1)> 100 & (_+1) <= 200:
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)#原1e-5
        elif (_+1) > 200:
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-7)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = 'min',factor = 0.8,verbose=1, patience=10, min_lr = 0, eps=1e-08)
        for batch_idx, x in tqdm(enumerate(train_loader)):
            #token_ids, seg_ids, sub_ids = x[0].to(device), x[1].to(device), x[2].to(device)
            token_ids, seg_ids, sub_ids = x[0].to(device), x[1].to(device), x[2].to(device) 
            #tokens_words, masks_out, head = x[5].to(device), x[6].to(device), x[7].to(device)
            #print (token_ids.shape)
            inputs = x[5].to(device), x[6].to(device), x[7].to(device), x[8].to(device), x[9].to(device), x[10].to(device),x[11].to(device),x[12].to(device)
            
            mask = (token_ids > 0).float()
            mask = mask.to(device)   # zero-mask
            sub_labels, obj_labels = x[3].float().to(device), x[4].float().to(device)
            sub_preds, obj_preds = model(token_ids, seg_ids, inputs, sub_ids)
            # (batch_size, maxlen, 2),  (batch_size, maxlen, 55, 2)
            
            # 计算loss
            loss_sub = F.binary_cross_entropy(sub_preds, sub_labels, reduction='none')  #[bs, ml, 2]
            loss_sub = torch.mean(loss_sub, 2)  # (batch_size, maxlen)
            loss_sub = torch.sum(loss_sub * mask) / torch.sum(mask)
            
            #loss_sub1 = RecallLoss(weight=None)(sub_preds, sub_labels)  #[bs, ml, 2]
            #loss_sub1 = torch.mean(loss_sub1, 2)  # (batch_size, maxlen)
            #loss_sub1 = torch.sum(loss_sub1 * mask) / torch.sum(mask)
            #print (loss_sub1)
            loss_obj = F.binary_cross_entropy(obj_preds, obj_labels, reduction='none')  # [bs, ml, 55, 2]
            loss_obj = torch.sum(torch.mean(loss_obj, 3), 2)   # (bs, maxlen)
            loss_obj = torch.sum(loss_obj * mask) / torch.sum(mask)
            
            #loss_obj1 = RecallLoss(weight=None)(obj_preds, obj_labels)  # [bs, ml, 55, 2]
            #loss_obj1 = torch.sum(torch.mean(loss_obj1, 3), 2)   # (bs, maxlen)
            #loss_obj1 = torch.sum(loss_obj1 * mask) / torch.sum(mask)
            #print (loss_obj1)
            loss = loss_sub + loss_obj
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step(loss)
            
            train_loss_sum += loss.cpu().item()
            
            if (batch_idx + 1) % 128 == 0:
                print('loss: ', train_loss_sum / (batch_idx+1), 'time: ', time.time() - start) 
        
        torch.save(net.state_dict(), "./checkpoints/bert_relation_all.pth")
       
        with torch.no_grad():
            #model.eval()
            #print (valid_data[:5])
            val_f1, pre, rec = evaluate(valid_data, valid_loader, net, device)
            print ('F1_score: %.5f, Precision: %.5f, Recall: %.5f' % (val_f1, pre, rec))
            #print("f1, pre, rec: ", val_f1, pre, rec)
if __name__ == '__main__':
    #net.load_state_dict(torch.load("RE/data/bert_re.pth"))
    train(net, train_loader, 300, DEVICE)