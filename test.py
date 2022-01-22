import json
#from train import REModel
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertModel, BertTokenizer
import unicodedata
import numpy as np
import torch.nn.functional as F 
from tqdm import tqdm 
from axial_attention import AxialAttention
import statistics
from einops import rearrange
from config import args
from graphModule import *
from pyhanlp import *

maxlen = 256
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

valid_data =  load_data("/home/cug210/data/wang/geo_triple/data/CMED/test_triples.json")
    
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
        #out1 = out + sub_start + sub_end
        
        #word_re_embed = word_re_embed.permute(1,0,2,3)
        #print (word_re_embed.shape)
        #print ('out1.shape',out1.shape)
        out1 = torch.reshape(out,(-1, 16, 16, 768))
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
        out1 = out1 + sub_start + sub_end
       
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
    start = np.where(sub_preds[0, :, 0] > 0.4)[0] #case0.3,0.2
    end = np.where(sub_preds[0, :, 1] > 0.4)[0]
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
            start = np.where(obj_pred[:, :, 0] > 0.4)
            end = np.where(obj_pred[:, :, 1] > 0.4)
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

def predict_to_file(valid_data, out_file,valid_loader,DEVICE):
    """预测结果到文件，方便提交
    """
    F1 = []
    P = []
    Re = []
    net = REModel().to(DEVICE)
    net.load_state_dict(torch.load("/home/cug210/data/wang/geo_triple/checkpoints1/bert_relation_all.pth"))
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(out_file, 'w', encoding='utf-8')
    pbar = tqdm()
       
    for idx, data in tqdm(enumerate(valid_loader)):   
        #print (valid_data[idx]['text'])
        #print (data)
        input = data[0], data[1], valid_data[idx]['text'], data[3], data[4], data[5], data[6], data[7], data[8],data[9],data[10]
        #input = data[0], data[1], valid_data[idx]['text'], valid_data[idx]['triple_list']
        R = extract_spoes(input, net, DEVICE)
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



if __name__ == "__main__":
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BERT_PATH = "./chinese_roberta_wwm_ext_pytorch"
    # 初始化分词器
    tokenizer = OurTokenizer(vocab_file="/home/cug210/data/wang/geo_triple/bert/vocab.txt")  
    with open("/home/cug210/data/wang/geo_triple/data/CMED/rel2id.json", encoding='utf-8') as f:
        l = json.load(f)
        id2predicate = l[0] 
        predicate2id = l[1]  
    out = predict_to_file(valid_data, "/home/cug210/data/wang/geo_triple/data/CMED/test_out_11final.json",valid_loader,DEVICE)
    print ('F1_score: %.5f, Precision: %.5f, Recall: %.5f' % (out[0], out[1], out[2]))
    print ("预测完成!")
    