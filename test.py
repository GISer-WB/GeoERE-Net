import json
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
#from pyhanlp import *

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

valid_data =  load_data("./data/CMED/test_triples.json")
    
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
        self.relu = nn.ReLU()
        self.sub_output = nn.Linear(768, 2)

        self.obj_output = nn.Linear(768, len(predicate2id)*2)
        
        self.sub_pos_emb = nn.Embedding(256, 768)   # subject位置embedding
        self.layernorm = BertLayerNorm(768, eps=1e-12)

        self.word_lstm = nn.LSTM(
        input_size=768,
        hidden_size=768 // 2,
        num_layers=1,
        bias=True,
        batch_first=True,
        bidirectional=True)
        
        self.gcu1 = GraphConv2(batch = args.batch1, h=[16,32,64,128,256], w=[16,32,64,128,256], d=[768,512], V=[2,4,8,16],outfeatures=[64,32])

        self.cov = nn.Conv2d(832, 768 ,1)

        self.out = nn.Linear(768, 2)

        self.axialatt = AxialAttention(
                     dim = 768,               # embedding dimension
                     dim_index = 1,         # where is the embedding dimension
                     dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                     heads = 1,             # number of heads for multi-head attention
                     num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
                     sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
                    )


    def forward(self, token_ids, seg_ids, sub_ids=None):
        out, _ = self.bert(token_ids, token_type_ids=seg_ids, 
                           output_all_encoded_layers=False)   # [batch_size, maxlen, size]       
        #print ('out.shape',token_ids.shape)
        batch_size = out.size(0)

        initial_hidden = init_hidden(batch_size)
        output, hidden = self.word_lstm(out, initial_hidden)
        output = torch.reshape(output,(-1, 16, 16, 768))
        output = output.permute(0,3,1,2)
        output = self.axialatt(output)

        b1, c1, h1 , w1 = output.shape
        output = rearrange(output, 'b1 c1 h1 w1 -> b1 c1 (h1 w1)')
        output = output.permute(0,2,1)

        sub_preds = self.sub_output(output)   # [batch_size, maxlen, 2]
        sub_preds = torch.sigmoid(sub_preds) 
        # sub_preds = sub_preds ** 2 

        if sub_ids is None:
            return sub_preds
            
        sub_pos_start = self.sub_pos_emb(sub_ids[:, :1]) #Take the start position of the subject entity
        sub_pos_end = self.sub_pos_emb(sub_ids[:, 1:])   # [batch_size, 1, size] #Take the tail position of the subject entity
        
        #print(sub_pos_start)

        sub_id1 = sub_ids[:, :1].unsqueeze(-1).repeat(1, 1, out.shape[-1])
        #print (sub_id1)
        sub_id2 = sub_ids[:, 1:].unsqueeze(-1).repeat(1, 1, out.shape[-1])     # [batch_size, 1, size]
        sub_start = torch.gather(out, 1, sub_id1)   #The sub_id1 position index is used to find the Bert encoded value  
        #print(sub_start.shape)
        sub_end = torch.gather(out, 1, sub_id2)   # [batch_size, 1, size]
        
        sub_start = sub_pos_start + sub_start #Position code vector + Bert word code vector
        sub_end = sub_pos_end + sub_end
        out1 = out + sub_start + sub_end

        out1 = torch.reshape(out1,(-1, 16, 16, 768))
        #print ('out1:',out1.shape)
        out1 = out1.permute(0,3,1,2)
        #print (out.shape)
        if out1.shape[0] == args.batch1:
            out1  = self.gcu1(out1)
        else:
            out1  = GraphConv2(batch = out1.shape[0], h=[16,32,64,128,256], w=[16,32,64,128,256], d=[768,512], V=[2,4,8,16],outfeatures=[64,32])(out1)

        out1 = self.cov(out1)

        b, c, h , w = out1.shape
        out1 = rearrange(out1, 'b c h w -> b c (h w)')
        out1 = out1.permute(0,2,1)
        out1 = self.layernorm(out1)
        out1 = F.dropout(out1, p=0.5, training=self.training)
        
        output = self.relu(self.linear(out1))  
        output = F.dropout(output, p=0.4, training=self.training)
        output = self.obj_output(output)  # [batch_size, maxlen, 2*plen]

        output = torch.sigmoid(output)
        
        obj_preds = output.view(-1, output.shape[1], len(predicate2id), 2)
        return sub_preds, obj_preds
        
def extract_spoes(data, model, device):
 
    token_ids = data[0]
    
    #seg_ids = data['seg_ids']
    seg_ids = data[1]
    sub_preds = model(token_ids.to(device), 
                      seg_ids.to(device))
    sub_preds = sub_preds.detach().cpu().numpy()  # [1, maxlen, 2]
    # print(sub_preds[0,])
    start = np.where(sub_preds[0, :, 0] > 0.5)[0] #case0.3,0.2
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
        #print (len(subjects)) 
        token_ids = np.repeat(token_ids, len(subjects), 0)   # [len_subjects, seqlen]
        #print(token_ids.shape)
        seg_ids = np.repeat(seg_ids, len(subjects), 0)
        subjects = np.array(subjects)   # [len_subjects, 2]
        _, object_preds = model(token_ids.to(device), 
                            seg_ids.to(device), 
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
        
        token_ids = torch.LongTensor(self.sequence_padding(token_ids, maxlen=maxlen))
        seg_ids = torch.LongTensor(self.sequence_padding(seg_ids, maxlen=maxlen))
        

        return token_ids, seg_ids, t
        
    def __len__(self):
        data_len = len(self.data)
        return data_len
    
    def sequence_padding(self, x, maxlen, padding=0):
        output = np.concatenate([x, [padding]*(maxlen-len(x))]) if len(x)<maxlen else np.array(x[:maxlen])
        return output 

valid_dataset = ValidDataset(valid_data)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch2, shuffle=False, drop_last = True)

def predict_to_file(valid_data, out_file,valid_loader,DEVICE):

    F1 = []
    P = []
    Re = []
    net = REModel().to(DEVICE)
    net.load_state_dict(torch.load("./checkpoints/bert_relation_all.pth"))
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(out_file, 'w', encoding='utf-8')
    pbar = tqdm()
       
    for idx, data in tqdm(enumerate(valid_loader)):   
        input = data[0], data[1], valid_data[idx]['text']
        R = extract_spoes(input, net, DEVICE)
        #print ('R:',R)
        T = valid_data[idx]['triple_list']
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
    tokenizer = OurTokenizer(vocab_file="./geo_triple/bert/vocab.txt")  
    with open("./data/rel2id.json", encoding='utf-8') as f:
        l = json.load(f)
        id2predicate = l[0] 
        predicate2id = l[1]  
    out = predict_to_file(valid_data, "./data/test_out_11final.json",valid_loader,DEVICE)
    print ('F1_score: %.5f, Precision: %.5f, Recall: %.5f' % (out[0], out[1], out[2]))
    print ("Finished!")
    