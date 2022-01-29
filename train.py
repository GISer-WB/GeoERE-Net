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

from graphModule import *
from einops import rearrange
from config import args
from torch_multi_head_attention import MultiHeadAttention
import statistics
from axial_attention import AxialAttention
from torch import optim

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_PATH = "./chinese_roberta_wwm_ext_pytorch"
maxlen = 256 ####256 

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


# Load dataset
train_data = load_data('./data/train_triples.json') 
valid_data = load_data('./data/dev_triples.json') 
print ("Validation size：", len(valid_data))

def search(pattern, sequence):
    """
    Find substring Pattern from sequence.
    If found, return the first subscript; Otherwise -1 is returned.
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

# Removes text that exceeds the 256 size length
train_data_new = []
for data in tqdm(train_data):
    #print (data)
    flag = 1
    for s, p, o in data['triple_list']:
        s_begin = search(s, data['text'])
        o_begin = search(o, data['text'])
        if s_begin == -1 or o_begin == -1 or s_begin + len(s) > 256 or o_begin + len(o) > 256:
            flag = 0
            break 
    if flag == 1:
        train_data_new.append(data)
print(len(train_data_new))


with open('./data/CMED/rel2id.json', encoding='utf-8') as f:
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

# Initialize the Tokenizer
tokenizer = OurTokenizer(vocab_file="./chinese_roberta_wwm_ext_pytorch/vocab.txt")   


class TorchDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        t = self.data[i]
        
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
                o = (o_idx, o_idx + len(o) - 1, p)  # Predict both o and P
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
            # Pick a subject at random  
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
        
        return (torch.LongTensor(token_ids), torch.LongTensor(seg_ids), torch.LongTensor(sub_ids),  
               torch.LongTensor(sub_labels), torch.LongTensor(obj_labels))
 
    def __len__(self):
        data_len = len(self.data)
        return data_len
    
    def sequence_padding(self, x, maxlen, padding=0):
        output = np.concatenate([x, [padding]*(maxlen-len(x))]) if len(x)<maxlen else np.array(x[:maxlen])
        return output 

train_dataset = TorchDataset(train_data_new)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch1, shuffle=True,drop_last = True)

        
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
        

        token_ids = torch.LongTensor(self.sequence_padding(token_ids, maxlen=maxlen))
        seg_ids = torch.LongTensor(self.sequence_padding(seg_ids, maxlen=maxlen))
        
        #tri = t['triple_list']
        #print('tri',tri)
        '''
        return {'token_ids':token_ids,
                'seg_ids':seg_ids,
                'text':t['text'],
                'triple_list':t['triple_list']}
        '''
        return token_ids, seg_ids, t
        
    def __len__(self):
        data_len = len(self.data)
        return data_len
    
    def sequence_padding(self, x, maxlen, padding=0):
        output = np.concatenate([x, [padding]*(maxlen-len(x))]) if len(x)<maxlen else np.array(x[:maxlen])
        return output 
    
valid_dataset = ValidDataset(valid_data)

valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch2, shuffle=False, drop_last = True)
    
def extract_spoes(data, model, device):

    token_ids = data[0]
    
    #seg_ids = data['seg_ids']
    seg_ids = data[1]
    
    sub_preds = model(token_ids.to(device), 
                      seg_ids.to(device))
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
    """
    Evaluation function, calculate F1, precision, recall
    """
    F1 = []
    P = []
    Re = []
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open("./data/dev_pred.json", 'w', encoding='utf-8')
    pbar = tqdm()
    #for d in data:
    #with torch.no_grad:
    #print (type(valid_load))
    #return
    for idx, data in tqdm(enumerate(valid_load)):   
        #print (valid_data[idx]['text'])
        #print (data)
        input = data[0], data[1], valid_data[idx]['text']
        #input = data[0], data[1], valid_data[idx]['text'], valid_data[idx]['triple_list']
        R = extract_spoes(input, model, device)
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


def train(model, train_loader, epoches, device):
    #model.train()
    for _ in range(epoches):
        print('epoch: ', _ + 1)
        start = time.time()
        train_loss_sum = 0.0
        if (_+1) <= 100:
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
        elif (_+1)> 100 & (_+1) <= 200:
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)
        elif (_+1) > 200:
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-7)

        for batch_idx, x in tqdm(enumerate(train_loader)):
            #token_ids, seg_ids, sub_ids = x[0].to(device), x[1].to(device), x[2].to(device)
            token_ids, seg_ids, sub_ids = x[0].to(device), x[1].to(device), x[2].to(device) 
            #tokens_words, masks_out, head = x[5].to(device), x[6].to(device), x[7].to(device)
            #print (token_ids.shape)
            
            mask = (token_ids > 0).float()
            mask = mask.to(device)   # zero-mask
            sub_labels, obj_labels = x[3].float().to(device), x[4].float().to(device)
            sub_preds, obj_preds = model(token_ids, seg_ids, sub_ids)
            # (batch_size, maxlen, 2),  (batch_size, maxlen, 55, 2)
            
            # loss
            loss_sub = F.binary_cross_entropy(sub_preds, sub_labels, reduction='none')  #[bs, ml, 2]
            loss_sub = torch.mean(loss_sub, 2)  # (batch_size, maxlen)
            loss_sub = torch.sum(loss_sub * mask) / torch.sum(mask)
            
            loss_obj = F.binary_cross_entropy(obj_preds, obj_labels, reduction='none')  # [bs, ml, 55, 2]
            loss_obj = torch.sum(torch.mean(loss_obj, 3), 2)   # (bs, maxlen)
            loss_obj = torch.sum(loss_obj * mask) / torch.sum(mask)
            

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

            val_f1, pre, rec = evaluate(valid_data, valid_loader, net, device)
            print ('F1_score: %.5f, Precision: %.5f, Recall: %.5f' % (val_f1, pre, rec))
           
if __name__ == '__main__':
    train(net, train_loader, 300, DEVICE)