import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel


class TokenNet(nn.Module):

    def __init__(self, bert_base, finetuning=False):
        super().__init__()
        
        self.bert = bert_base
        
        self.dropout = nn.Dropout(0.3)        
       
        self.fc_AO = nn.Linear(768, 9)
        self.fc_CP = nn.Linear(768, 38)

        self.finetuning = finetuning

    def forward(self, input_ids, attention_mask):
        '''
        x: (N, T). int64
        y: (N, T). int64
        Returns
        enc: (N, T, VOCAB)
        '''
        # x = x.to(self.device)
        # y = y.to(self.device)

        if self.training and self.finetuning:
            # print("->bert.train()")
            self.bert.train()
            # encoded_layers, _ = self.bert(x)
            encoded_layers, _ = self.bert(input_ids,attention_mask=attention_mask)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                # encoded_layers, _ = self.bert(x)
                encoded_layers, _ = self.bert(input_ids,attention_mask=attention_mask)
                enc = encoded_layers[-1]
        
        enc = self.dropout(enc)
            
        
        logit_AO = self.fc_AO(enc) 
        logit_CP = self.fc_CP(enc) 
        return logit_AO, logit_CP
        
        
class SentimentNet(nn.Module):

    def __init__(self, bert_base, finetuning=False):
        super().__init__()
        self.bert = bert_base

        
        # self.fc = nn.Linear(768, vocab_size)
        self.dropout = nn.Dropout(0.3) # 
        self.cls_categories = nn.Linear(768, 13)
        self.cls_polarities = nn.Linear(768, 3)
        # self.device = device
        self.finetuning = finetuning

    def forward(self, input_ids, attention_mask):
        '''
        x: (N, T). int64
        y: (N, T). int64
        Returns
        enc: (N, T, VOCAB)
        '''

        if self.training and self.finetuning:
            # print("->bert.train()")
            self.bert.train()
            # encoded_layers, _ = self.bert(x)
            outputs = self.bert(input_ids,attention_mask=attention_mask)
            # enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                # encoded_layers, _ = self.bert(x)
                outputs = self.bert(input_ids,attention_mask=attention_mask)
                # enc = encoded_layers[-1]
        
        # enc = self.dropout(enc)        
        
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits_categories = self.cls_categories(pooled_output)
        logits_polarities = self.cls_polarities(pooled_output)
        
        return logits_categories, logits_polarities
        

class UnionNet(nn.Module):

    def __init__(self, token_net, sent_net):
        super().__init__()
        self.token_net = token_net
        self.sent_net = sent_net
        
        

    def forward(self, input_view, mask_view, input_sent, mask_sent):
        
        
        logits_view = self.token_net(input_view, mask_view)
        logits_categories, logits_polarities = self.sent_net(input_sent, mask_sent)

        
        return logits_view, logits_categories, logits_polarities
        
        
