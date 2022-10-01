import torch.nn as nn
from transformers import BertModel, AutoModel

class bert_labeler(nn.Module):
    def __init__(self, p=0.1, clinical=False, freeze_embeddings=False, pretrain_path=None):
        """ Init the labeler module
        @param p (float): p to use for dropout in the linear heads, 0.1 by default is consistant with 
                          transformers.BertForSequenceClassification
        @param clinical (boolean): True if Bio_Clinical BERT desired, False otherwise. Ignored if
                                   pretrain_path is not None
        @param freeze_embeddings (boolean): true to freeze bert embeddings during training
        @param pretrain_path (string): path to load checkpoint from
        """
        super(bert_labeler, self).__init__()

        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif clinical:
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
                
        self.dropout = nn.Dropout(p)
        #size of the output of transformer's last layer
        hidden_size = self.bert.pooler.dense.in_features
        #classes: present, absent, unknown, blank for 12 conditions + support devices
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        #classes: yes, no for the 'no finding' observation
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def forward(self, source_padded, attention_mask):
        """ Forward pass of the labeler
        @param source_padded (torch.LongTensor): Tensor of word indices with padding, shape (batch_size, max_len)
        @param attention_mask (torch.Tensor): Mask to avoid attention on padding tokens, shape (batch_size, max_len)
        @returns out (List[torch.Tensor])): A list of size 14 containing tensors. The first 13 have shape 
                                            (batch_size, 4) and the last has shape (batch_size, 2)  
        """
        #shape (batch_size, max_len, hidden_size)
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        #shape (batch_size, hidden_size)
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        cls_hidden = self.dropout(cls_hidden)
        out = []
        for i in range(14):
            out.append(self.linear_heads[i](cls_hidden))
        return out
