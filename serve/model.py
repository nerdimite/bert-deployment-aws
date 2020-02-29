from transformers import BertModel
import torch
from torch import nn


class BERTClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, freeze_bert = True):
        super(BERTClassifier, self).__init__()

        # Instantiating the BERT model object
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        # Defining layers like dropout and linear
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 1)

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Getting contextualized representations from BERT Layer
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)

        # Obtaining the representation of [CLS] head
        cls_rep = cont_reps[:, 0]
        # print('CLS shape: ',cls_rep.shape)

        # Feeding cls_rep to the classifier layer
        logits = self.classifier(cls_rep)
        # print('Logits shape: ',logits.shape)

        return logits