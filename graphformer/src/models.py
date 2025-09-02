import torch
import torch.nn as nn
from transformers import BertModel
from .encoder_layer import GraphFormerEncoderLayer

class GraphFormerModel(nn.Module):
    def __init__(self, 
                 pretrained_model_name='bert-base-uncased',
                 hidden_size=768,
                 num_heads=12,
                 num_layers=4):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.encoder_layers = nn.ModuleList([
            GraphFormerEncoderLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        self.hidden_size = hidden_size

    def forward(self, input_ids_batch, attention_masks_batch, relation_matrix):
        B, N, T = input_ids_batch.shape
        H = self.hidden_size

        # 1. flatten the input
        input_ids_flat = input_ids_batch.view(B * N, T)
        attention_masks_flat = attention_masks_batch.view(B * N, T)

        # 2. bert encoding
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids_flat,
                                     attention_mask=attention_masks_flat,
                                     return_dict=True)
            
        token_embeddings = bert_outputs.last_hidden_state
        token_embeddings = token_embeddings.view(B, N, T, H)

        # 3. apply nested GNN + Transfomer layers
        H_g = token_embeddings
        for encoder_layer in self.encoder_layers:
            H_g = encoder_layer(H_g, relation_matrix)

        # 4. extract center node [CLS] token embedding
        h_center = H_g[:, 0, 0, :]
        return h_center
            