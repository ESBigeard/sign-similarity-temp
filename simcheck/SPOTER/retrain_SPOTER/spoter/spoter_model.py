
import copy
import torch

import torch.nn as nn
from typing import Optional


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Edited TransformerDecoderLayer implementation omitting the redundant self-attention operation as opposed to the
    standard implementation.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(SPOTERTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)

        del self.self_attn

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class SPOTER(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes, hidden_dim=50):
        super().__init__()

        # self.row_embed = nn.Parameter(torch.rand(50, hidden_dim)) # Learneable paramameter of shape (50, hidden_dim), initialized randomly
        # print(f"Row embedding 0: {self.row_embed[0]}")
        # self.pos = nn.Parameter(torch.cat([self.row_embed[0].unsqueeze(0).repeat(1, 1, 1)], dim=-1).flatten(0, 1).unsqueeze(0))
        # # Positional embedding of shape (1, 50, hidden_dim), initialized with the first row of row_embed
        # self.class_query = nn.Parameter(torch.rand(1, hidden_dim))
        # self.transformer = nn.Transformer(hidden_dim, 10, 6, 6)
        # self.linear_class = nn.Linear(hidden_dim, num_classes)


        self.class_query = nn.Parameter(torch.rand(1, hidden_dim))
        self.transformer = nn.Transformer(hidden_dim, 10, 6, 6)
        self.linear_class = nn.Linear(hidden_dim, num_classes)

        # Deactivate the initial attention decoder mechanism
        custom_decoder_layer = SPOTERTransformerDecoderLayer(self.transformer.d_model, self.transformer.nhead, 2048,
                                                             0.1, "relu")
        self.transformer.decoder.layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)

    def forward(self, inputs, return_embeddings=False):

        # # Inputs is o   of shape (num_frame, num_joints). Here, on the WordNet dataset, num_joints is 100 (because joints * 2)
        # h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        # # We change the shape to (num_frame, 1, 100) to match the expected input shape of the transformer
        # h = self.transformer(self.pos + h, self.class_query.unsqueeze(0)).transpose(0, 1)    
        # # We add the positional embedding to the input and pass it through the transformer.
        # res = self.linear_class(h)
        # if return_embeddings:
        #     return h, res

        h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        h = self.transformer(h, self.class_query.unsqueeze(0)).transpose(0, 1)
        res = self.linear_class(h)
        if return_embeddings:
            return h, res

        return res


if __name__ == "__main__":
    pass
