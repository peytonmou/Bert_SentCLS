from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # this attention is applied after calculating the attention score following the original implementation of transformer
    # although it is a bit unusual, we empirically observe that it yields better performance
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # next, we need to produce multiple heads for the proj 
    # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
    # attention scores are calculated by multiply query and key 
    # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
    # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
    dk = key.size(-1) 
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)

    # before normalizing the scores, use the attention mask to mask out the padding token scores
    # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number 
    if attention_mask is not None:
      scores = scores + attention_mask
    
    # normalize the scores
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = self.dropout(attn_weights)

    # multiply the attention scores to the value and get back V' 
    context = torch.matmul(attn_weights, value)

    # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
    bs, num_heads, seq_len, head_size = context.size()
    context = context.transpose(1, 2).contiguous().view(bs, seq_len, self.all_head_size) 
    return context 

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
    # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    # calculate the multi-head attention 
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # self attention
    self.self_attention = BertSelfAttention(config)
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # layer out
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    # apply dense layer and dropout to the sublayer output
    output = dense_layer(output)
    output = dropout(output)
    # add residual connection and apply layer norm
    output = ln_layer(input + output)
    return output

  def forward(self, hidden_states, attention_mask):
    # multi-head attention w/ self.self_attention
    attention_output = self.self_attention(hidden_states, attention_mask)   # [bs, seq_len, hidden_size]

    # add-norm layer
    hidden_states = self.add_norm(
      input=hidden_states,
      output=attention_output,
      dense_layer=self.attention_dense,   
      dropout=self.attention_dropout,
      ln_layer=self.attention_layer_norm)                  # [bs, seq_len, hidden_size]

    # feed forward
    interm_output = self.interm_dense(hidden_states)       # [bs, seq_len, interm_size]
    interm_output = self.interm_af(interm_output)          

    # another add-norm layer
    hidden_states = self.add_norm(
      input=hidden_states,
      output=interm_output,
      dense_layer=self.out_dense,        
      dropout=self.out_dropout,
      ln_layer=self.out_layer_norm)                       # [bs, seq_len, hidden_size]
    return hidden_states

# Below modified for Task 2 to add embeds of linguistic features
class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Standard BERT embeddings
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # Calculate active feature count
        self.active_features = sum([config.use_pos, config.use_dep, config.use_wn])
        self.feature_dim = config.hidden_size // max(1, self.active_features)  # Prevent division by zero
        
        # Initialize only enabled feature embeddings
        if config.use_pos:
            self.pos_tag_embedding = nn.Embedding(config.pos_tag_vocab_size, self.feature_dim)
        if config.use_dep:
            self.dep_embedding = nn.Embedding(config.dep_vocab_size, self.feature_dim)
        if config.use_wn:
            self.wn_linear = nn.Linear(1, self.feature_dim)
        
        # Only create combiner if features are enabled
        if self.active_features > 0:
            self.ling_combiner = nn.Linear(
                self.feature_dim * self.active_features,
                config.hidden_size
            )
        
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Rest of the model remains unchanged
        self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

    def embed(self, input_ids, pos_tag_ids=None, dep_ids=None, wn_ids=None):
        input_shape = input_ids.size()
        
        # Base embeddings
        inputs_embeds = self.word_embedding(input_ids)
        position_ids = torch.arange(input_shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.pos_embedding(position_ids)
        tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)
        
        # Linguistic features
        ling_features = []
        if self.config.use_pos and pos_tag_ids is not None:
            ling_features.append(self.pos_tag_embedding(pos_tag_ids))
        if self.config.use_dep and dep_ids is not None:
            ling_features.append(self.dep_embedding(dep_ids))
        if self.config.use_wn and wn_ids is not None:
            ling_features.append(self.wn_linear(wn_ids.unsqueeze(-1).float()))
        
        # Combine features if any exist
        if ling_features:
            ling_combined = self.ling_combiner(torch.cat(ling_features, dim=-1))
        else:
            ling_combined = 0
        
        # Final embedding combination
        embeds = inputs_embeds + pos_embeds + tk_type_embeds + ling_combined
        embeds = self.embed_layer_norm(embeds)
        embeds = self.embed_dropout(embeds)
        
        return embeds
        
    def encode(self, hidden_states, attention_mask):
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)
        for i, layer_module in enumerate(self.bert_layers):
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states
    
    def forward(self, input_ids, attention_mask, pos_tag_ids=None, dep_ids=None, wn_ids=None):
        embed_args = {
            'input_ids': input_ids}
        if self.config.use_pos and pos_tag_ids is not None:
            embed_args['pos_tag_ids'] = pos_tag_ids
        if self.config.use_dep and dep_ids is not None:
            embed_args['dep_ids'] = dep_ids 
        if self.config.use_wn and wn_ids is not None:
            embed_args['wn_ids'] = wn_ids
            
        embedding_output = self.embed(**embed_args)  
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)
        return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
