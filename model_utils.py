import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist


class MeanPooling(nn.Module):  
    """Averages token embeddings, useful for summarization and classification tasks."""
    def __init__(self):
        super().__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        attention_mask = attention_mask.unsqueeze(-1).float()  # Shape: (batch_size, seq_len, 1)
        sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
        sum_mask = attention_mask.sum(dim=1)  # Sum of valid tokens
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
        mean_embeddings = sum_embeddings / sum_mask  # Mean across valid tokens
        return mean_embeddings   


class CLSPooling(nn.Module):  
    """Uses the [CLS] token, which aggregates information from all tokens. Common for BERT models."""
    def __init__(self):
        super().__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        return last_hidden_state[:, 0, :]  # Extract [CLS] token embedding


class LastTokenPooling(nn.Module):  
    """Uses the last non-padding token's embedding. Useful for autoregressive models like GPT."""
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state, attention_mask):
        # Find the index of the last non-padding token for each sequence in the batch
        last_non_padding_index = (attention_mask.sum(dim=1) - 1).long()
        # Gather the corresponding token embeddings
        pooled_embeddings = last_hidden_state[torch.arange(last_hidden_state.size(0)), 
                                              last_non_padding_index, :]
        return pooled_embeddings


class Net(nn.Module):
    def __init__(self, config):
        
        super().__init__()

        self.transformer = AutoModel.from_pretrained(config.transformer, 
                                                    torch_dtype=config.torch_dtype,
                                                    attention_probs_dropout_prob=0.,
                                                    hidden_dropout_prob=0.,
                                                    add_pooling_layer=(config.pooling=="pooler")
                                                    )

        if config.pooling == "cls":
            self.pooler = CLSPooling()
        elif config.pooling == "pooler":
            self.pooler = "pooler"
        elif config.pooling == "last":
            self.pooler = LastTokenPooling()
        else:
            self.pooler = MeanPooling()
    
        if config.gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable({"use_reentrant": config.use_reentrant})
         
        # According to CLIP paper. temperature range: [0, inf], scale range: [inf, 0], ln(scale) range: [-inf, inf] 
        # It better to have unrestricted range for parameter so we take ln(scale)
        self.logit_scale = torch.nn.Parameter(torch.ones([], dtype=config.torch_dtype) * np.log(1 / 0.07)) 
    
    
    def forward(self, input_ids, attention_mask):
        '''
        Find pooled embeddings/logits
        '''
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = out.last_hidden_state

        if self.pooler == "pooler":
            pooled_output = out.pooler_output
        else:
            pooled_output = self.pooler(sequence_output, attention_mask)

        return pooled_output


class CLIPLoss(nn.Module):
    def __init__(self, distributed=False, **kwargs):
        super().__init__()
        self.loss_function = nn.CrossEntropyLoss(**kwargs)
        if distributed:
            self.distributed = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.distributed = False

    def forward(self, t_features, c_features, logit_scale):
        t_features = F.normalize(t_features, p=2, dim=-1)
        c_features = F.normalize(c_features, p=2, dim=-1)
        
        if self.distributed:
            t_features = self._dist_gather_tensor(t_features)
            c_features = self._dist_gather_tensor(c_features)
            
        logits = logit_scale * (t_features @ c_features.T)
        labels = torch.arange(len(logits), device=logits.device, dtype=torch.long)
        loss = self.loss_function(logits, labels)

        # Distributed interpertation:
        # if we have computation graph with 2 paths (w-a-L, w-b-L) and we want dL/dw which passes
        # through intermediates [a, b] then In orginal case, It is propagated through both paths and sum
        # the path gradients (dL/dw = dL/dw|_a + dL/dw|_b) but In distributed case, gpu who found 'a' will 
        # find dL/dw|_a passing through 'a' while other gpu will find dL/dw|_b passing through b.
         
        return loss

    def _dist_gather_tensor(self, t):
        t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        # Following Step is important because:
        # Require atleast one node to propagate gradients through because gathered tensors have no gradient
        all_tensors[self.rank] = t 
        all_tensors = torch.cat(all_tensors, dim=0)
        
        return all_tensors