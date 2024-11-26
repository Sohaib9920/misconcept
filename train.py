from dataclasses import dataclass
import torch
import pandas as pd
import torch.distributed as dist
from transformers import set_seed, AutoTokenizer
from torch.utils.data import DataLoader
from data import TrainDataset, prepare_eval_loaders
from utils import read_pkl, process_correlations
from model_utils import Net
from trainer import Trainer
from sklearn.model_selection import train_test_split
import os
import deepspeed

@dataclass
class Configuration:
    
    # Transformer
    transformer: str = "sentence-transformers/LaBSE"
    pooling: str = 'cls'                   # 'mean' | 'cls' | 'pooler' | 'last'

    # Eval
    eval_val = False
    eval_train = False
    margin: float = 0.16
    eval_batch_size: int = 768
    max_contents = 128

    # Others
    fp16: bool = True
    bf16: bool = False
    
    # Debugging
    debug = True                     
        
    # Training 
    seed: int = 42
    epochs: int = 1
    train_batch_size: int = 512
    gradient_checkpointing: bool = True 
    weight_decay = 0.01

    # model_config
    torch_dtype = torch.bfloat16
    # Only use quantization when model size is very huge because when it is
    # small and use_reentrant=False, then net gain is less then only using LORA
    load_in_8bit = False
    load_in_4bit = False
    bnb_4bit_quant_type = "nf4"
    use_bnb_nested_quant = False

    use_peft = False
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = "all-linear"

    # Optimizer
    max_grad_norm = 1.0                   
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.0002                   
    scheduler: str = None
    warmup_ratio: float = 0.25/1
    
    # Data
    fold: int = 0                        
    max_len: int = 96          
     
    # Sampling
    max_wrong: int = 128  
    missing_freq: float = 0.5

    # deepspeed
    zero: int = 2
    offload: bool = False


# Setup
config = Configuration() 

if os.environ.get("LOCAL_RANK") is not None:
    config.distributed = True
    deepspeed.init_distributed('nccl')
    config.rank = dist.get_rank()
    config.world_size = dist.get_world_size()
else:
    config.distributed = False
    config.rank = 0
    config.world_size = 1

config.device = torch.device(config.rank if torch.cuda.is_available() else "cpu")
if config.device.type != "cpu":
    torch.cuda.set_device(config.device) # must required for dist.barrier to work in case of DDP

set_seed(config.seed)

# loading correlation data
df_correlations = pd.read_csv('/kaggle/input/curriculum-split-data-prep/correlations.csv')
if config.debug:
    _, df_correlations = train_test_split(df_correlations, stratify=df_correlations["fold"], random_state=config.seed, test_size=1000)
    df_correlations.reset_index(drop=True, inplace=True)

# Preparing data loaders
tokenizer = AutoTokenizer.from_pretrained(config.transformer)

train_pairs, topic2content, content2topic, train_topics, eval_topics, train_contents, eval_contents = process_correlations(df_correlations, config)
topic2text = read_pkl("/kaggle/input/curriculum-split-data-prep/topic2text.pkl")
content2text = read_pkl("/kaggle/input/curriculum-split-data-prep/content2text.pkl")

assert config.train_batch_size % config.world_size == 0, "Train Batch size must be divisible by world size to evenly distribute data"
assert config.eval_batch_size % config.world_size == 0, "Eval Batch size must be divisible by world size to evenly distribute data"
  
train_dataset = TrainDataset(train_pairs, topic2content, content2topic, config.train_batch_size, tokenizer, topic2text, content2text, config.max_len)
train_loader = DataLoader(train_dataset, batch_size=None, collate_fn=train_dataset.collater, shuffle=True)

if config.eval_train:
    print("train loader (eval)")
    train_loader_topic, train_loader_content = prepare_eval_loaders(config, tokenizer, train_topics, train_contents, topic2text, content2text)
else:
    train_loader_topic, train_loader_content = None, None

if config.eval_val:
    print("eval loader")
    eval_loader_topic, eval_loader_content = prepare_eval_loaders(config, tokenizer, eval_topics, eval_contents, topic2text, content2text)
else:
    eval_loader_topic, eval_loader_content = None, None

# Preparing model
model = Net(config)

# Training 
trainer = Trainer(config, model, train_loader, 
                  train_loader_topic=train_loader_topic, train_loader_content=train_loader_content, 
                  eval_loader_topic=eval_loader_topic, eval_loader_content=eval_loader_content, 
                  topic2content=topic2content, content2topic=content2topic)

trainer.train()

print("After training {}|{}".format(torch.cuda.max_memory_allocated()/1e6, torch.cuda.max_memory_reserved()/1e6))

if config.distributed:
    dist.destroy_process_group()