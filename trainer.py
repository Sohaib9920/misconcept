import torch
from torch import optim
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, set_seed
from tqdm import tqdm
from model_utils import CLIPLoss
from eval import evaluate_eval, evaluate_train
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


class Trainer:
    def __init__(self, config, model, train_loader, 
                 train_loader_topic=None, train_loader_content=None, 
                 eval_loader_topic=None, eval_loader_content=None,
                 topic2content=None, content2topic=None):
        
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.train_loader_topic = train_loader_topic
        self.train_loader_content = train_loader_content
        self.eval_loader_topic = eval_loader_topic
        self.eval_loader_content = eval_loader_content
        self.topic2content = topic2content
        self.content2topic = content2topic

        self.distributed = self.config.distributed

        if self.eval_loader_topic is not None and self.eval_loader_content is not None:
            assert (self.topic2content is not None)
            self.eval_val = True
        else:
            self.eval_val = False
        
        if self.train_loader_topic is not None and self.train_loader_content is not None:
            assert ((self.topic2content is not None) and (self.content2topic is not None))
            self.eval_train = True
        else:
            self.eval_train = False
        
        ########### Loss function ###########
        self.loss_function = CLIPLoss(label_smoothing=config.label_smoothing, distributed=self.distributed)
        
        ########## Optimizer ###########
        decay_params = [p for p in self.model.parameters() if p.ndim >= 2]
        non_decay_params = [p for p in self.model.parameters() if p.ndim < 2]
        param_groups = [{"params": decay_params, "weight_decay": config.weight_decay}, {"params": non_decay_params, "weight_decay": 0.0}]
        AdamOptimizer = DeepSpeedCPUAdam if config.offload else FusedAdam
        self.optimizer = AdamOptimizer(param_groups, lr=config.lr, betas=(0.9, 0.95))
        
        ############ Scheduler ##############
        steps_per_epoch = len(train_loader)
        total_steps = config.epochs * steps_per_epoch * (1.5 if self.eval_train else 1) # investigate scedular for dynamic dataloader later
        if config.scheduler:
            num_warmup_steps = int(total_steps * config.warmup_ratio)
            if config.scheduler == "cosine":
                self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, total_steps)
            elif config.scheduler == "constant":
                self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps)
            else:
                self.scheduler = None
        else:
            self.scheduler = None
        
        ########## Wrapped Model Model ###########

        if self.distributed:

            ds_config = {
                "train_micro_batch_size_per_gpu": config.train_batch_size // config.world_size,
                "gradient_accumulation_steps": 2,
                "gradient_clipping": config.max_grad_norm,
                "fp16": {
                    "enabled": config.fp16,
                    "loss_scale_window": 100
                },
                "bf16": {
                    "enabled": config.bf16
                },
                "zero_optimization": {
                    "stage": config.zero,
                    "offload_param": {
                        "device": "cpu" if config.offload else "none"
                    },
                    "offload_optimizer": {
                        "device": "cpu" if config.offload else "none",
                    },
                    "stage3_gather_fp16_weights_on_model_save": True,
                }
            }

            self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(model=self.model,
                                                                                optimizer=self.optimizer,
                                                                                config=ds_config,
                                                                                lr_scheduler=self.scheduler,
                                                                                dist_init_required=True)
        
        else:
            self.model.to(self.config.device)

        if self.config.rank == 0:
            for n, p in self.model.named_parameters():
                print("{}|{}|{}|{}".format(n, p.dtype, p.requires_grad, p.device))

        print("Before training {}|{}".format(torch.cuda.max_memory_allocated()/1e6, torch.cuda.max_memory_reserved()/1e6))
            

    def train_epoch(self):
        self.model.train()
        
        steps_per_epoch = len(self.train_loader) // 2 # found at each epoch as it is dynamic
        bar = tqdm(total=steps_per_epoch, desc=f"{self.config.rank} Training Steps")

        device = self.config.device
        epoch_loss = 0
        step_loss = 0
        log_info = {}
        step = 0
                
        for i, batch in enumerate(self.train_loader):
            t_input_ids, t_attention_mask = batch["t_input_ids"].to(device, non_blocking=True), batch["t_attention_mask"].to(device, non_blocking=True)
            c_input_ids, c_attention_mask = batch["c_input_ids"].to(device, non_blocking=True), batch["c_attention_mask"].to(device, non_blocking=True)
            t_features = self.model(input_ids=t_input_ids, attention_mask=t_attention_mask)
            c_features = self.model(input_ids=c_input_ids, attention_mask=c_attention_mask)
            logit_scale = self.model.logit_scale.squeeze().exp()
            loss = self.loss_function(t_features, c_features, logit_scale) # same in both single and distributed case due to gather
            step_loss += loss.item() / 2

            if self.distributed:
                self.model.backward(loss)
            else:
                loss = loss / 2
                loss.backward()
            
            if self.distributed:
                self.model.step()
            else:
                if (i + 1) % 2 == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()

            if (i + 1) % 2 == 0:
                log_info["step_loss"] = f"{step_loss:.5f}"
                log_info["step_lr"] = f"{self.optimizer.param_groups[0]['lr']:.3e}"
                with torch.no_grad():
                    self.model.logit_scale.clamp_(0, math.log(100))
                    log_info["step_scale"] = f"{self.model.logit_scale.item():.5f}"
                
                    epoch_loss += step_loss / steps_per_epoch
                    step_loss = 0
                    step += 1
    
                    bar.set_postfix(ordered_dict=log_info)
                    bar.update(1)
            
            if step == steps_per_epoch:
                break
        
        bar.close()
        return epoch_loss

    
    def evaluate(self):
        self.model.eval()

        if self.eval_val:
            print('\n{}[{}| {}]{}'.format(30*'-', self.config.rank, 'Evaluate (Val)', 30*'-'))
            f, p, r = evaluate_eval(self.model, self.eval_loader_topic, self.eval_loader_content, self.topic2content,
                                    margin=self.config.margin, fp16=self.config.fp16, bf16=self.config.bf16, 
                                    max_contents=self.config.max_contents)
        
        if self.eval_train:
            print('\n{}[{}| {}]{}'.format(30*'-', self.config.rank, 'Evaluate (Train)', 30*'-'))
            missing_pairs, topic2wrong = evaluate_train(self.model, self.train_loader_topic, self.train_loader_content, self.topic2content,
                                                        self.content2topic, margin=self.config.margin, fp16=self.config.fp16, bf16=self.config.bf16, 
                                                        max_contents=self.config.max_contents)
                                                            
            self.train_loader.dataset.shuffle(missing_pairs, topic2wrong, max_wrong=self.config.max_wrong, missing_freq=self.config.missing_freq)

    
    def train(self):
        set_seed(self.config.seed) 
        
        for epoch in range(1, self.config.epochs + 1):
            print('\n{}[{}| Epoch: {}]{}'.format(30*'-', self.config.rank, epoch, 30*'-'))
            epoch_loss = self.train_epoch()
            print('{}| Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}'.format(self.config.rank, epoch, epoch_loss, self.optimizer.param_groups[0]['lr']))
            self.evaluate()
            if self.distributed:
                dist.barrier()

        print(f"{self.config.rank}: End")