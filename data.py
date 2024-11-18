from typing import List, Tuple, Dict, Set
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset
from collections import deque
import random
from tqdm import tqdm
import numpy as np
import torch.distributed as dist


class TrainDataset(Dataset):
    
    def __init__(self, 
                 pairs: List[Tuple[int, int]], 
                 topic2content: Dict[str, Set[str]], 
                 content2topic: Dict[str, Set[str]],
                 batch_size: int,
                 tokenizer,
                 topic2text: Dict[str, str],
                 content2text: Dict[str, str],
                 max_len: int):
        
        super().__init__()
        
        # Initializing pairs and mappings for topics and contents
        self.pairs = pairs
        self.topic2content = topic2content
        self.content2topic = content2topic
        self.batch_size = batch_size
        
        # Shuffle to create initial batches
        self.shuffle(None, None)
        
        # For tokenization
        self.tokenizer = tokenizer
        self.topic2text = topic2text
        self.content2text = content2text
        self.max_len = max_len

        self.splitter = BatchSplitter() if dist.is_initialized() else None

    
    def __getitem__(self, index):
        """
        Fetch a batch of pairs based on the given index.
        """
        pairs_batch = self.batches[index]
        return pairs_batch
    

    def collater(self, batch):
        """
        Split the batch of pairs if it is distributed and then tokenize that split.
        """
        if self.splitter is not None:
            batch = self.splitter(batch)
            print(f"I am working {dist.get_rank()}")
        inputs = self.tokenize_pairs_batch(batch, self.tokenizer, self.topic2text, self.content2text, self.max_len)
        return inputs

    
    def tokenize_pairs_batch(self, batch, tokenizer, topic2text, content2text, max_len):
        """
        Tokenize batch of topics and contents
        """
        topics = []
        contents = []
        topic_texts = []
        content_texts = []
    
        for pair in batch:
            topic, content = pair
            # Context = Title # Description # Text (cut to 32 based on white space splitting)
            content_text = content2text[content]
            # Topic = Title # Parent # Grandparent # … # Description
            topic_text = topic2text[topic]
    
            topics.append(topic)
            contents.append(content)
            topic_texts.append(topic_text)
            content_texts.append(content_text)
    
    
        tokenized_topics = tokenizer(topic_texts, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=max_len)
        tokenized_contents = tokenizer(content_texts, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=max_len)
        inputs = {f"t_{k}": v for k,v in tokenized_topics.items()} | {f"c_{k}": v for k,v in tokenized_contents.items()} 
        inputs["content_id"] = contents
        inputs["topic_id"] = topics
        
        return inputs

    
    def __len__(self):
        """
        Returns the total number of batches available.
        """
        return len(self.batches)

    
    def shuffle(self, 
                missing_pairs:List[Tuple[str, str]], 
                topic2wrong: Dict[str, List[Tuple[str, str]]], 
                max_wrong=25, 
                missing_freq=0.5):
        """
        Shuffle the dataset to create balanced batches of topic-content pairs, including 
        hard negatives (wrong predictions) and oversampled missing pairs for model training.
        
        Parameters:
        ----------
        missing_pairs: 
            List of topic-content pairs that were previously missed by the model (low similarity).
            These pairs are oversampled during batch creation to help the model focus on difficult examples.
        
        topic2wrong: 
            Dictionary mapping each topic to a list of all possible pairs of other topics and wrong
            predictions of that topic (hard negatives for that topic).
            These pairs are used as negative samples when forming batches with the topic.
        
        max_wrong: int, optional (default=25)
            The maximum number of hard negative pairs to add for each topic. Adjust based on batch size; 
            smaller batches should use fewer hard negatives.
        
        missing_freq: float, optional (default=0.5)
            The frequency at which missing pairs are sampled into batches. 
            A higher frequency means more emphasis on oversampling missing pairs.
        
        Returns:
        -------
        None
        """
        
        # Initialize the deque for efficient popping from the front (O(1) time complexity)
        pair_pool = deque(self.pairs)
        missing_pool = deque(missing_pairs) if missing_pairs else None
        wrong_pool = topic2wrong  # No deepcopy for efficiency
                
        print("\nShuffling pairs and creating suitable batches:")
        random.shuffle(pair_pool)  
        if missing_pool:
            random.shuffle(missing_pool)  
        
        # Track topics and contents to avoid overlap within a batch
        topics_to_avoid = set()
        contents_to_avoid = set()
        
        pairs_epoch = set()  # To track pairs already added to this epoch
        current_batch = []  # Collect pairs for the current batch
        batches = []  # List of final batches
        break_counter = 0  # Count how many consecutive unsuitable pairs were encountered
        max_break_limit = 512  # Threshold to avoid getting stuck on unsuitable pairs
        oversample_missing = 0  # Count how many missing pairs were added
        hard_topics = set()  # Track which topics had hard negatives added
        
        pbar = tqdm()  # Progress bar to track shuffling process
        
        while pair_pool:
            pbar.update()
            
            # Fetch the next pair from the front of the deque
            pair = pair_pool.popleft()
            topic, content = pair
            
            # Skip pairs that have already been used (during negative pairs sampling)
            if pair in pairs_epoch:
                continue

            # Check if the current pair is suitable for the current batch
            if (topic not in topics_to_avoid) and (content not in contents_to_avoid):
                # Avoid adding any topic or content related to the current pair
                topics_to_avoid.update(self.content2topic[content])
                contents_to_avoid.update(self.topic2content[topic])
                
                # Add the valid pair to the current batch and mark it as used
                current_batch.append(pair)
                pairs_epoch.add(pair)
                break_counter = 0  # Reset break counter since a valid pair was found
                
                # Add hard negatives to the batch
                if wrong_pool and len(current_batch) < self.batch_size:
                    wrong_pairs = wrong_pool.get(topic, [])
                    random.shuffle(wrong_pairs)
                    wrong_added = 0
                    for wp in wrong_pairs:
                        wt, wc = wp
                        if (wp not in pairs_epoch) and (wt not in topics_to_avoid) and (wc not in contents_to_avoid):
                            # Add wrong pair and update avoidance sets
                            topics_to_avoid.update(self.content2topic[wc])
                            contents_to_avoid.update(self.topic2content[wt])
                            current_batch.append(wp)
                            pairs_epoch.add(wp)
                            wrong_added += 1
                            hard_topics.add(wt)  # Track topics with added hard negatives
                            if wrong_added >= max_wrong or len(current_batch) >= self.batch_size:
                                break
            
            else:
                # Add unsuitable pairs back to the deque for later consideration
                pair_pool.append(pair)
                break_counter += 1
            
            # Occasionally oversample a missing pair with a frequency controlled by `missing_freq`
            if missing_pool and np.random.rand() < missing_freq and len(current_batch) < self.batch_size:
                mp = missing_pool.popleft()
                mt, mc = mp
                if (mt not in topics_to_avoid) and (mc not in contents_to_avoid):
                    # Add missing pair to batch if it's valid
                    topics_to_avoid.update(self.content2topic[mc])
                    contents_to_avoid.update(self.topic2content[mt])
                    current_batch.append(mp)
                    oversample_missing += 1
                else:
                    # If not valid, return the pair back to the missing pool
                    missing_pool.append((mt, mc))
                        
            # Stop if no suitable pairs are found after checking 512 consecutive pairs
            if break_counter >= max_break_limit: 
                break

            # When the batch is full, save it and reset for the next batch
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
                topics_to_avoid.clear()
                contents_to_avoid.clear()

        pbar.close()
        
        # Store the batches
        self.batches = batches

        # Summary of the shuffling process
        print(f"Estimated Batches: {len(self.pairs) // self.batch_size} - (Oversampled) Batches Created: {len(self.batches)}")
        print(f"Break Counter: {break_counter}")
        print(f"Pairs left: ({len(pair_pool)}/{len(self.pairs)}) - {len(pair_pool) / len(self.pairs) * 100:.2f}%")
        print(f"First Batch first element: {self.batches[0][0]} - Last Batch first element: {self.batches[-1][0]}")
        
        if missing_pool:
            print(f"Oversampled missing: ({oversample_missing}/{len(missing_pairs)}) - {oversample_missing / len(missing_pairs) * 100:.2f}%")
        
        if wrong_pool:
            print(f"Hard negatives added for topics: ({len(hard_topics)}/{len(topic2wrong)}) - {len(hard_topics) / len(topic2wrong) * 100:.2f}%")


class EvalDataset(Dataset):
    
    def __init__(self, ids: List[str], mapping2text: Dict[str, str], tokenizer, max_len):
        super().__init__()
        self.ids = ids
        self.mapping2text = mapping2text
        self.max_len = max_len
        self.tokenizer = tokenizer
    
    
    def __getitem__(self, index):      
        text_id = self.ids[index]
        text_input = self.mapping2text[text_id] 
        
        tok = self.tokenizer(
            text_input,
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=self.max_len,
            return_token_type_ids=False
        )
        
        return {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"], "id": text_id}
    
    
    def __len__(self):
        return len(self.ids)

    
    def collater(self, batch):
        
        tok = self.tokenizer.pad([
            {"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]} for x in batch
        ], return_tensors="pt")
        
        other_keys = [k for k in batch[0].keys() if k not in ["input_ids", "attention_mask"]]
        other_data = {}
        for k in other_keys:
            other_data[k] = [x[k] for x in batch]
        out = tok | other_data
        
        return out


class BatchSplitter:
    '''
    Split batch for distributed training
    '''
    def __init__(self, rank=None, world_size=None):
        self.rank = dist.get_rank() if rank is None else rank
        self.world_size = dist.get_world_size() if world_size is None else world_size

    def __call__(self, inputs):
        if isinstance(inputs, dict):
            split_inputs = {}
            for k, v in inputs.items():
                split_inputs[k] = BatchSplitter.chunk(v, self.world_size)[self.rank]
        else:
            split_inputs = BatchSplitter.chunk(inputs, self.world_size)[self.rank]
            
        return split_inputs

    @staticmethod
    def chunk(data, num_chunks):
        if num_chunks <= 0:
            return []
        
        chunk_size = len(data) // num_chunks
        remainder = len(data) % num_chunks
        
        chunks = []
        start = 0
        
        for i in range(num_chunks):
            end = start + chunk_size + (1 if i < remainder else 0) # fill starting chunks with remainder
            chunks.append(data[start:end])
            start = end
        
        return chunks


def prepare_eval_loaders(config, tokenizer, topics, contents, topic2text, content2text):
    dataset_topic = EvalDataset(topics, topic2text, tokenizer, config.max_len)
    topic_sampler = DistributedSampler(dataset_topic) if config.distributed else None
    batch_size = config.eval_batch_size // config.world_size if config.distributed else config.eval_batch_size
    loader_topic = DataLoader(
        dataset=dataset_topic, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        sampler=topic_sampler,
        collate_fn=dataset_topic.collater
    )
    dataset_content = EvalDataset(contents, content2text, tokenizer, config.max_len)
    content_sampler = DistributedSampler(dataset_content) if config.distributed else None
    loader_content = DataLoader(
        dataset=dataset_content, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        sampler=content_sampler,
        collate_fn=dataset_content.collater
    )

    print(f"\tTopics (examples, batches): ({len(dataset_topic)}, {len(loader_topic)})")
    print(f"\tContents (examples, batches): ({len(dataset_content)}, {len(loader_content)})")

    return loader_topic, loader_content