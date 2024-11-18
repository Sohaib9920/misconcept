import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import gc
from typing import Set, Dict
import torch.distributed as dist


def encode(model, dataloader, fp16=False, bf16=False):
    '''
    Find normalized pooled embeddings for whole data under evaluation
    '''
    
    model.eval()
    
    if fp16 and bf16:
        raise ValueError("Both fp16 and bf16 cannot be enabled simultaneously.")
        
    device = next(model.parameters()).device
    amp_dtype = torch.float16 if fp16 else torch.bfloat16 if bf16 else None
    
    bar = tqdm(dataloader, total=len(dataloader))
    
    features_list = []
    ids_list = []

    with torch.no_grad():
        for batch in bar:
            input_ids, attention_mask, ids = batch["input_ids"], batch["attention_mask"], batch["id"]
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            ids_list.extend(ids)
            
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(fp16 or bf16)):
                
                feature = model(input_ids=input_ids, attention_mask=attention_mask)
                feature = F.normalize(feature, p=2, dim=-1)
                
                # Keep features on GPU for faster eval if using CUDA
                # cast because model in float32 may give output in float32 under autocast 
                # (e.g when last layer is layernorm or float() is used on ouput)
                feature = feature.to(dtype=amp_dtype)
                features_list.append(feature)  
                
    features = torch.cat(features_list, dim=0)
    ids = np.array(ids_list) # array needed for indexing

    if dist.is_initialized():
        world_size = dist.get_world_size()
        gathered_features = [torch.empty_like(features) for _ in range(world_size)]
        dist.all_gather(gathered_features, features)
        
        gathered_ids = [None] * world_size
        dist.all_gather_object(gathered_ids, ids)
        
        features = torch.cat(gathered_features, dim=0)
        ids = np.concatenate(gathered_ids)

    bar.close()

    return features, ids


def predict(model, topic_dataloader, content_dataloader, fp16=False, bf16=False, margin=0.16, max_contents=128, chunk_size=1000):
    '''
    Encode the topics and contents and then find most related contents of topics
    '''
    
    print("Encode Topics:")
    topic_features, topic_ids = encode(model, topic_dataloader, fp16=fp16, bf16=bf16)
    
    print("Encode Contents:")
    content_features, content_ids = encode(model, content_dataloader, fp16=fp16, bf16=bf16) 
    
    print("Predicting:")
    pd_topic2content = {}
    
    def split_into_chunks(array, chunk_size):
        return [array[i:i + chunk_size] for i in range(0, len(array), chunk_size)]
    
    chunks = zip(split_into_chunks(topic_features, chunk_size), split_into_chunks(topic_ids, chunk_size))

    for (topic_chunk, topic_chunk_ids) in chunks:
        sim = (topic_chunk @ content_features.T)
        pd_topic2content.update(predict_contents(sim, topic_chunk_ids, content_ids, margin, max_contents))
    
    print("Completed")
    
    del content_features, topic_features, topic_chunk, sim, chunks
    torch.cuda.empty_cache()
    gc.collect()
    
    return pd_topic2content


def predict_contents(similarity_matrix, topic_ids, content_ids, margin=0.16, max_contents=128):
    '''
    Find most related contents of topics given the similarity matrix
    '''
    
    pd_topic2content = {}
    # sim must be in [-1, 1] range
    for i in range(len(similarity_matrix)):

        # Getting at most `max_contents` ordered indices. Ordering required for scores like MAP
        sim = similarity_matrix[i]
        values, indices = sim.topk(k=min(len(sim), max_contents))

        # Filter indices using dynamic range which increases with max sim. 
        # No selection when max sim is negative.
        # Useful when score is based on precision and hence adding wrong contents decrease score
        if margin is not None:
            th = sim.max() - margin * sim.max()
            indices = indices[values >= th]

        # Get relevent predicted contents and add them to predicted topic2content
        pd_contents = content_ids[indices.cpu()]
        topic = topic_ids[i]
        pd_topic2content[topic] = list(pd_contents) # need the order so keep it in list rather than set
    
    return pd_topic2content


def compute_metrics(gt, pd):

    gt = set(gt)
    pd = set(pd)

    if len(pd) == 0:
        precision = 0.0
    else:
        precision = len(gt.intersection(pd)) / len(pd)
        
        
    if len(gt) == 0:
        recall = 0.0
    else:
        recall = len(gt.intersection(pd)) / len(gt)


    if (4 * precision + recall) == 0.0:
        f2 = 0.0
    else:
        f2 = (5 * precision * recall) / (4 * precision + recall)
        
    return f2, precision, recall  


def score_predictions(pd_topic2content, gt_topic2content):
    '''
    Find the score from language-wise and true predictions
    '''
    metrics = [
        compute_metrics(gt_topic2content[t], pd_cs) + (len(pd_cs),)
        for t, pd_cs in pd_topic2content.items()
    ]

    f, p, r, s = map(np.mean, zip(*metrics))
    s = int(s)

    print("-" * 80)
    print(f"Eval Score: {f:.5f} - Precision: {p:.5f} - Recall: {r:.3f} - Selected: {s}")
    print("-" * 80)
    
    return f, p, r


def evaluate_eval(model, eval_loader_topic, eval_loader_content, gt_topic2content: Dict[str, Set[str]], 
                 margin=0.16, fp16=True, bf16=False, max_contents=128, chunk_size=10000):
    '''
    Predict the contents associated with each topic and evaluate these predictions against the ground truth 
    provided in `gt_topic2content`.
    '''
    
    pd_topic2content = predict(model, eval_loader_topic, eval_loader_content, 
                               margin=margin, fp16=fp16, bf16=bf16, 
                               max_contents=max_contents, chunk_size=chunk_size)
    
    f, p, r = score_predictions(pd_topic2content, gt_topic2content)
    
    return f, p, r


def evaluate_train(model, train_loader_topic, train_loader_content, gt_topic2content: Dict[str, Set[str]], 
                   content2topic: Dict[str, Set[str]], margin=0.16, fp16=True, bf16=False, max_contents=128, 
                   chunk_size=10000):
    '''
    Predict the contents associated with each topic and evaluate these predictions against the ground truth 
    provided in `gt_topic2content`. This function identifies missing pairs (i.e., topic-content pairs that 
    were not successfully predicted) and generates a `topic2wrong` mapping, which links each topic to the 
    incorrectly predicted content pairs associated with it.
    '''
    
    pd_topic2content = predict(model, train_loader_topic, train_loader_content, 
                               margin=margin, fp16=fp16, bf16=bf16, 
                               max_contents=max_contents, chunk_size=chunk_size)
    
    f, p, r = score_predictions(pd_topic2content, gt_topic2content)

    # Making unordered set ordered using sorted is must required otherwise DDP devices would have different order
    # and hence different batches leading to hang

    missing_dict = {}
    wrong_dict = {}
    for t, cs in pd_topic2content.items():
        cs = set(cs)
        gt = gt_topic2content[t]
        missing = gt - cs
        wrong = cs - gt
        if len(wrong) > 0:
            wrong_dict[t] = sorted(wrong)
        if len(missing) > 0:
            missing_dict[t] = sorted(missing)

    missing_pairs = []
    for t, mcs in missing_dict.items():
        for mc in mcs:
            missing_pairs.append((t, mc))

    topic2wrong = dict()
    for t, wcs in wrong_dict.items():
        candidates = []
        for wc in wcs:
            other_topics = sorted(content2topic[wc])
            for ot in other_topics:
                candidates.append((ot, wc))
        topic2wrong[t] = candidates

    return missing_pairs, topic2wrong
    