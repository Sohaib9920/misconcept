import torch.distributed as dist

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
