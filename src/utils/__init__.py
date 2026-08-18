from utils.seed import set_global_seed, seed_worker

def collate_fn(batch):
    return tuple(zip(*batch))
