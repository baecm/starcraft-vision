import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def loss_summary(self) -> str:
        """'loss 0.5320 [classifier 0.0990 mask 0.3100 ...]': the epoch-to-date
        mean of the total loss and of each term (the "loss_" prefix dropped)."""
        terms = [
            f"{name[len('loss_'):]} {meter.global_avg:.4f}"
            for name, meter in self.meters.items()
            if name.startswith("loss_") and meter.count
        ]
        total = self.meters["loss"].global_avg if "loss" in self.meters and self.meters["loss"].count else float("nan")
        return f"loss {total:.4f} [{' '.join(terms)}]"

    def log_every(self, iterable, print_interval_s, header=None, on_report=None):
        """Yield from iterable, printing one progress line on the first
        iteration and then at most once every print_interval_s seconds.

        The line holds what is worth watching mid-epoch: position, ETA, the
        epoch-to-date losses, lr, the last gradient norm, speed and memory.
        Skipped batches are reported by the training loop as they happen.
        on_report(i, seconds_per_iter), if given, is called at each printed
        line (train.py sends the same numbers to W&B).
        After the loop, total_time, seconds_per_iter and data_seconds_per_iter
        hold the epoch's timing.
        """
        header = header or ""
        n = len(iterable)
        start_time = time.time()
        end = start_time
        last_print = None
        iter_time = SmoothedValue(window_size=50)
        data_time = SmoothedValue(window_size=50)
        for i, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            now = time.time()
            iter_time.update(now - end)
            end = now
            if last_print is None or now - last_print >= print_interval_s:
                last_print = now
                eta = datetime.timedelta(seconds=int(iter_time.global_avg * (n - i - 1)))
                parts = [
                    f"{header} {i + 1}/{n}",
                    f"eta {eta}",
                    self.loss_summary(),
                ]
                if "lr" in self.meters:
                    parts.append(f"lr {self.meters['lr'].value:.6f}")
                if "grad_norm" in self.meters:
                    parts.append(f"grad {self.meters['grad_norm'].value:.3g}")
                parts.append(f"{iter_time.avg:.2f} s/it (data {data_time.avg:.3f})")
                if torch.cuda.is_available():
                    parts.append(f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
                print("  ".join(parts), flush=True)
                if on_report is not None:
                    on_report(i, iter_time.avg)
        self.total_time = time.time() - start_time
        self.seconds_per_iter = iter_time.global_avg if iter_time.count else float("nan")
        self.data_seconds_per_iter = data_time.global_avg if data_time.count else float("nan")


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
