import torch
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWithWarmup(LRScheduler):
    def __init__(self, optimizer, min_lr, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)
        for lr in self.base_lrs:
            assert lr > min_lr, "Base learning rate must be greater than min_lr"

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.min_lr] * len(self.optimizer.param_groups)
        elif self.last_epoch < self.warmup_steps:
            return [
                self.min_lr + (base_lr - self.min_lr) * (self.last_epoch / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]
