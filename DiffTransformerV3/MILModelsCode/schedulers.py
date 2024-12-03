from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR, CyclicLR

class MyCyclicLR(SequentialLR):
    def __init__(self, optimizer, max_lr, base_lr, step_size_up):
        scheduler1 = LinearLR(optimizer, start_factor=1, end_factor=base_lr/max_lr, total_iters=step_size_up)
        scheduler2 = CyclicLR(optimizer, base_lr=base_lr, max_lr=base_lr + (max_lr - base_lr)/2, mode='triangular2', 
                             step_size_up=step_size_up, cycle_momentum=False)
        super().__init__(optimizer, [scheduler1, scheduler2], milestones=[step_size_up])

class MyCosineLR(SequentialLR):
    def __init__(self, optimizer, max_lr, base_lr, T_max):
        scheduler1 = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=base_lr)
        scheduler2 = LinearLR(optimizer, start_factor=base_lr/max_lr, end_factor=base_lr/max_lr)
        super().__init__(optimizer, [scheduler1, scheduler2], milestones=[T_max])