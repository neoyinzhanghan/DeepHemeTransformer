from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR, CyclicLR


class MyCyclicLR(SequentialLR):
    def __init__(self, optimizer, max_lr, base_lr, step_size_up):
        scheduler1 = LinearLR(
            optimizer,
            start_factor=1,
            end_factor=base_lr / max_lr,
            total_iters=step_size_up,
        )
        scheduler2 = CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=base_lr + (max_lr - base_lr) / 2,
            mode="triangular2",
            step_size_up=step_size_up,
            cycle_momentum=False,
        )
        super().__init__(optimizer, [scheduler1, scheduler2], milestones=[step_size_up])


class MyCosineWarmupLR(SequentialLR):
    def __init__(self, optimizer, max_lr, base_lr, warmup_epochs, T_max):
        # Linear warm-up phase
        scheduler1 = LinearLR(
            optimizer,
            start_factor=base_lr / max_lr,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        # Cosine annealing phase
        scheduler2 = CosineAnnealingLR(
            optimizer, T_max=T_max - warmup_epochs, eta_min=base_lr
        )

        # Combine the schedulers
        super().__init__(
            optimizer, [scheduler1, scheduler2], milestones=[warmup_epochs]
        )


class MyCosineLR(CosineAnnealingLR):
    def __init__(self, optimizer, base_lr, T_max):
        super().__init__(optimizer, T_max=T_max, eta_min=base_lr)
