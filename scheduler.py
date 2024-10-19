import torch


class PrefixStepLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.9,
                 constant_epochs=330, initial_lr=0.001,
                 new_lr=0.0005, last_epoch=-1):
        """
        Custom LR scheduler that keeps the learning rate constant for a specified number of epochs,
        then follows a step decay schedule.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            step_size (int): Period of learning rate decay after the constant period.
            gamma (float): Multiplicative factor of learning rate decay. Default: 0.9.
            constant_epochs (int): Number of epochs to keep the learning rate constant.
            last_epoch (int): The index of the last epoch. Default: -1 (automatically sets initial lr).
        """
        self.constant_epochs = constant_epochs  # Epochs to keep LR constant
        self.step_size = step_size  # Step size for decay after constant period
        self.gamma = gamma  # Decay factor for LR
        self.initial_lr = initial_lr  # Initial LR for the first `constant_epochs`
        self.new_lr = new_lr  # LR to switch to after `constant_epochs`
        super(PrefixStepLRScheduler, self).__init__(optimizer, last_epoch)

        # Initialize learning rate to `initial_lr`
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr

    def get_lr(self):
        # If current epoch is within constant period, return the initial LR
        if self.last_epoch < self.constant_epochs:
            return [self.initial_lr for _ in self.optimizer.param_groups]
        # After constant period, set the learning rate to new_lr and apply StepLR decay
        else:
            # Calculate the decay from `new_lr` using StepLR logic
            step_adjusted_epoch = self.last_epoch - self.constant_epochs
            return [self.new_lr * self.gamma ** (step_adjusted_epoch // self.step_size)
                    for _ in self.optimizer.param_groups]
