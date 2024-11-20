import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, StepLR


def main():
    # Configuration for testing
    config = {
        "scheduler": True,
        "scheduler_step": 200,  # Step decay interval
        "warmup_ratio": 0.001,
        "lr": 0.0002,# Decay factor
        "warmup_epochs": 15,  # Warmup period in steps
    }

    # Dummy model parameter for optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=0.0002)
    warmup_epochs = 15
    warmup_ratio = 0.001
    # Define the warmup and step decay schedulers
    warmup_epochs = config["warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            lr = warmup_ratio + (1.0 - warmup_ratio) * (epoch / warmup_epochs)
            return lr
        else:
            # After warm-up, switch to a StepLR-like decay by returning 1 (base_lr)
            return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    step_scheduler = StepLR(optimizer, step_size=config["scheduler_step"], gamma=0.9)

    # Wrap schedulers for warmup and decay
    scheduler = (warmup_scheduler, step_scheduler) if config["scheduler"] else None

    # Initialize training variables
    global_step = 0
    num_epochs = 250
    steps_per_epoch = 15  # Simulate 15 steps per epoch

    # Training loop with warmup + decay scheduler
    for epoch in range(num_epochs):
        for batch_idx in range(steps_per_epoch):
            # Simulate a training step
            optimizer.step()

            # Print the current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch}, Step: {global_step}, Learning Rate: {current_lr:.8f}")

            # Increment global step
            global_step += 1

        # Step the decay scheduler once per epoch
        if scheduler is not None:
            warmup_scheduler.step()

            # Once warm-up phase is over, switch to StepLR
            if epoch >= warmup_epochs:
                step_scheduler.step()


if __name__ == "__main__":
    main()
