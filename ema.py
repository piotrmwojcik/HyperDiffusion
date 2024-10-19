from copy import deepcopy

import torch


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999, rampup_rate=0.05, rampup_kimg=4, batch_size=16, eps=1e-8):
        """
        Initializes the EMA object.

        Args:
            model (torch.nn.Module): The model to track the parameters.
            decay (float): The base decay rate (EMA momentum).
            rampup_rate (float): Fraction of total training iterations over which momentum ramps up.
            rampup_kimg (int): A scaling factor for ramping up EMA momentum.
            batch_size (int): Batch size for scaling the ramp-up.
            eps (float): A small constant to prevent division by zero in momentum updates.
        """
        self.model = model
        self.decay = decay
        self.rampup_rate = rampup_rate
        self.rampup_kimg = rampup_kimg
        self.batch_size = batch_size
        self.eps = eps

        # Clone the model structure and parameters to store the EMA weights
        self.ema_model = self._initialize_ema_model()
        self.current_iter = 0

    def _initialize_ema_model(self):
        """Clones the model and initializes the EMA parameters."""
        ema_model = deepcopy(self.model)  # Create a clone of the model
        #ema_model.load_state_dict(self.model.state_dict())  # Copy the initial parameters
        for param in ema_model.parameters():
            param.requires_grad = False  # EMA parameters should not require gradients
        return ema_model

    def _get_momentum(self):
        """
        Returns the current EMA momentum based on ramp-up.
        Uses ramp-up over a certain number of iterations as specified in the config.
        """
        rampup_steps = self.rampup_kimg * 1000 / (self.batch_size + self.eps)
        if self.current_iter < rampup_steps:
            return min(self.decay, 1 - (1 - self.decay) * (self.current_iter / rampup_steps))
        return self.decay

    def update(self):
        """Updates the EMA parameters using the model's current parameters."""
        self.current_iter += 1
        momentum = self._get_momentum()

        with torch.no_grad():
            model_params = list(self.model.parameters())
            ema_params = list(self.ema_model.parameters())

            for model_param, ema_param in zip(model_params, ema_params):
                # Apply linear interpolation (LERP) to update EMA parameters
                ema_param.data.mul_(momentum).add_(model_param.data, alpha=(1 - momentum))

    def set_ema_params(self):
        """Copies EMA parameters back to the model for evaluation (optional)."""
        self.model.load_state_dict(self.ema_model.state_dict())
