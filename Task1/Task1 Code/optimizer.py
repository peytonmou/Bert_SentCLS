from typing import Callable, Iterable, Tuple
import math 
import torch
from torch.optim import Optimizer

#### VLJ: For Task 1, step() need to be implemented !####

class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)        # 1st moment vector
                    state['v'] = torch.zeros_like(p.data)        # 2nd moment vector
 
                # Access hyperparameters from the `group` dictionary
                beta1, beta2 = group['betas']
                eps = group['eps']
                alpha = group['lr']
                weight_decay = group['weight_decay']
                
                # Increment step counter first
                state['step'] += 1
                step = state['step']
                
                # Get state variables
                m, v = state['m'], state['v']
                
                # Apply weight decay 
                if weight_decay != 0:
                    p.data.add_(p.data, alpha= -alpha * weight_decay)
                
                # Update first and second moments of the gradients
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                
                # Efficient method for computing bias correction
                if group['correct_bias']:
                    step_size = alpha * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
                else:
                    step_size = alpha
                
                # Update parameters (incorporating learning rate) 
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)

        return loss