import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    # """Implements stochastic gradient descent (optionally with momentum).

    # Nesterov momentum is based on the formula from
    # `On the importance of initialization and momentum in deep learning`__.

    # Args:
        # params (iterable): iterable of parameters to optimize or dicts defining
            # parameter groups
        # lr (float): learning rate
        # momentum (float, optional): momentum factor (default: 0)
        # weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        # dampening (float, optional): dampening for momentum (default: 0)
        # nesterov (bool, optional): enables Nesterov momentum (default: False)

    # Example:
        # >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        # >>> optimizer.zero_grad()
        # >>> loss_fn(model(input), target).backward()
        # >>> optimizer.step()

    # __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    # .. note::
        # The implementation of SGD with Momentum/Nesterov subtly differs from
        # Sutskever et. al. and implementations in some other frameworks.

        # Considering the specific case of Momentum, the update can be written as

        # .. math::
                  # v = \rho * v + g \\
                  # p = p - lr * v

        # where p, g, v and :math:`\rho` denote the parameters, gradient,
        # velocity, and momentum respectively.

        # This is in contrast to Sutskever et. al. and
        # other frameworks which employ an update of the form

        # .. math::
             # v = \rho * v + lr * g \\
             # p = p - v

        # The Nesterov version is analogously modified.
    # """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        self.name = "SGD"
        self.lr = lr
        self.momnt = momentum
        self.wDecay = weight_decay
        if  lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)


    def set_params(self, params = None, lr = None, momentum = None, dampening = None, weight_decay = None, nesterov
                  = None):

        lrIn = lr if lr is not None else self.lr 
        momentumIn = momentum if momentum is not None else self.momnt
        dampeningIn = dampening if dampening is not None else self.param_groups[0]['dampening']
        WDecayIn = weight_decay if weight_decay is not None else self.param_groups[0]['weight_decay']
        nesterovIn = nesterov if nesterov is not None else self.param_groups[0]['nesterov']
        params = params if params is not None else self.param_groups[0]['params']
        self.name = "SGD"
        self.lr = lrIn
        self.momnt = momentumIn
        print(WDecayIn, nesterovIn, params, self.name)


        defaults = dict(lr=lrIn, momentum=momentumIn, dampening=dampeningIn,
                        weight_decay=WDecayIn, nesterov=nesterovIn)
        if nesterovIn and (momentumIn <= 0 or dampeningIn != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, t=1, closure=None):
        # """Performs a single optimization step.

        # Arguments:
            # closure (callable, optional): A closure that reevaluates the model
                # and returns the loss.
        # """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
