import torch
from torch.optim import Optimizer
import math


class TSSGD(Optimizer):
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

    def __init__(self, params, name = "A-TSSGD", lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, w = 1, nesterov=False, a =1, wScheme = 'simple', lrScheme = 'constant'):
        # Nameing style: decay factor, combination shceme - window scheme - learning rate scheme - TSSGD

        # Compination scheme: Differentialy weighed (DW)
        # Decay Factors: Simple averaging(-), Exponential decay(ED)
        # Window Schemes: Simple w window(-), Heritage window(H)
        # Lr rate Scheme: Constant(-), Fixed reduction based on epoch(FT), Adaptive on grad energy(AL), BB
        
        # TSSGD : Simple averaged Time Smoothoed SGD. 
        # ED-TSSGD: exponentially decaying
        # DWED-TSSGD: Differentialy Weighed Time smoothed SGD
        #  
        self.name = name+'-'+lrScheme
        self.descr = '-'.join((self.name, str(lr), str(momentum), str(weight_decay), wScheme, str(w), str(a)))
        self.lr = lr
        self.momnt = momentum
        self.wDecay = weight_decay
        self.w = w 
        self.a = a
        self.history = []
        self.entryIdx = 0
        self.lrFactor = 0
        self.learnDecay = list(map(lambda x: self.lr ** x, reversed(range(0,w+1))))  # This struct stores the decaying learning rates for the stored gradients.
        self.learnDecay = [i if i  >= 10 **-9 else 10 ** -9 for i in self.learnDecay]
        if  lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, w=w, a = a)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        if w < 1:
            raise ValueError("Window size cannot be less than 1")
        if w % 1 != 0:
            raise ValueError("Window size cannot be negative")
        print(self.learnDecay)
        print(a,w)

        if lrScheme == "fixed-reduction":
            self.lrFactor = lambda x: 1/ math.sqrt(x) 
        elif lrScheme == "grad-based":
            self.lrFactor = lambda x: 1 
        else:
            self.lrFactor = lambda x: 1  # this is essential just 1.form it like this so it can be callable.
        
        super(TSSGD, self).__init__(params, defaults)


    def set_params(self, params = None, lr = None, momentum = None, dampening = None, weight_decay = None, nesterov
                  = None, window = 1):

        lrIn = lr if lr is not None else self.lr 
        momentumIn = momentum if momentum is not None else self.momnt
        dampeningIn = dampening if dampening is not None else self.param_groups[0]['dampening']
        WDecayIn = weight_decay if weight_decay is not None else self.param_groups[0]['weight_decay']
        nesterovIn = nesterov if nesterov is not None else self.param_groups[0]['nesterov']
        params = params if params is not None else self.param_groups[0]['params']
        wIn = w if w == 1  else self.param_groups[0]['w']
        aIn = a if a == 0.5  else self.param_groups[0]['a']
        # self.name = "TSSGD"
        self.lr = lrIn
        self.momnt = momentumIn
        print(WDecayIn, nesterovIn, params, self.name)


        defaults = dict(lr=lrIn, momentum=momentumIn, dampening=dampeningIn,
                        weight_decay=WDecayIn, nesterov=nesterovIn, w = wIn, a = aIn)
        if nesterovIn and (momentumIn <= 0 or dampeningIn != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TSSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, t=1, closure=None):
        """Performs a single optimization step.

            Arguments:
                        # closure (callable, optional): A closure that reevaluates the model
                        # and returns the loss.
            Description: This function follows the standard pytorch SGD implementation wqith the following exceptions.
                         1. It logs all previous gradients up to the given parameter value w.
                         2. For each new step, instead of using only the current gradient it uses the average value
                            of the past w gradients, if they exist; elsewise it uses all available.
                         3. It then performs  the update as x_t+1 = x_t - lr * SUM_(i:0-w-1){ grad_(x_i) } 
        """
        loss = None
        # This is used for the rolling window history param group index. There are multiple param groups
        # each one with its own gradient.
        paramIdx = 0
        if closure is not None:
            loss = closure()

        self.history.append([])
        storedElems = len(self.history)
        # If stored elementes in history exceed w, drop the oldest one, located at the start of list
        w = self.w
        if storedElems > w:
            self.history = self.history[1:]
            storedElems = w
        # print("Stored elems: " + str(storedElems))
        # print(self.entryIdx)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            # w = group['w']
            # a = group['a']
            # print("Len of group_params: {}".format(len(group['params'])))
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    print("MOMENTUM NOT ZERO!!!!")
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
                # Need to log the computed gradient at each step, so we can use it to smooth out the
                # SGD jumps in the following fashion x_t+1 = x_t - (lr / w) * SUM_0-w { grad(x_t) }
                self.history[-1].append(d_p)
                # print("--------------------------------\n")
                # print(paramIdx)
                # this is the factor that multiplies the learning rate, at each epoch or step. Used for
                # adaptive learning rate tuning.
                lrFactor = self.lrFactor
                # Steps for i=0, that is for the current gradient. Some schemes require different handling
                # expecially for the diffferentailly weighed scheme.
                if 'A-TSSGD' in self.name:
                    grad_sum =  lrFactor(t) * group['lr'] * d_p.clone()
                elif "ED-TSSGD" in self.name:
                    grad_sum = (self.learnDecay[-1]* lrFactor(t) * group['lr']) * d_p.clone()
                if "DW" in self.name:
                    grad_sum *= a
                else:
                    grad_sum *= 1/ storedElems 
                # print(grad_sum)
                # print("--------------------------------\n")
                for i in reversed(range(0, storedElems-1)):
                    # print("i is {} paramIdx: {}".format(i, paramIdx))
                    if len(self.history[i]) != 0:
                        # learn decay weights are increasing left->right. smallest for oldest grad is at idx 0 and largest
                        # for newest grad is at idx -1.
                        if 'A-TSSGD' in self.name:
                            factor2 = lrFactor(t) * group['lr'] * 1/ storedElems
                        elif "ED-TSSGD" in self.name:
                            # print(i,self.learnDecay[-storedElems +i-1])
                            # oldest grad has smallest weight
                            factor2 = self.learnDecay[-storedElems+i-1]* lrFactor(t) * group['lr'] * 1/ storedElems
                        if "DW" in self.name:
                            factor2 *= a

                        grad_sum.add(factor2, self.history[i][paramIdx])
                # add the grad sum to the original parameters.
                p.data.add_(-grad_sum)
                # increase the index to update the next set of params.
                paramIdx += 1

        return loss 
