"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import torch
import utils
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
from meters import AverageMeter, TimeMeter
from trainer_OE import Trainer


class DynamicLossScaler:

    def __init__(self, init_scale=2.**15, scale_factor=2., scale_window=2000):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self._iter = 0
        self._last_overflow_iter = -1

    def update_scale(self, overflow):
        if overflow:
            self.loss_scale /= self.scale_factor
            self._last_overflow_iter = self._iter
        elif (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
        self._iter += 1

    @staticmethod
    def has_overflow(grad_norm):
        # detect inf and nan
        if grad_norm == float('inf') or grad_norm != grad_norm:
            return True
        return False


class FP16Trainer(Trainer):
    """Modified trainer for FP16.

    We maintain two copies of the model's parameters, both in FP16 and FP32.
    We do forward/backward with FP16 and compute the loss + optimize with FP32.
    """

    def __init__(self, args, model, criterion, lr, optimizer=None):
        super().__init__(args, model, criterion, optimizer)

        # convert model to FP16 (but keep criterion FP32)
        self.model.half()
        # dynamically scale loss to reduce overflow
        self.scaler = DynamicLossScaler(init_scale=2.**7, scale_window=2000)
        self.meters['loss_scale'] = AverageMeter()
        self.lr = lr
        if not hasattr(self, 'fp32_params'):
            self._build_optimizer()

    def _build_optimizer(self):
        # create FP32 copy of parameters and grads
        params = [p for p in self.model.parameters() if p.requires_grad]
        total_param_size = sum(p.data.numel() for p in params)
        self.fp32_params = params[0].new(0).float().new(total_param_size)
        offset = 0
        for p in params:
            numel = p.data.numel()
            self.fp32_params[offset:offset+numel].copy_(p.data.view(-1))
            offset += numel
        self.fp32_params = torch.nn.Parameter(self.fp32_params)
        self.fp32_params.grad = self.fp32_params.data.new(total_param_size)

        # create optimizer using the copied FP32 params
        self._optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, [self.fp32_params]), lr=self.lr)
        # self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        extra_state['loss_scale'] = self.scaler.loss_scale
        super().save_checkpoint(filename, extra_state)

    def load_checkpoint(self, filename):
        """Load all training state from a checkpoint file."""
        extra_state = super().load_checkpoint(filename)
        if extra_state is not None and 'loss_scale' in extra_state:
            self.scaler.loss_scale = extra_state['loss_scale']
        return extra_state

    def zero_grad(self):
        # zero both the FP16 and FP32 grads
        self.model.zero_grad()      # FP16
        self.optimizer.zero_grad()  # FP32

    def _backward(self, loss):
        self.meters['loss_scale'].reset()
        self.meters['loss_scale'].update(self.scaler.loss_scale)
        if loss is not None:
            # dynamically rescale loss to stay in FP16 range
            loss = loss * self.scaler.loss_scale
        return super()._backward(loss)

    def _all_reduce_and_rescale(self, grad_denom):
        # undo effect of dynamic loss scaling on gradients
        grad_denom *= self.scaler.loss_scale

        # single worker: copy grads directly to FP32
        self._get_flat_grads(out=self.fp32_params.grad.data)

        # rescale and clip grads
        self.fp32_params.grad.data.div_(grad_denom)
        grad_norm = utils.clip_grad_norm_(self.fp32_params.grad.data, self.args.clip_norm)

        # detect overflow and adjust loss scale
        overflow = DynamicLossScaler.has_overflow(grad_norm)
        self.scaler.update_scale(overflow)
        if overflow:
            if self.scaler.loss_scale <= self.args.min_loss_scale:
                raise Exception((
                                    'Minimum loss scale reached ({}). Your loss is probably exploding. '
                                    'Try lowering the learning rate, using gradient clipping or '
                                    'increasing the batch size.'
                                ).format(self.args.min_loss_scale))
            raise OverflowError('setting loss scale to: ' + str(self.scaler.loss_scale))

        return grad_norm

    def _opt(self):
        # take an optimization step using the FP32 params and grads
        super()._opt()

        # copy FP32 params back into FP16 model
        offset = 0
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            numel = p.data.numel()
            p.data.copy_(self.fp32_params.data[offset:offset+numel].view_as(p.data))
            offset += numel

    def get_loss_scale(self):
        return self.scaler.loss_scale
