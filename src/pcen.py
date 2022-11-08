import torch
import torch.nn as nn
from torch import Tensor

class PCEN(nn.Module):
    """Per-Channel Energy Normalization Class

    This applies a fixed or learnable normalization by an exponential moving
    average smoother, and a compression.
    See https://arxiv.org/abs/1607.05666 for more details.
    """
    def __init__(
        self,
        n_filters: int,
        s_coef: float = 0.05,
        alpha: float = 0.98,
        delta: float = 2,
        r_coef: float = 2,
        eps: float = 1E-6,
        trainable: bool = True,
        learn_s_coef: bool = True,
        per_channel_s: bool = True,
        name: str = "PCEN"
    ):
        """PCEN Constructor

        Args:
          alpha: float, exponent of EMA smoother
          smooth_coef: float, smoothing coefficient of EMA
          delta: float, bias added before compression
          root: float, one over exponent applied for compression (r in the paper)
          floor: float, offset added to EMA smoother
          trainable: bool, False means fixed_pcen, True is trainable_pcen
          learn_smooth_coef: bool, True means we also learn the smoothing
            coefficient
          per_channel_smooth_coef: bool, True means each channel has its own smooth
            coefficient
          name: str, name of the layer
        """
        super(PCEN, self).__init__()
        self._alpha_init = alpha
        self._delta_init = delta
        self._r_init = r_coef
        self._s_init = s_coef
        self._eps = eps
        self._trainable = trainable
        self._learn_s_coef = learn_s_coef
        self._per_channel_s = per_channel_s

        self._build(n_filters)

    def _build(self, n_filters):
        alpha_tensor = torch.zeros((n_filters)).type(torch.float32)
        alpha_tensor[:] = self._alpha_init
        self.alpha = nn.Parameter(alpha_tensor, requires_grad=self._trainable)

        delta_tensor = torch.zeros((n_filters)).type(torch.float32)
        delta_tensor[:] = self._delta_init
        self.delta = nn.Parameter(delta_tensor, requires_grad=self._trainable)

        r_tensor = torch.zeros((n_filters)).type(torch.float32)
        r_tensor[:] = self._r_init
        self.root = nn.Parameter(r_tensor, requires_grad=self._trainable)

        if self._learn_s_coef:
            self.ema = ExponentialMovingAverage(
                coeff_init=self._s_init,
                per_channel=self._per_channel_s,
                trainable=True
            )
            self.ema.build(n_filters)
        else:
            self.ema = ExponentialMovingAverage(
                coeff_init=self._s_init,
                per_channel=False,
                trainable=False
            )
            self.ema.build(n_filters)

    def forward(self, x):
        alpha = torch.minimum(self.alpha, torch.ones_like(self.alpha))
        root = torch.maximum(self.root, torch.ones_like(self.root))
        x = torch.squeeze(x, 1)
        ema_smoother = self.ema(x, x[:,:,0])
        one_over_root = 1. / root
        output = (x.permute(0,2,1) / (self._eps + ema_smoother) ** alpha + self.delta) \
                  ** one_over_root - self.delta ** one_over_root
        output = torch.permute(output, (0,2,1))
        return output


class ExponentialMovingAverage(nn.Module):
    """Computes of an exponential moving average of an sequential input."""
    def __init__(
        self,
        coeff_init: float,
        per_channel: bool = False,
        trainable: bool = False
    ):
        """Initializes the ExponentialMovingAverage.

        Args:
          coeff_init: the value of the initial coeff.
          per_channel: whether the smoothing should be different per channel.
          trainable: whether the smoothing should be trained or not.
        """
        super(ExponentialMovingAverage, self).__init__()
        self._coeff_init = coeff_init
        self._per_channel = per_channel
        self._trainable = trainable

    def build(self, num_channels):
        if self._per_channel:
            ema_tensor = torch.zeros((num_channels)).type(torch.float32)
        else:
            ema_tensor = torch.zeros((1))
        ema_tensor[:] = self._coeff_init
        self._weights = nn.Parameter(ema_tensor, requires_grad=self._trainable)

    def forward(self, x, initial_state):
        w = torch.clamp(self._weights, 0.0, 0.2)
        func = lambda a, y: w * y + (1.0 - w) * a

        def scan(foo, x):
            res = []
            res.append(x[0].unsqueeze(0))
            a_ = x[0].clone()

            for i in range(1, len(x)):
                res.append(foo(a_, x[i]).unsqueeze(0))
                a_ = foo(a_, x[i])

            return torch.cat(res)

        res = scan(func, x.permute(2,0,1))
        return res.permute(1,0,2)
