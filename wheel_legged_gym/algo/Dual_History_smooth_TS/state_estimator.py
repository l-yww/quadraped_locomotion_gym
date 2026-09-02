import torch.nn as nn
import torch
import numpy as np
from .utils import get_activation, check_cnnoutput
from torch.distributions import Normal
from torch.nn import functional
from torchsummary import summary

class estimator_TS(nn.Module):
    def __init__(
        self,
        num_obs,
        output_dim=3,
        activation="elu",
        estimator_hidden_dims=[512, 256, 256],
        device = "cuda"
    ):
        super().__init__()
        self.device = device
        self.num_obs = num_obs

        self.mlp_est = MLP_TS(
            num_obs,
            output_dim,
            activation,
            estimator_hidden_dims,
        )


    def forward(self, obs_history):
        output= self.mlp_est(obs_history)
        return output

    def loss_fn(self, estimator_latent, critic_obs, num_est_prob):
        prob_real = critic_obs[:, -num_est_prob:]
        loss = functional.mse_loss(estimator_latent, prob_real, reduction="none").mean(-1)
        return loss

class MLP_TS(nn.Module):
    def __init__(self, input_size, output_size, activation, hidden_dims):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        module = []
        module.append(nn.Linear(self.input_size, hidden_dims[0]))
        module.append(self.activation)
        for i in range(len(hidden_dims) - 1):
            module.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            module.append(self.activation)
        module.append(nn.Linear(hidden_dims[-1], self.output_size))
        self.encoder = nn.Sequential(*module)

    def forward(self, obs_history):
        RS_obs_history = obs_history.reshape(obs_history.shape[0],-1)
        return self.encoder(RS_obs_history)


class MlpModel(torch.nn.Module):
    """Multilayer Perceptron with last layer linear.

    Args:
        input_size (int): number of inputs
        hidden_sizes (list): can be empty list for none (linear model).
        output_size: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
    """

    def __init__(
            self,
            input_size,
            hidden_sizes,  # Can be empty list or None for none.
            output_size=None,  # if None, last layer has nonlinearity applied.
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        elif hidden_sizes is None:
            hidden_sizes = []
        if isinstance(nonlinearity, str):
            nonlinearity = getattr(torch.nn, nonlinearity)
        hidden_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in
            zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend([layer, nonlinearity()])
        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            sequence.append(torch.nn.Linear(last_size, output_size))
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = (hidden_sizes[-1] if output_size is None
            else output_size)

    def forward(self, input):
        """Compute the model on the input, assuming input shape [B,input_size]."""
        return self.model(input)

    @property
    def output_size(self):
        """Retuns the output size of the model."""
        return self._output_size

def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Returns output H, W after convolution/pooling on input H, W.
    """
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
    sh, sw = stride if isinstance(stride, tuple) else (stride,) * 2
    ph, pw = padding if isinstance(padding, tuple) else (padding,) * 2
    d = dilation
    h = (h + (2 * ph) - (d * (kh - 1)) - 1) // sh + 1
    w = (w + (2 * pw) - (d * (kw - 1)) - 1) // sw + 1
    return h, w


class Conv2dModel(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            head_sizes=None,  # Put an MLP head on top.
            normlayer= None, # If None, will not be used
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        if isinstance(normlayer, str):
            normlayer = getattr(torch.nn, normlayer)
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, oc, maxp_stride in zip(conv_layers, channels, maxp_strides):
            if normlayer is not None:
                sequence.extend([conv_layer, normlayer(oc), nonlinearity()])
            else:
                sequence.extend([conv_layer, nonlinearity()])
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        return self.conv(input)

    def conv_out_size(self, h, w, c=None):
        """Helper function ot return the output size for a given input shape,
        without actually performing a forward pass through the model."""
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        return h * w * c

    def conv_out_resolution(self, h, w):
        """Helper function that return the resolution (H, W) for a giben input resolution"""
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        return h, w

class Conv2dHeadModel(torch.nn.Module):
    """Model component composed of a ``Conv2dModel`` component followed by 
    a fully-connected ``MlpModel`` head.  Requires full input image shape to
    instantiate the MLP head.
    """

    def __init__(
            self,
            image_shape,
            channels,
            kernel_sizes,
            strides,
            hidden_sizes,
            output_size=None,  # if None: nonlinearity applied to output.
            paddings=None,
            nonlinearity=torch.nn.ReLU,
            use_maxpool=False,
            normlayer= None, # if None, will not be used
            ):
        super().__init__()
        if isinstance(nonlinearity, str): nonlinearity = getattr(torch.nn, nonlinearity)
        c, h, w = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            nonlinearity=nonlinearity,
            use_maxpool=use_maxpool,
            normlayer= None, # if None, will not be used
        )
        conv_out_size = self.conv.conv_out_size(h, w)
        if hidden_sizes or output_size:
            self.head = MlpModel(conv_out_size, hidden_sizes,
                output_size=output_size, nonlinearity=nonlinearity)
            if output_size is not None:
                self._output_size = output_size
            else:
                self._output_size = (hidden_sizes if
                    isinstance(hidden_sizes, int) else hidden_sizes[-1])
        else:
            self.head = lambda x: x
            self._output_size = conv_out_size

    def forward(self, input):
        """Compute the convolution and fully connected head on the input;
        assumes correct input shape: [B,C,H,W]."""
        return self.head(self.conv(input).view(input.shape[0], -1))

    @property
    def output_size(self):
        """Returns the final output size after MLP head."""
        return self._output_size