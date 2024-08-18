import math

from torch.nn import Module, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, MaxPool2d, ModuleList, AvgPool2d


class ConvBlock(Module):
    """
    A convolutional block module for neural networks, supporting downsampling, upsampling,
    and optional batch normalization.

    Parameters
    ----------
    in_channels : int
        The number of input channels to the first convolutional layer.
    out_channels : int
        The number of output channels for each convolutional layer.
    kernel : int, optional
        The size of the convolutional kernel. Default is 3.
    chain_conv : int, optional
        The number of chained convolutional layers within the block. Default is 3.
    pool : str, optional
        The type of pooling operation to apply during downsampling. Can be 'MAX' for max pooling,
        'AVG' for average pooling, or 'none' for no pooling. Default is 'MAX'.
    use_batchnorm : bool, optional
        Whether to apply batch normalization after each convolutional layer. Default is True.
    mode : str, optional
        Determines the operation of the block. 'downsample' for reducing spatial dimensions using pooling,
        'upsample' for increasing spatial dimensions using transposed convolution, or 'none' for neither.
        Default is 'downsample'.
    *args, **kwargs :
        Additional arguments passed to the parent class `Module`.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, chain_conv: int = 3, pool: str = 'MAX',
                 use_batchnorm: bool = True,
                 mode: str = 'downsample', *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mode not in ['downsample', 'upsample', 'none']: raise ValueError(
            f'Unrecognized mode: {mode}. Must be either downsample, upsample or none')
        if pool not in ['MAX', 'AVG', 'none']: raise ValueError(
            f'Unrecognized pooling: {pool}. Must be either MAX, AVG or none')
        self.in_channels = in_channels
        self.out_channels = out_channels
        tmp_layers = []
        # If kernel is 1x1 padding is not needed since the activations will not be downsampled
        if kernel == 1:
            padding = 0
        else:
            padding = 1
        for i in range(chain_conv):
            # If it's the first convolutional layer use in_channels as input dim
            if i == 0:
                tmp_layers.append(Conv2d(in_channels, out_channels, kernel, padding=padding))
            # From the first onward, input is out_channels and output is out_channels
            else:
                tmp_layers.append(Conv2d(out_channels, out_channels, kernel, padding=padding))
            tmp_layers.append(ReLU())
            if use_batchnorm:
                tmp_layers.append(BatchNorm2d(out_channels))
            # If we reached the last layer, add downsampling after it if needed
            if i == chain_conv - 1:
                if mode != 'none':
                    if mode == 'downsample':
                        if pool == 'AVG':
                            tmp_layers.append(AvgPool2d(2, 2))
                        else:
                            tmp_layers.append(MaxPool2d(2, 2))
                    else:
                        tmp_layers.append(ConvTranspose2d(out_channels, out_channels, 2, stride=2))
        self.layers = ModuleList(tmp_layers)

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x


class Encoder(Module):
    """
    An encoder module that progressively downsamples the input tensor while increasing the channel
    dimension, ultimately producing an embedding of specified dimensions. The architecture is
    composed of multiple convolutional blocks.

    Parameters
    ----------
    in_channels : int
        The number of input channels to the encoder.
    emb_dim : int
        The dimensionality of the output embedding.
    chain_conv : int, optional
        The number of chained convolutional layers within each block. Default is 3.
    pool : str, optional
        The type of pooling operation to apply during downsampling. Can be 'MAX' for max pooling or
        'AVG' for average pooling. Default is 'MAX'.
    start_conv : int, optional
        The number of output channels of the first convolutional block. Default is 32.
    input_dim : int, optional
        The spatial dimensions (height and width) of the input tensor. Must be a multiple of 4. Default is 64.
    channel_cap : int, optional
        The maximum number of channels that any convolutional block can output. Default is 128.
    use_batchnorm : bool, optional
        Whether to apply batch normalization after each convolutional layer. Default is True.
    *args, **kwargs :
        Additional arguments passed to the parent class `Module`.
    """

    def __init__(self, in_channels: int, emb_dim: int, chain_conv: int = 3, pool: str = 'MAX', start_conv: int = 32,
                 input_dim: int = 64, channel_cap: int = 128,
                 use_batchnorm: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if input_dim % 4 != 0:
            raise ValueError('Input dimension must be a multiple of 4')
        necessary_blocks = int(math.log2(input_dim / 16)) + 1
        tmp_layers = []
        for i in range(necessary_blocks):
            if i == 0:
                tmp_layers.append(
                    ConvBlock(in_channels, start_conv, chain_conv=chain_conv, pool=pool, use_batchnorm=use_batchnorm,
                              mode='downsample'))
            else:
                out = 2 * tmp_layers[-1].out_channels if 2 * tmp_layers[-1].out_channels <= channel_cap else channel_cap
                if i == necessary_blocks - 1:
                    tmp_layers.append(
                        ConvBlock(tmp_layers[-1].out_channels, emb_dim // (16 * 16), kernel=1, chain_conv=chain_conv,
                                  use_batchnorm=use_batchnorm, mode='none'))
                else:
                    tmp_layers.append(ConvBlock(tmp_layers[-1].out_channels, out, chain_conv=chain_conv, pool=pool,
                                                use_batchnorm=use_batchnorm, mode='downsample'))
        self.blocks = ModuleList(tmp_layers)
        self.out_channels = self.blocks[-1].out_channels

    def forward(self, x):
        for module in self.blocks:
            x = module(x)
        return x


class Decoder(Module):
    """
    A decoder module that progressively upsamples the input tensor while reducing the channel
    dimension, ultimately restoring the spatial dimensions to the specified output size. The architecture
    is composed of multiple convolutional blocks.

    Parameters
    ----------
    in_channels : int
        The number of input channels to the decoder.
    chain_conv : int, optional
        The number of chained convolutional layers within each block. Default is 3.
    channel_cap : int, optional
        The minimum number of channels that any convolutional block can output. Default is 64.
    out_dim : int, optional
        The spatial dimensions (height and width) of the output tensor. Must be a multiple of 4. Default is 64.
    use_batchnorm : bool, optional
        Whether to apply batch normalization after each convolutional layer. Default is True.
    *args, **kwargs :
        Additional arguments passed to the parent class `Module`.
    """

    def __init__(self, in_channels, chain_conv=3, channel_cap=64, out_dim=64, use_batchnorm=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if out_dim % 4 != 0: raise ValueError('Input dimension must be a multiple of 4')
        necessary_blocks = int(math.log2(out_dim / 16)) + 1
        tmp_layers = []
        current = in_channels
        for i in range(necessary_blocks):
            out = max(channel_cap, int(current / 2))
            if i == necessary_blocks - 1:
                tmp_layers.append(
                    ConvBlock(current, out, chain_conv=chain_conv, use_batchnorm=use_batchnorm, mode='none'))
            elif i == 0:
                tmp_layers.append(ConvBlock(current, out, kernel=1, chain_conv=chain_conv, use_batchnorm=use_batchnorm,
                                            mode='upsample'))
            else:
                tmp_layers.append(
                    ConvBlock(current, out, chain_conv=chain_conv, use_batchnorm=use_batchnorm, mode='upsample'))
            current = out
        self.blocks = ModuleList(tmp_layers)
        self.out_channels = current

    def forward(self, x):
        for module in self.blocks:
            x = module(x)
        return x


class Autoencoder(Module):
    """
    An autoencoder module composed of an encoder and a decoder, designed to encode input data into a
    lower-dimensional embedding and then reconstruct it back to the original or specified output dimensions.

    Parameters
    ----------
    in_channels : int
        The number of input channels to the autoencoder.
    emb_dim : int
        The dimensionality of the encoded embedding.
    chain_conv : int, optional
        The number of chained convolutional layers within each block of the encoder and decoder. Default is 3.
    start_conv : int, optional
        The number of output channels of the first convolutional block in the encoder. Default is 32.
    out_channels : int, optional
        The number of output channels of the final output layer. Default is 3.
    input_dim : int, optional
        The spatial dimensions (height and width) of the input tensor. Must be a multiple of 4. Default is 64.
    out_dim : int, optional
        The spatial dimensions (height and width) of the output tensor. Must be a multiple of 4. Default is 64.
    channel_cap : int, optional
        The maximum number of channels that any convolutional block in the encoder can output. Default is 128.
    use_batchnorm : bool, optional
        Whether to apply batch normalization after each convolutional layer in the encoder and decoder. Default is True.
    pool : str, optional
        The type of pooling operation to apply during downsampling in the encoder. Can be 'MAX' for max pooling
        or 'AVG' for average pooling. Default is 'MAX'.
    dropout : float, optional
        The dropout rate to apply after each block in the encoder. Default is 0.
    *args, **kwargs :
        Additional arguments passed to the parent class `Module`.
    """

    def __init__(self, in_channels: int, emb_dim: int, chain_conv: int = 3, start_conv: int = 32, out_channels: int = 3,
                 input_dim: int = 64, out_dim: int = 64,
                 channel_cap: int = 128, use_batchnorm: bool = True, pool: str = 'MAX', dropout: float = .0, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(in_channels, emb_dim, chain_conv=chain_conv, pool=pool, start_conv=start_conv,
                               input_dim=input_dim, channel_cap=channel_cap, use_batchnorm=use_batchnorm)
        self.decoder = Decoder(self.encoder.out_channels, chain_conv=chain_conv, out_dim=out_dim,
                               use_batchnorm=use_batchnorm)
        self.output_head = Conv2d(self.decoder.out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_head(x)
        return x

    def encode(self, x):
        return self.encoder(x)
