from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from dwtls.dwt_layers import DWT, MultiStageLWT, WeightNormalizedMultiStageLWT

from asteroid.models.base_models import BaseModel

EPS = 1e-5

#########################################################
def crop(to_be_cropped, y):
    shape = to_be_cropped.shape
    target_shape = y.shape
    diff = shape[-1] - target_shape[-1]
    if diff != 0:
        crop_start = diff // 2
        crop_end = diff - crop_start
        if to_be_cropped.ndim == 3:
            to_be_cropped = to_be_cropped[:, :, crop_start:-crop_end]
        elif to_be_cropped.ndim == 4:
            to_be_cropped = to_be_cropped[:, :, :, crop_start:-crop_end]
        else:
            raise ValueError
    return to_be_cropped

def crop_and_concat(to_be_cropped, y):
    '''Crop and concat

    Args:
        to_be_cropped (torch.Tensor): To-be-corpped input (batch x ch x time)
        y (torch.Tensor): To-be-concatenated input (batch x ch x time)
    
    Return:
        torch.Tensor: Concatenation of two inputs with the time length of min(to_be_cropped, y)
    '''
    # crop
    cropped = crop(to_be_cropped, y)
    # concat
    return torch.cat((cropped, y), dim=1)

class _MRDLA_Base(BaseModel):
    """Base class for MRDLA

    References:
        [1] Tomohiko Nakamura, Shihori Kozuka, and Hiroshi Saruwatari, "Time-domain audio source separation with neural networks based on multiresolution analysis," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 1687--1701, Apr. 2021.
    """
    def __init__(self, sample_rate: float, in_channels: Optional[int] = 1):
        super().__init__(sample_rate, in_channels)

    def get_activation(self, ch=None, dim=-1):
        if self.activation == "LeakyReLU":
            Activation = nn.LeakyReLU(negative_slope=0.2)
        elif self.activation == "GELU":
            Activation = nn.GELU()
        else:
            raise ValueError(f'Unknown activation [{self.activation}]')
        return Activation

    def get_output_length(self, input_length):
        output_length = input_length
        pad_history = []
        for l in range(self.L):
            output_length -= self.f_enc-1 # Conv1d
            pad_history.append(output_length%2 == 1)
            if pad_history[-1]:
                output_length += 1 
            output_length //= 2 # DWT
        output_length -= self.f_enc-1 # Conv1d
        for l in range(self.L):
            output_length = output_length*2 # inverse DWT
            if pad_history.pop():
                output_length -= 1
            output_length -= self.f_dec-1 # Conv1d
        return output_length

    def _sequential_forward(self, input_mix):
        '''Predict source signals sequentially with an interval of input_length

        Args:
            input_mix (torch.Tensor): batch x ch x time

        Returns:
            estimates (torch.Tensor): batch x n_srcs x ch x time
        '''
        input_length = self.input_length
        output_length = self.output_length
        # pad zeros for shorter input than input_length
        if input_mix.shape[-1] < input_length:
            extra_pad = input_length - input_mix.shape[2]
            input_mix = F.pad(input_mix, (0, extra_pad))
        else:
            extra_pad = 0

        # allocate estimates
        source_time_frames = input_mix.shape[-1]
        estimates = torch.zeros((input_mix.shape[0], self.n_srcs, input_mix.shape[1], input_mix.shape[2])).type_as(input_mix) # batch x n_srcs x ch x time

        # pad zeros at beginning and end
        pad_time_frames = (input_length - output_length) // 2
        input_mix_padded = F.pad(input_mix, (pad_time_frames, pad_time_frames))

        # main loop
        for source_pos in range(0, source_time_frames, output_length):
            # for last slice
            if source_pos + output_length > source_time_frames:
                source_pos = source_time_frames - output_length

            # get mixture excerpt and estimates
            # batch x ch x time (sliced)
            mix_part = input_mix_padded[:, :, source_pos:source_pos+input_length]
            curr_estimates = self._forward_core(mix_part) # batch x n_srcs x ch x time

            # save estimates
            estimates[:,:,:,source_pos:source_pos + output_length] = curr_estimates.detach()
            del curr_estimates

        if extra_pad > 0:
            estimates = estimates[:,:,:,:-extra_pad]
        return estimates

    def _forward_core(self, input_mix):
        '''Core of the forward process of MRDLA

        This function includes standardization of the input mixture.
        
        Args:
            input_mix (torch.Tensor): Input mixture (batch x channel x time)

        Return:
            torch.Tensor: Source estimates (batch x sources x channel x time)
        '''
        mix = input_mix

        # normalize
        monaural_mix = mix.mean(dim=1, keepdim=True)
        monaural_mean = monaural_mix.mean(dim=2, keepdim=True)
        monaural_std = EPS+monaural_mix.std(dim=2, keepdim=True) # batch x 1 x 1
        mix = (mix - monaural_mean)/monaural_std

        h = mix
        h_list = []
        pad_history = []
        for l, m in enumerate(self.encoder):
            h = m(h)
            h_list.append(h)
            # if self.context:
            pad_history.append(h.shape[-1]%2 == 1)
            if pad_history[-1]:
                h = F.pad(h, pad=(0, 1), mode=self.padding_type)
            h = self.dwt.forward(h)
        h = self.intermediate_layers(h)
        for l, m in enumerate(self.decoder):
            h = self.dwt.inverse(h)
            # if self.context:
            if pad_history.pop():
                h = h[:, :, :-1]
            skipped_h = h_list.pop()
            h = m(crop_and_concat(skipped_h, h))
        estimates = self.out_layers(crop_and_concat(mix, h)).view(mix.shape[0], self.n_srcs, self.signal_ch, -1) # shape: B x N x signal_ch x T
        estimates = estimates*monaural_std.unsqueeze(2) + monaural_mean.unsqueeze(2)

        return estimates

    def forward(self, input_mix):
        if self.training:
            return self._forward_core(input_mix)
        else:
            return self._sequential_forward(input_mix)
    
    def sequential_predict_w_shifts(self, input_mix, shift_length: int):
        '''Predict source signals sequentially with an uniform interval

        Args:
            input_mix (torch.Tensor): batch x ch x time
            shift_length (int): Shift length
            input_length (int): Time length of input

        Returns:
            torch.Tensor: Source estimates (batch x n_srcs x ch x time)
        '''
        input_length = self.input_length
        output_length = self.output_length
        #
        if shift_length > output_length:
            raise ValueError(f'shift_length must be smaller than output_length [shift_length={shift_length}, output_length={output_length}]')
        # pad zeros for shorter input than input_length
        if input_mix.shape[-1] < input_length:
            extra_pad = input_length - input_mix.shape[2]
            input_mix = F.pad(input_mix, (0, extra_pad))
        else:
            extra_pad = 0

        # allocate estimates
        source_time_frames = input_mix.shape[-1]
        estimates = torch.zeros((input_mix.shape[0], self.n_srcs, input_mix.shape[1], input_mix.shape[2])).type_as(input_mix) # batch x n_srcs x ch x time
        counts = torch.zeros((input_mix.shape[0], self.n_srcs, input_mix.shape[1], input_mix.shape[2])).type_as(input_mix)

        # pad zeros at beginning and end
        pad_time_frames = (input_length - output_length) // 2
        input_mix_padded = F.pad(input_mix, (pad_time_frames, pad_time_frames))

        # main loop
        for source_pos in range(0, source_time_frames, shift_length):
            # for last slice
            if source_pos + output_length > source_time_frames:
                source_pos = source_time_frames - output_length

            # get mixture excerpt and estimates
            # batch x ch x time (sliced)
            mix_part = input_mix_padded[:, :, source_pos:source_pos+input_length]
            curr_estimates = self._forward_core(mix_part) # batch x n_srcs x ch x time

            # save estimates
            estimates[:,:,:,source_pos:source_pos + output_length] += curr_estimates.detach()
            counts[:,:,:,source_pos:source_pos + output_length] += torch.ones_like(curr_estimates)
            del curr_estimates

        # remove the extra padded elements
        assert (counts>0).all()
        estimates = estimates / counts
        if extra_pad > 0:
            estimates = estimates[:,:,:,:-extra_pad]
        return estimates


class MRDLA(_MRDLA_Base):
    """MultiResolution Deep Layered Analysis (MRDLA) using the discrete wavelet transform (DWT) layer with the Haar wavelet

    Attributes:
        signal_ch (int): The number of channels of input signal
        n_srcs (int): The number of sources
        L (int): The number of levels
        C_enc (int): The base number of channels of the encoder
        C_mid (int): The number of channels of the intermediate layer
        C_dec (int): The base number of channels of the decoder
        f_enc (int): Kernel size of the convolutional layers in the encoder
        f_dec (int): Kernel size of the convolutional layers in the decoder
        context (bool): If True, the convolutional layers have no paddings.
        padding_type (str): Padding type for the convolutional layers
        input_length (int): Input signal length (Time interval for sequential processing in the inference stage)
        output_length (int): Output signal length
        weight_initialization (str): Weight initialization method. If None, use the pytorch's default initialization.
        sample_rate (int): Sampling rate

    References:
        [1] Tomohiko Nakamura, Shihori Kozuka, and Hiroshi Saruwatari, "Time-domain audio source separation with neural networks based on multiresolution analysis," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 1687--1701, Apr. 2021.
    """
    def __init__(self, signal_ch=1, n_srcs=4, L=12, C_enc=36, C_mid=432, C_dec=36, f_enc=9, f_dec=9, wavelet="haar", context=True, padding_type="reflect", activation="LeakyReLU", input_length=147443, sample_rate=48000, weight_initialization=None):
        super().__init__(sample_rate=sample_rate)
        self.signal_ch = signal_ch
        self.n_srcs = n_srcs
        self.L = L
        self.C_enc = C_enc
        self.C_mid = C_mid
        self.C_dec = C_dec
        self.f_enc = f_enc
        self.f_dec = f_dec
        self.context = context
        self.padding_type = padding_type
        self.activation = activation
        self.input_length = input_length
        #
        if self.context:
            self.input_length = input_length
            self.output_length = self.get_output_length(input_length)
            print(f'Network in->output length: {self.input_length} -> {self.output_length}')
        else:
            self.input_length = input_length
            self.output_length = input_length
        #
        self.dwt = DWT(wavelet=wavelet)
        # Encoder
        self.encoder = nn.ModuleList()
        for l in range(L):
            in_ch = C_enc*l if l > 0 else signal_ch
            out_ch = (C_enc * (l + 1)) // 2
            layers = [
                nn.Conv1d(in_ch, out_ch, f_enc, stride=1, padding=0 if self.context else (f_enc-1)//2),
                self.get_activation(ch=out_ch)
            ]
            self.encoder.append(nn.Sequential(*layers))
        ####
        layers = [
            nn.Conv1d(out_ch*2, C_mid, f_enc, 1, 0 if self.context else (f_enc-1)//2),
            self.get_activation(ch=C_mid)
        ]
        self.intermediate_layers = nn.Sequential(*layers)
        ####
        self.decoder = nn.ModuleList()
        in_ch = C_mid
        for l in reversed(range(L)):
            # channel of residual feature map
            in_ch = (C_enc * (l + 1)) // 2
            in_ch += C_mid//2 if l == L-1 else C_dec*(l+2)//2
            out_ch = C_dec*(l+1)
            layers = [
                nn.Conv1d(in_ch, out_ch, f_dec, 1, 0 if self.context else (f_dec-1)//2),
                self.get_activation(ch=out_ch)
            ]
            self.decoder.append(nn.Sequential(*layers))
        self.out_layers = nn.Conv1d(signal_ch + C_dec, signal_ch*n_srcs, 1, 1, 0)

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        model_args = {
            "signal_ch": self.signal_ch,
            "n_srcs": self.n_srcs,
            "L": self.L,
            "C_enc": self.C_enc,
            "C_mid": self.C_mid,
            "C_dec": self.C_dec,
            "f_enc": self.f_enc,
            "f_dec": self.f_dec,
            "wavelet": self.dwt.wavelet,
            "context": self.context,
            "padding_type": self.padding_type,
            "activation": self.activation,
            "input_length": self.input_length,
            "sample_rate": self.sample_rate,
            "weight_initialization": self.weight_initialization,
        }
        return model_args

class MRDLA_WNTDWTL(_MRDLA_Base):
    """MultiResolution Deep Layered Analysis (MRDLA) using the trainable discrete wavelet transform (DWT) layer

    Attributes:
        signal_ch (int): The number of channels of input signal
        n_srcs (int): The number of sources
        L (int): The number of levels
        C_enc (int): The base number of channels of the encoder
        C_mid (int): The number of channels of the intermediate layer
        C_dec (int): The base number of channels of the decoder
        f_enc (int): Kernel size of the convolutional layers in the encoder
        f_dec (int): Kernel size of the convolutional layers in the decoder
        context (bool): If True, the convolutional layers have no paddings.
        padding_type (str): Padding type for the convolutional layers
        input_length (int): Input signal length (Time interval for sequential processing in the inference stage)
        output_length (int): Output signal length
        weight_initialization (str): Weight initialization method. If None, use the pytorch's default initialization.
        sample_rate (int): Sampling rate

    References:
        [1] Tomohiko Nakamura, Shihori Kozuka, and Hiroshi Saruwatari, "Time-domain audio source separation with neural networks based on multiresolution analysis," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 1687--1701, Apr. 2021.
    """
    def __init__(self, signal_ch=1, n_srcs=4, L=12, C_enc=24, C_mid=13*24, C_dec=24, f_enc=15, f_dec=5, 
        wavelet_params_list=[
             dict(predict_ksize=3, update_ksize=3, requires_grad={"predict": True, "update": True}, initial_values={"predict": [0,1,0], "update": [0,0.5,0]}),
        ],
        context=True, padding_type="reflect", activation="LeakyReLU", input_length=147443, weight_normalized=True, sample_rate=48000, weight_initialization=None, wavelet=None):
        super().__init__(sample_rate=sample_rate)
        self.signal_ch = signal_ch
        self.n_srcs = n_srcs
        self.L = L
        self.C_enc = C_enc
        self.C_mid = C_mid
        self.C_dec = C_dec
        self.f_enc = f_enc
        self.f_dec = f_dec
        self.context = context
        self.padding_type = padding_type
        self.activation = activation
        self.input_length = input_length
        self.weight_initialization = weight_initialization
        ####
        if self.context:
            self.input_length = input_length
            self.output_length = self.get_output_length(input_length)
            print(f'Network in->output length: {self.input_length} -> {self.output_length}')
        else:
            self.input_length = input_length
            self.output_length = input_length
        ### set DWT
        self.dwt = WeightNormalizedMultiStageLWT(wavelet_params_list) if weight_normalized else MultiStageLWT(wavelet_params_list)
        self.dwt.init_params()
        ### encoder
        self.encoder = nn.ModuleList()
        for l in range(L):
            in_ch = C_enc*l if l > 0 else signal_ch
            out_ch = (C_enc * (l + 1)) // 2
            layers = [
                nn.Conv1d(in_ch, out_ch, f_enc, stride=1, padding=0 if self.context else (f_enc-1)//2),
                self.get_activation(out_ch, dim=1)
            ]
            self.encoder.append(nn.Sequential(*layers))
        ####
        layers = [
            nn.Conv1d(out_ch*2, C_mid, f_enc, 1, 0 if self.context else (f_enc-1)//2),
            self.get_activation(out_ch, dim=1)
        ]
        self.intermediate_layers = nn.Sequential(*layers)
        ####
        self.decoder = nn.ModuleList()
        in_ch = C_mid
        for l in reversed(range(L)):
            # channel of residual feature map
            in_ch = (C_enc * (l + 1)) // 2
            in_ch += C_mid//2 if l == L-1 else C_dec*(l+2)//2
            out_ch = C_dec*(l+1)
            layers = [
                nn.Conv1d(in_ch, out_ch, f_dec, 1, 0 if self.context else (f_dec-1)//2),
                self.get_activation(out_ch, dim=1)
            ]
            self.decoder.append(nn.Sequential(*layers))
        self.out_layers = nn.Conv1d(signal_ch + C_dec, signal_ch*n_srcs, 1, 1, 0)

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        model_args = {
            "signal_ch": self.signal_ch,
            "n_srcs": self.n_srcs,
            "L": self.L,
            "C_enc": self.C_enc,
            "C_mid": self.C_mid,
            "C_dec": self.C_dec,
            "f_enc": self.f_enc,
            "f_dec": self.f_dec,
            "wavelet_params_list": self.dwt.params_list,
            "context": self.context,
            "padding_type": self.padding_type,
            "activation": self.activation,
            "input_length": self.input_length,
            "weight_normalized": isinstance(self.dwt, WeightNormalizedMultiStageLWT),
            "sample_rate": self.sample_rate,
            "weight_initialization": self.weight_initialization,
        }
        return model_args
