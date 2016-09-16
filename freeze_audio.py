import dzr_audio.stft as stft
from dzr_audio.signals import Signal
import numpy as np
Nfft = 2048
of = 0.5

start = 2.15

#s = Signal(67014843, crop=[0,5])
s = Signal(116348656, crop=[0,20])


stft_param = { "window_size": Nfft, "overlap_factor": of, "window_type": "hanning"}



step_size = (1.-of)*Nfft/s.fs
start_frame = int(start / step_size)



spectrogram_in = stft.stft(s.data, **stft_param)

dphi = np.angle(spectrogram_in[:,:,start_frame-1]) - np.angle(spectrogram_in[:,:,start_frame-2])
total_dphi = np.angle(spectrogram_in[:,:,start_frame-1])
for k in xrange(spectrogram_in.shape[-1]-start_frame):
    total_dphi+=dphi%(2*np.pi)
    spectrogram_in[:,:,start_frame+k] =  np.abs(spectrogram_in[:,:,start_frame-1])*np.exp(1j*total_dphi)

freezed_waveform = stft.istft(spectrogram_in, original_signal_length=s.data.shape[0], **stft_param)
s_freezed = Signal(freezed_waveform, s.fs)

# s_freezed.play()



# real time style

import scipy.signal as ss
window = np.sqrt(ss.hanning(Nfft,sym=False).reshape(-1,1))


freeze_on = False
first_buffer = False

mix = 1.0
buffer_freezed_signal = np.zeros((2*Nfft, s.n_chan))
output_signal = np.zeros(s.data.shape)


buffer_len = 512

circ_buffer = np.zeros((2*Nfft, s.n_chan))

for k in range(int(s.length/buffer_len)-2):

    if k == start_frame:
        freeze_on = True
        first_buffer = True

    start = k * buffer_len
    end = start + Nfft
    input_buffer = s.data[start:end,:]

    fourier_transform  = np.fft.fft(input_buffer*window)
    previous_fourier_transform = fourier_transform
    if first_buffer:
        dphi = np.angle(fourier_transform) - np.angle(fourier_transform_previous)
        start_fourier_transform = np.abs(fourier_transform_previous)
        total_dphi = np.angle(fourier_transform)
        first_buffer = False


    if freeze_on:

        total_dphi+=dphi%(2*np.pi)

        buffer_freezed_signal = np.roll(buffer_freezed_signal,int(-Nfft*(1-of)),axis=0)
        buffer_freezed_signal[-Nfft*(1-of):] = 0
        buffer_freezed_signal[:Nfft] += np.real(np.fft.ifft(start_fourier_transform*np.exp(1j*total_dphi)))*window
        output_buffer = 0.0*input_buffer + mix * buffer_freezed_signal[:Nfft]

    else:
        output_buffer = input_buffer

    previous_input = input_buffer
    # fourier_transform_previous = fourier_transform
    output_signal[start:end]+= output_buffer

s_out = Signal(output_signal, fs = s.fs)