import numpy as np
import scipy.signal as ss

max_buffer_len = 2048

def inplace_shift_buffer(buffer, shift):
    """
    shift an array along 1st axis by "shift" bin.
    """
    buffer[:] = np.roll(buffer, shift, axis=0)
    if shift<0:
        buffer[shift:] = 0
    else:
        buffer[:shift] = 0

class FreezeProcessor(object):

    def __init__(self, Nfft=2048, overlap_factor=0.5, n_channel=2):
        self.sliding_buffer = np.zeros((Nfft + max_buffer_len, n_channel))
        self.output_buffer = np.zeros((Nfft + max_buffer_len, n_channel))
        self.fourier_transform = np.zeros((Nfft, n_channel), dtype=np.complex128)

        self.overlap_factor = overlap_factor
        self.Nfft = Nfft
        self.hop_size = int(self.Nfft*(1 - self.overlap_factor))
        self.index_sliding = self.Nfft - self.hop_size
        self._is_on = False
        self.just_on = False

        self.window = np.sqrt(ss.hanning(Nfft,sym=False).reshape(-1,1))


    @property
    def is_on(self):
        return self._is_on

    @is_on.setter
    def is_on(self, value):
        if value and not self.is_on:
            self.just_on = True

        self._is_on = value

    def process_buffer(self, buffer):
        """
        Process one buffer of audio.
        Return a same sized buffer.
        """

        buffer_length = buffer.shape[0]

        # slide output
        inplace_shift_buffer(self.output_buffer, -buffer_length)

        self.sliding_buffer[self.index_sliding:self.index_sliding+buffer_length] = buffer
        self.index_sliding+= buffer_length

        out_buffer_offset = 0 # for input buffer greater than Nfft
        while self.index_sliding>=self.Nfft:
            self.previous_fourier_transform = self.fourier_transform
            self.fourier_transform = np.fft.rfft(self.sliding_buffer[:self.Nfft]*self.window, axis=0)

            if self.just_on:
                self.dphi = np.angle(self.fourier_transform) - np.angle(self.previous_fourier_transform)
                self.freeze_ft_magnitude = np.abs(self.fourier_transform)
                self.total_dphi = np.angle(self.fourier_transform)
                self.just_on = False

            if self.is_on:
                self.total_dphi+= self.dphi
                self.total_dphi%= (2*np.pi)
                self.output_buffer[out_buffer_offset:out_buffer_offset+self.Nfft]+= np.real(np.fft.irfft(self.freeze_ft_magnitude*np.exp(1j*self.total_dphi), axis=0))*self.window
                out_buffer_offset+= self.hop_size

            # slide input
            inplace_shift_buffer(self.sliding_buffer, -self.hop_size)
            self.index_sliding-= self.hop_size

        return self.output_buffer[:buffer_length]