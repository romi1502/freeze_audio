import numpy as np
from freeze_processor import FreezeProcessor
import librosa

input_filename = "test_sample.wav"
output_filename = "test_sample_processed.wav"

input_signal, fs = librosa.load(input_filename, sr=None, mono=False)
input_signal = input_signal.T


buffer_len = 256
processor = FreezeProcessor(Nfft=2048, overlap_factor=0.5, n_channel=2)


# time at which effect is switched on
start_sample = 0.9*fs
# time at which effect is switched off
end_sample = 3.0*fs

#  start gain of dry signal
dry_gain = 1


output_signal = np.zeros(input_signal.shape)
for k in range(0, int(input_signal.shape[0]), buffer_len):

    # effect is switched on
    if k <= start_sample < k+buffer_len:
        processor.is_on = True

    # effect is switched off
    if k <= end_sample < k+buffer_len:
        processor.is_on = False

    if processor.is_on:
        dry_gain*=0.8
    else:
        dry_gain = 1 - (1-dry_gain)*0.8

    output_signal[k:k+buffer_len]+= processor.process_buffer(input_signal[k:k+buffer_len,:])
    output_signal[k:k+buffer_len]+= dry_gain*input_signal[k:k+buffer_len,:]

librosa.output.write_wav(output_filename, output_signal, fs, norm=True)