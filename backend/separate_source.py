import numpy as np
# import matplotlib.pyplot as plt
from IPython.display import Audio

import librosa
import soundfile as sf

y, sr = librosa.load(input_file, duration=120)


# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))

S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))

# The output of the filter shouldn't be greater than the input
# if we assume signals are additive.  Taking the pointwise minimum
# with the input spectrum forces this.
S_filter = np.minimum(S_full, S_filter)
margin_i, margin_v = 1, 15
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Once we have the masks, simply multiply them with the input spectrum
# to separate the components

S_foreground = mask_v * S_full
S_background = mask_i * S_full
y_foreground = librosa.istft(S_foreground * phase)
Audio(data=y_foreground[10*sr:15*sr], rate=sr)


sf.write(output_file, y_foreground, sr, 'PCM_24')