# PCEN PyTorch Layer

A fully trainable Per-Channel Energy Normalisation layer for PyTorch.
This repo will be in sporadic development, as there are a number of issues with the implementation that could be improved for usage with PyTorch. From a signal processing perspective however the layer is complete, the issues are primarily on the optimisation end.

See the following papers for details:
 - [Trainable frontend for robust and far-field keyword spotting](10.1109/ICASSP.2017.7953242)
 - [Per-Channel Energy Normalization: Why and How](10.1109/LSP.2018.2878620)

## Usage

``` python
from pcen import PCEN

# Fully Learnable PCEN layer
pcen = PCEN(
    n_filters=40,
    s_coef=0.05,
    alpha=0.98,
    delta=2.,
    r_coef=2.,
    trainable=True,
    learn_s_coef=True,
    per_channel_s=True
)

...

x = torchaudio.transforms.MelSpectrogram(n_mels=40)(audio)
x_pcen = pcen(x)
```

`
