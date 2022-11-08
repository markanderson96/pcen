import platform
import torchaudio
import unittest

if platform.system() == 'Windows':
    torchaudio.set_audio_backend('soundfile')

from pcen import PCEN


class PCENTest(unittest.TestCase):
    def setUp(self):
        self.audio, self.sample_rate = torchaudio.load(
            "tests/resources/test_voice.wav"
        )

    def pcen_test(self):
        x = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=512,
            n_mels=40,
            power=1.0,
        )(self.audio)
        pcen = PCEN(n_filters=40, trainable=False)(x)
        pcen = pcen.detach().numpy()


if __name__ == '__main__':
    breakpoint()
    unittest.main()
