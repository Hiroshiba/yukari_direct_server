from pathlib import Path

import numpy
from become_yukarin import SuperResolution
from become_yukarin.config.sr_config import create_from_json as create_sr_config
from become_yukarin.data_struct import AcousticFeature as BYAcousticFeature
from yukarin import AcousticConverter
from yukarin.config import create_from_json as create_config
from yukarin.f0_converter import F0Converter


class YukarinWrapper(object):
    def __init__(
            self,
            voice_changer_model: Path,
            voice_changer_config: Path,
            super_resolution_model: Path,
            super_resolution_config: Path,
            input_statistics: Path,
            target_statistics: Path,
            gpu: int,
    ):
        # f0 converter
        if input_statistics is not None:
            f0_converter = F0Converter(input_statistics=input_statistics, target_statistics=target_statistics)
        else:
            f0_converter = None

        # acoustic converter
        config = create_config(voice_changer_config)
        acoustic_converter = AcousticConverter(
            config,
            voice_changer_model,
            gpu=gpu,
            f0_converter=f0_converter,
        )

        # super resolution
        sr_config = create_sr_config(super_resolution_config)
        super_resolution = SuperResolution(sr_config, super_resolution_model, gpu=gpu)

        self.acoustic_converter = acoustic_converter
        self.super_resolution = super_resolution

    def convert(self, p_in: Path):
        return self._convert(
            p_in,
            acoustic_converter=self.acoustic_converter,
            super_resolution=self.super_resolution,
        )

    @staticmethod
    def _convert(p_in: Path, acoustic_converter: AcousticConverter, super_resolution: SuperResolution):
        w_in = acoustic_converter.load_wave(p_in)
        f_in = acoustic_converter.extract_acoustic_feature(w_in)
        f_in_effective, effective = acoustic_converter.separate_effective(wave=w_in, feature=f_in)
        f_low = acoustic_converter.convert(f_in_effective)
        f_low = acoustic_converter.combine_silent(effective=effective, feature=f_low)
        f_low = acoustic_converter.decode_spectrogram(f_low)
        s_high = super_resolution.convert(f_low.sp.astype(numpy.float32))

        f_low_sr = BYAcousticFeature(
            f0=f_low.f0,
            spectrogram=f_low.sp,
            aperiodicity=f_low.ap,
            mfcc=f_low.mc,
            voiced=f_low.voiced,
        )

        rate = acoustic_converter.out_sampling_rate
        wave = super_resolution(s_high, acoustic_feature=f_low_sr, sampling_rate=rate)
        return wave
