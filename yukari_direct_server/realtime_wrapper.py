from multiprocessing import Process
from multiprocessing import Queue
from pathlib import Path
from typing import Any
from typing import NamedTuple

import librosa
import numpy
from become_yukarin import SuperResolution
from become_yukarin.config.sr_config import SRConfig
from become_yukarin.config.sr_config import create_from_json as create_sr_config
from realtime_voice_conversion.voice_changer_stream import VoiceChangerStream
from realtime_voice_conversion.voice_changer_stream import VoiceChangerStreamWrapper
from realtime_voice_conversion.yukarin_wrapper.vocoder import RealtimeVocoder
from realtime_voice_conversion.yukarin_wrapper.vocoder import Vocoder
from realtime_voice_conversion.yukarin_wrapper.voice_changer import AcousticFeatureWrapper
from realtime_voice_conversion.yukarin_wrapper.voice_changer import VoiceChanger
from yukarin import AcousticConverter
from yukarin.acoustic_feature import AcousticFeature
from yukarin.config import Config
from yukarin.config import create_from_json as create_config
from yukarin.f0_converter import F0Converter
from yukarin.wave import Wave


class AudioConfig(NamedTuple):
    in_rate: int
    out_rate: int
    frame_period: float
    in_audio_chunk: int
    out_audio_chunk: int
    vocoder_buffer_size: int
    in_norm: float
    out_norm: float
    silent_threshold: float


class Item(object):
    def __init__(
            self,
            original: numpy.ndarray,
            item: Any,
            index: int,
    ):
        self.original = original
        self.item = item
        self.index = index


def input_worker(
        audio_config: AudioConfig,
        queue_input: Queue,
        queue_output: Queue,
):
    in_audio_chunk = audio_config.in_audio_chunk

    index_input = 0
    wave_fragment = numpy.empty(0, dtype=numpy.float32)
    while True:
        wave: numpy.ndarray = queue_input.get()
        wave = librosa.resample(wave, 44100, 16000)

        wave_fragment = numpy.concatenate([wave_fragment, wave])
        while len(wave_fragment) >= in_audio_chunk:
            wave = wave_fragment[:in_audio_chunk]
            wave_fragment = wave_fragment[in_audio_chunk:]

            item = Item(
                original=wave,
                item=wave,
                index=index_input,
            )
            index_input += 1
            queue_output.put(item)


def encode_worker(
        config: Config,
        wrapper: VoiceChangerStreamWrapper,
        audio_config: AudioConfig,
        queue_input: Queue,
        queue_output: Queue,
):
    wrapper.voice_changer_stream.vocoder = Vocoder(
        acoustic_param=config.dataset.acoustic_param,
        out_sampling_rate=audio_config.out_rate,
    )

    start_time = 0
    time_length = audio_config.in_audio_chunk / audio_config.in_rate

    # padding 1s
    prev_original = numpy.zeros(round(time_length * audio_config.in_rate), dtype=numpy.float32)
    w = Wave(wave=prev_original, sampling_rate=audio_config.in_rate)
    wrapper.voice_changer_stream.add_wave(start_time=start_time, wave=w)
    start_time += time_length

    while True:
        item: Item = queue_input.get()
        item.original, prev_original = prev_original, item.original
        wave = item.item

        w = Wave(wave=wave, sampling_rate=audio_config.in_rate)
        wrapper.voice_changer_stream.add_wave(start_time=start_time, wave=w)
        start_time += time_length

        feature_wrapper = wrapper.pre_convert_next(time_length=time_length)
        item.item = feature_wrapper
        queue_output.put(item)


def convert_worker(
        config: Config,
        voice_changer_model: Path,
        f0_converter: F0Converter,
        sr_config: SRConfig,
        super_resolution_model: Path,
        gpu: int,
        wrapper: VoiceChangerStreamWrapper,
        audio_config: AudioConfig,
        queue_input: Queue,
        queue_output: Queue,
):
    acoustic_converter = AcousticConverter(
        config,
        voice_changer_model,
        gpu=gpu,
        f0_converter=f0_converter,
    )
    super_resolution = SuperResolution(sr_config, super_resolution_model, gpu=gpu)
    wrapper.voice_changer_stream.voice_changer = VoiceChanger(
        super_resolution=super_resolution,
        acoustic_converter=acoustic_converter,
        threshold=80,
    )

    start_time = 0
    time_length = audio_config.in_audio_chunk / audio_config.in_rate
    while True:
        item: Item = queue_input.get()
        in_feature: AcousticFeatureWrapper = item.item
        wrapper.voice_changer_stream.add_in_feature(
            start_time=start_time,
            feature_wrapper=in_feature,
            frame_period=audio_config.frame_period,
        )
        start_time += time_length

        out_feature = wrapper.convert_next(time_length=time_length)
        item.item = out_feature
        queue_output.put(item)


def decode_worker(
        config: Config,
        wrapper: VoiceChangerStreamWrapper,
        audio_config: AudioConfig,
        queue_input: Queue,
        queue_output: Queue,
):
    wrapper.voice_changer_stream.vocoder = RealtimeVocoder(
        acoustic_param=config.dataset.acoustic_param,
        out_sampling_rate=audio_config.out_rate,
        buffer_size=audio_config.vocoder_buffer_size,
        number_of_pointers=16,
    )

    start_time = 0
    time_length = audio_config.out_audio_chunk / audio_config.out_rate
    while True:
        item: Item = queue_input.get()
        feature: AcousticFeature = item.item
        wrapper.voice_changer_stream.add_out_feature(
            start_time=start_time,
            feature=feature,
            frame_period=audio_config.frame_period,
        )
        start_time += time_length

        wave = wrapper.post_convert_next(time_length=time_length).wave
        queue_output.put(wave)


def output_worker(
        audio_config: AudioConfig,
        queue_input: Queue,
        queue_output: Queue,
):
    out_audio_chunk = audio_config.out_audio_chunk

    wave_fragment = numpy.empty(0)
    while True:
        wave: numpy.ndarray = queue_input.get()

        wave_fragment = numpy.concatenate([wave_fragment, wave])
        while len(wave_fragment) >= out_audio_chunk:
            wave = wave_fragment[:out_audio_chunk]
            wave_fragment = wave_fragment[out_audio_chunk:]

            power = librosa.core.power_to_db(numpy.abs(librosa.stft(wave)) ** 2).mean()
            if power < audio_config.silent_threshold:
                continue  # pass

            wave *= audio_config.out_norm
            b = wave.astype(numpy.float32).tobytes()
            queue_output.put(b)


class RealtimeWrapper(object):
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

        # super resolution
        sr_config = create_sr_config(super_resolution_config)

        self.queue_input_array = queue_input_array = Queue()
        queue_input_wave = Queue()
        queue_input_feature = Queue()
        queue_output_feature = Queue()
        queue_output_wave = Queue()
        self.queue_output_binary = queue_output_binary = Queue()

        self.audio_config = audio_config = AudioConfig(
            in_rate=config.dataset.acoustic_param.sampling_rate,
            out_rate=24000,
            frame_period=config.dataset.acoustic_param.frame_period,
            in_audio_chunk=config.dataset.acoustic_param.sampling_rate,
            out_audio_chunk=24000,
            vocoder_buffer_size=config.dataset.acoustic_param.sampling_rate // 16,
            in_norm=1,
            out_norm=1,
            silent_threshold=-80.0,
        )

        # stream
        voice_changer_stream = VoiceChangerStream(
            in_sampling_rate=audio_config.in_rate,
            frame_period=config.dataset.acoustic_param.frame_period,
            order=config.dataset.acoustic_param.order,
            in_dtype=numpy.float32,
        )

        wrapper = VoiceChangerStreamWrapper(
            voice_changer_stream=voice_changer_stream,
            extra_time_pre=0.2,
            extra_time=0.5,
        )

        # processes
        process_input = Process(target=input_worker, kwargs=dict(
            audio_config=audio_config,
            queue_input=queue_input_array,
            queue_output=queue_input_wave,
        ))
        process_input.start()

        process_encoder = Process(target=encode_worker, kwargs=dict(
            config=config,
            wrapper=wrapper,
            audio_config=audio_config,
            queue_input=queue_input_wave,
            queue_output=queue_input_feature,
        ))
        process_encoder.start()

        process_converter = Process(target=convert_worker, kwargs=dict(
            config=config,
            voice_changer_model=voice_changer_model,
            f0_converter=f0_converter,
            sr_config=sr_config,
            super_resolution_model=super_resolution_model,
            gpu=gpu,
            wrapper=wrapper,
            audio_config=audio_config,
            queue_input=queue_input_feature,
            queue_output=queue_output_feature,
        ))
        process_converter.start()

        process_decoder = Process(target=decode_worker, kwargs=dict(
            config=config,
            wrapper=wrapper,
            audio_config=audio_config,
            queue_input=queue_output_feature,
            queue_output=queue_output_wave,
        ))
        process_decoder.start()

        process_output = Process(target=output_worker, kwargs=dict(
            audio_config=audio_config,
            queue_input=queue_output_wave,
            queue_output=queue_output_binary,
        ))
        process_output.start()

    def addWaveBinary(self, binary: bytes, dtype=numpy.float32):
        wave = numpy.frombuffer(binary, dtype=dtype)
        self.queue_input_array.put(wave)
