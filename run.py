import argparse
import os
import tempfile
from io import BytesIO
from pathlib import Path

import librosa
import numpy
import tornado
import tornado.ioloop
import tornado.web
from tornado_cors import CorsMixin

from yukari_direct_server.yukarin_wrapper import YukarinWrapper


class YukarinHandler(CorsMixin, tornado.web.RequestHandler):
    CORS_ORIGIN = '*'
    CORS_HEADERS = 'Content-Type'
    CORS_METHODS = 'POST, OPTIONS'

    def initialize(self, yukarin_wrapper: YukarinWrapper):
        self.yukarin_wrapper = yukarin_wrapper

    def post(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(self.request.files['data'][0]['body'])

        wave_wrapper = self.yukarin_wrapper.convert(Path(f.name))
        sio = BytesIO()
        librosa.output.write_wav(sio, wave_wrapper.wave.astype(numpy.float32), sr=wave_wrapper.sampling_rate)
        os.remove(f.name)

        self.set_header('Content-type', 'audio/wave')
        self.write(sio.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voice_changer_model', '-vcm', type=Path)
    parser.add_argument('--voice_changer_config', '-vcc', type=Path)
    parser.add_argument('--super_resolution_model', '-srm', type=Path)
    parser.add_argument('--super_resolution_config', '-src', type=Path)
    parser.add_argument('--input_statistics', '-is', type=Path)
    parser.add_argument('--target_statistics', '-ts', type=Path)
    parser.add_argument('--gpu', type=int)
    arguments = parser.parse_args()

    yukarin_wrapper = YukarinWrapper(
        voice_changer_model=arguments.voice_changer_model,
        voice_changer_config=arguments.voice_changer_config,
        super_resolution_model=arguments.super_resolution_model,
        super_resolution_config=arguments.super_resolution_config,
        input_statistics=arguments.input_statistics,
        target_statistics=arguments.target_statistics,
        gpu=arguments.gpu,
    )

    app = tornado.web.Application(
        [
            (r"/yukari", YukarinHandler, dict(
                yukarin_wrapper=yukarin_wrapper,
            )),
        ],
        debug=True,
    )
    app.listen(9999)

    print('ready...')
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
