import argparse
from pathlib import Path

import tornado
import tornado.ioloop
import tornado.web
import tornado.websocket

from yukari_direct_server.realtime_wrapper import RealtimeWrapper


class YukarinHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, realtime_wrapper: RealtimeWrapper):
        self.realtime_wrapper = realtime_wrapper

    def check_origin(self, origin):
        return True

    def open(self):
        pass

    def on_message(self, binary: bytes):
        self.realtime_wrapper.addWaveBinary(binary=binary, dtype='float32')

        try:
            binary = self.realtime_wrapper.queue_output_binary.get_nowait()
            self.write_message(binary, binary=True)
        except:
            pass


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

    realtime_wrapper = RealtimeWrapper(
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
                realtime_wrapper=realtime_wrapper,
            )),
        ],
        debug=True,
    )
    app.listen(9999)

    print('ready...')
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
