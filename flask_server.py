import argparse
from hparams import hparams, hparams_debug_string
import os
from synthesizer import Synthesizer
from flask import Flask, make_response

synthesizer = Synthesizer()
app = Flask(__name__)

@app.route('/')
def index():
    return 'TTS'

@app.route('/synthesize')
def synthesize():
    synthesizer.load('tmp/model.ckpt')
    response = make_response(synthesizer.synthesize('Miễn phí giao hàng toàn quốc!'))
    response.headers['Content-Type'] = 'audio/wav'
    return response


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)