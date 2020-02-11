#!flask/bin/python
# import argparse
from argparse import ArgumentParser
import os

HERE = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))
import sys
sys.path.append(HERE + '/../')

from configuration import load_config
from synthesiser import Synthesiser, CMULexSynthesiser, HausaSynthesiser

from flask import Flask, request, render_template, send_file
#from TTS.server.synthesizer import Synthesizer


# def create_argparser():
 
#     parser = argparse.ArgumentParser()
#     a.add_argument('-c', dest='config', required=True, type=str)
#     # a.add_argument('-controllable', action='store_true', default=False) 
#     return parser



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/tts', methods=['GET'])
def tts():
    text = request.args.get('text')
    print(" > Model input: {}".format(text))
    data = synthesizer.tts(text)
    return send_file(data, mimetype='audio/wav')


if __name__ == '__main__':
    #args = create_argparser().parse_args()

    # Setup synthesizer from CLI args if they're specified or no embedded model
    # is present.
    # if not config or not synthesizer or args.tts_checkpoint or args.tts_config:
    #     synthesizer = Synthesizer(args)

      
    # ============= Process command line ============

    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    # a.add_argument('-i', dest='textdir', required=True, type=str)    
    # a.add_argument('-o', dest='synthdir', required=True, type=str)
    a.add_argument('-controllable', action='store_true', default=False) 

    opts = a.parse_args()
    
    # ===============================================


    hp = load_config(opts.config)

    if hp.language=='en_cmulex':
        synthesizer = CMULexSynthesiser(hp, controllable=opts.controllable, t2m_epoch=1000, ssrn_epoch=1000) ## TODO
    elif hp.language=='hausa':
        synthesizer = HausaSynthesiser(hp, controllable=opts.controllable, t2m_epoch=1000, ssrn_epoch=1000) ## TODO
    else:
        synthesizer = Synthesiser(hp, controllable=opts.controllable)

    server_config = {'debug': True, 'port': 5002}

    app.run(debug=server_config['debug'], host='0.0.0.0', port=server_config['port'])
