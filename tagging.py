"""
Tagging example.
"""
import argparse
import os

from bilstm_crf.tagger import Tagger
from bilstm_crf.models import BiLSTMCRF
from bilstm_crf.preprocessing import IndexTransformer
from bilstm_crf.utils import load_data

def tag_message(text_msg):
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/medical')
    SAVE_DIR = os.path.join(os.path.dirname(__file__), 'Models')

    parser = argparse.ArgumentParser(description='Tagging.')
    parser.add_argument('--weights_file', default=os.path.join(SAVE_DIR, 'model_weights.h5'))
    parser.add_argument('--params_file', default=os.path.join(SAVE_DIR, 'params.json'))
    parser.add_argument('--preprocessor_file', default=os.path.join(SAVE_DIR, 'preprocessor.json'))
    args = parser.parse_args()
    
    print('Loading model...')
    model = BiLSTMCRF.load(args.weights_file, args.params_file)
    p = IndexTransformer.load(args.preprocessor_file)
    tagger = Tagger(model, preprocessor=p)

    res = tagger.analyze(text_msg)
    print(res)
    return res

if __name__ == '__main__':
    tag_message("I had fever today so I took a Crocin.")