import numpy as np
from keras import backend as k

from label_utils import labels_to_text
from spell import Spell

class Decoder(object):
    def __init__(self, greedy=True, beam_width=200, top_paths=1, postprocessors=None):
        self.greedy = greedy
        self.beam_width = beam_width
        self.top_paths = top_paths
        self.postprocessors = postprocessors if postprocessors is not None else []


    def decode(self, y_pred, input_lengths):
        decoded = self.__keras_decode(y_pred, input_lengths)

        postprocessed = []
        for d in decoded:
            for f in self.postprocessors: d = f(d)
            postprocessed.append(d)

        return postprocessed

    #FIXME @staticmethod
    def __keras_decode(self, y_pred, input_lengths):
        decoded = k.ctc_decode(
            y_pred=y_pred, 
            input_length=input_lengths, 
            greedy=self.greedy, 
            beam_width=self.beam_width, 
            top_paths=self.top_paths
            )
        results = [path.numpy() for path in decoded[0]]
        return results[0]


def spellchecked_decoder(dict_path):
    spell = Spell(dict_path)
    return Decoder(postprocessors=[labels_to_text, spell.sentence])