from pyltp import Segmentor
from pyltp import SentenceSplitter
from pyltp import Parser
from pyltp import Postagger

LTP_DATA_DIR = 'ltp_data_v3.4.0'

class Sentence():
    def __init__(self):
        self.parser = Parser()
        self.postagger = Postagger()
        self.parser.load(f'{LTP_DATA_DIR}/parser.model')
        self.postagger.load(f'{LTP_DATA_DIR}/pos.model')

    def split(self, myStr):
        return list(SentenceSplitter.split(myStr))

    def parse(self, words):
        postags = self.postagger.postag(words)
        arcs = self.parser.parse(words, postags)
        return list(map(lambda arc: (arc.head, arc.relation), arcs))

class Word():
    def __init__(self, dictDir):
        self.segmentor = Segmentor()
        self.segmentor.load_with_lexicon(f'{LTP_DATA_DIR}/cws.model', f'{dictDir}/dict.txt')
    
    def split(self, myStr):
        return list(self.segmentor.segment(myStr))