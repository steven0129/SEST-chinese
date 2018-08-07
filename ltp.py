from pyltp import Segmentor
from pyltp import SentenceSplitter

LTP_DATA_DIR = 'ltp_data_v3.4.0'

class Sentence():
    def split(self, myStr):
        return list(SentenceSplitter.split(myStr))

class Word():
    def __init__(self, dictDir):
        self.segmentor = Segmentor()
        self.segmentor.load_with_lexicon(f'{LTP_DATA_DIR}/cws.model', f'{dictDir}/dict.txt')
    
    def split(self, myStr):
        return list(self.segmentor.segment(myStr))