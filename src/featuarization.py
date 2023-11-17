import pandas as pd
import re
import spacy
from tqdm import tqdm
import stanza
from unidecode import unidecode
from collections import Counter
from utils import save_json, load_yaml_file
import numpy as np
import os, sys
tqdm.pandas()

class CleanCorpus:
    def __init__(self, INPUT_PATH):
        self.nlp = spacy.load('en_core_web_trf')
        self.data = pd.read_csv(INPUT_PATH)
        self.column = 'input'
    
    @staticmethod    
    def lemma_doc(text: str, nlp) -> str:
        doc = nlp(text)
        return(" ".join([token.lemma_ for token in doc]))
    
    @staticmethod
    def clean_doc(text: str) -> str:
        text = text.lower()
        text = re.sub('@[^\s]+', '', text)
        text = re.sub('<[^<]+?>', '', text)
        text = ''.join(c for c in text if c.isalpha() or c.isspace())
        url_pattern = re.compile(r'https?://[^\s]+|www\.[^\s]+')
        text = re.sub(url_pattern, '', text)
        text = ' '.join(text.split())
        
        return text

    def clean_corpus(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.column] = df[self.column].progress_apply(self.clean_doc)
        return(df)

    def lemma_corpus(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.column] = df[self.column].progress_apply(lambda text: self.lemma_doc(text, self.nlp))
        return(df)
    
    def class2number(self, df: pd.DataFrame) -> pd.DataFrame:
        replace_map = {'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0}
        df['predictions'] = df['predictions'].replace(replace_map)
        return(df)
    
    def pipeline(self):
        self.data = (self.data.pipe(self.lemma_corpus)
                              .pipe(self.clean_corpus)
                              .pipe(self.class2number))
            
class build_terms_ratios_by_class:
    def __init__(self,data: pd.DataFrame, sentiments_column: str, reviews_column: str, OUTPUT_PATH: str):
        # load configurations
        self.config = load_yaml_file('params.yaml')['FEATUARIZATION']
        self.data = data
        self.sentiments_column = sentiments_column 
        self.reviews_column = reviews_column 
        self.OUT_PATH = OUTPUT_PATH
        
        # counters
        self.positive_sentiment_counter = Counter()
        self.negative_sentiment_counter = Counter()
        self.neutral_sentiment_counter = Counter()
        self.total_counter = Counter()
        self.positive_negative_ratio = Counter()
        
    @staticmethod
    def append_word_count(sentiment_counter: Counter(), total_counter: Counter(), word: str) -> Counter:
        sentiment_counter[word] += 1
        total_counter[word] += 1 
        return(sentiment_counter, total_counter)    
         
    def clip_weights(self):
        """ Clip weights between range min max."""
        
        weights = self.positive_negative_ratio
        self.clipped_weights = Counter()
        for word, weight in zip(weights.keys(), weights.values()):
            if weight < self.config['min_clip'] or weight > self.config['max_clip']:
                self.clipped_weights[word] = weight
   
    def word_counter(self):
        for _, row  in tqdm(self.data.iterrows()):
            for word in row[self.reviews_column].split(" "):
                if row[self.sentiments_column] == 1: # POSITIVE
                    self.append_word_count(self.positive_sentiment_counter, self.total_counter, word)
                elif row[self.sentiments_column] == -1: # NEGATIVE
                    self.append_word_count(self.negative_sentiment_counter, self.total_counter, word)
                    
    def positive_negative_ratio_calc(self):
        for term,cnt in list(self.total_counter.most_common()):
            if(cnt > self.config['min_frequency']):
                ratio = self.positive_sentiment_counter[term]/((float(self.negative_sentiment_counter[term]) + 1)/self.config['balanced'])
                self.positive_negative_ratio[term] = float(ratio)
                
    def ratios2logs(self):
        for word,ratio in self.positive_negative_ratio.most_common():
            self.positive_negative_ratio[word] = np.log(ratio + 0.01)     
        
    def fit(self):
        self.word_counter()
        self.positive_negative_ratio_calc()
        self.ratios2logs()
        self.clip_weights()

def main():
    os.makedirs(os.path.join("data", "to_train"), exist_ok=True)
    os.makedirs(os.path.join("data", "model"), exist_ok=True)
    
    TRAIN_INPUT = os.path.join(sys.argv[1], "train.csv")
    TEST_INPUT = os.path.join(sys.argv[1], "test.csv")
    VALID_INPUT = os.path.join(sys.argv[1], "valid.csv")
    
    TRAIN_OUTPUT = os.path.join(sys.argv[2], "train.csv")
    TEST_OUTPUT = os.path.join(sys.argv[2], "test.csv")
    VALID_OUTPUT = os.path.join(sys.argv[2], "valid.csv")
    
    WEIGHTS_PATH = os.path.join(sys.argv[3], "weights.json")

    # fit & transform TRAIN
    cc_train = CleanCorpus(TRAIN_INPUT)
    cc_train.pipeline()
    
    build = build_terms_ratios_by_class(cc_train.data, 'predictions', 'input', TRAIN_OUTPUT)
    build.fit()
    save_json(dict(build.clipped_weights.most_common()), WEIGHTS_PATH)
    
    # transform TEST
    cc_test = CleanCorpus(TEST_INPUT)
    cc_test.pipeline()
    
    # transform VALID
    cc_valid = CleanCorpus(VALID_INPUT)
    cc_valid.pipeline()
        
    # save
    cc_train.data.to_csv(TRAIN_OUTPUT, index=False)
    cc_test.data.to_csv(TEST_OUTPUT, index=False)
    cc_valid.data.to_csv(VALID_OUTPUT, index=False)
        
if __name__ == '__main__':
    main()
    