import pandas as pd
import re
import spacy
from tqdm import tqdm
import stanza
from unidecode import unidecode
from collections import Counter
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
    def __init__(self,data: pd.DataFrame, sentiments_column: str, reviews_column: str, OUTPUT_PATH: str,
                      min_frequency: int = 5, balanced: float = 0.38, padding: int = 50,
                      min_clip: float = -0.2, max_clip: float = 0.2):
        self.data = data
        self.sentiments_column = sentiments_column 
        self.reviews_column = reviews_column 
        self.min_frequency = min_frequency
        self.balanced_weight = balanced
        self.padding = padding
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.OUT_PATH = OUTPUT_PATH
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
     
    def text2weights(self, data: pd.DataFrame) -> pd.DataFrame:
        data['weights'] = data[self.reviews_column].apply(lambda review: self.replace_text(review))
        return(data)
    
    def clip_weights(self):
        """ Clip weights between range min max."""
        
        weights = self.positive_negative_ratio
        self.clipped_weights = Counter()
        for word, weight in zip(weights.keys(), weights.values()):
            if weight < self.min_clip or weight > self.max_clip:
                self.clipped_weights[word] = weight
   
    def replace_text(self, text: str) -> list():
        """ Tokenize text"""
        
        tokens = Counter()
        sentiment_ratios = self.clipped_weights
        for word in text.split(" "):
            try:
                if sentiment_ratios[word] != 0:
                    tokens[word] = sentiment_ratios[word]
            except KeyError: # jump word
                continue
        
        # padding
        tokens = [float(weight) for weight in tokens.values()]
        if len(tokens) < self.padding:
            size_tokens = len(tokens)
            to_append = self.padding - size_tokens
            tokens = np.append(tokens, np.zeros(to_append))
        else:
            tokens = tokens[0:self.padding]
        return(tokens)

    def word_counter(self):
        for _, row  in tqdm(self.data.iterrows()):
            for word in row[self.reviews_column].split(" "):
                if row[self.sentiments_column] == 1: # POSITIVE
                    self.append_word_count(self.positive_sentiment_counter, self.total_counter, word)
                elif row[self.sentiments_column] == -1: # NEGATIVE
                    self.append_word_count(self.negative_sentiment_counter, self.total_counter, word)
                    
    def positive_negative_ratio_calc(self):
        for term,cnt in list(self.total_counter.most_common()):
            if(cnt > self.min_frequency):
                ratio = self.positive_sentiment_counter[term]/((float(self.negative_sentiment_counter[term]) + 1)/self.balanced_weight)
                self.positive_negative_ratio[term] = ratio
                
    def ratios2logs(self):
        for word,ratio in self.positive_negative_ratio.most_common():
            self.positive_negative_ratio[word] = np.log(ratio + 0.01)     
        
    def fit(self):
        self.word_counter()
        self.positive_negative_ratio_calc()
        self.ratios2logs()
        self.clip_weights()
        self.text2weights(self.data)

def main():
    os.makedirs(os.path.join("data", "to_train"), exist_ok=True)
    TRAIN_INPUT = os.path.join(sys.argv[1], "train.csv")
    TEST_INPUT = os.path.join(sys.argv[1], "test.csv")
    VALID_INPUT = os.path.join(sys.argv[1], "valid.csv")
    
    TRAIN_OUTPUT = os.path.join(sys.argv[2], "train.csv")
    TEST_OUTPUT = os.path.join(sys.argv[2], "test.csv")
    VALID_OUTPUT = os.path.join(sys.argv[2], "valid.csv")

    # fit & transform TRAIN
    cc = CleanCorpus(TRAIN_INPUT)
    cc.pipeline()
    
    build = build_terms_ratios_by_class(cc.data, 'predictions', 'input', TRAIN_OUTPUT)
    build.fit()
    
    # transform TEST
    cc = CleanCorpus(TEST_INPUT)
    cc.pipeline()
    
    test_data = build.text2weights(cc.data)
    
    # transform VALID
    cc = CleanCorpus(VALID_INPUT)
    cc.pipeline()
    
    valid_data = build.text2weights(cc.data)
    
    # save
    build.data.to_csv(TRAIN_OUTPUT, index=False)
    test_data.to_csv(TEST_OUTPUT, index=False)
    valid_data.to_csv(VALID_OUTPUT, index=False)
        
if __name__ == '__main__':
    main()
    