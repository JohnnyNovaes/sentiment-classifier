import pandas as pd
import numpy as np
from model import Model
import sys, os
from utils import load_json

class Prepare2Train:
    def __init__(self, data, WEIGHTS_PATH, padding: int = 10):
        self.weights = load_json(WEIGHTS_PATH)
        self.data = data
        self.y = self.data.predictions
        self.padding = padding
        
    @staticmethod
    def pad_list(lst, target_size):
        if len(lst) >= target_size:
            return lst[:target_size]
        num_zeros = target_size - len(lst)
        padded_list = lst + [0.0] * num_zeros
        
        return padded_list
        
    def words2weights(self):
        self.lists = []
        for _, row in self.data.iterrows():
            self.lists.append([self.weights[word] for word in row.input.split(' ') if word in self.weights.keys()])

    def build_X(self):
        self.x =[self.pad_list(lst, 5) for lst in self.lists]
              
    def transform(self):
        self.words2weights()
        self.build_X()
        

def main():
    os.makedirs(os.path.join("data", "to_validate"), exist_ok=True)
    os.makedirs(os.path.join("data", "model"), exist_ok=True)
    
    # paths
    TRAIN_OUTPUT = os.path.join(sys.argv[2], "train.csv")
    VALID_OUTPUT = os.path.join(sys.argv[2], "valid.csv")
    MODEL_PATH = os.path.join(sys.argv[3], 'model.joblib')
    WEIGHTS_PATH = os.path.join(sys.argv[3], 'weights.json')   
    
    #load data
    train = pd.read_csv(os.path.join(sys.argv[1], "train.csv"))
    prepare = Prepare2Train(train, WEIGHTS_PATH, padding=10)
    prepare.transform()
    X_train, y_train = prepare.x, prepare.y
    
    valid = pd.read_csv(os.path.join(sys.argv[1], "valid.csv"))
    prepare = Prepare2Train(valid, WEIGHTS_PATH, padding=10)
    prepare.transform()
    X_valid = prepare.x 
        
    # load & fit model
    model = Model()

    model.classifier.fit(X_train, y_train)
    valid['y_pred'] = model.classifier.predict(X_valid)
    
    # save data & model
    valid.to_csv(VALID_OUTPUT,index=False)
    model.save_model(MODEL_PATH)

if __name__ == '__main__':
    main()
    # data = pd.read_csv('./data/to_train/train.csv')
    # y_train = data.predictions
    # lists = words2weights(data)
    # padded_lists = np.array([pad_list(lst, 5) for lst in lists])
    # model = Model()
    # model.classifier.fit(padded_lists, y_train)
    # print(model.classifier.predict(padded_lists))

