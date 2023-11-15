import pandas as pd
from model import Model

class Predictions:
    def __init__(self, IN_PATH, OUT_PATH):
        self.data = data
        self.OUT_PATH = OUT_PATH
        self.model = Model()
        
    def predictions(self):
        predictions = self.data.input.apply(lambda input: self.model(input))
        self.data['y_pred'] = predictions
        
    def save(self):
        self.data.to_csv(self.OUT_PATH)
        
    def pipeline(self):
        self.predictions()
        self.save()

def main():
    train_input = os.path.join(sys.argv[1], "train.csv")
    train_output = os.path.join(sys.argv[2], "train.csv")
    
    pred = Predictions(train_input, train_output)
    pred.pipeline()