import pandas as pd
import numpy as np
from model import Model
import sys, os

def list2Xtrain(data):
    X_train = data['predictions'].tolist()
    X_train = np.array([np.array(x) for x in X_train])
    X_train = np.vstack(X_train)
    return(X_train)

def main():
    os.makedirs(os.path.join("data", "to_validate"), exist_ok=True)
    os.makedirs(os.path.join("data", "model"), exist_ok=True)
    #load data
    train = pd.read_csv(os.path.join(sys.argv[1], "train.csv"))
    X_train, y_train = list2Xtrain(train), train['predictions']
        
    valid = pd.read_csv(os.path.join(sys.argv[1], "valid.csv"))
    X_valid, y_valid = list2Xtrain(valid), valid['predictions']
    
    # paths
    TRAIN_OUTPUT = os.path.join(sys.argv[2], "train.csv")
    VALID_OUTPUT = os.path.join(sys.argv[2], "valid.csv")
    MODEL_PATH = os.path.join(sys.argv[3], 'model.joblib')
        
    # load & fit model
    model = Model()
    model.classifier.fit(X_train, y_train)
    valid['y_pred'] = model.classifier.predict(X_valid)
    
    # save data & model
    valid.to_csv(VALID_OUTPUT,index=False)
    model.save_model(MODEL_PATH)

if __name__ == '__main__':
    main()

    

