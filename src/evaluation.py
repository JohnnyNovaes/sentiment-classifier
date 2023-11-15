import pandas as pd
from utils import load_yaml_file
import numpy as np
from dvclive import Live
import os, sys
from sklearn.metrics import (balanced_accuracy_score,
                             f1_score,
                             recall_score,
                             precision_score)

class Evaluation:
    def __init__(self, INPUT_PATH: str) -> None:
        self.config = load_yaml_file('params.yaml')['EVALUATION']
        self.y_true, self.y_pred = self.map_ytrue_ypred(pd.read_csv(INPUT_PATH),
                                                        self.config['llm_predictions_labels_map'],
                                                        self.config['model_predictions_labels_map'])    
        self.labels = self.y_true.unique()
        self.weights = pd.read_csv(INPUT_PATH).predictions.value_counts(.1).to_dict()
        
    @staticmethod
    def map_ytrue_ypred(data: pd.DataFrame, llm_map: str, model_map: str) -> pd.Series:
        data.predictions.replace(llm_map, inplace=True)
        data.y_pred.replace(model_map, inplace=True)
        y_true = data['predictions']
        y_pred = data['y_pred']
        return(y_true, y_pred)
    
    def metric_f1_score(self): 
        f1_scores = f1_score(self.y_true, self.y_pred, average=None, labels=self.labels, zero_division=0)
        f1_score_dict = {label:score for label,score in zip(self.labels, f1_scores)}
        
    def metric_precision_score(self):
        precision = precision_score(self.y_true, self.y_pred, average=None, labels=self.labels, zero_division=0)
        precision_dict = {label:score for label,score in zip(self.labels, precision)}   
        
        self.precision_POSITIVE = precision_dict[1]
        self.precision_NEUTRAL = precision_dict[0]
        self.precision_NEGATIVE = precision_dict[-1]    
        
    def metric_recall_score(self):
        recall = recall_score(self.y_true, self.y_pred, average=None, labels=self.labels, zero_division=0)
        recall_dict = {label:score for label,score in zip(self.labels, recall)}
        
        self.recall_POSITIVE = recall_dict[1]
        self.recall_NEUTRAL = recall_dict[0]
        self.recall_NEGATIVE = recall_dict[-1]
        
    def fbeta_score_modified(self, beta: float = 1):
        # get precision  and recall for each sentiment 

        beta = self.config['BETA']['positive']**2
        FBETA_POSITIVE = (1 + beta) * ((self.precision_POSITIVE * self.recall_POSITIVE)/
                                       (beta * self.precision_POSITIVE + self.recall_POSITIVE + 0.01))
        
        beta = self.config['BETA']['negative']**2
        FBETA_NEGATIVE = (1 + beta) * ((self.precision_NEGATIVE * self.recall_NEGATIVE)/
                                       (beta * self.precision_NEGATIVE + self.recall_NEGATIVE + 0.01))
        
        beta = self.config['BETA']['neutral']**2
        FBETA_NEUTRAL = (1 + beta) * ((self.precision_NEUTRAL * self.recall_NEUTRAL)/
                                      (beta * self.precision_NEUTRAL + self.recall_NEUTRAL + 0.01))
        
        self.fbeta_score = (FBETA_POSITIVE*self.weights['POSITIVE'] +
                            FBETA_NEGATIVE*self.weights['NEGATIVE'] +
                            FBETA_NEUTRAL*self.weights['NEUTRAL'])
        
    def metrics2dvc(self):
        with Live(save_dvc_exp=False) as live:
            # recall
            live.log_metric("recall_positive", self.recall_POSITIVE)
            live.log_metric("recall_negative", self.recall_NEGATIVE)
            live.log_metric("recall_neutral", self.recall_NEUTRAL)
            
            # precision
            live.log_metric("precision_positive", self.precision_POSITIVE)
            live.log_metric("precision_negative", self.precision_NEGATIVE)
            live.log_metric("precision_neutral", self.precision_NEUTRAL)  
            
            # fbeta
            live.log_metric("fbeta", self.fbeta_score)  
            
            # confusion matrix
            live.log_sklearn_plot("confusion_matrix", self.y_true, self.y_pred)
            
    def build_metrics(self):
        self.metric_f1_score()
        self.metric_precision_score()
        self.metric_recall_score()
        self.fbeta_score_modified()
        self.metrics2dvc()
                    
def main():
    train_input = os.path.join(sys.argv[1], "train.csv")
    eval = Evaluation(train_input)
    eval.build_metrics()

    
main()
        
    