from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

class Model:
    def __init__(self):
        clf1 = LogisticRegression(multi_class='multinomial', random_state=42, class_weight='balanced')
        clf2 = KNeighborsClassifier(n_neighbors=3)
        self.classifier = VotingClassifier(estimators=[('lr',  clf1),
                                                       ('knn', clf2)], voting='soft')
    @staticmethod
    def load_model(PATH):
        self.classifier = load(PATH)    
                
    def save_model(self, PATH):
        dump(self.classifier, PATH)
        
        
