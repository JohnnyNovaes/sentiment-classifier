from transformers import pipeline

class Model:
    def __init__(self, model_name: str = "lxyuan/distilbert-base-multilingual-cased-sentiments-student") -> None:
        self.model = pipeline(model=model_name)
    
    def predict(self, to_predict: str) -> str:
        return(self.model(to_predict))
    
if __name__ == '__main__':
    model = Model()
    print(model.predict('I hated the salad.'))
        
        
        
        
