import pandas as pd
import openai
import hydra
from omegaconf import DictConfig, OmegaConf
from langchain.prompts import PromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from utils import save_json
from tqdm import tqdm


openai.api_key = os.getenv("OPENAI_API_KEY") 

class LLMSentimentClassifier:
    def __init__(self, config, path) -> None:
        self.JSON_PATH = config['PATH']['JSON']
        self.PROMPT_TEMPLATE = config['MODEL']['PROMPT_TEMPLATE']
        self.data = pd.read_csv(config['PATH']['CSV'], sep='|')
        
    def build_batch(self) -> None:
        chat_template = PromptTemplate.from_template(self.PROMPT_TEMPLATE)
        context_data = self.data.input.apply(lambda input: {"review": input}).to_list()
        self.batch_data = [data.text for data in chat_template.batch(context_data)]
        
    def gtp_predict(self) -> None:
        self.predictions = []
        for user_template in tqdm(self.batch_data):
            completion = openai.completions.create(
                            model="gpt-3.5-turbo-instruct",
                            temperature=.5,
                            max_tokens=5,
                            prompt=user_template)
            self.predictions.append(completion.choices[0].text.strip('\n'))
        
    def predictions2json(self) -> None:
        save_json(self.predictions, self.JSON_PATH)
        
    def pipeline(self) -> None:
        self.build_batch()
        self.gtp_predict()
        self.predictions2json()
                   
@hydra.main(version_base=None, config_path="../conf", config_name="config")         
def main(cfg : DictConfig) -> None:
    llm = LLMSentimentClassifier(cfg)
    llm.pipeline()
 
if __name__ == '__main__':
    main()
    