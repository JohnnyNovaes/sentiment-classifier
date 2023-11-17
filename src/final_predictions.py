from model import Model
from train import Prepare2Train
import pandas as pd
from utils import load_json
import joblib
from featuarization import CleanCorpus

final_df = pd.read_csv('./data/raw/dataset_valid.csv', sep='|')
print(final_df.shape[0])
cc = CleanCorpus(final_df)
final_df = (final_df.pipe(cc.lemma_corpus)
                    .pipe(cc.clean_corpus))
model = joblib.load('./data/model/model.joblib')
prepare = Prepare2Train(final_df, './data/model/weights.json', padding=10)
prepare.transform()
X_valid = prepare.x
final_df['y_pred'] = model.predict(X_valid)
final_df.to_csv('./data/raw/final_pred.csv')
