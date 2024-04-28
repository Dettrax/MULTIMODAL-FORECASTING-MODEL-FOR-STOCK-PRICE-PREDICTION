from tqdm import tqdm
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertModel, BertTokenizer

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the BERT model
model = BertModel.from_pretrained('yiyanghkust/finbert-tone').to(device)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

final_df = pd.read_excel('final_df.xlsx')
targets = final_df.iloc[:, 4:]  # sentiment scores
targets = targets.apply(lambda row: [1 if val == max(row) else 0 for val in row], axis=1, result_type='broadcast')

embeddings = []
for i in tqdm(range(len(final_df))):
    text = final_df['Content'].loc[i]
    tokens = tokenizer.encode(text, max_length=510,return_tensors='pt').to(device)
    embeddings.append(model(tokens)[0][0][0].detach().cpu().numpy())

features = pd.DataFrame(embeddings)



X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.5, 1.0, 0.1),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.4, 1.0, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 10, 500),
        'objective': 'binary:logistic',
        'use_label_encoder': False,
        'tree_method': 'gpu_hist'
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy


study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=100)

best_trial = study.best_trial

print("Best trial:")
print(" Value: ", best_trial.value)
print(" Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

