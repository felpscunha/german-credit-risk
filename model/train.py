import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
import pickle

def balance_dataset(df: pd.DataFrame, target: str, percentual=.45) -> pd.DataFrame:
    df_balanced = df.copy()
    while True:
        majority, minority = df_balanced[target].value_counts(normalize=True).index.to_list()
        
        if np.random.randint(2): # Maioria
            idx = df_balanced[df_balanced[target] == majority].sample(1).index
            df_balanced.drop(index=idx, inplace=True)
            df_balanced.reset_index(drop=True, inplace=True)
        else: # Minoria
            idx = df[df[target] == minority].sample(1).index
            df_balanced = pd.concat([df_balanced, df.iloc[idx, :]], ignore_index=True)
        
        if percentual-.05 < df_balanced[target].value_counts(normalize=True).values[1] <= percentual+.05:
            break
        
    return df_balanced

def train(df_train, y_train, C=1.0):
    dicts = df_train[object_cols + numeric_cols].to_dict(orient='records')


    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df, dv, model, threshold=0.5):
    dicts = df[object_cols + numeric_cols].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)

    return y_pred


# parameter utilized in the model

C = 10
n_splits = 5

output_file = f'model_C={C}.pkl' # file name that will be generated

# DATA PREPARATION

columns = ['status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings', 'employment_since', 'installment_rate', 'personal_status_and_sex', 'others_debtors_guarantors', 'residence_since', 'property', 'age', 'installment_plans', 'housing', 'existing_credits', 'job', 'people_maitenance', 'telephone', 'foreign_worker', 'risk']

df = pd.read_csv('german.data', sep='\s+', header=None, names=columns)
df['risk'] = df['risk'].apply(lambda x: 0 if x == 1 else 1 if x == 2 else x)

df['installment_rate'] = df['installment_rate'].astype('category')
df['residence_since'] = df['residence_since'].astype('category')
df['existing_credits'] = df['existing_credits'].astype('category')
df['people_maitenance'] = df['people_maitenance'].astype('category')

purpose_pct = df['purpose'].value_counts(normalize=True)

categories_to_replace = purpose_pct[purpose_pct <= 0.05].index.tolist()

df.loc[df['purpose'].isin(categories_to_replace), 'purpose'] = 'others'

object_cols = df.select_dtypes(include=['object']).columns.tolist()

df_balanced = balance_dataset(df, 'risk', percentual=.5)

# DATA VALIDATION

# pipeline for model training
object_cols = df.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()
numeric_cols.remove('risk')

df_full_train, df_test = train_test_split(df_balanced, test_size=0.2, random_state=42)

print(f'doing validation with C={C}')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.risk.values
    y_val = df_val.risk.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    recall = recall_score(y_val, y_pred)
    scores.append(recall)

    print(f'recall on fold {fold} is {recall}')
    fold = fold + 1


print('validation results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# TRAINING THE FINAL MODEL

print('training the final model')

dv, model = train(df_full_train, df_full_train.risk.values, C=10)

final_predict = predict(df_test, dv, model)

y_test = df_test.risk.values

recall_final = recall_score(y_test, final_predict)

print(f'recall={recall_final}')

# Saving model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}.')