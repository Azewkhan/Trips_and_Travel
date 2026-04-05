# Importing Libraries
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,classification_report,roc_auc_score,roc_curve
from sklearn.model_selection import RandomizedSearchCV
warnings.filterwarnings("ignore")

df = pd.read_csv("D:\\Trips_and_Travel\\dataset\\Cleaned_data.csv")

# Train test Split
X = df.drop("ProdTaken",axis=1)
y= df["ProdTaken"]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.20,random_state=42)

#Scaling and Encoding

cat = [col for col in X.columns if X[col].dtype=="O" ]
num = [col for col in X.columns if X[col].dtype!="O" ]

scaler = StandardScaler()
oh_encoder = OneHotEncoder(drop="first")

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoding",oh_encoder,cat),
        ("Scaling",scaler,num)
    ]
)

X_train= preprocessor.fit_transform(X_train)
X_test= preprocessor.transform(X_test)

Models = {
    "Random Forest Classifier":RandomForestClassifier()
    ,"Decision Tree Classifier": DecisionTreeClassifier()
    ,"Logistic Regression" : LogisticRegression()
    }

for m in range(len(list(Models))):

    model= list(Models.values())[m]
    model.fit(X_train,y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    y_train_prob = model.predict_proba(X_train)[:,1]
    y_test_prob = model.predict_proba(X_test)[:,1]

    print("=="*50)
    print(f"Model Name  = {list(Models.keys())[m]}")
    #y_train scores
    accuracy_train =  accuracy_score(y_train_pred,y_train)
    recall_train =recall_score(y_train_pred,y_train)
    precision_train = precision_score(y_train_pred,y_train)
    classification_train = classification_report(y_train_pred,y_train)
    Roc_auc_score_train = roc_auc_score(y_train,y_train_prob)

    print(classification_train)
    print(f"Accuracy Score for Training set is {accuracy_train}")
    print(f"Recall Score for Training set is {recall_train}")
    print(f"Precision Score for Training set is {precision_train}")
    print(f"ROC Score for Training set is {Roc_auc_score_train}")

    print("--"*50)
    #y_test scores
    accuracy_test =  accuracy_score(y_test_pred,y_test)
    recall_test =recall_score(y_test_pred,y_test)
    precision_test = precision_score(y_test_pred,y_test)
    classification_test = classification_report(y_test_pred,y_test)
    Roc_auc_score_test = roc_auc_score(y_test,y_test_prob)

    print(classification_test)
    print(f"Accuracy Score for Test set is {accuracy_test}")
    print(f"Recall Score for Test set is {recall_test}")
    print(f"Precision Score for Test set is {precision_test}")
    print(f"ROC Score for Test set is {Roc_auc_score_test}")

## Hyperparameter Training
rf_params = {"max_depth": [5, 8, 15, None, 10],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]}

# Models list for Hyperparameter tuning
randomcv_models = [
                   ("RF", RandomForestClassifier(), rf_params)
                   
                   ]

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   n_jobs=-1)
    random.fit(X_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])