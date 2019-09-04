#%% [markdown]
# # Predicting Diabetes
# ## Import Libraries
# 

#%%
import pandas as pd # pandas is a dataframe library
import matplotlib.pyplot as plt # matplotlib.pyplot plots data
import numpy as np # numpy provides N-dim object support

#%% [markdown]
# ## Load and review data

#%%
df = pd.read_csv("./data/pima-data.csv") # load Pima data


#%%
df.head(5)


#%%
def check(df, size=11):
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.
    
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
        
    Displays:
        matrix of correlation between columns. Blue-cyan-yellow-red-darkred => less to more correlated
                                               0------------------------->1
                                               Expect a darkred line running from top to bottom right
    """
    
    corr = df.corr() # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr) # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns) # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns) # draw y tick marks
    


#%%
check(df)


#%%
del df['skin']

#%% [markdown]
# ## Check Data Types

#%%
diabetes_map = {True:1, False:0}


#%%
df['diabetes'] = df['diabetes'].map(diabetes_map)


#%%
df.head(5)

#%% [markdown]
# ## Spliting the data
# 70% for training, 30% for testing

#%%
from sklearn.model_selection import train_test_split

feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

x = df[feature_col_names].values # predictor feature columns (8 X m)
y = df[predicted_class_names].values # predicted class (1=true, 0=false) column (1 X m)
split_test_size = 0.30

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_test_size, random_state=42)
# test_size = 0.3 is 30%, 42 is the answer to everything

#%% [markdown]
# We check to ensure we have the desired 70% train, 30% test split of the data

#%%
print("{0:0.2f}% in training set".format((len(x_train)/len(df.index))*100))
print("{0:0.2f}% in test set".format((len(x_test)/len(df.index))*100))

#%% [markdown]
# ## Post-split Data Preparation
#%% [markdown]
# ### Impute with the mean

#%%
from sklearn.preprocessing import Imputer

# Impute with mean all 0 readings
fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)

x_train = fill_0.fit_transform(x_train)
x_test = fill_0.fit_transform(x_test)

#%% [markdown]
# ## Training Initial Algorithm = Naive Bayes

#%%
from sklearn.naive_bayes import GaussianNB

# create Gaussian Naive Bayes model object and train it with the data
nb_model = GaussianNB()

nb_model.fit(x_train, y_train.ravel())

#%% [markdown]
# ### Performance on Training data

#%%
# predict values using the training data
nb_predict_train = nb_model.predict(x_train)

# import the performance metrics library
from sklearn import metrics

# Accuracy
print("Accurary: {0:.4f}".format(metrics.accuracy_score(y_train, nb_predict_train)))
print("")

#%% [markdown]
# ### Performance on Testing data

#%%
# predict values using the training data
nb_predict_test = nb_model.predict(x_test)

# import the performance metrics library
# from sklearn import metrics

# Accuracy
print("Accurary: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))
print("")

#%% [markdown]
# ### Metrics

#%%
print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(y_test, nb_predict_test))

#%% [markdown]
# ## Retrain =  Random Forest 

#%%
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42) # Create random forest object

rf_model.fit(x_train, y_train.ravel())

#%% [markdown]
# ### Performance on Training data

#%%
# predict values using the training data
rf_predict_train = rf_model.predict(x_train)

# import the performance metrics library
from sklearn import metrics

# Accuracy
print("Accurary: {0:.4f}".format(metrics.accuracy_score(y_train, rf_predict_train)))
print("")

#%% [markdown]
# ### Performance on Testing data

#%%
# predict values using the testing data
rf_predict_test = rf_model.predict(x_test)

# import the performance metrics library
from sklearn import metrics

# Accuracy
print("Accurary: {0:.4f}".format(metrics.accuracy_score(y_test, rf_predict_test)))
print("")

#%% [markdown]
# ### Metrics

#%%
print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, rf_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(y_test, rf_predict_test))

#%% [markdown]
# ## Retrain = Logistic Regression

#%%
from sklearn.linear_model import LogisticRegression

lf_model = LogisticRegression(C=0.7, class_weight="balanced", random_state=42)
lf_model.fit(x_train, y_train.ravel())

#%% [markdown]
# ### Performance on Training data

#%%
# predict values using the training data
lf_predict_train = lf_model.predict(x_train)

# import the performance metrics library
from sklearn import metrics

# Accuracy
print("Accurary: {0:.4f}".format(metrics.accuracy_score(y_train, lf_predict_train)))
print("")

#%% [markdown]
# ### Performance on Testing data

#%%
# predict values using the training data
lf_predict_test = lf_model.predict(x_test)

# import the performance metrics library
from sklearn import metrics

# Accuracy
print("Accurary: {0:.4f}".format(metrics.accuracy_score(y_test, lf_predict_test)))
print("")

#%% [markdown]
# ### Metrics

#%%
print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, lf_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(y_test, lf_predict_test))

#%% [markdown]
# ### Setting regularization parameter

#%%
C_start = 0.1
C_end = 10
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while (C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C=C_val, random_state=42)
    lr_model_loop.fit(x_train, y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(x_test)
    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if (recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test

    C_val = C_val + C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))
plt.plot(C_values, recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall score")

#%% [markdown]
# ## Retrain with class_weight='balanced' and C=0.3

#%%
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(C=best_score_C_val, class_weight="balanced", random_state=42)
lr_model.fit(x_train, y_train.ravel())

#%% [markdown]
# ### Performance on Testing data

#%%
# predict values using the training data
lr_predict_test = lr_model.predict(x_test)

# import the performance metrics library
from sklearn import metrics

# Accuracy
print("Accurary: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
print("")

#%% [markdown]
# ### Metrics

#%%
print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, lr_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(y_test, lr_predict_test))
print(metrics.recall_score(y_test,lr_predict_test))




#%%
