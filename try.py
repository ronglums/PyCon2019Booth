# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\..\AppData\Local\Temp\2'))
	print(os.getcwd())
except:
	pass

#%%
# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#%%
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


#%%
# shape
print(dataset.shape)


#%%
# head
print(dataset.head(20))


#%%
# descriptions
print(dataset.describe())


#%%
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


#%%
# histograms
dataset.hist()
plt.show()


#%%
# scatter plot matrix
scatter_matrix(dataset)
plt.show()


#%%
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


#%%
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

#%% [markdown]

#%%
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


#%%
# evaluate each model in turn
results = []
names = []
print("Training Accuracy")
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)


#%%
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


#%%
scores = []
print("Validiation Accuracy")
for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    accuracy = accuracy_score(Y_validation, predictions)
    scores.append((name, accuracy))
    msg = "%s: %f" % (name, accuracy)
    print(msg)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
best_model = scores[0]

#%% [markdown]
# ### Saving the model
# The code below saves the best model under the filename [MODEL_TYPE]-model.sav. This model can be loaded again with the code:
# 
# ~~~python
# loaded_model = pickle.load(open(filename, 'rb'))
# ~~~

#%%
import pickle
filename = '%s-model.sav' % best_model[0]
pickle.dump(model, open(filename, 'wb'))
print("Model saved!")
