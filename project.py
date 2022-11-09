#!/usr/bin/env python3
# Let's import the needed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import tensorflow as tf
tf.get_logger().setLevel(tf._logging.ERROR)
from scikeras.wrappers import KerasClassifier
import keras_tuner

# 1. FUNCTIONS AND CLASSES DECLARATIONS
# this function plots a dataframe as a table
def plot_table(df, n=4):
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')

    table = ax.table(cellText=df.values[:n,:], colLabels=df.columns, loc='center')
    table.set_fontsize(40)
    table.scale(10.0, 10.0)

    plt.show()
    plt.close()


# this class is used in the Pipeline to delete columns from dataframe
class columnDropperTransformer():
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        return X.drop(self.columns,axis=1)

    def fit(self, X, y=None):
        return self


# Plots PCA graphs
def plot_2d_space(X, y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


# MLP architecture
# hp is needed by KerasTuner to perform hyperparameter tuning
def create_baseline(hp=None):
	# create model
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(11, input_shape=(11,), activation='relu')) # input layer
	if hp:
		hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
		model.add(tf.keras.layers.Dense(units=hp_units, activation="relu"))
	else:
		model.add(tf.keras.layers.Dense(100, activation="relu"))
	model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # output layer
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# Confusion Matrix for models performance evaluation
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sb.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
    plt.show()


# 2. PREP DATA
# dataset available at https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection
df = pd.read_csv("onlinefraud.csv")

train_set, test_set = train_test_split(df, test_size=0.1)

# 3. DATA EXPLORATION

# visualize the first rows
plot_table(df)

# 3.1 Exploring features

# 3.1.1 Step
step = train_set["step"]
max_hours = step.max()

plt.title('Step')
plt.hist(step, max_hours, density=True)
plt.show()
plt.close()

# 3.1.2 Type
type = train_set["type"].value_counts()
transaction_names = type.index.to_list()
quantity = type.values
expl = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
figure = plt.pie(quantity, labels=transaction_names, explode=expl, autopct='%1.1f%%')
plt.show()
plt.close()

# 3.1.3 isFraud
train_set["isFraud"].value_counts().plot(kind='bar')
pd.pivot_table(train_set, index=train_set["isFraud"], values='isFraud', aggfunc='count')

# 3.1.4 Amount
plot_table(train_set.nlargest(10, "amount"), 10)
# we can show that no fraud is committed in first 100 transactions
x = train_set.nlargest(100, "amount").isFraud
counter = 0
if 1 in x.values:
    counter +=1
print("\n\nFrauds in top 100 transactions: " + str(counter))

# 3.1.5 oldBalanceOrig and newBalanceOrig
# Before Transaction
plt.figure(figsize = [10,7])
plt.bar(train_set.nlargest(10, 'oldbalanceOrg').nameOrig, train_set.nlargest(10, 'oldbalanceOrg').oldbalanceOrg)
plt.xticks(rotation = 17.5) # to give more spacing for IDs
plt.title('the largest 10 balances before transaction')
plt.xlabel('Client Id')
plt.ylabel('The Balance')
plt.show()
plt.close()

# After Transaction
plt.figure(figsize = [10,7])
plt.bar(train_set.nlargest(10, 'newbalanceOrig').nameOrig, train_set.nlargest(10, 'newbalanceOrig').newbalanceOrig)
plt.xticks(rotation = 17.5) # to give more spacing for IDs
plt.title('the largest 10 balances after transaction')
plt.xlabel('Client Id')
plt.ylabel('The Balance')
plt.show()
plt.close()

# 3.2 Finding correlations

# show correlation between oldBalanceOrig and newBalanceOrig
print("\n\nCorrelation between oldBalanceOrig and new BalanceOrig")
print(train_set[['oldbalanceOrg', 'newbalanceOrig']].corr())
# plot correlation between isFraud and the others
corr_isFraud = train_set.corr(numeric_only=True)
print("\n\nCorrelation between isFraud and the others")
print(corr_isFraud["isFraud"].sort_values(ascending=False))

sb.heatmap(corr_isFraud, annot=True)

# 4. PREPROCESSING AND FEATURE SELECTION

# show that there are no missing values for features
print("\n\nNo missing values for features")
print(train_set.isnull().sum())

# show that isFlaggedFraud is not helpful
print("\n\nisFlaggedFraud number of occurrencies in dataset")
print(train_set["isFlaggedFraud"].value_counts())

# Pipeline
# This pipeline handles both numerical and categorical data
# But it also returns a numpy array instead of a dataframe, so we need to do some tricks
cat_attribs = ["type"]
num_attribs = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
drop_attribs = ['isFlaggedFraud', 'nameOrig', 'nameDest']

full_pipeline = ColumnTransformer([
    ("dropCol", "drop", drop_attribs),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
    ("num", StandardScaler(), num_attribs)
], verbose_feature_names_out=False, remainder='passthrough')

# here we use fit_transform for the training set
train_set = pd.DataFrame(full_pipeline.fit_transform(train_set), columns=full_pipeline.get_feature_names_out())
plot_table(train_set, 5)

# 5. MODELS COMPARISON

# 5.1 First struggles
y = train_set.loc[:, "isFraud"]
X = train_set.loc[:, train_set.columns != "isFraud"]
labels = ["True Neg","False Pos","False Neg","True Pos"]
categories = ["notFraud", "isFraud"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
cf_matrix = confusion_matrix(y_val, y_pred)
make_confusion_matrix(cf_matrix, group_names=labels, categories=categories)

# cfmatrix with only one column
y = train_set.loc[:, "isFraud"]
X_reduced = train_set.loc[:, ["amount"]]

X_train, X_val, y_train, y_val = train_test_split(X_reduced, y, test_size=0.15, random_state=42)

model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
cf_matrix = confusion_matrix(y_val, y_pred)
make_confusion_matrix(cf_matrix, group_names=labels, categories=categories)

# 5.1.2 Balance dataset

# plot imbalanced dataset first
colors = ['#1F77B4', '#FF7F0E']
markers = ['o', 's']
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)
plot_2d_space(X_pca, y, 'Imbalanced dataset (2 PCA components)')

# perform undersampling
class_count_0, class_count_1 = train_set['isFraud'].value_counts()
class_0 = train_set[train_set['isFraud'] == 0]
class_1 = train_set[train_set['isFraud'] == 1]

class_0_under = class_0.sample(class_count_1)
train_set = pd.concat([class_0_under, class_1], axis=0)
y = train_set.loc[:, "isFraud"]
X = train_set.loc[:, train_set.columns != "isFraud"]

# plot balanced dataset
X_pca = pca.fit_transform(X)
plot_2d_space(X_pca, y, 'Balanced dataset (2 PCA components)')

# 5.2 Cross validation for LR and RF
kf = KFold(n_splits=10)

models = {
    "LR": LogisticRegression(max_iter=300),
    "RF": RandomForestClassifier(max_depth=4, n_estimators=100),
}

final_scores = np.zeros((2, 4)) # each row: accuracy, recall, precision, f1
i = 0 # model index

for name, model in models.items():
    k = 0
    model_scores = np.zeros((10, 4)) # each row: accuracy, recall, precision, f1
    print(f'Training Model {name} \n--------------')
    for train_index, val_index in kf.split(X, y):
        # get the actual arrays
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        # train model and get prediction
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        model_scores[k][0] = accuracy_score(y_val, y_pred)
        model_scores[k][1] = recall_score(y_val, y_pred, average='micro')
        model_scores[k][2] = precision_score(y_val, y_pred, average='micro')
        model_scores[k][3] = f1_score(y_val, y_pred, average='micro')
        k += 1

    final_scores[i][0] = model_scores[:, 0].mean()
    final_scores[i][1] = model_scores[:, 1].mean()
    final_scores[i][2] = model_scores[:, 2].mean()
    final_scores[i][3] = model_scores[:, 3].mean()

    print(f"Avarage accuracy:  {final_scores[i][0]}")
    print(f"Avarage recall:  {final_scores[i][1]}")
    print(f"Avarage precision:  {final_scores[i][2]}")
    print(f"Avarage f1:  {final_scores[i][3]}")

# 5.3 Neural Networks

models = {
    "LR": LogisticRegression(max_iter=300),
    "RF": RandomForestClassifier(max_depth=4, n_estimators=100),
    "MLP": KerasClassifier(model=create_baseline(), epochs=60),
}

kfold = StratifiedKFold(n_splits=10, shuffle=True)
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro'}

final_scores = pd.DataFrame()
for name, model in models.items():
    print(f'Training Model {name} \n--------------')
    # This scores are for each fold
    scores = cross_validate(model, X, y, scoring=scoring,
                         cv=10, return_train_score=True)
    final_scores = pd.concat([final_scores, pd.DataFrame([{"models": name, "accuracy": scores['test_acc'].mean()}])])

# plot results
print(final_scores)

# 5.3.1 Hyperparameter optimization with KerasTuner

tuner = keras_tuner.RandomSearch(create_baseline, objective='val_accuracy')
tuner.search(X, y, epochs=50, validation_split=0.2)
best_model = tuner.get_best_models()[0]

# 6. TESTING FINAL MODEL

# prepare test set
# apply transformation also to test set
test_set = pd.DataFrame(full_pipeline.transform(test_set), columns=full_pipeline.get_feature_names_out())
y_test = test_set.loc[:, "isFraud"]
X_test = test_set.loc[:, test_set.columns != "isFraud"]

# train choosen model
best_model.fit(X_train, y_train)

# make predictions for test data and print evaluation
eval_res = best_model.evaluate(X_test, y_test)
print("[test loss, test accuracy]:", eval_res)
