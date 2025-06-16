import pandas as pd, numpy as np, csv, math, os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, metrics
from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# GitHub notebook with the classifier model enacted on latent space
#https://github.com/aaxwaz/Fraud-detection-using-deep-learning/blob/master/auto-encoder/autoencoder_demo.ipynb

# fixed seed for same model initialisation and training behaviour
tf.keras.utils.set_random_seed(seed=100)

# --------------- Define functions ---------------
def performanceMetrics(prediction, name):
    accuracy = 100 * accuracy_score(testLabels, prediction); print(f"Accuracy: {accuracy}")
    precision = 100 * precision_score(testLabels, prediction); print(f"Precision: {precision}")
    recall = 100 * recall_score(testLabels, prediction); print(f"Recall: {recall}")
    f1Score = 100 * f1_score(testLabels, prediction); print(f"F1 score: {f1Score}")
    tn, fp, fn, tp = confusion_matrix(testLabels, prediction).ravel()
    anomalies = testData[prediction.flatten() == 1, -1]
    print(f"Number of anomalies predicted ({name}):{len(anomalies)}")
    print(f"True Negatives (TN): {tn}"); print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}"); print(f"True Positives (TP): {tp}")
    return tn, fp, fn, tp
def plotFraudCases(score, threshold, name):
    plt.hist(score[np.where(testLabels == 1)[0]], bins=len(testLabels[testLabels == 1]), color='blue')
    plt.axvline(x=threshold, color='orange', label=f'Threshold: {threshold:.2f}')
    plt.title(f'Reconstruction Loss of Fraud Cases: {name}'); plt.legend(loc='upper right')
    plt.xlabel('Value'); plt.ylabel('Frequency'); plt.show()
def plotROCAUC(score, name):
    fpr, tpr, thresholds1 = roc_curve(testLabels, score)
    auc = roc_auc_score(testLabels, score)
    plt.figure(); plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.title(f'Receiver Operating Characteristic Curve: {name}')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0]); plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.show()
def plotScores(score, threshold, bins, title):
    plt.hist(score, bins=bins, color='blue')
    plt.axvline(x=threshold, color='orange', label=f'Threshold: {threshold:.2f}')
    plt.title(f'{title}'); plt.legend(loc='upper right')
    plt.xlabel('Value'); plt.ylabel('Frequency'); plt.show()

# --------------- Data management ---------------
# import kaggle database with 0.172% anomalies
project_root = os.path.dirname(os.path.abspath(__file__))
kaggleData = pd.read_csv(os.path.join(project_root, "creditcard.csv"))

kaggleData = kaggleData.to_numpy() # pandas -> numpy array
kaggleNormal = kaggleData[kaggleData[:, -1] == 0] # normal
kaggleAnomaly = kaggleData[kaggleData[:, -1] == 1] # anomalous

split = 0.2; seed = 42 # split data into train/test
trainNormal, testNormal = train_test_split(kaggleNormal,test_size=split,random_state=seed,shuffle=True)
trainAnomaly, testAnomaly = train_test_split(kaggleAnomaly,test_size=split,random_state=seed,shuffle=True)
trainData = np.concatenate((trainNormal, trainAnomaly)) # add anomalies to train data
testData = np.concatenate((testNormal, testAnomaly)) # add anomalies to test data

trainLabels = trainData[:,-1]; testLabels = testData[:,-1] # get labels
trainData = trainData[:,:-1]; testData = testData[:,:-1] # drop class

# --------------- Data preprocessing ---------------
qt = QuantileTransformer(output_distribution="normal", random_state=0) # apply quantile transformer
trainData = qt.fit_transform(trainData) # .fit_transform() to training data
testData = qt.transform(testData) # .transform() to prevent data leakage

zscoreScaler = StandardScaler() # z-score standardisation
minmaxScaler = MinMaxScaler(feature_range=(-1.0, 1.0)) # min-max feature scaling
trainData = zscoreScaler.fit_transform(trainData); trainData = minmaxScaler.fit_transform(trainData)
testData = zscoreScaler.transform(testData); testData = minmaxScaler.transform(testData)

# --------------- Classifier training/testing ---------------
model = keras.Sequential([
    layers.Input(shape=(trainData.shape[1],)), # input layer size of train data
    layers.Dense(22, activation='elu'),
    layers.Dense(11, activation='elu'),
    layers.Dense(8, activation='elu'),
    layers.Dense(6, activation='elu'),
    layers.Dense(1, activation='sigmoid') # output layer for binary classification - 0/1
])
# define early stopping parameters
earlyStopping = EarlyStopping(
    monitor='AUC', # metric to monitor
    patience=5, # number of epochs with no improvement
    min_delta=1e-2, # require at least 0.01 improvement
    restore_best_weights=True, # restore weights from the best epoch
    mode='max'
)
model.compile(keras.optimizers.Adam(learning_rate=0.002), loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(name='AUC')])
model.fit(trainData, trainLabels, epochs=50, batch_size=250)
trainOutput = model.predict(trainData) # results after data has gone through classifier
threshold1 = np.percentile(trainOutput, 99.5) # assume 0.5% contamination

classifierScore = model.predict(testData) # each value is probability of being an anomaly
classifierPrediction = (classifierScore >= threshold1).astype(int) # apply the anomaly threshold

performanceMetrics(classifierPrediction, "Classification Model")
plotROCAUC(classifierScore, "Classification Model")
#plotFraudCases(classifierScore, threshold1, "Classification Model")

# --------------- Calculate indices of points ---------------
testLabels = testLabels.reshape(-1,1)
TP = (testLabels == 1) & (classifierPrediction == 1); TPIndices = np.where(TP)[0]
TPScore = classifierScore[TPIndices]; TPScoreMean = np.mean(TPScore, axis=0)
FN = (testLabels == 1) & (classifierPrediction == 0); FNIndices = np.where(FN)[0]
FNScore = classifierScore[FNIndices]; FNScoreMean = np.mean(FNScore, axis=0)
FP = (testLabels == 0) & (classifierPrediction == 1); FPIndices = np.where(FP)[0]
FPScore = classifierScore[FPIndices]; FPScoreMean = np.mean(FPScore, axis=0)
TN = (testLabels == 0) & (classifierPrediction == 0); TNIndices = np.where(TN)[0]
TNScore = classifierScore[TNIndices]; TNScoreMean = np.mean(TNScore, axis=0)

#plotScores(TPScore, threshold1, len(TPScore), "Classifier Score for True Positives")
#plotScores(FNScore, threshold1, len(FNScore), "Classifier Score for False Negatives")
#plotScores(FPScore, threshold1, len(FPScore), "Classifier Score for False Positives")
#plotScores(TNScore, threshold1, len(TNScore), "Classifier Score for True Negatives")

# --------------- plotting distributions ---------------
# plot anomaly scores of normal data
plt.hist(classifierScore[np.where(testLabels == 0)[0]], bins=len(testLabels[testLabels == 1]), color='blue')
plt.axvline(x=threshold1, color='orange', label=f'Threshold: {threshold1:.2f}')
plt.title(f'Classification Score for Unseen Normal Data'); plt.legend(loc='upper right')
plt.xlabel('Classifier Score'); plt.ylabel('Frequency'); plt.show()

# plot anomaly scores of anomalous data
plt.hist(classifierScore[np.where(testLabels == 1)[0]], bins=len(testLabels[testLabels == 1]), color='blue')
plt.axvline(x=threshold1, color='orange', label=f'Threshold: {threshold1:.2f}')
plt.title(f'Classification Score for Unseen Anomalous Data'); plt.legend(loc='upper right')
plt.xlabel('Classifier Score'); plt.ylabel('Frequency'); plt.show()
