import pandas as pd, numpy as np, csv, math, os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Input, Dropout
from keras.regularizers import l1, l2
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# --------------- Define functions ---------------
def performanceMetrics(prediction, name, cont):
    accuracy = 100 * accuracy_score(testLabels, prediction); print(f"Accuracy: {accuracy}")
    precision = 100 * precision_score(testLabels, prediction); print(f"Precision: {precision}")
    recall = 100 * recall_score(testLabels, prediction); print(f"Recall: {recall}")
    f1Score = 100 * f1_score(testLabels, prediction); print(f"F1 score: {f1Score}")
    tn, fp, fn, tp = confusion_matrix(testLabels, prediction).ravel()
    anomalies = testData[prediction.flatten() == 1, -1]
    print(f"Number of anomalies predicted ({name}):{len(anomalies)} at {cont}% contamination")
    print(f"True Negatives (TN): {tn}"); print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}"); print(f"True Positives (TP): {tp}")
def plotFraudCases(score, threshold, name):
    plt.hist(score[np.where(testLabels == 1)[0]], bins=len(testLabels[testLabels == 1]), color='blue')
    plt.axvline(x=threshold, color='orange', label=f'Threshold: {threshold:.2f}')
    plt.title(f'Anomaly Score of Fraud Cases: {name}'); plt.legend(loc='upper right')
    plt.xlabel('Value'); plt.ylabel('Frequency'); plt.show()
def plotNormalCases(score, threshold, name):
    plt.hist(score[np.where(testLabels == 0)[0]], bins=len(testLabels[testLabels == 1]), color='blue')
    plt.axvline(x=threshold, color='orange', label=f'Threshold: {threshold:.2f}')
    plt.title(f'Anomaly Score of Normal Cases: {name}'); plt.legend(loc='upper right')
    plt.xlabel('Value'); plt.ylabel('Frequency'); plt.show()
def plotROCAUC(score, name):
    fpr, tpr, thresholds1 = roc_curve(testLabels, score)
    auc = roc_auc_score(testLabels, score)
    plt.figure(); plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f}), seed={seed}')
    plt.title(f'Receiver Operating Characteristic Curve: {name}')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0]); plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.show()
def plot2ROCAUC(score1, score2, name1, name2):
    fpr1, tpr1, thresholds1 = roc_curve(testLabels, score1)
    fpr2, tpr2, thresholds2 = roc_curve(testLabels, score2)
    auc1 = roc_auc_score(testLabels, score1)
    auc2 = roc_auc_score(testLabels, score2)

    plt.figure(); plt.title(f'Receiver Operating Characteristic Comparison')
    plt.plot(fpr1, tpr1, label=f'{name1} (AUC = {auc1:.4f})')
    plt.plot(fpr2, tpr2, label=f'{name2} (AUC = {auc2:.4f})')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0]); plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.show()
def plot3ROCAUC(score1, score2, score3, name1, name2, name3):
    fpr1, tpr1, thresholds1 = roc_curve(testLabels, score1)
    fpr2, tpr2, thresholds2 = roc_curve(testLabels, score2)
    fpr3, tpr3, thresholds3 = roc_curve(testLabels, score3)
    auc1 = roc_auc_score(testLabels, score1)
    auc2 = roc_auc_score(testLabels, score2)
    auc3 = roc_auc_score(testLabels, score3)

    plt.figure(); plt.title(f'Receiver Operating Characteristic Comparison')
    plt.plot(fpr1, tpr1, label=f'{name1} (AUC = {auc1:.4f})')
    plt.plot(fpr2, tpr2, label=f'{name2} (AUC = {auc2:.4f})')
    plt.plot(fpr3, tpr3, label=f'{name3} (AUC = {auc3:.4f})')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0]); plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.show()
def plotBarChart(variable, title, ylim):
    plt.figure(figsize=(12, 6)); plt.xticks(rotation=45, ha='right')
    plt.bar(cols, variable); plt.xlabel(f'Feature'); plt.ylabel(f'Reconstruction Loss')
    plt.title(f'{title}'); plt.tight_layout(); plt.ylim(*ylim); plt.show()
def displayAnomaly(num, label):
    labelDictionary = {"TP": TPReconstruction, "FN": FNReconstruction,"FP": FPReconstruction, "TN": TNReconstruction}
    if label not in labelDictionary:
        raise ValueError("Invalid type. Must be one of: TP, FN, FP, TN")
    data = labelDictionary[label]
    if num > len(data):
        raise IndexError(f"There are {len(data)} samples available for type '{label}', but {num} was input.")
    indices = np.random.choice(len(data), size=min(num, len(data)), replace=False)
    for j, idx in enumerate(indices):
        plotBarChart(data[idx], f"Sample {j+1} Reconstruction error for {label}: {idx}", ylim=(0, 2))

# --------------- Autoencoder class ---------------
# fixed seed for same model initialisation and training behaviour
tf.keras.utils.set_random_seed(seed=100)
class AutoEncoder(tf.keras.Model):
    def __init__(self, x_train):
        super().__init__()
        self.encoder = self.encoder_model()
        self.decoder = self.decoder_model()
        self.total_loss_tracker = keras.metrics.Mean(name="Total loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="Reconstruction loss")
        self.train_data = x_train
        self.loss_fn = tf.keras.losses.MeanSquaredLogarithmicError()
    @staticmethod
    def encoder_model():
        encoder_inputs = keras.Input(shape=(np.shape(trainData1)[1],))
        # shape is the number of dimensions and length of data
        encoded = layers.Dense(24, activation='tanh')(encoder_inputs)
        encoded = layers.Dense(18, activation='tanh')(encoded)
        latent_space = layers.Dense(12, activation='tanh')(encoded) # the latent space (output of the encoder)
        encoder = keras.Model(encoder_inputs, latent_space) # defines the model in the keras computational graph
        encoder.compile(optimizer='Adam', loss='mse') # the compile step of the keras model
        #encoder.summary()
        return encoder # returns the compiled model
    @staticmethod
    def decoder_model():
        decoder_inputs = keras.Input(shape=(12,)) # the inputs of the decoder
        decoded = layers.Dense(18, activation='tanh')(decoder_inputs)
        decoded = layers.Dense(24, activation='tanh')(decoded)
        output_space = layers.Dense(np.shape(trainData1)[1], activation='tanh')(decoded) # the output (decoded latent space, should be the inputs)
        decoder = keras.Model(decoder_inputs, output_space) # defining the decoder as a model
        decoder.compile(optimizer='Adam', loss='mse') # compiling the model
        #decoder.summary()
        return decoder # returning the compiled model
    # as we have two parts of the model, we have to define how the model should be trained to access the latent space
    def train_step(self, data):
        with tf.GradientTape() as tape:
            latent_output = self.encoder(data)
            reconstructed_data = self.decoder(latent_output)
            reconstruction_loss = tf.reduce_mean(self.loss_fn(data, reconstructed_data))
            total_loss = reconstruction_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {'Total loss': self.total_loss_tracker.result(),
                'Reconstruction loss': self.reconstruction_loss_tracker.result()}

# --------------- Data management ---------------
# import kaggle database with 0.172% anomalies
project_root = os.path.dirname(os.path.abspath(__file__))
kaggleData = pd.read_csv(os.path.join(project_root, "creditcard.csv"))

cols = np.array(kaggleData.columns.tolist(), dtype=object); cols = cols[cols != 'Class'] # get used columns
kaggleData = kaggleData.to_numpy() # pandas -> numpy array
kaggleNormal = kaggleData[kaggleData[:, -1] == 0] # normal
kaggleAnomaly = kaggleData[kaggleData[:, -1] == 1] # anomalous

contam = 1.0 # set the contamination level
testSplit = 0.2; trainSplit = 0.25; seed = 42 # split data into train/test sets
trainData, testData = train_test_split(kaggleNormal,test_size=testSplit,random_state=seed,shuffle=True)
trainData1, trainData2 = train_test_split(trainData,test_size=trainSplit,random_state=seed,shuffle=True)
trainAnomaly, testAnomaly = train_test_split(kaggleAnomaly,test_size=testSplit,random_state=seed,shuffle=True)
trainData2 = np.concatenate((trainData2, trainAnomaly)) # classifier mixed set
testData = np.concatenate((testData, testAnomaly)) # test mixed set

trainLabels1 = trainData1[:,-1]; trainLabels2 = trainData2[:,-1]; testLabels = testData[:,-1] # get labels
trainData1 = trainData1[:,:-1]; trainData2 = trainData2[:,:-1]; testData = testData[:,:-1] # drop class

# --------------- Data preprocessing ---------------
qt = QuantileTransformer(output_distribution="normal", random_state=0) # apply quantile transformer separately
trainData1 = qt.fit_transform(trainData1) # fit_transform() to training data
trainData2 = qt.transform(trainData2) # transform() to prevent data leakage
testData = qt.transform(testData)

zscoreScaler = StandardScaler() # z-score standardisation
minmaxScaler = MinMaxScaler(feature_range=(-1.0, 1.0)) # min-max feature scaling
trainData1 = zscoreScaler.fit_transform(trainData1); trainData1 = minmaxScaler.fit_transform(trainData1)
trainData2 = zscoreScaler.transform(trainData2); trainData2 = minmaxScaler.transform(trainData2)
testData = zscoreScaler.transform(testData); testData = minmaxScaler.transform(testData)

# --------------- Autoencoder training/testing ---------------
earlyStopping1 = EarlyStopping( # define early stopping parameters
    monitor='Total loss',       # metric to monitor
    patience=10,                # number of epochs with no improvement
    min_delta=1e-4,             # require at least 0.0001 improvement
    restore_best_weights=True,  # restore weights from the best epoch
    mode='min')

AutoEncoder = AutoEncoder(trainData1) # train the autoencoder
AutoEncoder.compile(keras.optimizers.Adam(learning_rate=0.001))
AutoEncoder.fit(trainData1, epochs=50, batch_size=250, callbacks=[earlyStopping1])

latentSpaceTrain1 = AutoEncoder.encoder.predict(trainData1) # latent space after encoder
outputDataTrain1 = AutoEncoder.decoder.predict(latentSpaceTrain1) # output results after decoder
reconstructionLossTrain1 = np.mean(np.square(trainData1 - outputDataTrain1), axis=1)
threshold1 = np.percentile(reconstructionLossTrain1, 100-contam) # calculate a threshold on normal data

latentSpaceTest = AutoEncoder.encoder.predict(testData) # latent space for unseen test data
outputDataTest = AutoEncoder.decoder.predict(latentSpaceTest) # outputs for test data
reconstructionLossTest = np.square(testData - outputDataTest) # reconstruction error for each feature

# --------------- Contribution per feature to anomaly score ---------------
latentSpaceTrain2 = AutoEncoder.encoder.predict(trainData2) # latent space for mixed train set
outputDataTrain2 = AutoEncoder.decoder.predict(latentSpaceTrain2)
reconstructionLossTrain2 = np.square(trainData2 - outputDataTrain2) # reconstruction error for each feature
normalLossTrain2 = np.mean(reconstructionLossTrain2[np.where(trainLabels2 == 0)[0]], axis=0) # normal loss
fraudLossTrain2 = np.mean(reconstructionLossTrain2[np.where(trainLabels2 == 1)[0]], axis=0) # fraud loss

weights = fraudLossTrain2 - normalLossTrain2 # difference between normal/fraud reconstruction errors
weights = np.maximum(weights, 0) # don't include cases with worse normal data reconstruction
weights = weights / weights.sum() # normalise weightings

autoencoderScore1 = np.mean(reconstructionLossTest, axis=1).reshape(-1,1) # all features contribute equal
autoencoderScore2 = np.sum(reconstructionLossTest * weights, axis=1).reshape(-1,1) # weighted contributions

autoencoderScore = autoencoderScore2 # relative contributions
# if a test data point has higher reconstruction error than the threshold, it is output as an anomaly
autoencoderPrediction = autoencoderScore > threshold1

# ---------- Autoencoder-classifier ensemble model ----------
# define the binary classification neural network model
classifier = keras.Sequential([
    layers.Input(shape=(latentSpaceTrain2.shape[1],)), # input layer size of latent space
    layers.Dense(11, activation='elu'),
    layers.Dense(8, activation='elu'),
    layers.Dense(6, activation='elu'),
    layers.Dense(4, activation='elu'),
    layers.Dense(3, activation='elu'),
    layers.Dense(2, activation='elu'),
    layers.Dense(1, activation='sigmoid') # output layer for classifier - 0/1
])

earlyStopping2 = EarlyStopping( # define early stopping parameters
    monitor='loss',             # metric to monitor
    patience=5,                 # number of epochs with no improvement
    min_delta=1e-4,             # require at least 0.0001 improvement
    restore_best_weights=True,  # restore weights from the best epoch
    mode='min'
)
classifier.compile(keras.optimizers.Adam(learning_rate=0.002),loss='binary_crossentropy') # train the classifier
classifier.fit(latentSpaceTrain2, trainLabels2, epochs=50, batch_size=250, callbacks=[earlyStopping2])
ensembleScoreTrain = classifier.predict(latentSpaceTrain2) # classifier outputs probability on [0,1]
threshold2 = np.percentile(ensembleScoreTrain, 100-contam) # calculate the threshold on train data

ensembleScore = classifier.predict(latentSpaceTest) # evaluate on the unseen encoded test set
ensemblePrediction = (ensembleScore >= threshold2).astype(int) # apply the anomaly threshold

# --------------- Hybrid approach ---------------
minmaxScaler = MinMaxScaler(feature_range=(0, 1.0)) # rescale autoencoder predictions to fit ensemble
autoencoderScore = minmaxScaler.fit_transform(autoencoderScore)

aucScore = []; weight = []; n = 100 # find the best combination of each model
for i in range(n+1):
    hybridScore = (i/n)*autoencoderScore + (1-i/n)*ensembleScore
    aucScore.append(roc_auc_score(testLabels, hybridScore))
    weight.append(i/n)
index = np.argmax(aucScore) # find index of highest-scoring model weighting

hybridScore = weight[index]*autoencoderScore + (1-weight[index])*ensembleScore # calculate hybrid score
threshold3 = weight[index]*threshold1 + (1-weight[index])*threshold2 # calculate the hybrid threshold
print(f"Hybrid Threshold:{threshold3}")
hybridPrediction = (hybridScore >= threshold3).astype(int) # apply the threshold

# --------------- Evaluate model with performance metrics ---------------
performanceMetrics(hybridPrediction, "Hybrid", contam)
plot2ROCAUC(autoencoderScore1, autoencoderScore2, "Equal", "Relative")
plot3ROCAUC(autoencoderScore, ensembleScore, hybridScore, "Autoencoder", "Ensemble", "Hybrid")
#plotFraudCases(hybridScore, threshold3, "Hybrid Model")

# --------------- Calculate indices of data points ---------------
testLabels = testLabels.reshape(-1,1) # could make this into a function
TP = (testLabels == 1) & (hybridPrediction == 1); TPIndices = np.where(TP)[0]
TPReconstruction = reconstructionLossTest[TPIndices]; TPMean = np.mean(TPReconstruction, axis=0)
FN = (testLabels == 1) & (hybridPrediction == 0); FNIndices = np.where(FN)[0]
FNReconstruction = reconstructionLossTest[FNIndices]; FNMean = np.mean(FNReconstruction, axis=0)
FP = (testLabels == 0) & (hybridPrediction == 1); FPIndices = np.where(FP)[0]
FPReconstruction = reconstructionLossTest[FPIndices]; FPMean = np.mean(FPReconstruction, axis=0)
TN = (testLabels == 0) & (hybridPrediction == 0); TNIndices = np.where(TN)[0]
TNReconstruction = reconstructionLossTest[TNIndices]; TNMean = np.mean(TNReconstruction, axis=0)

normIndices = np.where((testLabels == 0))[0]; anomIndices = np.where((testLabels == 1))[0]
normReconstruction = reconstructionLossTest[normIndices]; normalMean = np.mean(normReconstruction, axis=0)
anomReconstruction = reconstructionLossTest[anomIndices]; anomalyMean = np.mean(anomReconstruction, axis=0)

# --------------- plotting distributions ---------------
# plot line graph of hybrid weighting against AUC
plt.plot(weight, aucScore, label=f'Best AUC: {aucScore[index]:.4f} at weight: {weight[index]:.2f}')
plt.text(0, aucScore[0], 'Ensemble', ha='left', va='bottom')
plt.text(1, aucScore[-1], 'Autoencoder', ha='right', va='bottom')
plt.title('Weighting Effect on the Hybrid AUC Score'); plt.legend()
plt.xlabel('Model Weighting'); plt.ylabel('AUC Score'); plt.show()

# reconstruction of normal vs. anomalous points
plt.figure(figsize=(12, 6)); plt.xticks(rotation=45, ha='right')
plt.bar(cols, normalMean, color='blue', alpha=0.9, label='Normal')
plt.bar(cols, anomalyMean, color='red', alpha=0.5, label='Anomalous')
plt.xlabel(f'Feature'); plt.ylabel(f'Reconstruction Loss'); plt.legend(loc='upper right')
plt.title(f'Mean Reconstruction Errors for Normal vs Anomalous Data'); plt.tight_layout(); plt.ylim(0,0.75); plt.show()

# plot average reconstruction errors for different points
#plotBarChart(TPMean, "Autoencoder True Positives Reconstruction Errors", ylim=(0, 2))
#plotBarChart(FNMean, "Autoencoder False Negatives Reconstruction Errors", ylim=(0, 2))
#plotBarChart(FPMean, "Autoencoder False Positives Reconstruction Errors", ylim=(0, 2))
#plotBarChart(TNMean, "Autoencoder True Negatives Reconstruction Errors", ylim=(0, 2))

# plot case-by-case data points
#displayAnomaly(2, "TP"); displayAnomaly(2, "FN")
#displayAnomaly(2, "FP"); displayAnomaly(2, "TN")