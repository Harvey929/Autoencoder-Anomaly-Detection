import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.decomposition import PCA
import seaborn as sns

# --------------- Define functions ---------------
def performanceMetrics(labels, prediction):
    accuracy = 100 * accuracy_score(labels, prediction); print(f"Accuracy: {accuracy}")
    precision = 100 * precision_score(labels, prediction); print(f"Precision: {precision}")
    recall = 100 * recall_score(labels, prediction); print(f"Recall: {recall}")
    f1Score = 100 * f1_score(labels, prediction); print(f"F1 score: {f1Score}")
    tn, fp, fn, tp = confusion_matrix(labels, prediction).ravel()
    print(f"True Negatives (TN): {tn}"); print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}"); print(f"True Positives (TP): {tp}")
    return tn, fp, fn, tp
def plotROCAUC(score, name):
    fpr, tpr, thresholds1 = roc_curve(testLabels, score)
    auc = roc_auc_score(testLabels, score)
    plt.figure(); plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.title(f'Receiver Operating Characteristic Curve: {name}')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0]); plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.show()
def plotBarChart(variable, title, ylim):
    plt.figure(figsize=(12, 6)); plt.xticks(rotation=45, ha='right')
    plt.bar(cols, variable); plt.xlabel(f'Feature'); plt.ylabel(f'Reconstruction Loss')
    plt.title(f'{title}'); plt.tight_layout(); plt.ylim(*ylim); plt.show()
def plotScores(score, threshold, bins, title):
    plt.hist(score, bins=bins, color='blue')
    plt.axvline(x=threshold, color='orange', label=f'Threshold: {threshold:.2f}')
    plt.title(f'{title}'); plt.legend(loc='upper right')
    plt.xlabel('Value'); plt.ylabel('Frequency'); plt.show()

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
        encoder_inputs = keras.Input(shape=(trainData.shape[1],))
        # shape is the number of dimensions and length of data
        encoded = layers.Dense(24, activation='tanh')(encoder_inputs)
        encoded = layers.Dense(18, activation='tanh')(encoded)
        latent_space = layers.Dense(12, activation='tanh')(encoded) # the latent space (output of the encoder)
        encoder = keras.Model(encoder_inputs, latent_space) # defines the model in the keras computational graph
        return encoder # returns the compiled model
    @staticmethod
    def decoder_model():
        decoder_inputs = keras.Input(shape=(12,)) # the inputs of the decoder
        decoded = layers.Dense(18, activation='tanh')(decoder_inputs)
        decoded = layers.Dense(24, activation='tanh')(decoded)
        output_space = layers.Dense(np.shape(trainData)[1], activation='tanh')(decoded) # the output (decoded latent space, should be the inputs)
        decoder = keras.Model(decoder_inputs, output_space) # defining the decoder as a model
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

cols = np.array(kaggleData.columns.tolist(), dtype=object); cols = cols[cols != 'Class'] # get columns
kaggleData = kaggleData.to_numpy() # pandas -> numpy array
kaggleNormal = kaggleData[kaggleData[:, -1] == 0] # normal
kaggleAnomaly = kaggleData[kaggleData[:, -1] == 1] # anomalous

normalSplit = 0.2; anomalySplit = 0.2; seed = 42 # split data into train/test
trainData, testData = train_test_split(kaggleNormal,test_size=normalSplit,random_state=seed,shuffle=True)
anomaly1, anomaly2 = train_test_split(kaggleAnomaly,test_size=anomalySplit,random_state=seed,shuffle=True)
testData = np.concatenate((testData, anomaly2)) # add anomalies to test data
trainLabels = trainData[:,-1]; testLabels = testData[:,-1] # get labels
trainData = trainData[:,:-1]; testData = testData[:,:-1] # drop class

# --------------- Data preprocessing ---------------
qt = QuantileTransformer(output_distribution="normal", random_state=0) # apply quantile transformer
trainData = qt.fit_transform(trainData) # fit_transform() to training data
testData = qt.transform(testData) # transform() to prevent data leakage
# inverse quantile transformer with: data = qt.inverse_transform(transformedData)

zscoreScaler = StandardScaler() # z-score standardisation
minmaxScaler = MinMaxScaler(feature_range=(-1.0, 1.0)) # min max feature scaler
trainData = zscoreScaler.fit_transform(trainData); trainData = minmaxScaler.fit_transform(trainData)
testData = zscoreScaler.transform(testData); testData = minmaxScaler.transform(testData)

# --------------- Autoencoder training/testing ---------------
# define early stopping parameters
earlyStopping = EarlyStopping(
    monitor='Total loss', # metric to monitor
    patience=10, # number of epochs with no improvement before stopping
    min_delta=1e-4, # requires at least 0.0001 improvement
    restore_best_weights=True, # restores model weights from the best epoch
    mode='min'
)
AutoEncoder = AutoEncoder(trainData) # train the autoencoder on clean dataset
AutoEncoder.compile(keras.optimizers.Adam(learning_rate=0.001))
AutoEncoder.fit(trainData, epochs=50, batch_size=250, callbacks=[earlyStopping])

latentSpaceTrain = AutoEncoder.encoder.predict(trainData) # latent space results after encoder
outputDataTrain = AutoEncoder.decoder.predict(latentSpaceTrain) # output results after decoder
reconstructionLossTrain = np.square(trainData - outputDataTrain) # reconstruction by feature

latentSpaceTest = AutoEncoder.encoder.predict(testData) # latent space on unseen testing data
outputDataTest = AutoEncoder.decoder.predict(latentSpaceTest) # outputs for testing data
reconstructionLossTest = np.square(testData - outputDataTest) # reconstruction for each feature separate

reconstructionSplit = np.mean(reconstructionLossTrain, axis=0)
#plotBarChart(reconstructionSplit, "Reconstruction loss of training data")
weights = 1 / (reconstructionSplit + 1e-6); weights = weights / weights.sum()
autoencoderScore = np.mean(reconstructionLossTest, axis=1).reshape(-1,1) # equal contribution
#autoencoderScore = np.sum(reconstructionLossTest * weights, axis=1).reshape(-1,1) # relative contribution

# --------------- Evaluate model with performance metrics ---------------
reconstructionLossTrain = np.mean(reconstructionLossTrain, axis=1)
threshold1 = np.percentile(reconstructionLossTrain, 99.0) # assume a 1% contamination
print(f"Threshold:{threshold1}")
# if a test data point has higher reconstruction error than the threshold, it is output as an anomaly
autoencoderPrediction = autoencoderScore > threshold1 # apply the anomaly threshold
print(f"Anomalies predicted: {np.sum(autoencoderPrediction == 1)}")

performanceMetrics(testLabels, autoencoderPrediction) # output metrics
plotROCAUC(autoencoderScore, "Autoencoder")
#plotScores(autoencoderScore[np.where(testLabels == 1)[0]], threshold1,
# len(testLabels[testLabels == 1]), "Reconstruction loss of fraud cases") # plot score of fraud cases

# --------------- Calculate indices of points ---------------
testLabels = testLabels.reshape(-1,1)
TP = (testLabels == 1) & (autoencoderPrediction == 1); TPIndices = np.where(TP)[0]
TPReconstruction = reconstructionLossTest[TPIndices]; TPMean = np.mean(TPReconstruction, axis=0)
FN = (testLabels == 1) & (autoencoderPrediction == 0); FNIndices = np.where(FN)[0]
FNReconstruction = reconstructionLossTest[FNIndices]; FNMean = np.mean(FNReconstruction, axis=0)
FP = (testLabels == 0) & (autoencoderPrediction == 1); FPIndices = np.where(FP)[0]
FPReconstruction = reconstructionLossTest[FPIndices]; FPMean = np.mean(FPReconstruction, axis=0)
TN = (testLabels == 0) & (autoencoderPrediction == 0); TNIndices = np.where(TN)[0]
TNReconstruction = reconstructionLossTest[TNIndices]; TNMean = np.mean(TNReconstruction, axis=0)

#plotBarChart(TPMean, "Reconstruction error for true positives")
#plotBarChart(FNMean, "Reconstruction error for false negatives")
#plotBarChart(FPMean, "Reconstruction error for false positives")
#plotBarChart(TNMean, "Reconstruction error for true negatives")

normIndices = np.where((testLabels == 0))[0]; anomIndices = np.where((testLabels == 1))[0]
normReconstruction = reconstructionLossTest[normIndices]; normMean = np.mean(normReconstruction, axis=0)
anomReconstruction = reconstructionLossTest[anomIndices]; anomMean = np.mean(anomReconstruction, axis=0)

# --------------- plotting distributions ---------------
# plot a 2d PCA transformation of autoencoder latent space
pca = PCA(n_components=2); latent_2d = pca.fit_transform(latentSpaceTest)
df = pd.DataFrame(latent_2d, columns=["PC1", "PC2"])
df["Class"] = testLabels
df["Class"] = df["Class"].map({0: "Normal", 1: "Anomaly"})
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="Class", palette={"Normal": "#1f77b4", "Anomaly": "#d62728"})
plt.legend(title="Class", loc="lower right"); plt.tight_layout()
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
plt.title("PCA Projection of the 12 Dimensional Latent Space")
plt.xlim(-1, 1); plt.ylim(-1, 1); plt.show()

# plot reconstruction errors of each feature
plt.figure(figsize=(12, 6)); plt.xticks(rotation=45, ha='right')
plt.bar(cols, normMean, color='blue', alpha=0.9, label='Normal')
plt.bar(cols, anomMean, color='orange', alpha=0.5, label='Anomalous')
plt.xlabel(f'Feature'); plt.ylabel(f'Reconstruction Loss'); plt.legend(loc='upper right')
plt.title(f'Mean Reconstruction Errors for Normal vs Anomalous Data: Clean Training'); plt.tight_layout(); plt.ylim(0,1.5); plt.show()
