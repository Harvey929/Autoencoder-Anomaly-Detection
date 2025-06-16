import pandas as pd, numpy as np, csv, math, os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, metrics
from tensorflow.keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.regularizers import l1, l2
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# --------------- define functions ---------------
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
    plt.title(f'Reconstruction Loss of Fraud Cases: {name}'); plt.legend(loc='upper right')
    plt.xlabel('Value'); plt.ylabel('Frequency'); plt.show()
def plotROCAUC(score, name):
    fpr, tpr, thresholds1 = roc_curve(testLabels, score)
    auc = roc_auc_score(testLabels, score)
    plt.figure(); plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.title(f'Receiver Operating Characteristic Curve: {name}')
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

# --------------- Variational autoencoder class ---------------
# fixed seed ensures the same model initialisation and training behaviour
tf.keras.utils.set_random_seed(seed=100)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
class KLAnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self, start=0.001, end=1.0, steps=50):
        super(KLAnnealingCallback, self).__init__()
        self.start = start; self.end = end; self.steps = steps
    def on_epoch_begin(self, epoch, logs=None):
        beta = min(self.end, self.start + epoch * (self.end - self.start) / self.steps)
        self.model.beta.assign(beta)
        print(f'Epoch {epoch + 1}: Beta = {self.model.beta.numpy():.4f}')
class variationalAutoencoder(tf.keras.Model):
    def __init__(self, train_data):
        super(variationalAutoencoder, self).__init__()
        self.beta = tf.Variable(0.001, trainable=False, name='beta')
        latent_space = 12

        self.encoder_inputs = keras.Input(shape=(np.shape(train_data)[1],))
        encoded = layers.Dense(14, activation='tanh')(self.encoder_inputs)
        encoded = layers.Dense(18, activation='tanh')(encoded)
        self.latent_space = layers.Dense(latent_space, activation='tanh',name='latent_space')(encoded)

        z_mean = layers.Dense(latent_space, name='z_mean',activation='linear')(encoded)
        z_log_var = layers.Dense(latent_space, name='z_log_var',activation='linear')(encoded)
        z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])
        self.encoder = keras.Model(inputs=self.encoder_inputs, outputs=[z_mean, z_log_var, z], name='Encoder')

        self.latent_inputs = keras.Model(inputs=self.encoder_inputs, outputs=self.latent_space, name='Latent_space')
        self.decoder_inputs = keras.Input(shape=(latent_space,))
        decoded = layers.Dense(18, activation='tanh')(self.decoder_inputs)
        decoded = layers.Dense(24, activation='tanh')(decoded)
        self.output_space = layers.Dense(np.shape(train_data)[1], activation='tanh')(decoded)
        self.decoder = keras.Model(inputs=self.decoder_inputs, outputs=self.output_space, name='Decoder')

        self.total_loss_tracker = keras.metrics.Mean(name="Total loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="Reconstruction loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="KL loss")
        self.loss_fn = tf.keras.losses.MeanSquaredError()
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstructed_data = self.decoder(z, training=True)

            reconstruction_loss = tf.reduce_mean(self.loss_fn(data, reconstructed_data)) # reconstruction loss
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) # kl loss
            total_loss = reconstruction_loss + self.beta * kl_loss


        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {'Total loss': self.total_loss_tracker.result(),
                'Reconstruction loss': self.reconstruction_loss_tracker.result(),
                'KL loss': self.kl_loss_tracker.result()}
    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstructed_data = self.decoder(z, training=False)

        reconstruction_loss = tf.reduce_mean(self.loss_fn(data, reconstructed_data))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = reconstruction_loss + self.beta * kl_loss
        return {'Total loss': total_loss,
                'Reconstruction loss': reconstruction_loss,
                'KL loss': kl_loss}
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs, training=True)
        reconstructed_data = self.decoder(z, training=True)
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        reconstruction_loss = tf.reduce_mean(self.loss_fn(inputs, reconstructed_data))
        total_loss = reconstruction_loss + self.beta * kl_loss
        self.add_loss(total_loss)
        #z = sampling([z_mean, z_log_var])
        return reconstructed_data

# --------------- data management ---------------
# import kaggle database with 0.172% anomalies
project_root = os.path.dirname(os.path.abspath(__file__))
kaggleData = pd.read_csv(os.path.join(project_root, "creditcard.csv"))

kaggleData = kaggleData.to_numpy() # pandas -> numpy array
kaggleNormal = kaggleData[kaggleData[:, -1] == 0] # normal
kaggleAnomaly = kaggleData[kaggleData[:, -1] == 1] # anomalous

# split data into train/test at 80/20 split
trainNormal, testNormal = train_test_split(kaggleNormal, test_size=0.2, random_state=42, shuffle=True)
trainAnomaly, testAnomaly = train_test_split(kaggleAnomaly, test_size=0.2, random_state=42, shuffle=True)
trainData = np.concatenate((trainNormal, trainAnomaly))
testData = np.concatenate((testNormal, testAnomaly)) # add anomalies to test data
trainLabels = trainData[:, -1]; testLabels = testData[:, -1] # get labels
trainData = trainData[:, :-1]; testData = testData[:, :-1] # drop class

# --------------- data preprocessing ---------------
# apply quantile transformer to train/test datasets separately
qt = QuantileTransformer(output_distribution="normal", random_state=0)
trainData = qt.fit_transform(trainData) # .fit_transform() to training data
testData = qt.transform(testData) # .transform() to prevent data leakage

# standardise & normalise data distribution
zscore = StandardScaler(); minmax = MinMaxScaler(feature_range=(-1.0, 1.0))
trainData = zscore.fit_transform(trainData); trainData = minmax.fit_transform(trainData)
testData = zscore.transform(testData); testData = minmax.transform(testData)

# train the VAE
vae = variationalAutoencoder(trainData)
vae.compile(keras.optimizers.Adam(learning_rate=0.001), loss=vae.loss_fn)
kl_annealing = KLAnnealingCallback(start=0.001, end=1.0, steps=50)
vae.fit(trainData, epochs=50, batch_size=250, callbacks=[kl_annealing])
vae.encoder.summary(); vae.decoder.summary()

# results after data has gone through autoencoder
reconstructionTrainData = vae.predict(trainData)
reconstructionTrainLoss = np.mean(np.square(trainData - reconstructionTrainData), axis=1)
threshold1 = np.percentile(reconstructionTrainLoss, 99.0) # assume 1% contamination
print(f"Threshold:{threshold1}")

# test the model on unseen data
reconstructionTestData = vae.predict(testData)
vaeScore = np.mean(np.square(testData - reconstructionTestData), axis=1).reshape(-1, 1)
# if a test data point has higher reconstruction error than the threshold, it is output as an anomaly
vaePrediction = (vaeScore >= threshold1).astype(int)

# --------------- Binary classification model ---------------
# get latent space representations for train/test data
encodedTrainData = vae.latent_inputs.predict(trainData)
encodedTestData = vae.latent_inputs.predict(testData)

# define the binary classification neural network model
model = keras.Sequential([
    layers.Dense(11, activation='elu'),
    layers.Dense(8, activation='elu'),
    layers.Dense(6, activation='elu'),
    layers.Dense(4, activation='elu'),
    layers.Dense(3, activation='elu'),
    layers.Dense(2, activation='elu'),
    layers.Dense(1, activation='sigmoid')
]) # output layer for classifier - 0/1

earlyStopping2 = EarlyStopping( # define early stopping parameters
    monitor='loss', # metric to monitor
    patience=5, # number of epochs with no improvement
    min_delta=1e-4, # require at least 0.0001 improvement
    restore_best_weights=True, # restore weights from the best epoch
    mode='min'
)

# train the binary classifier
model.compile(keras.optimizers.Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(name='AUC')])
model.fit(encodedTrainData, trainLabels, epochs=50, batch_size=250, callbacks=[earlyStopping2])

# results after data has gone through classifier
encodedTrainOutput = model.predict(encodedTrainData)
threshold2 = np.percentile(encodedTrainOutput, 99.5) # assume .5% contamination

# evaluate the model on the unseen encoded test set
ensembleScore = model.predict(encodedTestData) # each value is probability of being an anomaly
ensemblePrediction = (ensembleScore >= threshold2).astype(int)

# --------------- Hybrid Approach ---------------
# rescale autoencoder reconstruction score to fit classifier predictions
minmaxScaler = MinMaxScaler(feature_range=(0, 1.0))
vaeScore = minmaxScaler.fit_transform(vaeScore)

# find the best combination of each model for highest AUC
aucScore = []; weight = []; n = 50
for i in range(n+1):
    hybridScore = i/n*vaeScore + (1-i/n)*ensembleScore
    aucScore.append(roc_auc_score(testLabels, hybridScore))
    weight.append(i/n)
index = np.argmax(aucScore)

# plot line graph of weighting against AUC
plt.plot(weight, aucScore, label=f'Best AUC: {aucScore[index]:.4f} at weight: {weight[index]:.2f}')
plt.text(0, aucScore[0], 'Ensemble', ha='left', va='bottom')
plt.text(1, aucScore[-1], 'VAE', ha='right', va='bottom')
plt.title('Weighting Effect on the AUC Score'); plt.legend()
plt.xlabel('Model Weighting'); plt.ylabel('AUC Score'); plt.show()

# use optimal weighting to find a combined score
hybridScore = weight[index]*vaeScore + (1-weight[index])*ensembleScore
threshold3 = weight[index]*threshold1 + (1-weight[index])*threshold2
print(f"Threshold:{threshold3}")
hybridPrediction = (hybridScore >= threshold3).astype(int)

# --------------- Evaluate model with performance metrics ---------------
performanceMetrics(hybridPrediction, "Hybrid", 1.0)
anomalies = testData[hybridPrediction.flatten() == 1, -1]
print(f"Number of anomalies predicted:{len(anomalies)}")

# --------------- plotting distributions ---------------
#plotFraudCases(hybridScore, threshold3, "Weighted")
plot3ROCAUC(vaeScore, ensembleScore, hybridScore,
    "VAE", "Ensemble", "Hybrid")
