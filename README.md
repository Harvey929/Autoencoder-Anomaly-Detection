# Autoencoder-Anomaly-Detection
Code from my research project on anomaly detection using autoencoders, trained on the Kaggle credit card fraud dataset, and includes full preprocessing, model training, and evaluation.

autoencoder.py: Baseline autoencoder with 2-way train/test split
autoencoder2.py: Autoencoder with feature-wise reconstruction weighting using 3-way split
binaryClassifier.py: Standalone binary classification model
ensembleMethod.py: Classifier on AE latent space, trained on mixed data
ensembleMethod2.py: Latent classifier using clean AE and feature weighting via 3-way split
hybridModel.py: Hybrid score combining AE and classifier predictions
variationalAutoencoder.py: Alternative hybrid using variational autoencoder

Important AUC Results:
'autoencoder2.py' achieves AUC = 0.9908 from the autoencoder alone. 'ensembleMethod2.py' achieves AUC = 0.9890 using the binary classifier on the latent space representation. 'hybridModel.py' combines the results of these approaches to achieve AUC = 0.9932.
