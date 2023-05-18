import tensorflow as tf
import tensorflow_hub
import tensorflow_text
import pandas as pd
import numpy as np

from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt



### Data Preprocessing ###

# Load the dataset into a pandas dataframe.
dataframe = pd.read_csv("financial_phrasebank.csv")
#print(dataframe.head(5))

# Apply one-hot encoding to the sentiment labels (positive, neutral, negative).
encoder = LabelEncoder()
labelEncode = encoder.fit_transform(dataframe["sentiment"])
categoryEncode = tf.keras.utils.to_categorical(labelEncode)

# Concatenate the old dataframe with the new encoded labels.
encodedDF = pd.DataFrame(categoryEncode, columns = encoder.classes_, index = dataframe["sentence"].index)
dataframe.drop(columns = ["sentiment"], inplace = True)
dataframe = pd.concat([dataframe, encodedDF], axis = 1)
#print(dataframe.head(5))

# Split the dataset into train and test.
xtrain, xtest, ytrain, ytest = train_test_split(dataframe["sentence"], dataframe.iloc[:, 1:], 
												test_size = 0.2, random_state = 42, shuffle = False)



### The BERT Model Implementation ###

# Selecting a BERT model from tensorflow hub.
BERTEncoder = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/2"
preprocessModel = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
BERTPreprocess = tensorflow_hub.KerasLayer(preprocessModel)

# Test example with the preprocessing model.
'''
textTest = ["Circulation revenue has increased by 5% in Finland and 4% in Sweden in 2008."]
textPreprocessed = BERTPreprocess(textTest)
print(f"Sentence   : {textTest}")
print(f'Keys       : {list(textPreprocessed.keys())}')
print(f'Shape      : {textPreprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {textPreprocessed["input_word_ids"][0, :20]}')
print(f'Input Mask : {textPreprocessed["input_mask"][0, :20]}')
print(f'Type Ids   : {textPreprocessed["input_type_ids"][0, :20]}')
'''

# Define the classifier model.
def classifierModel():
	# Preprocess the text input.
	textInput = tf.keras.layers.Input(shape = (), dtype = tf.string, name = "text")
	preprocessingLayer = tensorflow_hub.KerasLayer(preprocessModel, name = "preprocessing")
	encoderInput = preprocessingLayer(textInput)

	# Apply the BERT model.
	encoder = tensorflow_hub.KerasLayer(BERTEncoder, trainable = True, name = "BERT_encoder")
	output = encoder(encoderInput)

	# Pool the output for the entire sentence.
	network = output["pooled_output"]

	# Finally include dropout and dense layers.
	network = tf.keras.layers.Dropout(0.3)(network)
	network = tf.keras.layers.Dense(3, activation = "softmax", name = "financial_sentiment_classifier")(network)
	return tf.keras.Model(textInput, network)

# Assign the model.
BERTModel = classifierModel()
#tf.keras.utils.plot_model(BERTModel, to_file = "BERT_model.png")

# Define the training parameters.
epochs = 15
lr = 0.00002
batchSize = 32

# Define the optimizer for training the model.
optimizer = tf.keras.optimizers.Adam(learning_rate = lr, decay = (lr/epochs))

# Define the early stopping callback that prevents model over-fitting.
earlyStop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 4, restore_best_weights = True)

# Compile the model, ready for training.
BERTModel.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = "accuracy")

# Train the model.
train = BERTModel.fit(x = xtrain, y = ytrain, validation_data = (xtest, ytest),
							 epochs = epochs, batch_size = batchSize, callbacks = [earlyStop])

# Save the trained BERT model.
BERTModel.save("financial_sentiment.model")

# Plotting training history with matplotlib.
plt.style.use("ggplot")
plt.figure(figsize = (12, 12), dpi = 120)
plt.plot(np.arange(0, len(train.history["loss"])), train.history["loss"], label = "train_loss")
plt.plot(np.arange(0, len(train.history["val_loss"])), train.history["val_loss"], label = "validation_loss")
plt.plot(np.arange(0, len(train.history["accuracy"])), train.history["accuracy"], label = "train_accuracy")
plt.plot(np.arange(0, len(train.history["val_accuracy"])), train.history["val_accuracy"], label = "validation_accuracy")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend(loc = "best")
plt.savefig("model_training_history.png")
