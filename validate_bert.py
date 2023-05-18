import tensorflow as tf
import tensorflow_hub
import tensorflow_text
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import matplotlib.pyplot as plt

# Load the dataset into a pandas dataframe.
dataframe = pd.read_csv("yahoo_validation.csv")

# Apply one-hot encoding to the sentiment labels (positive, neutral, negative).
encoder = LabelEncoder()
labelEncode = encoder.fit_transform(dataframe["sentiment"])
categoryEncode = tf.keras.utils.to_categorical(labelEncode)

# Concatenate the old dataframe with the new encoded labels.
encodedDF = pd.DataFrame(categoryEncode, columns = encoder.classes_, index = dataframe["sentence"].index)
dataframe.drop(columns = ["sentiment"], inplace = True)
dataframe = pd.concat([dataframe, encodedDF], axis = 1)

# Seperate the data from the label.
validData = dataframe["sentence"]
labels = dataframe.iloc[:, 1:]
validLabels = []
for i in labels.index:
	# Assigning 0 for negative, 1 for neutral, and 2 for positive.
	# This will make it easier to create a confusion matrix.
	validLabels.append(np.argmax(labels[i:i + 1]))

# Load the pre-trained BERT model that was created.
BERTModel = tf.keras.models.load_model("financial_sentiment.model")

# Create predictions using the validation data from Yahoo Finance.
predictions = BERTModel.predict(validData)

# Evaluate the predictions.
predictions = np.argmax(predictions, axis = 1) # for finding the predicted label class.
labelClass = encoder.classes_

# Create the confusion matrix.
confusionMatrix = metrics.confusion_matrix(validLabels, predictions)
#print(confusionMatrix)

# Plot the confusion matrix.
fig, ax = plt.subplots()
ax.matshow(confusionMatrix)

for (i, j), z in np.ndenumerate(confusionMatrix):
    ax.text(j, i, "{}".format(z), ha = "center", va = "center", color = "red", fontsize = "large", fontweight = "normal")

plt.title("BERT Confusion Matrix")
plt.savefig("BERT_matrix.png")
