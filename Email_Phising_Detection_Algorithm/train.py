import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer 
from nltk.stem.snowball import SnowballStemmer
from keras.regularizers import l2

# Load and preprocess dataset
phish_data = pd.read_csv('phishing_site_urls.csv')

# Tokenize and stem the URLs
tokenizer_regexp = RegexpTokenizer(r'[A-Za-z]+')
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    tokens = tokenizer_regexp.tokenize(text)
    return ' '.join(stemmer.stem(token) for token in tokens)

phish_data['text_sent'] = phish_data['URL'].apply(tokenize_and_stem)

# Keras Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(phish_data['text_sent'])
sequences = tokenizer.texts_to_sequences(phish_data['text_sent'])

# Pad sequences
max_sequence_length = 300
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Labels
labels = phish_data['Label'].map({'bad': 1, 'good': 0}).values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Load GloVe embeddings
embeddings_index = {}
with open('glove.6B/glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Model configuration
additional_metrics = ['accuracy']
batch_size = 128
loss_function = BinaryCrossentropy()
number_of_epochs = 5
validation_split = 0.20
verbosity_mode = 1

# Model with GloVe Embedding Layer
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_sequence_length,
                    trainable=False))
model.add(LSTM(10))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))

# Compile the model
model.compile(optimizer='adam', loss=loss_function, metrics=additional_metrics)

# Train the model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=number_of_epochs, verbose=verbosity_mode, validation_split=validation_split)

# Save the model
model.save('phishing_detection_model.h5')

# Plot accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model
test_results = model.evaluate(X_test, y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}%')
