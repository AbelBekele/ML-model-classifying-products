import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
import mlflow
import mlflow.tensorflow
import time
import os

# Load data
data = pd.read_csv('../notebooks/content/amazon.csv')

# Select specific columns
selected_data = data[['product_name', 'about_product', 'actual_price', 'category']]

# Text Augmentation
def augment_text(text):
    # Adding a random word to the beginning of the sentence
    augmented_text = ' ' + text + ' augmented'
    return augmented_text

# Apply text augmentation
selected_data['about_product'] = selected_data['about_product'].apply(augment_text)

# Tokenization and encoding
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(selected_data['about_product'])
sequences = tokenizer.texts_to_sequences(selected_data['about_product'])
padded_sequences = pad_sequences(sequences, padding='post')
labels = selected_data['category']

# Encoding categorical labels
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2)

def create_model(embedding_dim, lstm_units, dense_units):
    # Define the model
    model = tf.keras.Sequential([
        Embedding(10000, embedding_dim, input_length=padded_sequences.shape[1]),
        LSTM(lstm_units, return_sequences=True),
        GlobalAveragePooling1D(),
        Dense(dense_units, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(len(set(labels)), activation='softmax')
    ])

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=AdamW(learning_rate=0.001), metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_test, y_test, embedding_dim=128, lstm_units=64, dense_units=128, epochs=20):
    # Create and train the model
    model = create_model(embedding_dim, lstm_units, dense_units)

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=2, callbacks=[early_stopping, lr_reduction])

    return model, history

def save_model(model, model_dir="saved_model"):
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    model.save(os.path.join(model_dir, "trained_model"))

def load_model(model_dir="saved_model"):
    # Load the model
    model = tf.keras.models.load_model(os.path.join(model_dir, "trained_model"))
    return model

def interact_with_model(model, tokenizer, text):
    # Tokenize input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=model.input_shape[1], padding='post')

    # Make predictions
    prediction = model.predict(padded_sequence)

    # Decode prediction
    predicted_label = encoder.inverse_transform([prediction.argmax()])[0]

    return predicted_label

if __name__ == "__main__":
    # Enable auto logging
    mlflow.tensorflow.autolog()

    # Start a new run
    with mlflow.start_run():
        # Hyperparameter tuning (replace with your search strategy)
        embedding_dim = 128  # Experiment with different values
        lstm_units = 64  # Experiment with different values
        dense_units = 128  # Experiment with different values

        # Train the model
        trained_model, history = train_model(X_train, y_train, X_test, y_test, embedding_dim, lstm_units, dense_units)

        # Evaluate the model
        loss, accuracy = trained_model.evaluate(X_test, y_test)
        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}')

        # Log metrics
        mlflow.log_metric("loss", loss)
        mlflow.log_metric("accuracy", accuracy)

        # Save the model separately
        save_model(trained_model)

        # Test interaction with the model
        test_text = "This product is amazing"
        loaded_model = load_model()
        predicted_category = interact_with_model(loaded_model, tokenizer, test_text)
        print(f"Predicted category for '{test_text}': {predicted_category}")
