# model.py
import os
import tensorflow as tf
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW

def create_model(embedding_dim, lstm_units, dense_units, vocab_size, num_classes):
    model = tf.keras.Sequential([
        Embedding(vocab_size, embedding_dim, input_length=padded_sequences.shape[1]),
        LSTM(lstm_units, return_sequences=True),
        GlobalAveragePooling1D(),
        Dense(dense_units, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=AdamW(learning_rate=0.001), metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs):
    # Training steps
    return trained_model, history

def save_model(model, model_dir="saved_model"):
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "trained_model"))

def load_model(model_dir="saved_model"):
    model = tf.keras.models.load_model(os.path.join(model_dir, "trained_model"))
    return model
