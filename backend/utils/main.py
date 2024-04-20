# main.py
from data_utils import load_data, preprocess_data, split_data
from text_augmentation import augment_text
from model import create_model, train_model, save_model, load_model
from interaction import interact_with_model

# Load data
data = load_data('../notebooks/content/amazon.csv')

# Preprocess data
processed_data = preprocess_data(data)

# Split data
X_train, X_test, y_train, y_test = split_data(processed_data)

# Text augmentation
selected_data['about_product'] = selected_data['about_product'].apply(augment_text)

# Create and train the model
model = create_model(embedding_dim, lstm_units, dense_units, vocab_size, num_classes)
trained_model, history = train_model(model, X_train, y_train, X_test, y_test, epochs)

# Save the model
save_model(trained_model)

# Load the model
loaded_model = load_model()

# Test interaction with the model
predicted_label = interact_with_model(loaded_model, tokenizer, test_text)
