# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.data_utils import load_data, preprocess_data, split_data
from utils.model import create_model, train_model, save_model, load_model
from utils.interaction import interact_with_model
from utils.text_augmentation import augment_text

app = FastAPI()

class TrainingParams(BaseModel):
    embedding_dim: int
    lstm_units: int
    dense_units: int
    epochs: int

@app.post("/train/")
async def train(params: TrainingParams):
    data = load_data('../notebooks/content/amazon.csv')
    processed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(processed_data)
    selected_data['about_product'] = selected_data['about_product'].apply(augment_text)
    model = create_model(params.embedding_dim, params.lstm_units, params.dense_units, vocab_size, num_classes)
    trained_model, history = train_model(model, X_train, y_train, X_test, y_test, params.epochs)
    save_model(trained_model)
    return {"message": "Model trained successfully"}

@app.get("/predict/")
async def predict(text: str):
    loaded_model = load_model()
    predicted_label = interact_with_model(loaded_model, tokenizer, text)
    return {"predicted_category": predicted_label}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
