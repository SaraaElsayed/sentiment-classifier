from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

app = FastAPI()

model = TFBertForSequenceClassification.from_pretrained("./my_finetuned_model")
tokenizer = BertTokenizer.from_pretrained("./my_finetuned_model")

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: InputText):
    inputs = tokenizer(data.text, return_tensors="tf", truncation=True, padding=True, max_length=256)

    outputs = model(inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1)
    label = int(tf.argmax(probs, axis=1))

    return {
        "prediction": "positive" if label == 1 else "negative",
        "confidence": float(tf.reduce_max(probs))
    }
