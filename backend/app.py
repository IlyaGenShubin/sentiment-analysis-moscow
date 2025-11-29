from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import io
from model import SentimentModel
import os
from sklearn.metrics import f1_score

app = FastAPI(title="Sentiment Analysis API for Moscow")

MODEL_PATH = os.getenv("MODEL_PATH", "./model")
model = SentimentModel(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Требуется CSV-файл")
    df = pd.read_csv(file.file)
    if 'text' not in df.columns:
        raise HTTPException(400, "Требуется колонка 'text'")
    predictions = model.predict(df['text'].tolist())
    df['label'] = [p['label_id'] for p in predictions]
    df['confidence'] = [p['confidence'] for p in predictions]
    output_file = "/tmp/predictions.csv"
    df.to_csv(output_file, index=False)
    return FileResponse(output_file, media_type='text/csv', filename='predictions.csv')

@app.post("/evaluate")
async def evaluate(pred_file: UploadFile = File(...), gt_file: UploadFile = File(...)):
    pred_df = pd.read_csv(pred_file.file)
    gt_df = pd.read_csv(gt_file.file)
    if 'label' not in pred_df.columns or 'label' not in gt_df.columns:
        raise HTTPException(400, "Оба файла должны содержать колонку 'label'")
    macro_f1 = f1_score(gt_df['label'], pred_df['label'], average='macro', zero_division=0)
    return {"macro_f1": macro_f1}
