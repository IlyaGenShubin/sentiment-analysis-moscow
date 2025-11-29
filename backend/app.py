from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import pandas as pd
import io
import os
from sklearn.metrics import f1_score

# Глобальная переменная (будет инициализирована при старте)
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("🔄 Загрузка модели при старте приложения...")
    from model import SentimentModel
    MODEL_PATH = os.getenv("MODEL_PATH", "./model")
    model = SentimentModel(MODEL_PATH)
    print("✅ Модель успешно загружена!")
    yield
    # Завершение (опционально)
    model = None
    print("🛑 Модель выгружена.")

app = FastAPI(title="Sentiment Analysis API for Moscow", lifespan=lifespan)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(503, "Модель ещё не загружена")

    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Требуется CSV-файл")

    try:
        # Загружаем весь файл в память (но обрабатываем по частям!)
        df = pd.read_csv(file.file)
        if 'text' not in df.columns:
            raise HTTPException(400, "Требуется колонка 'text'")

        all_preds = []
        batch_size = 32  # ← безопасный размер для MPS

        texts = df['text'].tolist()
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            preds = model.predict(batch)
            all_preds.extend(preds)

        df['label'] = [p['label_id'] for p in all_preds]
        df['confidence'] = [p['confidence'] for p in all_preds]

        output_file = "/tmp/predictions.csv"
        df.to_csv(output_file, index=False)
        return FileResponse(output_file, media_type='text/csv', filename='predictions.csv')

    except Exception as e:
        import traceback
        print("❌ Ошибка в /predict:", str(e))
        traceback.print_exc()
        raise HTTPException(500, f"Ошибка обработки: {str(e)}")

@app.post("/evaluate")
async def evaluate(pred_file: UploadFile = File(...), gt_file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(503, "Модель ещё не загружена")
    try:
        pred_df = pd.read_csv(pred_file.file)
        gt_df = pd.read_csv(gt_file.file)
        if 'label' not in pred_df.columns or 'label' not in gt_df.columns:
            raise HTTPException(400, "Оба файла должны содержать колонку 'label'")
        macro_f1 = f1_score(gt_df['label'], pred_df['label'], average='macro', zero_division=0)
        return {"macro_f1": macro_f1}
    except Exception as e:
        raise HTTPException(500, f"Ошибка оценки: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
