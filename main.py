from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from typing import List
import shutil
import os
from logic.pipeline import process_pipeline
from logic.file_utils import save_uploaded_files, create_zip_archive

app = FastAPI(title="Call Analysis API", description="Диаризация, разделение по спикерам и анализ эмоций")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
ZIP_RESULT = "result.zip"


@app.get("/info")
def get_info():
    return {
        "description": "API для обработки звонков: диаризация, разбиение по спикерам, анализ эмоций",
        "endpoints": [
            {"path": "/info", "method": "GET", "description": "Информация об API"},
            {"path": "/process", "method": "POST", "description": "Обработка аудиофайлов и транскриптов"}
        ],
        "input_format": "Несколько файлов (mp3 и соответствующие txt)",
        "output": ["ZIP архив с результатами", "Excel таблица callset_summary.xlsx"]
    }


@app.post("/process")
def process_files(files: List[UploadFile] = File(...)):
    # Очистка и подготовка
    if os.path.exists(UPLOAD_DIR): shutil.rmtree(UPLOAD_DIR)
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Сохранение файлов
    save_uploaded_files(files, UPLOAD_DIR)

    try:
        # Запуск пайплайна
        process_pipeline(upload_dir=UPLOAD_DIR, output_dir=OUTPUT_DIR)

        # Создание архива
        create_zip_archive(OUTPUT_DIR, ZIP_RESULT)
        return FileResponse(ZIP_RESULT, filename="result.zip", media_type="application/zip")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})