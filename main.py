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
        "project": "🎙 Emotional Analyzer with Diarization",
        "description": (
            "API для комплексной обработки аудиозаписей разговоров. "
            "Выполняет спикерскую диаризацию, разбиение аудио по спикерам, "
            "эмоциональный анализ каждого сегмента и генерацию отчётов.\n\n"
            "Разработано в рамках проекта исследования удовлетворённости клиентов "
            "Единого Контактного Центра Дальневосточного Федерального Университета (ДВФУ)."
        ),
        "authors": [
            "Шорохов Константин",
            "Дмитрий Белов"
        ],
        "technologies": [
            "Whisper (ASR)",
            "Pyannote (speaker diarization)",
            "reverb_v2 pipeline",
            "GigaAMEmoAnalyzer (эмоциональный анализ)",
            "pydub, librosa, ffmpeg (аудиообработка)",
            "PDF/Excel генерация отчётов"
        ],
        "pipeline_steps": [
            "1. Диаризация: определение и маркировка голосов спикеров",
            "2. Разделение аудио по спикерам с сохранением тишины в паузах",
            "3. Анализ эмоций на каждом сегменте спикеров",
            "4. Генерация PDF отчёта по каждому звонку",
            "5. Создание общей Excel-таблицы 'callset_summary.xlsx'"
        ],
        "endpoints": [
            {
                "path": "/info",
                "method": "GET",
                "description": "Возвращает информацию о текущем API, его назначении и функциональности"
            },
            {
                "path": "/process",
                "method": "POST",
                "description": (
                    "Обрабатывает загруженные аудиофайлы и соответствующие транскрипты. "
                    "Результатом являются сегментированные аудиофайлы по спикерам, "
                    "анализ эмоций, и итоговые отчёты."
                )
            }
        ],
        "input_format": {
            "type": "multipart/form-data",
            "description": "Загрузка нескольких файлов",
            "expected": [
                "*.mp3 — аудиофайлы разговоров",
                "*.txt — текстовые транскрипты, соответствующие аудиофайлам"
            ]
        },
        "output_format": {
            "type": "application/zip",
            "description": "ZIP-архив со следующими данными:",
            "contents": [
                "Оригинальные транскрипты (*.txt)",
                "Аудиодорожки по каждому спикеру (*.mp3)",
                "PDF отчёты с визуализацией эмоций (*.pdf)",
                "Общая таблица по всем звонкам — callset_summary.xlsx"
            ]
        }
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