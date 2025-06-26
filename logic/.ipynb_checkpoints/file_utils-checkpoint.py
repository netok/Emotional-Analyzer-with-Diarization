import os
import zipfile
from fastapi import UploadFile
from typing import List


def save_uploaded_files(files: List[UploadFile], upload_dir: str):
    """
    Сохраняет загруженные файлы в указанную директорию.
    """
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())


def create_zip_archive(source_dir: str, zip_path: str):
    """
    Упаковывает содержимое source_dir в zip-файл zip_path.
    """
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, source_dir)
                zipf.write(abs_path, arcname=rel_path)