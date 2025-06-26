import os
import glob
import subprocess
from tqdm import tqdm
from typing import List, Union, Optional

class CallDiarizer:
    def __init__(
        self,
        input_dir: str = 'calls',
        batch_file: str = 'batch.txt',
        output_dir: str = 'Transcripts',
        model_name: str = 'large-v2',
        task: str = 'transcribe',
        diarization_method: str = 'reverb_v2',
        output_formats: List[str] = ['txt'],
        device: str = 'cuda',
        compute_type: str = 'float16',
        language: Optional[str] = 'ru',
        verbose: bool = True
    ):
        self.input_dir = input_dir
        self.batch_file = batch_file
        self.output_dir = output_dir
        self.model_name = model_name
        self.task = task
        self.diarization_method = diarization_method
        self.output_formats = output_formats
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.verbose = verbose
        self.whisper_path = "./Faster-Whisper-XXL/faster-whisper-xxl"

    def collect_calls(self) -> None:
        """Собирает пути к аудиофайлам в batch-файл"""
        audio_files = glob.glob(os.path.join(self.input_dir, '*.*'))
        
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in {self.input_dir}")
        
        with open(self.batch_file, 'w') as f:
            for file in audio_files:
                f.write(file + '\n')
        
        if self.verbose:
            print(f"[✓] Collected {len(audio_files)} files in {self.batch_file}")

    def run_diarization(self) -> None:
        """Запускает процесс диаризации с правильными аргументами"""
        if not os.path.exists(self.batch_file):
            raise FileNotFoundError(f"Batch file {self.batch_file} not found")
        
        # Формируем команду с правильными аргументами
        command = [
            self.whisper_path,
            self.batch_file,
            f"--model={self.model_name}",
            f"--task={self.task}",
            f"--diarize={self.diarization_method}",
            f"--output_dir={self.output_dir}",
            f"--output_format={','.join(self.output_formats)}",  # Форматы через запятую
            f"--device={self.device}",
            f"--compute_type={self.compute_type}"
        ]
        
        # Добавляем язык, если указан
        if self.language:
            command.append(f"--language={self.language}")
        
        if self.verbose:
            print("[•] Running command:")
            print(" ".join(command))
        
        try:
            # Используем subprocess для более надежного выполнения
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if self.verbose:
                print(result.stdout)
                print(f"[✓] Diarization completed! Results in {self.output_dir}")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Diarization failed with error {e.returncode}:\n"
            error_msg += f"STDOUT: {e.stdout}\n"
            error_msg += f"STDERR: {e.stderr}"
            raise RuntimeError(error_msg)
        except Exception as e:
            raise RuntimeError(f"Diarization failed: {str(e)}")

    def process(self) -> None:
        """Полный процесс: сбор файлов и диаризация"""
        self.collect_calls()
        self.run_diarization()