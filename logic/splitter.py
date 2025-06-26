import os
import re
from typing import Union
from pydub import AudioSegment
from tqdm.notebook import tqdm



class SpeakerSplitter:
    """
    Класс для разделения аудиофайлов на отдельные дорожки по спикерам.

    Использует транскрипты в формате:
    [HH:MM.SS --> HH:MM.SS] [SPEAKER_XX]: текст

    Parameters
    ----------
    None

    Attributes
    ----------
    audio : AudioSegment or None
        Загруженный аудиофайл
    speaker_segments : dict
        Словарь с аудиодорожками спикеров в формате {speaker: AudioSegment}

    Examples
    --------
    >>> splitter = SpeakerSplitter()
    >>> splitter.process("audio_dir", "transcripts_dir", "output_dir")
    """

    def __init__(self):
        """
        Инициализирует экземпляр SpeakerSplitter.

        Returns
        -------
        None
        """
        self.audio = None
        self.speaker_segments = {}

    def _time_to_millis(self, timestamp: str) -> int:
        """
        Конвертирует временную метку в миллисекунды.

        Parameters
        ----------
        timestamp : str
            Временная метка в формате 'HH:MM.SS'

        Returns
        -------
        int
            Время в миллисекундах

        Examples
        --------
        >>> self._time_to_millis("01:23.456")
        83456
        """
        minutes, sec_millis = timestamp.split(":")
        seconds, millis = sec_millis.split(".")
        return (int(minutes) * 60 + int(seconds)) * 1000 + int(millis)

    def _parse_transcript(self, lines: list) -> list:
        """
        Парсит транскрипт и извлекает сегменты спикеров.

        Parameters
        ----------
        lines : list of str
            Список строк транскрипта

        Returns
        -------
        list of tuple
            Список кортежей (speaker, start_ms, end_ms)

        Examples
        --------
        >>> lines = ["[01:23.456 --> 01:45.789] [SPEAKER_01]: текст"]
        >>> self._parse_transcript(lines)
        [('SPEAKER_01', 83456, 105789)]
        """
        pattern = re.compile(r"\[(\d+:\d+\.\d+)\s*-->\s*(\d+:\d+\.\d+)\]\s+\[(SPEAKER_\d+)\]:")
        segments = []
        for line in lines:
            match = pattern.match(line)
            if match:
                start = self._time_to_millis(match.group(1))
                end = self._time_to_millis(match.group(2))
                speaker = match.group(3)
                segments.append((speaker, start, end))
        return segments

    def _process_single_pair(self, audio_path: str, transcript_path: str, output_root: str):
        """
        Обрабатывает один аудиофайл и соответствующий транскрипт.

        Parameters
        ----------
        audio_path : str
            Путь к аудиофайлу (.mp3)
        transcript_path : str
            Путь к файлу транскрипта (.txt)
        output_root : str
            Директория для сохранения результатов

        Raises
        ------
        FileNotFoundError
            Если файлы не существуют

        Returns
        -------
        None
        """
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = os.path.join(output_root, filename)
        os.makedirs(output_dir, exist_ok=True)

        # Загрузка аудио
        self.audio = AudioSegment.from_file(audio_path)
        audio_duration = len(self.audio)

        # Загрузка транскрипта
        with open(transcript_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Сохранение оригинального текста
        with open(os.path.join(output_dir, "original_transcript.txt"), "w", encoding="utf-8") as f:
            f.writelines(lines)

        # Парсинг и инициализация дорожек
        segments = self._parse_transcript(lines)
        speakers = {s for s, _, _ in segments}
        self.speaker_segments = {s: AudioSegment.silent(duration=audio_duration) for s in speakers}

        # Вставка аудио
        for speaker, start_ms, end_ms in segments:
            self.speaker_segments[speaker] = (
                self.speaker_segments[speaker][:start_ms] +
                self.audio[start_ms:end_ms] +
                self.speaker_segments[speaker][end_ms:]
            )

        # Сохранение
        for speaker, combined_audio in self.speaker_segments.items():
            output_path = os.path.join(output_dir, f"{speaker}.mp3")
            combined_audio.export(output_path, format="mp3")
            print(f"[✓] Сохранено: {output_path}")

    def process(self, audio_dir: str, transcript_dir: str, output_dir: str):
        """
        Обрабатывает все аудиофайлы и транскрипты в указанных директориях.

        Parameters
        ----------
        audio_dir : str
            Директория с аудиофайлами (.mp3)
        transcript_dir : str
            Директория с транскриптами (.txt)
        output_dir : str
            Директория для сохранения результатов

        Returns
        -------
        None

        Examples
        --------
        >>> splitter = SpeakerSplitter()
        >>> splitter.process("audio", "transcripts", "output")
        """
        os.makedirs(output_dir, exist_ok=True)
        audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".mp3")]
        
        for audio_file in tqdm(audio_files):
            base = os.path.splitext(audio_file)[0]
            audio_path = os.path.join(audio_dir, f"{base}.mp3")
            transcript_path = os.path.join(transcript_dir, f"{base}.txt")

            if os.path.exists(transcript_path):
                print(f"[•] Обработка пары: {base}")
                self._process_single_pair(audio_path, transcript_path, output_dir)
            else:
                print(f"[!] Пропущено: {base} — нет соответствующего транскрипта.")