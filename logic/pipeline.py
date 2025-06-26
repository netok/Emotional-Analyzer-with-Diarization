import os
from logic.diarizer import CallDiarizer
from logic.splitter import SpeakerSplitter
from logic.analyzer import GigaAMEmoAnalyzer


def process_pipeline(upload_dir: str, output_dir: str):
    temp_transcript_dir = os.path.join(output_dir, "Transcripts")
    os.makedirs(temp_transcript_dir, exist_ok=True)

    # 1. Диаризация
    diarizer = CallDiarizer(
        input_dir=upload_dir,
        output_dir=temp_transcript_dir,
        model_name='turbo',
        language='ru',
        output_formats=['txt'],
        verbose=True
    )
    diarizer.process()

    # 2. Разделение на дорожки по спикерам
    splitter = SpeakerSplitter()
    splitter.process(audio_dir=upload_dir, transcript_dir=temp_transcript_dir, output_dir=output_dir)

    # 3. Эмоциональный анализ и PDF + Excel
    analyzer = GigaAMEmoAnalyzer(input_dir=output_dir, output_dir=output_dir)
    analyzer.analyze()