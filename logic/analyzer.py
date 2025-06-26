import os
import torch
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import gigaam
import re
import tempfile
import soundfile as sf
import io
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, NextPageTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Frame, PageTemplate, Image, FrameBreak
from reportlab.lib.units import inch


class GigaAMEmoAnalyzer:
    """
    Analyzer for emotion recognition in phone call datasets using the GigaAM model.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing call folders (each folder is a call).
    output_dir : str
        Path to the directory where PDF reports and the summary Excel file will be saved.
    target_sr : int, optional
        Target sample rate for audio processing (default is 16000).

    Notes
    -----
    Requires the GigaAM model and all dependencies installed.
    """

    def __init__(self, input_dir, output_dir, target_sr=16000):
        """
        Initialize the analyzer with input/output directories and target sample rate.

        Parameters
        ----------
        input_dir : str
            Path to the directory containing call folders.
        output_dir : str
            Path to the directory for saving results.
        target_sr : int, optional
            Target sample rate for audio (default is 16000).
        """

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_sr = target_sr
        self.model = gigaam.load_model('emo')
        self.font_name = self._register_font()
        plt.rcParams['font.sans-serif'] = [self.font_name, 'DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
    
    def _register_font(self):
        """
        Register a system font for correct PDF rendering (Cyrillic support).
    
        Returns
        -------
        str
            Name of the registered font to use in ReportLab.
        """
        font_name = 'SystemFont'
    
        possible_font_paths = [
            # macOS
            '/Library/Fonts/Arial.ttf',
            '/System/Library/Fonts/Supplemental/Arial.ttf',
            '/Library/Fonts/HelveticaNeue.ttc',
            os.path.expanduser('~/Library/Fonts/Arial.ttf'),
            os.path.expanduser('~/Library/Fonts/HelveticaNeue.ttc'),
    
            # Windows
            'C:\\Windows\\Fonts\\arial.ttf',
            'C:\\Windows\\Fonts\\ARIALUNI.TTF',
            'C:\\Windows\\Fonts\\segoeui.ttf',
            'C:\\Windows\\Fonts\\tahoma.ttf',
    
            # Linux (common fonts)
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf',
            '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            os.path.expanduser('~/.fonts/DejaVuSans.ttf'),
            os.path.expanduser('~/.local/share/fonts/DejaVuSans.ttf'),
        ]
    
        for path in possible_font_paths:
            if os.path.exists(path):
                try:
                    if path.lower().endswith('.ttc'):
                        pdfmetrics.registerFont(TTFont(font_name, path, subfontIndex=0))
                    else:
                        pdfmetrics.registerFont(TTFont(font_name, path))
                    return font_name
                except Exception as e:
                    # Опционально логировать ошибку
                    print(f"Failed to register font from {path}: {e}")
                    continue
    
        # Если не удалось зарегистрировать ни один шрифт
        return 'Helvetica'
    
    def load_audio(self, file_path):
        """
        Load an audio file and resample it to the target sample rate.

        Parameters
        ----------
        file_path : str
            Path to the audio file.

        Returns
        -------
        audio : np.ndarray or None
            Loaded audio data.
        actual_sr : int or None
            Actual sample rate after loading.
        """

        try:
            audio, actual_sr = librosa.load(file_path, sr=self.target_sr)
            return audio, actual_sr
        except Exception:
            return None, None

    def extract_audio_segment(self, audio, sr, start_time, end_time):
        """
        Extract a segment from audio data given start and end times.

        Parameters
        ----------
        audio : np.ndarray
            Audio data.
        sr : int
            Sample rate.
        start_time : float
            Start time in seconds.
        end_time : float
            End time in seconds.

        Returns
        -------
        np.ndarray
            Audio segment for the specified interval.
        """

        if audio is None:
            return np.array([])
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        if start_sample >= len(audio) or start_sample >= end_sample:
            return np.array([])
        return audio[start_sample:end_sample]
    
    def read_transcript(self, file_path):
        """
        Read transcript segmentation from a text file.

        Parameters
        ----------
        file_path : str
            Path to the transcript file.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: start_time, end_time, speaker, text.
        """

        segments = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'\[(\d+:\d+\.\d+)\s*-->\s*(\d+:\d+\.\d+)\]\s*\[(SPEAKER_\d+)\]:\s*(.*)', line)
                if match:
                    start_time, end_time, speaker, text = match.groups()
                    start_min, start_sec = start_time.split(':')
                    end_min, end_sec = end_time.split(':')
                    start_seconds = float(start_min) * 60 + float(start_sec)
                    end_seconds = float(end_min) * 60 + float(end_sec)
                    segments.append({
                        'start_time': start_seconds,
                        'end_time': end_seconds,
                        'speaker': speaker,
                        'text': text
                    })
        return pd.DataFrame(segments)
    
    def create_emotion_pie_chart(self, results, speaker_name):
        """
        Create a pie chart of emotion distribution for a speaker.

        Parameters
        ----------
        results : list of dict
            Segment analysis results.
        speaker_name : str
            Name of the speaker.

        Returns
        -------
        io.BytesIO or None
            Buffer with the chart image, or None if no data.
        """

        EMOTION_COLORS = {
            'positive': '#8fd9b6',
            'neutral': '#b6c9f0',
            'sad': '#f7b7a3',
            'angry': '#f67280'
        }
        all_emotions = {}
        for result in results:
            if 'emotions' in result and result['emotions'] is not None:
                for emotion, prob in result['emotions'].items():
                    if emotion not in all_emotions:
                        all_emotions[emotion] = 0
                    all_emotions[emotion] += prob

        total_prob_sum = sum(all_emotions.values())
        if total_prob_sum == 0:
            return None

        # Сохраняем порядок эмоций для цветов
        emotions = [e for e in EMOTION_COLORS if e in all_emotions]
        values = [all_emotions[e] / total_prob_sum for e in emotions]
        colors = [EMOTION_COLORS[e] for e in emotions]

        if not values:
            return None

        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            values, labels=emotions, autopct='%1.1f%%', colors=colors,
            startangle=90, pctdistance=0.8, labeldistance=1.05,
            wedgeprops={'edgecolor': 'white'}
        )

        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(9)
        for text in texts:
            text.set_fontsize(10)

        ax.axis('equal')
        ax.set_title(f'Распределение эмоций', fontsize=12)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    def create_emotion_timeline(self, results, speaker_name):
        """
        Create a heatmap timeline of emotions for a speaker.

        Parameters
        ----------
        results : list of dict
            Segment analysis results.
        speaker_name : str
            Name of the speaker.

        Returns
        -------
        io.BytesIO or None
            Buffer with the timeline image, or None if no data.
        """

        analyzed_results = [r for r in results if 'emotions' in r and r['emotions'] is not None]
        if not analyzed_results:
            return None

        emotions_df = pd.DataFrame([r['emotions'] for r in analyzed_results])

        # Кастомный colormap от голубого к зеленому
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "blue2green", ['#b6c9f0', '#8fd9b6']
        )

        annot_rotation = 90 if len(analyzed_results) > 15 else 0

        fig, ax = plt.subplots(figsize=(max(18, len(analyzed_results) * 0.5), 10))

        sns.heatmap(
            emotions_df.T,
            ax=ax,
            cmap=cmap,
            cbar_kws={'label': 'Вероятность', "shrink": .8},
            annot=True, 
            fmt=".2f", 
            annot_kws={"size": 10, "color": "black", "rotation": annot_rotation},
            linewidths=1, linecolor='white'
        )

        ax.set_title('Временная шкала эмоций', fontsize=20, pad=18)
        ax.set_xlabel('Проанализированный сегмент', fontsize=14, labelpad=10)
        ax.set_ylabel('Эмоция', fontsize=14, labelpad=10)

        ax.set_yticklabels(emotions_df.columns, rotation=0, va='center', fontsize=13)
        segment_labels = [f"{r['start_time']:.1f}-{r['end_time']:.1f}" for r in analyzed_results]
        N = max(1, int(len(segment_labels) / 15))
        ax.set_xticks(np.arange(len(segment_labels))[::N] + 0.5)
        ax.set_xticklabels(segment_labels[::N], rotation=45, ha='right', fontsize=11)

        plt.tight_layout(rect=[0.05, 0.05, 0.98, 0.98])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close(fig)
        return buf
    
    def create_segment_duration_chart(self, results, speaker_name):
        """
        Create a bar chart of segment durations for a speaker.

        Parameters
        ----------
        results : list of dict
            Segment analysis results.
        speaker_name : str
            Name of the speaker.

        Returns
        -------
        io.BytesIO or None
            Buffer with the chart image, or None if no data.
        """

        analyzed_results = [r for r in results if 'emotions' in r and r['emotions'] is not None]
        if not analyzed_results:
            return None

        durations = [r['duration'] for r in analyzed_results]
        segments = range(1, len(durations) + 1)

        EMOTION_COLORS = {
            'positive': '#8fd9b6',
            'neutral': '#b6c9f0',
            'sad': '#f7b7a3',
            'angry': '#f67280'
        }
        bar_colors = []
        for r in analyzed_results:
            if r['emotions']:
                dominant = max(r['emotions'], key=r['emotions'].get)
                bar_colors.append(EMOTION_COLORS.get(dominant, '#cccccc'))
            else:
                bar_colors.append('#cccccc')

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(segments, durations, color=bar_colors)
        ax.set_title(f'Длительность проанализированных сегментов', fontsize=12)
        ax.set_xlabel('Номер проанализированного сегмента', fontsize=10)
        ax.set_ylabel('Длительность (секунды)', fontsize=10)

        N = max(1, int(len(segments) / 15))
        ax.set_xticks(np.arange(len(segments))[::N] + 1)
        ax.set_xticklabels(list(segments)[::N], rotation=45, ha='right', fontsize=9)

        # Легенда по цветам эмоций
        handles = [plt.Rectangle((0,0),1,1, color=color) for color in EMOTION_COLORS.values()]
        ax.legend(handles, EMOTION_COLORS.keys(), title="Доминирующая эмоция", fontsize=9)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    def create_emotion_stacked_area_plot(self, results, speaker_name):
        """
        Create a stacked area plot of emotion probabilities for a speaker.

        Parameters
        ----------
        results : list of dict
            Segment analysis results.
        speaker_name : str
            Name of the speaker.

        Returns
        -------
        io.BytesIO or None
            Buffer with the chart image, or None if no data.
        """

        EMOTION_COLORS = {
            'positive': '#8fd9b6',
            'neutral': '#b6c9f0',
            'sad': '#f7b7a3',
            'angry': '#f67280'
        }
        EMOTION_ORDER = list(EMOTION_COLORS.keys())

        analyzed_results = [r for r in results if 'emotions' in r and r['emotions'] is not None]
        if not analyzed_results:
            return None

        # DataFrame с фиксированным порядком эмоций
        emotions_df = pd.DataFrame([r['emotions'] for r in analyzed_results])[EMOTION_ORDER]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.stackplot(
            range(len(emotions_df)),
            emotions_df.values.T,
            labels=EMOTION_ORDER,
            colors=[EMOTION_COLORS[e] for e in EMOTION_ORDER],
            alpha=0.8
        )
        ax.set_title('Emotion probabilities (stacked area)')
        ax.set_xlabel('Segment')
        ax.set_ylabel('Probability')
        ax.legend(loc='upper right')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf


    def generate_pdf_report(self, results_speaker_00, results_speaker_01, audio_duration, transcript_segment_count, output_path, call_id):
        """
        Generate a PDF report for a single call.

        Parameters
        ----------
        results_speaker_00 : list of dict
            Analysis results for SPEAKER_00.
        results_speaker_01 : list of dict
            Analysis results for SPEAKER_01.
        audio_duration : float
            Total call duration in seconds.
        transcript_segment_count : int
            Number of segments in the transcript.
        output_path : str
            Path to save the PDF report.
        call_id : str
            Call identifier.

        Returns
        -------
        None
        """

        doc = BaseDocTemplate(output_path, pagesize=landscape(letter))
        styles = getSampleStyleSheet()
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontName=self.font_name,
            fontSize=12,
            leading=14,
            spaceAfter=0
        )
        heading1_style = ParagraphStyle(
            'CustomHeading1',
            parent=styles['Heading1'],
            fontName=self.font_name,
            fontSize=24,
            spaceAfter=20,
            alignment=1
        )
        heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontName=self.font_name,
            fontSize=18,
            spaceBefore=20,
            spaceAfter=10
            # alignment=1
        )
        heading3_style = ParagraphStyle(
            'CustomHeading3',
            parent=styles['Heading3'],
            fontName=self.font_name,
            fontSize=14,
            spaceBefore=10,
            spaceAfter=5
        )
        heading4_style = ParagraphStyle(
            'CustomHeading3',
            parent=styles['Heading3'],
            fontName=self.font_name,
            fontSize=18,
            spaceBefore=10,
            spaceAfter=5,
            alignment=1
        )
        table_style = TableStyle([
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), self.font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
            ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
        ])

        EMOTION_COLORS = {
            'positive': '#8fd9b6',
            'neutral': '#b6c9f0',
            'sad': '#f7b7a3',
            'angry': '#f67280'
        }

        margin = inch
        col_width = (landscape(letter)[0] - 2*margin - 0.5*inch) / 2
        # Две колонки для страниц со спикером
        frame_left = Frame(margin, margin, col_width, landscape(letter)[1] - 2*margin, id='left')
        frame_right = Frame(margin + col_width + 0.5*inch, margin, col_width, landscape(letter)[1] - 2*margin, id='right')
        two_col_template = PageTemplate(id='two_col', frames=[frame_left, frame_right])
        # Одиночная страница
        single_frame = Frame(margin, margin, landscape(letter)[0] - 2*margin, landscape(letter)[1] - 2*margin, id='single')
        single_template = PageTemplate(id='single', frames=[single_frame])
        doc.addPageTemplates([single_template, two_col_template])
        # --- Данные ---
        analyzed_results_00 = [r for r in results_speaker_00 if not r.get('skipped_analysis', False) and r.get('emotions')]
        analyzed_results_01 = [r for r in results_speaker_01 if not r.get('skipped_analysis', False) and r.get('emotions')]
        # --- 1 страница: анализ диалога ---
        Story = []
        Story.append(Paragraph(f"Анализ диалога {call_id}", heading1_style))
        Story.append(Spacer(1, inch))
        info_data = [
            [Paragraph("Общая длительность звонка", normal_style), Paragraph(f"{audio_duration:.2f} секунд", normal_style)],
            [Paragraph("Всего сегментов в разметке", normal_style), Paragraph(str(transcript_segment_count), normal_style)],
            [Paragraph("Проанализировано сегментов SPEAKER_00", normal_style), Paragraph(str(len(analyzed_results_00)), normal_style)],
            [Paragraph("Проанализировано сегментов SPEAKER_01", normal_style), Paragraph(str(len(analyzed_results_01)), normal_style)],
            [Paragraph("Общая длительность проанализированной речи SPEAKER_00", normal_style), Paragraph(f"{sum(r['duration'] for r in analyzed_results_00):.2f} секунд", normal_style)],
            [Paragraph("Общая длительность проанализированной речи SPEAKER_01", normal_style), Paragraph(f"{sum(r['duration'] for r in analyzed_results_01):.2f} секунд", normal_style)],
        ]
        info_table = Table(info_data, colWidths=[4*inch, 3*inch])
        info_table.setStyle(table_style)
        Story.append(info_table)
        Story.append(NextPageTemplate('two_col'))
        Story.append(PageBreak())
        # --- 2 страница: анализ спикера 00 ---
        # Левая колонка
        Story.append(Spacer(2, inch))
        Story.append(Paragraph("Анализ SPEAKER_00", heading2_style))
        Story.append(Spacer(1, 10))
        emotion_counts_00 = {}
        for res in analyzed_results_00:
            dominant_emotion = max(res['emotions'], key=res['emotions'].get)
            emotion_counts_00[dominant_emotion] = emotion_counts_00.get(dominant_emotion, 0) + 1
        Story.append(Paragraph("Доминирующие эмоции:", normal_style))
        for emotion, count in sorted(emotion_counts_00.items(), key=lambda x: x[1], reverse=True):
            Story.append(Paragraph(f"- {emotion}: {count}", normal_style))
        Story.append(Spacer(1, 10))
        Story.append(Paragraph(f"Всего сегментов (разметка): {len(results_speaker_00)}", normal_style))
        Story.append(Paragraph(f"Проанализировано: {len(analyzed_results_00)}", normal_style))
        Story.append(Paragraph(f"Общая длительность: {sum(r['duration'] for r in analyzed_results_00):.2f} сек", normal_style))
        # Правая колонка: график длительности
        Story.append(FrameBreak())
        Story.append(Spacer(2, inch))
        duration_chart_00 = self.create_segment_duration_chart(analyzed_results_00, "SPEAKER_00")
        if duration_chart_00:
            Story.append(Image(duration_chart_00, width=5*inch, height=4*inch))
        Story.append(PageBreak())
        # --- 3 страница: Эмоциональный анализ спикера 00 ---
        Story.append(Paragraph("Эмоциональный анализ SPEAKER_00", heading2_style))
        Story.append(Spacer(2, inch))
        # Левая колонка: круговая диаграмма
        pie_chart_00 = self.create_emotion_pie_chart(analyzed_results_00, "SPEAKER_00")
        if pie_chart_00:
            Story.append(Image(pie_chart_00, width=4*inch, height=4*inch))
        # Правая колонка: временная шкала
        Story.append(FrameBreak())
        Story.append(Spacer(10, 50))
        emo_stacked_area_00 = self.create_emotion_stacked_area_plot(analyzed_results_00, "SPEAKER_00")
        if emo_stacked_area_00:
            Story.append(Image(emo_stacked_area_00, width=5*inch, height=4*inch))
        Story.append(NextPageTemplate('single'))
        Story.append(PageBreak())
        # --- 4 страница: временная шкала спикера 00 ---
        Story.append(Spacer(2, inch))
        timeline_00 = self.create_emotion_timeline(analyzed_results_00, "SPEAKER_00")
        if timeline_00:
            Story.append(Image(timeline_00, width=7*inch, height=5*inch))
        Story.append(NextPageTemplate('two_col'))
        Story.append(PageBreak())
        # --- 5 страница: анализ спикера 01 ---
        # Левая колонка
        Story.append(Spacer(2, inch))
        Story.append(Paragraph("Анализ SPEAKER_01", heading2_style))
        Story.append(Spacer(1, 10))
        emotion_counts_01 = {}
        for res in analyzed_results_01:
            dominant_emotion = max(res['emotions'], key=res['emotions'].get)
            emotion_counts_01[dominant_emotion] = emotion_counts_01.get(dominant_emotion, 0) + 1
        Story.append(Paragraph("Доминирующие эмоции:", normal_style))
        for emotion, count in sorted(emotion_counts_01.items(), key=lambda x: x[1], reverse=True):
            Story.append(Paragraph(f"- {emotion}: {count}", normal_style))
        Story.append(Spacer(1, 10))
        Story.append(Paragraph(f"Всего сегментов (разметка): {len(results_speaker_01)}", normal_style))
        Story.append(Paragraph(f"Проанализировано: {len(analyzed_results_01)}", normal_style))
        Story.append(Paragraph(f"Общая длительность: {sum(r['duration'] for r in analyzed_results_01):.2f} сек", normal_style))
        # Правая колонка: график длительности
        Story.append(FrameBreak())
        Story.append(Spacer(2, inch))
        duration_chart_01 = self.create_segment_duration_chart(analyzed_results_01, "SPEAKER_01")
        if duration_chart_01:
            Story.append(Image(duration_chart_01, width=5*inch, height=4*inch))
        Story.append(PageBreak())
        # --- 6 страница: Эмоциональный анализ спикера 01 ---
        Story.append(Paragraph("Эмоциональный анализ SPEAKER_01", heading2_style))
        Story.append(Spacer(2, inch))
        doc.handle_nextPageTemplate('two_col')
        # Левая колонка: круговая диаграмма
        pie_chart_01 = self.create_emotion_pie_chart(analyzed_results_01, "SPEAKER_01")
        if pie_chart_01:
            Story.append(Image(pie_chart_01, width=4*inch, height=4*inch))
        # Правая колонка: временная шкала
        Story.append(FrameBreak())
        Story.append(Spacer(2, inch))
        timeline_01 = self.create_emotion_stacked_area_plot(analyzed_results_01, "SPEAKER_01")
        if timeline_01:
            Story.append(Image(timeline_01, width=5*inch, height=4*inch))
        Story.append(NextPageTemplate('single'))
        Story.append(PageBreak())
        # --- 7 страница: временная шкала спикера 01 ---
        # Story.append(Spacer(2, inch))
        timeline_01 = self.create_emotion_timeline(analyzed_results_01, "SPEAKER_01")
        if timeline_01:
            Story.append(Image(timeline_01, width=8.5*inch, height=6*inch))
        Story.append(PageBreak())
        # --- 8 страница и далее: посегментный анализ ---
        doc.handle_nextPageTemplate('single')
        Story.append(Paragraph("Посегментный анализ диалога", heading4_style))
        Story.append(Spacer(1, inch))
        all_results = sorted(results_speaker_00 + results_speaker_01, key=lambda x: x['start_time'])
        for i, result in enumerate(all_results):
            speaker = result['speaker']
            text_header = f"Сегмент {i+1}: [{result['start_time']:.2f} --> {result['end_time']:.2f}] {speaker}"
            Story.append(Paragraph(text_header, heading3_style))
            Story.append(Paragraph("Текст: " + result['text'], normal_style))
            Story.append(Paragraph(f"Длительность по разметке: {result['duration']:.2f} сек", normal_style))
            Story.append(Paragraph(f"Длительность извлеченного аудио: {result['extracted_duration']:.2f} сек", normal_style))
            if result.get('skipped_analysis', False):
                Story.append(Paragraph("Анализ эмоций: Сегмент пропущен", normal_style))
            elif result.get('emotions'):
                emotions = [f"- {e}: {p:.3f}" for e, p in result['emotions'].items()]
                Story.append(Paragraph("Эмоции: " + ", ".join(emotions), normal_style))
            else:
                Story.append(Paragraph("Эмоции: Нет данных", normal_style))
            Story.append(Spacer(1, 8))
        try:
            doc.build(Story)
            # print(f"PDF отчет сохранен в {output_path}")
        except Exception as e:
            print(f"Ошибка при создании PDF: {e}")
            import traceback
            traceback.print_exc()
    
    def find_valid_calls(self):
        """
        Find all valid calls in the input directory.

        Returns
        -------
        list of dict
            List of dicts with paths to transcript and audio files for each call.
        """

        valid_calls = []
        for call_id in sorted(os.listdir(self.input_dir)):
            call_path = os.path.join(self.input_dir, call_id)
            if not os.path.isdir(call_path):
                continue
            transcript = None
            for fname in os.listdir(call_path):
                if fname.startswith("transcript") and fname.endswith(".txt"):
                    transcript = os.path.join(call_path, fname)
                    break
                if fname == "original_transcript.txt":
                    transcript = os.path.join(call_path, fname)
                    break
            speaker_00 = None
            speaker_01 = None
            for fname in os.listdir(call_path):
                if fname.startswith("SPEAKER_00"):
                    speaker_00 = os.path.join(call_path, fname)
                if fname.startswith("SPEAKER_01"):
                    speaker_01 = os.path.join(call_path, fname)
            if transcript and speaker_00 and speaker_01:
                valid_calls.append({
                    "id": call_id,
                    "transcript": transcript,
                    "speaker_00": speaker_00,
                    "speaker_01": speaker_01
                })
        return valid_calls

    def process_call(self, call):
        """
        Process a single call: analyze segments and generate a PDF report.

        Parameters
        ----------
        call : dict
            Dict with paths to transcript and audio files.

        Returns
        -------
        results_speaker_00 : list of dict
        results_speaker_01 : list of dict
        audio_duration : float
        """

        speaker_00_audio, _ = self.load_audio(call['speaker_00'])
        speaker_01_audio, _ = self.load_audio(call['speaker_01'])
        sr = self.target_sr
        if speaker_00_audio is None or speaker_01_audio is None:
            print(f"Ошибка: Не удалось загрузить аудио для {call['id']}")
            return None, None, 0
        transcript_df = self.read_transcript(call['transcript'])
        transcript_segment_count = len(transcript_df)
        results_speaker_00 = []
        results_speaker_01 = []
        if not transcript_df.empty:
            for _, row in transcript_df.iterrows():
                audio = speaker_00_audio if row['speaker'] == 'SPEAKER_00' else speaker_01_audio
                audio_segment = self.extract_audio_segment(audio, sr, row['start_time'], row['end_time'])
                segment_duration = len(audio_segment) / sr if sr > 0 and len(audio_segment) > 0 else 0
                skipped = segment_duration < 0.1
                emotion_probs = None
                if not skipped:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                        sf.write(tmp_wav.name, audio_segment, sr)
                        tmp_wav_path = tmp_wav.name
                    try:
                        with torch.no_grad():
                            probs = self.model.get_probs(tmp_wav_path)
                        emotion_probs = dict(probs)
                    finally:
                        os.remove(tmp_wav_path)
                result = {
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'speaker': row['speaker'],
                    'text': row['text'],
                    'duration': row['end_time'] - row['start_time'],
                    'extracted_duration': segment_duration,
                    'emotions': emotion_probs,
                    'skipped_analysis': skipped
                }
                if row['speaker'] == 'SPEAKER_00':
                    results_speaker_00.append(result)
                else:
                    results_speaker_01.append(result)
            audio_duration = len(speaker_01_audio) / sr if speaker_01_audio is not None and sr > 0 else 0
        else:
            audio_duration = 0
        output_pdf = os.path.join(self.output_dir, f"{call['id']}_analysis_giga.pdf")
        self.generate_pdf_report(results_speaker_00, results_speaker_01, audio_duration, transcript_segment_count, output_pdf, call['id'])
        return results_speaker_00, results_speaker_01, audio_duration

    def analyze(self):
        """
        Run full analysis for all calls: generate reports and summary table.

        Returns
        -------
        None
            All results are saved in output_dir.
        """
        
        os.makedirs(self.output_dir, exist_ok=True)
        calls = self.find_valid_calls()
        print(f"Найдено {len(calls)} валидных звонков для обработки.")
        EMOTION_LABELS = ['positive', 'neutral', 'sad', 'angry']
        NEGATIVE_EMOTIONS = {'sad', 'angry'}
        rows = []
        for call in calls:
            results_00, results_01, audio_duration = self.process_call(call)
            for speaker, results in zip(['SPEAKER_00', 'SPEAKER_01'], [results_00, results_01]):
                dominant_emotions = []
                emotion_sums = {emo: 0.0 for emo in EMOTION_LABELS}
                count = 0
                for res in results:
                    if not res.get('skipped_analysis', False) and res.get('emotions'):
                        dominant = max(res['emotions'], key=res['emotions'].get)
                        dominant_emotions.append(dominant)
                        for emo in EMOTION_LABELS:
                            emotion_sums[emo] += res['emotions'].get(emo, 0.0)
                        count += 1
                if dominant_emotions:
                    most_common = pd.Series(dominant_emotions).value_counts()
                    for emo in most_common.index:
                        if emo in NEGATIVE_EMOTIONS:
                            main_emotion = emo
                            break
                    else:
                        main_emotion = most_common.index[0]
                else:
                    main_emotion = 'none'
                result = 'bad' if main_emotion in NEGATIVE_EMOTIONS else 'good'
                emotion_percents = [round(emotion_sums[emo] / count if count > 0 else 0, 3) for emo in EMOTION_LABELS]
                rows.append([
                    call['id'],
                    speaker,
                    main_emotion,
                    *emotion_percents,
                    round(audio_duration, 2),
                    result
                ])
        df = pd.DataFrame(rows, columns=['dialog_id', 'speaker', 'main_emotion', *EMOTION_LABELS, 'dialog_duration', 'result'])
        excel_path = os.path.join(self.output_dir, 'callset_summary.xlsx')
        df.to_excel(excel_path, index=False)
        print(f"Excel-отчет сохранён: {excel_path}")