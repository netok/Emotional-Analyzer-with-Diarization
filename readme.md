# 🎙 Emotional Analyzer with Diarization

This project performs comprehensive audio analysis, including speaker diarization, speaker-based audio segmentation, and emotional analysis of each speaker's segment.
It was developed for research into the emotional satisfaction of Contact Center clients at Far Eastern Federal University (FEFU).
---
The application uses the following pipeline:
1) Performs speaker diarization using Whisper and the reverb_v2 pipeline
2) Splits the audio file by speaker
3) Performs emotional analysis for each speaker segment
4) Generates a PDF report for each call that has been submitted
5) Also creates an Excel table with short information about each call

Also creates an Excel table with information about each call
---


## 🚀 Features

- **Speaker Diarization** — segment speech by different speakers
-  **Emotion Detection** — analyze emotions per speaker (via `GigaAMEmoAnalyzer`)
-  **Speaker Audio Tracks** — generate individual audio files for each speaker
-  **Transcript Preservation** — save and organize original transcripts
-  **Batch File Support** — analyze multiple audio files automatically
-  **ZIP Export** — output packaged into downloadable ZIP

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/netok/Emotional-Analyzer-with-Diarization.git
cd Emotional-Analyzer-with-Diarization
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```