# YouTube Video Summarizer

A web application that creates summaries of YouTube videos using OpenAI's GPT model and Whisper for transcription.

## Features
- YouTube video download and audio extraction
- Audio transcription using Whisper
- Text summarization using GPT
- Web interface for easy interaction
- Multiple summary detail levels
- Text-to-speech summary playback

## Installation

1. Clone the repository
2. Create a virtual environment
    python -m venv .venv
    source .venv/bin/activate
3. Install dependencies
    pip install -r requirements.txt
    For macOS, you may need to install ffmpeg: brew install ffmpeg
4. Create a .env file with the following variables:
    OPENAI_API_KEY=your_openai_api_key
    ANTHROPIC_API_KEY=your_anthropic_api_key
5. Run the application
    python app.py
6. Open the web interface at http://localhost:5000
7. Enter the URL of the YouTube video you want to summarize and select the summary detail level
8. Click the "Summarize" button
9. The summary will be displayed in the text box
10. Click the "Play" button to hear the summary
