import os
import yt_dlp
import openai
from openai import OpenAI
from dotenv import load_dotenv

def setup_environment():
    """Load environment variables and setup OpenAI"""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    return api_key

def download_audio(url, output_file='downloaded_audio.mp3'):
    """Download audio from YouTube video"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_file.replace('.mp3', ''),
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("✓ Audio downloaded")
        return True
    except Exception as e:
        print(f"Error downloading audio: {str(e)}")
        return False

def transcribe_audio(file_path):
    """Transcribe audio file using OpenAI Whisper API"""
    try:
        client = OpenAI()
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        print("✓ Audio transcribed")
        return transcription
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        return None

def get_summary_prompt(level):
    """Get prompt based on summarization level"""
    base_format = """Format your response as follows:
# Overview
[Provide a concise overview of the main topic]

# Key Points
* [First key point]
* [Second key point]
* [Additional key points as needed]

# Details
* [Important detail or explanation]
* [Additional details based on level]

# Conclusion
[Summarize the main takeaways]
"""
    
    prompts = {
        'brief': f"Provide a concise summary of the following text. Focus on the most important points only. {base_format}",
        'moderate': f"Provide a balanced summary of the following text. Include main points and some supporting details. {base_format}",
        'detailed': f"Provide a comprehensive summary of the following text. Include main points, supporting details, and examples where relevant. {base_format}"
    }
    
    return prompts.get(level, prompts['moderate'])

def generate_summary(transcription, api_key, level='moderate'):
    """Generate summary using OpenAI GPT"""
    client = OpenAI(api_key=api_key)
    
    try:
        prompt = get_summary_prompt(level)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at creating clear, well-structured summaries."},
                {"role": "user", "content": f"{prompt}\n\nText to summarize:\n{transcription}"}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        print("✓ Summary generated")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return None

def cleanup(file_path):
    """Clean up downloaded files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print("✓ Cleanup completed")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}") 