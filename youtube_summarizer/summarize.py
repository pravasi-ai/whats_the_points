import os
import yt_dlp
import openai
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
import math

SYSTEM_MESSAGE = """You are an expert content analyst and summarizer, specifically trained to:
1. Deeply comprehend complex topics and their interconnections
2. Identify and extract key arguments and supporting evidence
3. Maintain the speaker's original context and examples
4. Structure information hierarchically with main points and supporting details
5. Present information in a clear, logical flow

Your goal is to provide comprehensive summaries that preserve the speaker's expertise while making the content accessible and actionable for the reader."""

SUMMARY_PROMPTS = {
    "detailed": """Analyze the following transcript and provide a detailed summary that:
1. Captures ALL key points and main arguments presented by the speaker
2. Includes specific examples, data points, and case studies mentioned
3. Preserves important quotes or unique insights
4. Maintains the chronological flow of ideas
5. Groups related concepts together
6. Highlights practical takeaways or action items
7. Retains technical details and terminology where relevant

Transcript: {text}

Please structure the summary with:
- Overview (2-3 sentences)
- Key Takeaways (bullet points)
- Detailed Discussion (organized by main topics)
- Supporting Examples & Evidence
- Practical Applications/Conclusions""",

    "concise": """Analyze the following transcript and provide a focused summary that:
1. Identifies the 3-5 most important points
2. Includes key supporting evidence for each point
3. Maintains essential context and examples
4. Emphasizes practical takeaways

Transcript: {text}

Please structure the summary with:
- Brief Overview (1-2 sentences)
- Key Points (with supporting evidence)
- Main Takeaways""",

    "brief": """Provide a brief summary of the key message and main points from this transcript:
1. Core message/theme
2. 2-3 main points with brief supporting evidence
3. Key takeaway

Transcript: {text}"""
}

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

def split_audio(file_path, chunk_duration_minutes=10):
    """Split audio file into smaller chunks"""
    try:
        audio = AudioSegment.from_mp3(file_path)
        chunk_length_ms = chunk_duration_minutes * 60 * 1000  # Convert minutes to milliseconds
        chunks = []
        
        # Split audio into chunks
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            chunk_path = f"{file_path[:-4]}_chunk_{i//chunk_length_ms}.mp3"
            chunk.export(chunk_path, format="mp3")
            chunks.append(chunk_path)
        
        print(f"✓ Audio split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        print(f"Error splitting audio: {str(e)}")
        return None

def transcribe_audio(file_path):
    """Transcribe audio file using OpenAI Whisper API with chunking"""
    try:
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > 25 * 1024 * 1024:  # 25MB in bytes
            print("Large audio file detected, splitting into chunks...")
            audio_chunks = split_audio(file_path)
            if not audio_chunks:
                return None
            
            # Transcribe each chunk
            transcriptions = []
            client = OpenAI()
            
            for i, chunk_path in enumerate(audio_chunks):
                print(f"Transcribing chunk {i+1} of {len(audio_chunks)}...")
                with open(chunk_path, "rb") as audio_file:
                    chunk_transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )
                transcriptions.append(chunk_transcription)
                # Clean up chunk file
                os.remove(chunk_path)
            
            # Combine all transcriptions
            full_transcription = " ".join(transcriptions)
            print("✓ Audio transcribed (processed in chunks)")
            return full_transcription
            
        else:
            # Original processing for smaller files
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

def chunk_text(text, max_chunk_size=4000):
    """Split text into chunks of approximately equal size"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        if current_size + word_size > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def generate_summary(transcription, api_key, level='concise'):
    """Generate summary using OpenAI GPT with chunking for long transcripts"""
    client = OpenAI(api_key=api_key)
    
    try:
        # Validate the level
        if level not in SUMMARY_PROMPTS:
            print(f"Invalid level '{level}'. Using 'concise' instead.")
            level = 'concise'
            
        # If transcription is too long, process in chunks
        if len(transcription.encode('utf-8')) > 25000:  # Safe limit
            chunks = chunk_text(transcription)
            chunk_summaries = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks, 1):
                print(f"Processing chunk {i} of {len(chunks)}...")
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": SUMMARY_PROMPTS[level].format(text=chunk)}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                chunk_summaries.append(response.choices[0].message.content)
            
            # Combine chunk summaries
            combined_summary = "\n\n".join(chunk_summaries)
            
            # Generate final summary of summaries
            final_prompt = """Combine the following summaries into a single coherent summary. 
            Remove any redundancy while preserving all unique and important information.
            
            Follow this structure:
            # Overview
            # Key Points
            # Details
            # Conclusion
            
            Summaries to combine:\n""" + combined_summary
            
            final_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at creating clear, well-structured summaries."},
                    {"role": "user", "content": final_prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            print("✓ Summary generated (processed in chunks)")
            return final_response.choices[0].message.content
            
        else:
            # Original processing for shorter transcripts
            prompt = SUMMARY_PROMPTS[level]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt.format(text=transcription)}
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