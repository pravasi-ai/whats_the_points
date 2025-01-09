from flask import Flask, render_template, request, jsonify
import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from youtube_summarizer.summarize import setup_environment, download_audio, transcribe_audio, generate_summary, cleanup

app = Flask(__name__, 
    template_folder=os.path.join(current_dir, 'templates'),
    static_folder=os.path.join(current_dir, 'static')
)

# Basic config
app.config['SECRET_KEY'] = 'dev'

@app.route('/', methods=['GET'])
def index():
    try:
        print("Rendering index page...")
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        return str(e), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        print("Received summarize request")
        # Get form data
        url = request.form.get('url')
        level = request.form.get('level')
        
        print(f"URL: {url}, Level: {level}")
        
        if not url:
            return jsonify({'error': 'Please provide a YouTube URL'}), 400
        
        # Setup
        api_key = setup_environment()
        audio_file = 'downloaded_audio.mp3'
        
        # Process
        download_audio(url, audio_file)
        transcription = transcribe_audio(audio_file)
        summary = generate_summary(transcription, api_key, level)
        
        return jsonify({'summary': summary})
    
    except Exception as e:
        print(f"Error in summarize: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        if 'audio_file' in locals():
            cleanup(audio_file)

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, port=5000) 