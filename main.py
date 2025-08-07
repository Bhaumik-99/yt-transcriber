import os
import sys
import csv
import subprocess
import tempfile
import logging
from pathlib import Path
import pandas as pd
import whisper
import yt_dlp
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class VideoFactProcessor:
    def __init__(self, model_name="llama2"):
        """Initialize the processor with Whisper model and Ollama configuration."""
        self.model_name = model_name
        self.whisper_model = None
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Temporary directory created: {self.temp_dir}")
        
        # Initialize Whisper model once
        self._load_whisper_model()
        
        # Test Ollama availability
        self._test_ollama_connection()
    
    def _load_whisper_model(self):
        """Load Whisper model once for efficiency."""
        try:
            logger.info("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _test_ollama_connection(self):
        """Test if Ollama is available and the model is accessible."""
        try:
            # Test if ollama CLI is available
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise Exception(f"Ollama CLI not available: {result.stderr}")
            
            # Test if the specific model is available
            if self.model_name not in result.stdout:
                logger.warning(f"Model '{self.model_name}' not found in Ollama. Attempting to pull...")
                pull_result = subprocess.run(
                    ["ollama", "pull", self.model_name],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutes timeout for model download
                )
                if pull_result.returncode != 0:
                    raise Exception(f"Failed to pull model '{self.model_name}': {pull_result.stderr}")
            
            logger.info(f"Ollama connection successful with model: {self.model_name}")
            
        except subprocess.TimeoutExpired:
            raise Exception("Timeout while testing Ollama connection")
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            raise
def download_audio(self, video_url, output_path):
        """Download audio from video URL using yt-dlp."""
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'outtmpl': output_path,
                'quiet': True,
                'no_warnings': True,
            }
            
            # Download audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # yt-dlp adds .wav extension automatically
            wav_path = output_path + '.wav'
            if os.path.exists(wav_path):
                return wav_path
            else:
                raise Exception("Audio file was not created")
                
        except Exception as e:
            logger.error(f"Failed to download audio from {video_url}: {e}")
            return None
def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper."""
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            result = self.whisper_model.transcribe(audio_path)
            transcription = result["text"].strip()
            logger.info("Audio transcription completed")
            return transcription
        except Exception as e:
            logger.error(f"Failed to transcribe audio {audio_path}: {e}")
            return None
    
    def classify_fact(self, fact):
        """Classify fact as real (1) or myth (0) using Ollama LLaMA 2."""
        try:
            # Prepare the prompt
            prompt = f'Given the fact: "{fact}"\nIs this fact real or myth? Answer only with 1 for real, 0 for myth.'
            
            # Run Ollama command
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout
            )
            
            if result.returncode != 0:
                raise Exception(f"Ollama command failed: {result.stderr}")
            
            # Parse response to extract 0 or 1
            response = result.stdout.strip()
            logger.info(f"Ollama response: {response}")
            
            # Look for 0 or 1 in the response
            match = re.search(r'\b[01]\b', response)
            if match:
                classification = int(match.group())
                logger.info(f"Fact classification: {classification}")
                return classification
            else:
                logger.warning(f"Could not parse classification from response: {response}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Timeout while waiting for Ollama response")
            return None
        except Exception as e:
            logger.error(f"Failed to classify fact: {e}")
            return None
