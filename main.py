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

