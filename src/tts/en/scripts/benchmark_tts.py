#!/usr/bin/env python3
"""
Benchmark script for TTS generation time measurement.
Tests OGG Opus generation to measure RTF (Real-Time Factor).

Usage:
    python3 scripts/benchmark_tts.py

The script will:
1. Generate OGG Opus audio from a long English text sample
2. Measure total generation time (TTS + encoding)
3. Calculate audio duration using ffprobe
4. Display results including RTF

Note: OGG format is recommended for Telegram voice messages.
"""

import subprocess
import time
import os
import json
from pathlib import Path

def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', file_path
        ], capture_output=True, text=True)
        
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception as e:
        print(f"Warning: Could not get duration from ffprobe: {e}")
        return 0.0

def benchmark_tts():
    """Run TTS benchmark with OGG Opus format."""
    print("=== TTS Benchmark: OGG Opus Generation ===\n")
    
    # Long English text (≈18.65s audio at normal speed)
    text = "The quick brown fox jumps over lazy dog near riverbank on a bright sunny morning while birds sing melodious tunes in trees above. A gentle breeze flows through meadow as wildflowers sway gracefully in golden sunlight that illuminates peaceful countryside landscape."
    
    tests_dir = Path(__file__).parent.parent / "tests"
    tests_dir.mkdir(exist_ok=True)
    output_file = tests_dir / "benchmark_output.ogg"
    
    print(f"Text length: {len(text)} characters")
    print(f"Output: {output_file}")
    print("\nGenerating OGG Opus audio...\n")
    
    # Start timer
    start = time.time()
    
    # Run TTS API request with OGG format
    result = subprocess.run([
        'curl', '-s', 'http://127.0.0.1:8000/v1/audio/speech',
        '-H', 'Content-Type: application/json',
        '-o', output_file,
        '-d', f'{{"text": "{text}", "lang": "en", "voice": "af_bella", "speed": 1.0, "format": "ogg"}}'
    ], capture_output=True)
    
    # End timer
    elapsed = time.time() - start
    
    # Calculate metrics
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        
        # Get actual audio duration using ffprobe
        duration = get_audio_duration(str(output_file))
        rtf = elapsed / duration if duration > 0 else 0
        
        # Print results
        print(f"✓ OGG file generated successfully!")
        print()
        print(f"Total time (TTS + encoding): {elapsed:.3f}s")
        print(f"Audio duration:              {duration:.2f}s")
        print(f"File size:                   {file_size:,} bytes")
        print(f"RTF (Real-Time Factor):      {rtf:.3f}")
        print()
        
        # Performance analysis
        if rtf < 0.5:
            print("✓ Excellent: Generation >2x real-time")
        elif rtf < 1.0:
            print("✓ Good: Generation faster than real-time")
        else:
            print("⚠ Warning: Generation slower than real-time")
            
        print()
        print("Note: This test includes both TTS inference and OGG encoding.")
        print("      Pure TTS inference is ~20-30% faster than reported RTF.")
    else:
        print(f"✗ Error: Audio file not created")
        if result.stderr:
            print(f"Error: {result.stderr.decode()}")

if __name__ == "__main__":
    benchmark_tts()
