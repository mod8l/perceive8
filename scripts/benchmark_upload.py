#!/usr/bin/env python3
"""Benchmark script for testing large file uploads to perceive8 API."""
import argparse
import os
import sys
import time
import httpx

def main():
    parser = argparse.ArgumentParser(description="Benchmark large file uploads to perceive8 API")
    parser.add_argument("file_path", help="Path to the audio file to upload")
    parser.add_argument("user_id", nargs="?", default="benchmark-user", help="User ID (default: benchmark-user)")
    parser.add_argument("-l", "--language", default="auto", help="Language code (default: auto)")
    args = parser.parse_args()

    file_path = args.file_path
    user_id = args.user_id
    language = args.language
    api_url = "http://localhost:8000"
    
    file_size = os.path.getsize(file_path)
    print(f"File: {file_path}")
    print(f"Size: {file_size / (1024*1024):.1f} MB")
    print(f"API: {api_url}")
    print(f"User: {user_id}")
    print(f"Language: {language}")
    print("---")
    
    start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Starting upload...")
    
    with open(file_path, "rb") as f:
        with httpx.Client(timeout=None) as client:
            response = client.post(
                f"{api_url}/analysis/analyze",
                data={"user_id": user_id, "language": language},
                files={"audio_file": (os.path.basename(file_path), f)},
            )
    
    elapsed = time.time() - start
    print(f"[{time.strftime('%H:%M:%S')}] Complete!")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

if __name__ == "__main__":
    main()
