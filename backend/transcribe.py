import os
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from faster_whisper import WhisperModel


model = WhisperModel("small", device="cpu", compute_type="int8")
max_worker = os.cpu_count()

def transcribe_chunk(chunk_path):
    print(f"Transcribing chunk: {chunk_path}")
    segments, info = model.transcribe(chunk_path, beam_size=5, language="tl")
    return " ".join([seg.text for seg in segments])

def split_audio(file_path, chunk_length=60):
    # Split audio into chunks using ffmpeg (every `chunk_length` seconds)
    chunks = []
    out_pattern = os.path.join(tempfile.gettempdir(), "chunk_%03d.mp3")

    command = [
        "ffmpeg", "-i", file_path,
        "-f", "segment",
        "-segment_time", str(chunk_length),
        "-c", "copy", out_pattern, "-y"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Collect generated chunks
    i = 0
    while True:
        fname = os.path.join(tempfile.gettempdir(), f"chunk_{i:03d}.mp3")
        if os.path.exists(fname):
            chunks.append(fname)
            i += 1
        else:
            break

    return chunks

def transcribe_file(file_path, worker=max_worker):
    chunks = split_audio(file_path)
    result = []

    with ThreadPoolExecutor(max_workers=worker) as executor:
        futures = [executor.submit(transcribe_chunk, c) for c in chunks]
        for f in futures:
            print(f"Waiting for {f}")
            result.append(f.result())

    transcript = " ".join(result)

    # cleanup
    for c in chunks:
        os.remove(c)

    return transcript
