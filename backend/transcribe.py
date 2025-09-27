import os
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel

# ✅ Load model once (GPU + FP16 for speed & memory balance)
# Use "large-v3" instead of "large" if you want better accuracy with less VRAM
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# Use CPU count only for audio splitting (I/O), not transcription
max_worker = min(4, os.cpu_count())  # don’t oversubscribe threads


def transcribe_chunk(chunk_path):
    print(f"Transcribing chunk: {chunk_path}")
    segments, _ = model.transcribe(
        chunk_path,
        beam_size=5,           # higher beam_size = better accuracy, slower
        vad_filter=True,       # remove silences/long pauses
        temperature=0.0        # deterministic decoding (faster, less random)
    )
    return " ".join([seg.text for seg in segments])


def split_audio(file_path, chunk_length=300):
    """
    Split audio into ~5min chunks (default 300s) instead of 60s.
    Longer chunks = fewer context resets = better accuracy + speed.
    """
    chunks = []
    out_pattern = os.path.join(tempfile.gettempdir(), "chunk_%03d.mp3")

    command = [
        "ffmpeg", "-i", file_path,
        "-f", "segment",
        "-segment_time", str(chunk_length),
        "-c:a", "libmp3lame", "-q:a", "2",
        out_pattern, "-y"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    i = 0
    while True:
        fname = os.path.join(tempfile.gettempdir(), f"chunk_{i:03d}.mp3")
        if os.path.exists(fname):
            chunks.append(fname)
            i += 1
        else:
            break

    return chunks


def transcribe_file(file_path):
    chunks = split_audio(file_path)
    results = []

    # ✅ GPU already parallelizes work, so don’t spawn too many workers
    with ThreadPoolExecutor(max_workers=max_worker) as executor:
        futures = [executor.submit(transcribe_chunk, c) for c in chunks]
        for f in futures:
            results.append(f.result())

    transcript = " ".join(results)

    # Cleanup
    for c in chunks:
        os.remove(c)

    return transcript
