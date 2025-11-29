"""Audio/music file extraction for semantic vibe search.

Extracts musical features (tempo, key, energy, mood/genre tags) and converts
them to natural language descriptions that can be semantically searched.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

from .base import ExtractedDocument

AUDIO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"
})


@dataclass
class MusicFeatures:
    """Extracted music features."""

    duration_seconds: float
    bpm: float
    key: str
    scale: str  # "major" or "minor"
    energy: float  # 0-1
    mood_tags: list[str]  # ["sad", "happy", "relaxed", ...]
    genre_tags: list[str]  # ["rock", "jazz", "electronic", ...]


def extract_audio(data: bytes, filename: str = "audio") -> ExtractedDocument:
    """Extract searchable text from audio file."""
    # Write temp file (librosa needs file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        temp_path = f.name

    try:
        features = _analyze_audio(temp_path)
        text = _build_description(filename, features)

        return ExtractedDocument(
            text=text,
            images=[],
            page_count=None,
            metadata={
                "type": "audio",
                "bpm": str(int(features.bpm)),
                "key": f"{features.key} {features.scale}",
                "duration": f"{int(features.duration_seconds)}s",
            },
        )
    finally:
        os.unlink(temp_path)


def _analyze_audio(path: str) -> MusicFeatures:
    """Extract features using librosa + PANNs."""
    import librosa
    import numpy as np

    # Load audio
    y, sr = librosa.load(path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Key estimation via chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_idx = int(np.argmax(np.mean(chroma, axis=1)))
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key = keys[key_idx]

    # Major/minor detection using Krumhansl-Kessler profiles
    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )
    chroma_mean = np.mean(chroma, axis=1)
    chroma_rolled = np.roll(chroma_mean, -key_idx)
    major_corr = float(np.corrcoef(chroma_rolled, major_profile)[0, 1])
    minor_corr = float(np.corrcoef(chroma_rolled, minor_profile)[0, 1])
    scale = "major" if major_corr > minor_corr else "minor"

    # Energy (RMS normalized)
    rms = librosa.feature.rms(y=y)[0]
    energy = float(np.clip(np.mean(rms) * 10, 0, 1))

    # PANNs mood/genre tagging
    mood_tags, genre_tags = _get_panns_tags(path)

    return MusicFeatures(
        duration_seconds=duration,
        bpm=float(tempo),
        key=key,
        scale=scale,
        energy=energy,
        mood_tags=mood_tags,
        genre_tags=genre_tags,
    )


def _get_panns_tags(path: str) -> tuple[list[str], list[str]]:
    """Get mood and genre tags from PANNs."""
    import csv
    import os
    import librosa
    from panns_inference import AudioTagging

    # Load audio for PANNs (needs numpy array, not path)
    y, _ = librosa.load(path, sr=32000, mono=True)  # PANNs expects 32kHz

    at = AudioTagging(checkpoint_path=None, device="cpu")  # auto-downloads model
    clipwise_output, _ = at.inference(y[None, :])

    # Load class labels from CSV
    labels_path = os.path.expanduser("~/panns_data/class_labels_indices.csv")
    class_labels: list[str] = []
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_labels.append(row["display_name"])

    # Get top predictions
    scores = clipwise_output[0]
    top_indices = scores.argsort()[-50:][::-1]

    # Filter for mood and genre labels
    mood_keywords = {
        "happy", "sad", "angry", "tender", "exciting", "scary",
        "melancholic", "peaceful", "aggressive", "relaxed", "energetic",
    }
    genre_keywords = {
        "jazz", "rock", "pop", "classical", "electronic", "hip hop",
        "country", "blues", "metal", "folk", "soul", "reggae", "punk",
    }

    mood_tags: list[str] = []
    genre_tags: list[str] = []

    for idx in top_indices:
        if idx >= len(class_labels):
            continue
        label = class_labels[idx].lower()
        score = scores[idx]
        if score < 0.1:
            continue

        # Check for mood-related labels
        for mood in mood_keywords:
            if mood in label and mood not in mood_tags:
                mood_tags.append(mood)
                break

        # Check for genre labels
        for genre in genre_keywords:
            if genre in label and genre not in genre_tags:
                genre_tags.append(genre)
                break

    return mood_tags[:5], genre_tags[:3]


def _build_description(filename: str, f: MusicFeatures) -> str:
    """Build natural language description for semantic search."""
    # Tempo feel
    if f.bpm < 80:
        tempo_feel = "slow, contemplative"
    elif f.bpm < 120:
        tempo_feel = "moderate, relaxed"
    elif f.bpm < 140:
        tempo_feel = "upbeat, energetic"
    else:
        tempo_feel = "fast, driving"

    # Energy feel
    if f.energy < 0.3:
        energy_feel = "soft, gentle, quiet"
    elif f.energy < 0.6:
        energy_feel = "moderate energy"
    else:
        energy_feel = "loud, powerful, intense"

    # Key mood (music theory associations)
    key_moods = {
        "minor": "melancholic, emotional, introspective",
        "major": "bright, cheerful, uplifting",
    }
    key_feel = key_moods.get(f.scale, f.scale)

    # Build description
    mins = int(f.duration_seconds // 60)
    secs = int(f.duration_seconds % 60)

    mood_str = ", ".join(f.mood_tags) if f.mood_tags else "neutral"
    genre_str = ", ".join(f.genre_tags) if f.genre_tags else "unclassified"

    lines = [
        f"[Audio: {filename}]",
        "",
        f"This track has a {mood_str} vibe with {genre_str} influences.",
        f"The {f.key} {f.scale} key gives it a {key_feel} tonality.",
        f"At {int(f.bpm)} BPM, the tempo feels {tempo_feel}.",
        f"The overall energy is {energy_feel}.",
        "",
        "Musical characteristics:",
        f"- Tempo: {int(f.bpm)} BPM ({tempo_feel})",
        f"- Key: {f.key} {f.scale}",
        f"- Energy: {f.energy:.0%}",
        f"- Duration: {mins}:{secs:02d}",
        "",
        f"Mood tags: {mood_str}",
        f"Genre tags: {genre_str}",
    ]

    return "\n".join(lines)


def can_extract_audio(extension: str) -> bool:
    """Check if the extension is a supported audio format."""
    return extension.lower() in AUDIO_EXTENSIONS
