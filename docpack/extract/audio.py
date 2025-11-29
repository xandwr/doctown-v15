"""Audio/music file extraction for semantic vibe search.

Extracts rich musical features including:
- Melodic contour via CREPE F0 tracking
- Chord progressions from chroma vectors
- Spectral shape descriptors (brightness, warmth, texture)
- Emotion inference from feature combinations
- Sub-genre detection

Converts all features to natural language for semantic search.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

from .base import ExtractedDocument

AUDIO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"
})

# Chord templates for detection (root position triads)
CHORD_TEMPLATES = {
    "C": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "Cm": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "C#": [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "C#m": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    "D": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    "Dm": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    "D#": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "D#m": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    "E": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    "Em": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "F": [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    "Fm": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "F#": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    "F#m": [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    "G": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    "Gm": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    "G#": [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    "G#m": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    "A": [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "Am": [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "A#": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    "A#m": [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    "B": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    "Bm": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
}

# Sub-genre mappings based on feature combinations
SUBGENRE_PROFILES = {
    # Electronic sub-genres
    "ambient": {"tempo_range": (0, 100), "energy_range": (0, 0.3), "brightness_range": (0, 0.4)},
    "downtempo": {"tempo_range": (70, 110), "energy_range": (0.2, 0.5), "brightness_range": (0.3, 0.6)},
    "chillwave": {"tempo_range": (80, 120), "energy_range": (0.2, 0.5), "warmth_min": 0.5},
    "synthwave": {"tempo_range": (100, 130), "energy_range": (0.4, 0.7), "brightness_range": (0.5, 0.8)},
    "deep house": {"tempo_range": (115, 130), "energy_range": (0.4, 0.7), "bass_dominance_min": 0.5},
    "future garage": {"tempo_range": (125, 140), "energy_range": (0.3, 0.6), "texture": "atmospheric"},
    "trap": {"tempo_range": (130, 170), "energy_range": (0.5, 0.9), "bass_dominance_min": 0.6},
    "phonk": {"tempo_range": (130, 160), "energy_range": (0.5, 0.8), "darkness_min": 0.5},
    "drift phonk": {"tempo_range": (140, 170), "energy_range": (0.6, 0.9), "aggression_min": 0.5},
    # Other genres
    "lo-fi": {"tempo_range": (70, 100), "energy_range": (0.2, 0.5), "warmth_min": 0.5, "noisiness_min": 0.3},
    "shoegaze": {"tempo_range": (90, 130), "energy_range": (0.4, 0.7), "reverb_min": 0.6},
    "dream pop": {"tempo_range": (90, 130), "energy_range": (0.3, 0.6), "brightness_range": (0.4, 0.7)},
}


@dataclass
class SpectralFeatures:
    """Spectral shape descriptors."""
    brightness: float  # 0-1, spectral centroid normalized
    warmth: float  # 0-1, low frequency dominance
    noisiness: float  # 0-1, zero crossing rate
    harmonic_richness: float  # 0-1, spectral flatness inverse
    transient_sharpness: float  # 0-1, onset strength
    bass_dominance: float  # 0-1, low frequency energy ratio
    dynamic_range: float  # 0-1, RMS variance


@dataclass
class MelodicFeatures:
    """Melodic contour from F0 tracking."""
    contour_description: str  # e.g., "falling minor 3rd → rising perfect 4th"
    register: str  # "low", "mid", "high"
    stability: str  # "stable", "wandering", "dramatic"
    vibrato_amount: str  # "none", "subtle", "moderate", "heavy"
    intervals: list[str] = field(default_factory=list)  # detected interval movements


@dataclass
class HarmonicFeatures:
    """Chord progression and harmonic analysis."""
    chord_progression: list[str]  # ["Fm", "Db", "Eb", "C"]
    progression_text: str  # "Fm → Db → Eb → C"
    mode: str  # "major", "natural minor", "dorian", etc.
    harmonic_rhythm: str  # "slow", "moderate", "fast"
    color: str  # "bright", "dark", "modal", "chromatic"


@dataclass
class MusicFeatures:
    """Complete extracted music features."""
    # Basic
    duration_seconds: float
    bpm: float
    key: str
    scale: str

    # Spectral
    spectral: SpectralFeatures

    # Melodic (from CREPE)
    melodic: MelodicFeatures

    # Harmonic
    harmonic: HarmonicFeatures

    # Tags from PANNs
    mood_tags: list[str]
    genre_tags: list[str]

    # Inferred
    emotions: list[str]  # ["melancholic", "nostalgic", "floating"]
    subgenres: list[str]  # ["future garage", "chillwave"]
    texture_description: str  # "smooth, low-transient, reverb-rich"
    timbre_description: str  # "warm, wide, atmospheric"


def extract_audio(data: bytes, filename: str = "audio") -> ExtractedDocument:
    """Extract searchable text from audio file."""
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
                "emotions": ", ".join(features.emotions),
                "subgenres": ", ".join(features.subgenres),
            },
        )
    finally:
        os.unlink(temp_path)


def _analyze_audio(path: str) -> MusicFeatures:
    """Extract comprehensive features using librosa + CREPE + PANNs."""
    import librosa
    import numpy as np

    # Load audio
    y, sr = librosa.load(path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # Tempo detection
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if tempo > 0 else 0.0

    # Key and scale detection
    key, scale = _detect_key(y, sr)

    # Spectral features
    spectral = _extract_spectral_features(y, sr)

    # Melodic contour (CREPE)
    melodic = _extract_melodic_features(y, sr)

    # Harmonic analysis (chord progression)
    harmonic = _extract_harmonic_features(y, sr, key, scale)

    # PANNs tags
    mood_tags, genre_tags = _get_panns_tags(path)

    # Infer emotions from feature combinations
    emotions = _infer_emotions(bpm, scale, spectral, melodic, harmonic, mood_tags)

    # Detect sub-genres
    subgenres = _detect_subgenres(bpm, spectral, genre_tags)

    # Generate texture and timbre descriptions
    texture_desc = _describe_texture(spectral)
    timbre_desc = _describe_timbre(spectral)

    return MusicFeatures(
        duration_seconds=duration,
        bpm=bpm,
        key=key,
        scale=scale,
        spectral=spectral,
        melodic=melodic,
        harmonic=harmonic,
        mood_tags=mood_tags,
        genre_tags=genre_tags,
        emotions=emotions,
        subgenres=subgenres,
        texture_description=texture_desc,
        timbre_description=timbre_desc,
    )


def _detect_key(y, sr) -> tuple[str, str]:
    """Detect key and scale using chroma features."""
    import librosa
    import numpy as np

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Krumhansl-Kessler profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_corr = -1
    best_key = "C"
    best_scale = "major"

    for i, key in enumerate(keys):
        rolled = np.roll(chroma_mean, -i)
        major_corr = float(np.corrcoef(rolled, major_profile)[0, 1])
        minor_corr = float(np.corrcoef(rolled, minor_profile)[0, 1])

        if major_corr > best_corr:
            best_corr = major_corr
            best_key = key
            best_scale = "major"
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = key
            best_scale = "minor"

    return best_key, best_scale


def _extract_spectral_features(y, sr) -> SpectralFeatures:
    """Extract spectral shape descriptors."""
    import librosa
    import numpy as np

    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = float(np.clip(np.mean(centroid) / 5000, 0, 1))

    # Spectral rolloff (warmth - inverse of high frequency content)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    warmth = float(1.0 - np.clip(np.mean(rolloff) / sr, 0, 1))

    # Zero crossing rate (noisiness/texture)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    noisiness = float(np.clip(np.mean(zcr) * 10, 0, 1))

    # Spectral flatness (harmonic richness is inverse)
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    harmonic_richness = float(1.0 - np.clip(np.mean(flatness), 0, 1))

    # Onset strength (transient sharpness)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    transient_sharpness = float(np.clip(np.std(onset_env) / 2, 0, 1))

    # Bass dominance (energy in low frequencies)
    spec = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    low_freq_mask = freqs < 250
    low_energy = np.sum(spec[low_freq_mask, :])
    total_energy = np.sum(spec)
    bass_dominance = float(low_energy / total_energy if total_energy > 0 else 0)

    # Dynamic range (RMS variance)
    rms = librosa.feature.rms(y=y)[0]
    dynamic_range = float(np.clip(np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0, 0, 1))

    return SpectralFeatures(
        brightness=brightness,
        warmth=warmth,
        noisiness=noisiness,
        harmonic_richness=harmonic_richness,
        transient_sharpness=transient_sharpness,
        bass_dominance=bass_dominance,
        dynamic_range=dynamic_range,
    )


def _extract_melodic_features(y, sr) -> MelodicFeatures:
    """Extract melodic contour using CREPE for F0 tracking."""
    import numpy as np

    try:
        import crepe
        # Get F0 predictions (step size in ms for efficiency)
        _, frequency, confidence, _ = crepe.predict(y, sr, step_size=50, viterbi=True)

        # Filter by confidence
        confident_mask = confidence > 0.5
        f0_confident = frequency[confident_mask]

        if len(f0_confident) < 10:
            return MelodicFeatures(
                contour_description="minimal melodic content",
                register="mid",
                stability="stable",
                vibrato_amount="none",
                intervals=[],
            )

        # Analyze register
        median_f0 = np.median(f0_confident)
        if median_f0 < 200:
            register = "low"
        elif median_f0 < 400:
            register = "low-mid"
        elif median_f0 < 600:
            register = "mid"
        elif median_f0 < 800:
            register = "mid-high"
        else:
            register = "high"

        # Analyze stability (variance in cents)
        cents = 1200 * np.log2(f0_confident / np.median(f0_confident))
        cents_std = np.std(cents)
        if cents_std < 50:
            stability = "stable, focused"
        elif cents_std < 150:
            stability = "moderately varied"
        elif cents_std < 300:
            stability = "wandering, expressive"
        else:
            stability = "dramatic, wide-ranging"

        # Detect vibrato (periodic variation in F0)
        f0_diff = np.diff(f0_confident)
        sign_changes = np.sum(np.diff(np.sign(f0_diff)) != 0)
        vibrato_rate = sign_changes / len(f0_diff) if len(f0_diff) > 0 else 0
        if vibrato_rate < 0.1:
            vibrato = "none"
        elif vibrato_rate < 0.3:
            vibrato = "subtle"
        elif vibrato_rate < 0.5:
            vibrato = "moderate"
        else:
            vibrato = "heavy, expressive"

        # Analyze melodic intervals
        intervals = _analyze_intervals(f0_confident)
        contour_desc = _describe_contour(f0_confident, intervals)

        return MelodicFeatures(
            contour_description=contour_desc,
            register=register,
            stability=stability,
            vibrato_amount=vibrato,
            intervals=intervals[:5],  # Top 5 intervals
        )

    except Exception:
        # Fallback if CREPE fails
        return MelodicFeatures(
            contour_description="melodic analysis unavailable",
            register="mid",
            stability="unknown",
            vibrato_amount="unknown",
            intervals=[],
        )


def _analyze_intervals(f0: "np.ndarray") -> list[str]:
    """Analyze melodic intervals from F0 contour."""
    import numpy as np

    intervals = []
    interval_names = {
        0: "unison", 1: "minor 2nd", 2: "major 2nd", 3: "minor 3rd",
        4: "major 3rd", 5: "perfect 4th", 6: "tritone", 7: "perfect 5th",
        8: "minor 6th", 9: "major 6th", 10: "minor 7th", 11: "major 7th",
        12: "octave",
    }

    # Sample every N frames to get meaningful intervals
    step = max(1, len(f0) // 20)
    sampled = f0[::step]

    for i in range(len(sampled) - 1):
        if sampled[i] > 0 and sampled[i + 1] > 0:
            cents = 1200 * np.log2(sampled[i + 1] / sampled[i])
            semitones = int(round(cents / 100))
            direction = "rising" if semitones > 0 else "falling"
            semitones = abs(semitones) % 13

            if semitones > 0:
                interval_name = interval_names.get(semitones, f"{semitones} semitones")
                intervals.append(f"{direction} {interval_name}")

    # Return unique intervals in order of appearance
    seen = set()
    unique = []
    for i in intervals:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique


def _describe_contour(f0: "np.ndarray", intervals: list[str]) -> str:
    """Generate natural language description of melodic contour."""
    import numpy as np

    if len(f0) < 5:
        return "minimal melodic content"

    # Overall direction
    start_avg = np.mean(f0[:len(f0)//4])
    end_avg = np.mean(f0[-len(f0)//4:])
    if end_avg > start_avg * 1.1:
        overall = "ascending"
    elif end_avg < start_avg * 0.9:
        overall = "descending"
    else:
        overall = "centered"

    # Build description
    if intervals:
        top_intervals = " → ".join(intervals[:3])
        return f"{overall} contour: {top_intervals}"
    else:
        return f"{overall} melodic line"


def _extract_harmonic_features(y, sr, key: str, scale: str) -> HarmonicFeatures:
    """Extract chord progression from chroma features."""
    import librosa
    import numpy as np

    # Get chroma with longer hop for chord-level analysis
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=8192)

    # Detect chords frame by frame
    chords = []
    for i in range(chroma.shape[1]):
        frame = chroma[:, i]
        best_chord = _match_chord(frame)
        if best_chord:
            chords.append(best_chord)

    # Simplify to unique progression
    if not chords:
        return HarmonicFeatures(
            chord_progression=[],
            progression_text="no clear harmonic content",
            mode=scale,
            harmonic_rhythm="unclear",
            color="neutral",
        )

    # Remove consecutive duplicates
    simplified = [chords[0]]
    for c in chords[1:]:
        if c != simplified[-1]:
            simplified.append(c)

    # Take first 8 unique chords as the "progression"
    progression = simplified[:8]

    # Determine harmonic rhythm
    changes_per_second = len(simplified) / (len(chords) * 8192 / sr) if chords else 0
    if changes_per_second < 0.3:
        rhythm = "slow, sustained"
    elif changes_per_second < 0.8:
        rhythm = "moderate"
    else:
        rhythm = "fast, active"

    # Determine harmonic color
    minor_count = sum(1 for c in progression if "m" in c and "maj" not in c.lower())
    major_count = len(progression) - minor_count

    if minor_count > major_count * 1.5:
        color = "dark, minor-heavy"
    elif major_count > minor_count * 1.5:
        color = "bright, major-heavy"
    else:
        color = "modal, mixed"

    # Check for chromatic movement
    roots = [c.replace("m", "").replace("#", "s") for c in progression]
    if len(set(roots)) > 4:
        color += ", chromatic"

    return HarmonicFeatures(
        chord_progression=progression,
        progression_text=" → ".join(progression) if progression else "ambiguous",
        mode=f"{scale}" if scale else "ambiguous",
        harmonic_rhythm=rhythm,
        color=color,
    )


def _match_chord(chroma_frame: "np.ndarray") -> str | None:
    """Match a chroma frame to the best chord template."""
    import numpy as np

    best_score = 0.5  # Minimum threshold
    best_chord = None

    for chord_name, template in CHORD_TEMPLATES.items():
        template_arr = np.array(template, dtype=float)
        score = float(np.dot(chroma_frame, template_arr) / (np.linalg.norm(chroma_frame) * np.linalg.norm(template_arr) + 1e-6))
        if score > best_score:
            best_score = score
            best_chord = chord_name

    return best_chord


def _get_panns_tags(path: str) -> tuple[list[str], list[str]]:
    """Get mood and genre tags from PANNs."""
    import csv
    import librosa
    from panns_inference import AudioTagging

    y, _ = librosa.load(path, sr=32000, mono=True)

    at = AudioTagging(checkpoint_path=None, device="cpu")
    clipwise_output, _ = at.inference(y[None, :])

    labels_path = os.path.expanduser("~/panns_data/class_labels_indices.csv")
    class_labels: list[str] = []
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_labels.append(row["display_name"])

    scores = clipwise_output[0]
    top_indices = scores.argsort()[-50:][::-1]

    mood_keywords = {
        "happy", "sad", "angry", "tender", "exciting", "scary",
        "melancholic", "peaceful", "aggressive", "relaxed", "energetic",
    }
    genre_keywords = {
        "jazz", "rock", "pop", "classical", "electronic", "hip hop",
        "country", "blues", "metal", "folk", "soul", "reggae", "punk",
        "techno", "house", "ambient", "disco", "funk", "r&b",
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

        for mood in mood_keywords:
            if mood in label and mood not in mood_tags:
                mood_tags.append(mood)
                break

        for genre in genre_keywords:
            if genre in label and genre not in genre_tags:
                genre_tags.append(genre)
                break

    return mood_tags[:5], genre_tags[:5]


def _infer_emotions(
    bpm: float,
    scale: str,
    spectral: SpectralFeatures,
    melodic: MelodicFeatures,
    harmonic: HarmonicFeatures,
    mood_tags: list[str],
) -> list[str]:
    """Infer emotional qualities from feature combinations."""
    emotions = []

    # Melancholic: slow + minor + low brightness + descending melody
    if bpm < 100 and scale == "minor" and spectral.brightness < 0.5:
        emotions.append("melancholic")

    # Nostalgic: moderate tempo + warm + minor or modal
    if 80 < bpm < 120 and spectral.warmth > 0.5 and scale == "minor":
        emotions.append("nostalgic")

    # Floating/dreamy: slow + low transients + warm + stable melody
    if bpm < 110 and spectral.transient_sharpness < 0.3 and spectral.warmth > 0.4:
        emotions.append("floating")
        emotions.append("dreamy")

    # Expansive: wide dynamic range + harmonic richness
    if spectral.dynamic_range > 0.4 and spectral.harmonic_richness > 0.5:
        emotions.append("expansive")

    # Introspective: low energy + stable melody + minor
    if spectral.brightness < 0.4 and "stable" in melodic.stability and scale == "minor":
        emotions.append("introspective")

    # Bittersweet: major key but low brightness, or mixed harmony
    if scale == "major" and spectral.brightness < 0.4:
        emotions.append("bittersweet")

    # Aggressive: high transients + bright + fast
    if bpm > 120 and spectral.transient_sharpness > 0.5 and spectral.brightness > 0.5:
        emotions.append("aggressive")
        emotions.append("intense")

    # Euphoric: fast + major + bright + high energy
    if bpm > 120 and scale == "major" and spectral.brightness > 0.5:
        emotions.append("euphoric")
        emotions.append("uplifting")

    # Dark: low brightness + bass heavy + minor
    if spectral.brightness < 0.3 and spectral.bass_dominance > 0.4 and scale == "minor":
        emotions.append("dark")
        emotions.append("brooding")

    # Peaceful: slow + low transients + stable
    if bpm < 90 and spectral.transient_sharpness < 0.3:
        emotions.append("peaceful")
        emotions.append("serene")

    # Energetic: from mood tags or high tempo
    if "energetic" in mood_tags or bpm > 130:
        emotions.append("energetic")

    # Add mood tags that are emotional
    for tag in mood_tags:
        if tag not in emotions:
            emotions.append(tag)

    return list(dict.fromkeys(emotions))[:8]  # Unique, max 8


def _detect_subgenres(bpm: float, spectral: SpectralFeatures, genre_tags: list[str]) -> list[str]:
    """Detect sub-genres based on feature profiles."""
    matches = []

    for subgenre, profile in SUBGENRE_PROFILES.items():
        score = 0
        checks = 0

        if "tempo_range" in profile:
            checks += 1
            if profile["tempo_range"][0] <= bpm <= profile["tempo_range"][1]:
                score += 1

        if "energy_range" in profile:
            checks += 1
            # Use brightness + transients as energy proxy
            energy = (spectral.brightness + spectral.transient_sharpness) / 2
            if profile["energy_range"][0] <= energy <= profile["energy_range"][1]:
                score += 1

        if "brightness_range" in profile:
            checks += 1
            if profile["brightness_range"][0] <= spectral.brightness <= profile["brightness_range"][1]:
                score += 1

        if "warmth_min" in profile:
            checks += 1
            if spectral.warmth >= profile["warmth_min"]:
                score += 1

        if "bass_dominance_min" in profile:
            checks += 1
            if spectral.bass_dominance >= profile["bass_dominance_min"]:
                score += 1

        if "noisiness_min" in profile:
            checks += 1
            if spectral.noisiness >= profile["noisiness_min"]:
                score += 1

        # Require at least 60% match
        if checks > 0 and score / checks >= 0.6:
            matches.append((subgenre, score / checks))

    # Sort by match quality
    matches.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matches[:3]]


def _describe_texture(spectral: SpectralFeatures) -> str:
    """Generate texture description from spectral features."""
    parts = []

    # Smoothness based on transients
    if spectral.transient_sharpness < 0.3:
        parts.append("smooth")
    elif spectral.transient_sharpness > 0.6:
        parts.append("punchy")
    else:
        parts.append("textured")

    # Noise character
    if spectral.noisiness > 0.5:
        parts.append("gritty")
    elif spectral.noisiness < 0.2:
        parts.append("clean")

    # Dynamic character
    if spectral.dynamic_range > 0.5:
        parts.append("dynamic")
    elif spectral.dynamic_range < 0.2:
        parts.append("compressed")

    # Harmonic character
    if spectral.harmonic_richness > 0.6:
        parts.append("harmonically rich")
    elif spectral.harmonic_richness < 0.3:
        parts.append("sparse")

    return ", ".join(parts) if parts else "balanced texture"


def _describe_timbre(spectral: SpectralFeatures) -> str:
    """Generate timbre description from spectral features."""
    parts = []

    # Temperature
    if spectral.warmth > 0.6:
        parts.append("warm")
    elif spectral.warmth < 0.3:
        parts.append("cold")
    else:
        parts.append("balanced")

    # Brightness
    if spectral.brightness > 0.6:
        parts.append("bright")
    elif spectral.brightness < 0.3:
        parts.append("dark")

    # Bass
    if spectral.bass_dominance > 0.5:
        parts.append("bass-heavy")
    elif spectral.bass_dominance < 0.2:
        parts.append("thin")
    else:
        parts.append("full")

    # Width (approximated from harmonic richness + dynamic range)
    width = (spectral.harmonic_richness + spectral.dynamic_range) / 2
    if width > 0.5:
        parts.append("wide")
    elif width < 0.3:
        parts.append("narrow")

    return ", ".join(parts) if parts else "neutral timbre"


def _build_description(filename: str, f: MusicFeatures) -> str:
    """Build rich natural language description for semantic search."""
    mins = int(f.duration_seconds // 60)
    secs = int(f.duration_seconds % 60)

    # Tempo feel
    if f.bpm < 80:
        tempo_feel = "slow, contemplative"
    elif f.bpm < 100:
        tempo_feel = "downtempo, relaxed"
    elif f.bpm < 120:
        tempo_feel = "moderate groove"
    elif f.bpm < 140:
        tempo_feel = "upbeat, driving"
    else:
        tempo_feel = "fast, high-energy"

    # Build the rich description
    lines = [
        f"[Audio: {filename}]",
        "",
    ]

    # Opening vibe sentence
    emotions_str = ", ".join(f.emotions[:4]) if f.emotions else "atmospheric"
    subgenres_str = ", ".join(f.subgenres[:2]) if f.subgenres else ", ".join(f.genre_tags[:2]) if f.genre_tags else "electronic"
    lines.append(f"A {emotions_str} {subgenres_str} track with {f.timbre_description} character.")

    # Harmonic description
    if f.harmonic.chord_progression:
        lines.append(f"The {f.key} {f.scale} tonality features a {f.harmonic.color} harmonic palette.")
        lines.append(f"Chord progression: {f.harmonic.progression_text}")
    else:
        lines.append(f"Built around {f.key} {f.scale} with {f.harmonic.color} harmonic character.")

    # Melodic description
    if f.melodic.contour_description and "unavailable" not in f.melodic.contour_description:
        lines.append(f"Melodic contour: {f.melodic.contour_description}")
        lines.append(f"Vocal/lead line: {f.melodic.register} register, {f.melodic.stability}, {f.melodic.vibrato_amount} vibrato")

    lines.append("")

    # Spectral/timbre section
    lines.append("Sonic character:")
    lines.append(f"- Timbre: {f.timbre_description}")
    lines.append(f"- Texture: {f.texture_description}")
    lines.append(f"- Spectral shape: {'bass-heavy' if f.spectral.bass_dominance > 0.5 else 'balanced low-end'}"
                 f" with {'bright' if f.spectral.brightness > 0.5 else 'warm'} upper harmonics")

    lines.append("")

    # Musical characteristics
    lines.append("Musical characteristics:")
    lines.append(f"- Tempo: {int(f.bpm)} BPM ({tempo_feel})")
    lines.append(f"- Key: {f.key} {f.scale}")
    lines.append(f"- Harmonic rhythm: {f.harmonic.harmonic_rhythm}")
    lines.append(f"- Duration: {mins}:{secs:02d}")

    lines.append("")

    # Emotional qualities
    if f.emotions:
        lines.append(f"Emotional qualities: {', '.join(f.emotions)}")

    # Tags for exact matching
    all_tags = []
    all_tags.extend(f.emotions)
    all_tags.extend(f.mood_tags)
    all_tags.extend(f.subgenres)
    all_tags.extend(f.genre_tags)
    all_tags.append(f.scale)
    all_tags.append(tempo_feel.split(",")[0])
    unique_tags = list(dict.fromkeys(all_tags))

    lines.append(f"Genre/style: {', '.join(f.subgenres + f.genre_tags)}")
    lines.append(f"Searchable tags: {', '.join(unique_tags)}")

    return "\n".join(lines)


def can_extract_audio(extension: str) -> bool:
    """Check if the extension is a supported audio format."""
    return extension.lower() in AUDIO_EXTENSIONS
