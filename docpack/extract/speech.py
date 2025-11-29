"""Speech transcription using Distil-Whisper.

Provides local speech-to-text transcription with word-level timestamps
for chunking spoken content in audio files.

Uses distil-whisper/distil-large-v3 from HuggingFace for efficient
local inference with high accuracy.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field


@dataclass
class TranscriptionSegment:
    """A segment of transcribed speech with timing."""
    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SpeechFeatures:
    """Complete speech transcription with segments."""
    full_text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str = "en"
    has_speech: bool = False
    speech_ratio: float = 0.0  # Ratio of audio that contains speech

    def get_chunked_transcript(self, max_chunk_duration: float = 30.0) -> list[str]:
        """Get transcript split into time-based chunks."""
        if not self.segments:
            return [self.full_text] if self.full_text else []

        chunks = []
        current_chunk = []
        chunk_start = self.segments[0].start if self.segments else 0

        for segment in self.segments:
            current_chunk.append(segment.text)

            # Check if we've exceeded the chunk duration
            if segment.end - chunk_start >= max_chunk_duration:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                chunk_start = segment.end

        # Add remaining text
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


# Global model cache for efficiency
_whisper_model = None
_whisper_processor = None


def _load_whisper_model():
    """Lazily load the Distil-Whisper model."""
    global _whisper_model, _whisper_processor

    if _whisper_model is not None:
        return _whisper_model, _whisper_processor

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "distil-whisper/distil-large-v3"

    _whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    _whisper_model.to(device)

    _whisper_processor = AutoProcessor.from_pretrained(model_id)

    return _whisper_model, _whisper_processor


def _get_whisper_pipeline():
    """Get the Whisper ASR pipeline."""
    import torch
    from transformers import pipeline

    model, processor = _load_whisper_model()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
    )


def transcribe_audio(
    audio_path: str,
    *,
    language: str | None = None,
    chunk_length_s: int = 30,
    batch_size: int = 16,
) -> SpeechFeatures:
    """
    Transcribe speech from an audio file using Distil-Whisper.

    Args:
        audio_path: Path to audio file (WAV, MP3, etc.)
        language: Optional language code (auto-detected if None)
        chunk_length_s: Chunk length for processing long audio
        batch_size: Batch size for inference

    Returns:
        SpeechFeatures with full transcription and timed segments
    """
    try:
        pipe = _get_whisper_pipeline()

        # Configure generation
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language

        # Run transcription with timestamps
        result = pipe(
            audio_path,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )

        # Parse results
        full_text = result.get("text", "").strip()
        chunks = result.get("chunks", [])

        segments = []
        for chunk in chunks:
            timestamp = chunk.get("timestamp", (0, 0))
            start = timestamp[0] if timestamp[0] is not None else 0
            end = timestamp[1] if timestamp[1] is not None else start

            segments.append(TranscriptionSegment(
                text=chunk.get("text", "").strip(),
                start=start,
                end=end,
            ))

        # Calculate speech ratio
        import librosa
        duration = librosa.get_duration(path=audio_path)
        speech_duration = sum(s.duration for s in segments)
        speech_ratio = speech_duration / duration if duration > 0 else 0

        has_speech = len(full_text) > 10 and speech_ratio > 0.05

        return SpeechFeatures(
            full_text=full_text,
            segments=segments,
            language=language or "auto",
            has_speech=has_speech,
            speech_ratio=min(1.0, speech_ratio),
        )

    except Exception as e:
        # Return empty features on failure
        return SpeechFeatures(
            full_text="",
            segments=[],
            language="unknown",
            has_speech=False,
            speech_ratio=0.0,
        )


def transcribe_bytes(
    audio_data: bytes,
    *,
    language: str | None = None,
) -> SpeechFeatures:
    """
    Transcribe speech from audio bytes.

    Args:
        audio_data: Raw audio bytes
        language: Optional language code

    Returns:
        SpeechFeatures with transcription
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_data)
        temp_path = f.name

    try:
        return transcribe_audio(temp_path, language=language)
    finally:
        os.unlink(temp_path)


def detect_speech_segments(
    audio_path: str,
    *,
    threshold: float = 0.5,
    min_speech_duration: float = 0.5,
    min_silence_duration: float = 0.3,
) -> list[tuple[float, float]]:
    """
    Detect speech segments using Voice Activity Detection (VAD).

    Uses Silero VAD for efficient speech detection.

    Args:
        audio_path: Path to audio file
        threshold: VAD threshold (0-1)
        min_speech_duration: Minimum speech segment duration
        min_silence_duration: Minimum silence to split segments

    Returns:
        List of (start, end) tuples for speech segments
    """
    try:
        import torch
        import librosa

        # Load Silero VAD
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
        )

        get_speech_timestamps, _, read_audio, *_ = utils

        # Read audio at 16kHz (required by Silero VAD)
        wav = read_audio(audio_path, sampling_rate=16000)

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            threshold=threshold,
            min_speech_duration_ms=int(min_speech_duration * 1000),
            min_silence_duration_ms=int(min_silence_duration * 1000),
            return_seconds=True,
        )

        return [(s['start'], s['end']) for s in speech_timestamps]

    except Exception:
        # Fallback: assume entire audio might be speech
        return []


def is_speech_content(audio_path: str, *, threshold: float = 0.3) -> bool:
    """
    Quick check if audio contains significant speech.

    Args:
        audio_path: Path to audio file
        threshold: Minimum speech ratio to return True

    Returns:
        True if audio appears to contain speech
    """
    import librosa

    segments = detect_speech_segments(audio_path)
    if not segments:
        return False

    duration = librosa.get_duration(path=audio_path)
    speech_duration = sum(end - start for start, end in segments)

    return (speech_duration / duration) >= threshold if duration > 0 else False
