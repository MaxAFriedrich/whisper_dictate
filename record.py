import sounddevice as sd
import soundfile as sf
from whisper.audio import SAMPLE_RATE
import numpy as np

def write_audio_file(array, filename):
    # Used for debugging
    # Convert the array to 16-bit integer format
    array_int = np.int16(array * 32767)

    # Write the array to a WAV
    sf.write(filename, array_int, SAMPLE_RATE)


def record(duration: float, sr: int = SAMPLE_RATE):
    # Returns: A NumPy array containing the audio waveform, in float32 dtype
    try:
        # Record audio from default microphone for specified duration
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()  # Wait for recording to complete
    except sd.PortAudioError as e:
        raise RuntimeError(
            f"Failed to record audio from default microphone: {e}"
        ) from e

    return audio.flatten()
