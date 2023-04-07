import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
from pyautogui import typewrite

import whisper
from whisper.audio import SAMPLE_RATE

# MODEL = whisper.load_model("base.en")
STOP = False
newest_frame = 0

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


def transcribe(frame_number: int, audio):
    # concatinate audio
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(MODEL.device)

    # decode the audio
    # joined_memory = "".join(frame_memory)
    options = whisper.DecodingOptions(
        fp16=False,
        language="en",
        prompt=" ",
        temperature=0,
    )
    # run the transcription
    try:
        result = whisper.decode(MODEL, mel, options)
        text = result.text
    except RuntimeError:
        print(f"Failed to transcribe {frame_number}.")
        return None
    if type(text) != str:
        return None
    return text


audio_buffer = {}
from time import sleep


def check_for_repeating_chars(text):
    if len(text) == 0:
        return False
    first_chunk = text[: len(text) // 2]
    for i in range(len(first_chunk)):
        chunk_size = i + 1
        chunks = [text[j : j + chunk_size] for j in range(0, len(text), chunk_size)]
        if len(set(chunks)) == 1:
            return True  # All chunks are the same
    return False  # No repeating pattern found


def bulk_transcribe(start: int, end: int,trans_func):
    global audio_buffer
    full_audio = np.array([], dtype=np.float32)
    for i in range(start, end + 1):
        full_audio = np.concatenate([full_audio, audio_buffer.get(i)])
    return trans_func(end, full_audio)


def transcribe_main(frame_size: float,trans_func):
    global STOP, newest_frame, audio_buffer
    current_start = 0
    current_end = 0
    final_text = None
    last_text = 0
    print("Transcribing")
    while not STOP:
        audio = audio_buffer.get(current_start)
        audio1 = audio_buffer.get(current_end)
        if audio is None or audio1 is None:
            sleep(frame_size / 3)
        else:
            current_end = newest_frame

            text = bulk_transcribe(current_start,current_end,trans_func)
            if not (
                text == ""
                or text == None
                or all(c in ". " for c in text)
                or check_for_repeating_chars(text)
                or final_text == text
            ):
                final_text = text
                last_text = 0
                output(current_end,text)
            else:
                last_text -=1
                if last_text > -frame_size:
                    continue
                if final_text!=None:
                    typewrite(final_text)
                    final_text=None
                current_start = current_end+1
                for i in range(current_start, current_end):
                        audio_buffer.pop(i)
    print("Not transcribing")


def output(frame_no: int, text: str):
    print(f"{frame_no}: {text}")

import client
def main(block_size: float =3 ):
    threading.Thread(target=transcribe_main, args=[block_size,client.client]).start()
    global STOP
    global audio_buffer
    global newest_frame
    print("Listening...")
    try:
        frame_number = 0
        while not STOP:
            audio_buffer[frame_number] = record(block_size)
            newest_frame = frame_number
            frame_number += 1

    except KeyboardInterrupt:
        STOP = True

    print("Not listening")


if __name__ == "__main__":
    main()
