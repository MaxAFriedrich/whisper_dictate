from time import sleep

import numpy as np
from pyautogui import typewrite


class Output:
    def __init__(self, newest_frame: int = 0) -> None:
        self.stop = False
        self.newest_frame = newest_frame
        self.audio_buffer = {}

    def check_for_repeating_chars(self, text):
        if len(text) == 0:
            return False
        first_chunk = text[: len(text) // 2]
        for i in range(len(first_chunk)):
            chunk_size = i + 1
            chunks = [text[j : j + chunk_size] for j in range(0, len(text), chunk_size)]
            if len(set(chunks)) == 1:
                return True  # All chunks are the same
        return False  # No repeating pattern found

    def bulk_transcribe(self, start: int, end: int, trans_func):
        global audio_buffer
        full_audio = np.array([], dtype=np.float32)
        for i in range(start, end + 1):
            full_audio = np.concatenate([full_audio, self.audio_buffer.get(i)])
        return trans_func(end, full_audio)

    def transcribe_main(self, frame_size: float, trans_func):
        global STOP, newest_frame, audio_buffer
        current_start = 0
        current_end = 0
        final_text = None
        last_text = 0
        print("Transcribing")
        while not self.stop:
            audio = self.audio_buffer.get(current_start)
            audio1 = self.audio_buffer.get(current_end)
            if audio is None or audio1 is None:
                sleep(frame_size / 3)
            else:
                current_end = self.newest_frame

                text = self.bulk_transcribe(current_start, current_end, trans_func)
                if not (
                    text == ""
                    or text == None
                    or all(c in ". " for c in text)
                    or self.check_for_repeating_chars(text)
                    or final_text == text
                ):
                    final_text = text
                    last_text = 0
                    self.output(current_end, text)
                else:
                    last_text -= 1
                    if last_text > -2:
                        continue
                    if final_text != None:
                        typewrite(final_text)
                        final_text = None
                    current_start = current_end + 1
                    for i in range(current_start, current_end):
                        self.audio_buffer.pop(i)
        print("Not transcribing")

    def output(self, frame_no: int, text: str):
        print(f"{frame_no}: {text}")
