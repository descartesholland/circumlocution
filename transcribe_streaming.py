#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the streaming API.

Example usage:
    python transcribe_streaming.py resources/audio.raw
"""

from array import array
import argparse
import io
from sys import byteorder
from struct import pack
import time

## Pip Deps
# Audio file open as stream (for devel)
import soundfile

import pyaudio
import wave

## Protobuff 

# [START speech_transcribe_streaming]
def transcribe_streaming(recorder, stream_file=''):
    """Streams transcription of the given audio file."""
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()

    ## dont read from file
    #with io.open(stream_file, 'rb') as audio_file:
    #    content = audio_file.read()

    ## call live record(); output will be tuple (samp rate, array())
    print("Calling record()..")
    sr, data, data2 = recorder.record()
    print("Obtained data with sample rate: " + str(sr))

    print(str(data)[0:20])
    print(str(data2)[0:20])
    # In practice, stream should be a generator yielding chunks of audio data.
    stream = [data2] #[content]
    requests = (types.StreamingRecognizeRequest(audio_content=chunk)
                for chunk in stream)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US')
    streaming_config = types.StreamingRecognitionConfig(config=config)

    # streaming_recognize returns a generator.
    # [START speech_python_migration_streaming_response]
    print("Calling streaming_recognize()...")
    responses = client.streaming_recognize(streaming_config, requests)
    # [END speech_python_migration_streaming_request]

    if not responses or len(responses) == 0:
        print("No transcriptions obtained.")

    for response in responses:
        # Once the transcription has settled, the first result will contain the
        # is_final result. The other results will be for subsequent portions of
        # the audio.
        for result in response.results:
            print('Finished: {}'.format(result.is_final))
            print('Stability: {}'.format(result.stability))
            alternatives = result.alternatives
            # The alternatives are ordered from most likely to least.
            for alternative in alternatives:
                print('Confidence: {}'.format(alternative.confidence))
                print(u'Transcript: {}'.format(alternative.transcript))

    # [END speech_python_migration_streaming_response]
# [END speech_transcribe_streaming]

class Recorder(object):

    def __init__(self):
        self.THRESHOLD = 1000
        self.CHUNK_SIZE = 1024
        self.FORMAT = pyaudio.paInt16
        self.RATE = 16000

    def is_silent(self, snd_data):
        "Returns 'True' if below the 'silent' threshold"
        m = max(snd_data)
        print(str(m))
        return m < self.THRESHOLD

    def normalize(self, snd_data):
        "Average the volume out"
        MAXIMUM = 16384
        times = float(MAXIMUM)/max(abs(i) for i in snd_data)

        r = array('h')
        for i in snd_data:
            r.append(int(i*times))
        return r

    def trim(self, snd_data):
        "Trim the blank spots at the start and end"
        def _trim(snd_data):
            snd_started = False
            r = array('h')

            for i in snd_data:
                if not snd_started and abs(i)>self.THRESHOLD:
                    snd_started = True
                    r.append(i)

                elif snd_started:
                    r.append(i)
            return r

        # Trim to the left
        snd_data = _trim(snd_data)

        # Trim to the right
        snd_data.reverse()
        snd_data = _trim(snd_data)
        snd_data.reverse()
        return snd_data

    def add_silence(self, snd_data, seconds):
        "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
        r = array('h', [0 for i in range(int(seconds*self.RATE))])
        r.extend(snd_data)
        r.extend([0 for i in range(int(seconds*self.RATE))])
        return r

    def record(self):
        """
        Record from  micr; return as an array of signed shorts.
        """
        print("record() setup..")
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT, channels=1, rate=self.RATE,
            input=True, output=True,
            frames_per_buffer=self.CHUNK_SIZE)

        num_silent = 0
        max_silent = 30
        last_reported_silent = 0
        snd_started = False

        r = array('h')

        print("record() starting..")
        bstr = b''
        while 1:
            # little endian, signed short
            rr = stream.read(self.CHUNK_SIZE)
            for _r in rr:
                bstr  += pack('h', _r)
            snd_data = array('h', rr)
            if byteorder == 'big':
                snd_data.byteswap()
            r.extend(snd_data)

            print("record() checking silence.. " + str(len(r)))
            silent = self.is_silent(snd_data)

            if silent and snd_started:
                num_silent += 1
                if (num_silent - last_reported_silent) >= int(max_silent / 4.):
                    print("Num Silent: " + str(num_silent))
                    last_reported_silent = num_silent

            elif not silent and not snd_started:
                snd_started = True

            if snd_started and num_silent > max_silent:
                print("Enough Silence; breaking record loop")
                break

        print("record() stopping stream..")
        sample_width = p.get_sample_size(self.FORMAT)
        stream.stop_stream()
        stream.close()
        p.terminate()

        pp_st = time.time()
        print("record() post processing..")
        r = self.normalize(r)
        r = self.trim(r)
        r = self.add_silence(r, 0.5)
        pp_en = time.time()
        print("record() post proc took " + str(int((pp_en - pp_st) * 1000)))
        return sample_width, r, bstr

    def record_to_file(self, path):
        "Records from the microphone and outputs the resulting data to 'path'"
        sample_width, data = self.record()
        data = pack('<' + ('h'*len(data)), *data)
    
        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(self.RATE)
        wf.writeframes(data)
        wf.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    rec = Recorder()
    transcribe_streaming(rec)
