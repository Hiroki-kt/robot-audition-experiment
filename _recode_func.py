# coding:utf-8
import pyaudio
import wave
import sys
import os
# from msvcrt import getch  # Windowsでは使えるらしい


class RecodeFunc:
    @staticmethod
    def get_index(device_name):
        audio = pyaudio.PyAudio()
        audio_num = audio.get_device_count()
        print('Searching: ' + device_name + '...')
        for x in range(0, audio_num):
            device_info = audio.get_device_info_by_index(x)
            # print(device_info)
            if device_name in device_info.values():
                print('Find Mic!!')
                # print(device_info)
                channels = device_info['maxInputChannels']
                return x, channels
        print('can not find:' + device_name)
        sys.exit()

    @staticmethod
    def wave_save(data, channels=1, width=2, sampling=44100, wave_file='./out_put.wav'):
        wf = wave.Wave_write(wave_file)
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(sampling)
        wf.writeframes(data)
        wf.close()
        # print('saved', wave_file)

    @staticmethod
    def sound_read(filename):
        wf = wave.open(filename, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        return wf, stream, p

    @staticmethod
    def my_makedirs(path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def sound_recode(self, file_path, name, recode_seconds, channels, index, sampling_rate=44100, chunk=1024):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sampling_rate,
                        input=True,
                        input_device_index=index,
                        frames_per_buffer=chunk
                        )
        stream.start_stream()
        data = stream.read(chunk)
        # print('Recode start!')
        print('Now, recording ...')
        self.my_makedirs(file_path)
        filename = file_path + str(name) + ".wav"
        recording_data = [data]
        for i in range(0, int(sampling_rate / chunk * recode_seconds)):
            data = stream.read(chunk)
            recording_data.append(data)
        data = b''.join(recording_data)
        self.wave_save(data, channels=channels, sampling=sampling_rate, wave_file=filename)
        print("Saved." + str(name) + ".wav")
        stream.stop_stream()
        stream.close()
        p.terminate()

    def sound_out(self, file_name, chunk=1024):
        wf, stream, p = self.sound_read(file_name)
        data = wf.readframes(chunk)
        while data != b'':
            stream.write(data)  # ストリームへの書き込み(バイナリ)
            data = wf.readframes(chunk)  # ファイルから1024個*2個
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Speaker Finish")

    def sound_play_recode(self, file_name, chunk):
        wf, stream, p = self.sound_read(file_name)
        data = wf.readframes(chunk)


if __name__ == '__main__':
    rec = RecodeFunc()
    FILE = "./origin_sound_data/tsp_1num.wav"
    rec.sound_out(FILE)