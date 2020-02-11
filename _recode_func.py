# coding:utf-8
import pyaudio
import wave
import sys
import os
import numpy as np
# from _function import MyFunc
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

    def play_rec(self, out_file_name, recode_second, device_name='ReSpeaker 4 Mic Array (UAC1.0)',
                 CHUNK=1024, input_file_name='./test_out.wav', need_data=False, order_index=None, order_ch=None):
        # file_name = '../_exp/Speaker_Sound/up_tsp_1num.wav'
        wf = wave.open(out_file_name, 'rb')
        sampling = wf.getframerate()
        if order_index is not None:
            index = order_index
            channels = order_ch
        else:
            index, channels = self.get_index(device_name)
        p = pyaudio.PyAudio()

        stream1 = p.open(format=pyaudio.paInt16,
                         channels=channels,
                         rate=sampling,
                         frames_per_buffer=CHUNK,
                         input=True,
                         input_device_index=index,
                         )

        stream2 = p.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=sampling,
                         frames_per_buffer=CHUNK,
                         output=True
                         )

        if sampling * recode_second < wf.getnframes():
            print('Error recode time is not enough', wf.getnframes()/sampling)
            sys.exit()

        elif sampling * recode_second > wf.getnframes() * 2:
            print('Error recode time is too long')
            sys.exit()

        else:
            out_data = wf.readframes(CHUNK)
            in_data = stream1.read(CHUNK)
            recoding_data = [in_data]
            for i in range(0, int(sampling / CHUNK * recode_second)):
                input_data = stream1.read(CHUNK)
                recoding_data.append(input_data)
                if out_data != b'':
                    stream2.write(out_data)
                    out_data = wf.readframes(CHUNK)
            recoded_data = b''.join(recoding_data)
            # print(type(recoded_data))
            self.wave_save(recoded_data, channels=channels, sampling=sampling, wave_file=input_file_name)

            stream1.stop_stream()
            stream2.stop_stream()
            stream1.close()
            stream2.close()
            p.terminate()
            if need_data:
                # print('use data return data', np.frombuffer(np.array(recoding_data), dtype='int16').shape)
                recoded_input_data = np.array(np.frombuffer(np.array(recoding_data), dtype='int16'))\
                    .reshape((channels, -1), order='F')
                return recoded_input_data, sampling

    @staticmethod
    def wave_read_func(wave_path):
        with wave.open(wave_path, 'r') as wave_file:
            w_channel = wave_file.getnchannels()
            w_sanpling_rate = wave_file.getframerate()
            w_frames_num = wave_file.getnframes()
            w_sample_width = wave_file.getsampwidth()

            data = wave_file.readframes(w_frames_num)
            if w_sample_width == 2:
                data = np.frombuffer(data, dtype='int16').reshape((w_frames_num, w_channel)).T
            elif w_sample_width == 4:
                data = np.frombuffer(data, dtype='int32').reshape((w_frames_num, w_channel)).T

            '''
            print('*****************************')
            print('Read wave file:', wave_path)
            print('Mic channel num:', w_channel)
            print('Sampling rate:', w_sanpling_rate)
            print('Frame_num:', w_frames_num, ' time:', w_frames_num / float(w_sanpling_rate))
            print('sound data shape:', data.shape)
            print('*****************************')
            '''

            return data, w_channel, w_sanpling_rate, w_frames_num


if __name__ == '__main__':
    rec = RecodeFunc()
    FILE = "../_exp/Speaker_Sound/2_up_tsp_1num.wav"
    rec.sound_out(FILE)