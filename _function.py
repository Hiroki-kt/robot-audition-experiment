# coding:utf-8
import os
import numpy as np
from matplotlib import pyplot as plt
import math
import wave
from datetime import datetime
from scipy import signal


class MyFunc:
    def __init__(self):
        self.onedrive_path = '../../../../OneDrive/Research/'
        self.recode_data_path = self.onedrive_path + 'Recode_Data/'
        self.speaker_sound_path = self.onedrive_path + 'Speaker_Sound/'

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

    @staticmethod
    def my_makedirs(path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def make_dir_path(self, array=False, img=False, exp=False):
        path = '/20' + datetime.today().strftime("%m%d") + '/'
        if array:
            path = '../../_array' + path
        elif img:
            path = '../../_img' + path
        elif exp:
            path = '../../_exp' + path

        # print(path)
        self.my_makedirs(path)
        return path

    @staticmethod
    def reshape_sound_data(data, rate, interval_time, need_time, start_time, des_freq_list, time_range=0.1,
                           not_reshape=False):
        if not_reshape:
            return data
        # print('Sound Data:', data.shape)  # (Mic, Freq, Time)
        start_time = int(start_time * rate)
        time_range = int(time_range * rate / 2)
        use_sound_data = data[:, start_time + int(need_time * rate / 2) - time_range:
                                 start_time + int(need_time * rate / 2) + time_range]
        start_time = start_time + int(need_time * rate) + int(interval_time * rate)
        for k in range(len(des_freq_list)):
            use_sound_data = np.append(use_sound_data,
                                       data[:, start_time + int(need_time * rate / 2) - time_range:
                                               start_time + int(need_time * rate / 2) + time_range],
                                       axis=1)
            start_time += int(need_time * rate) + int(interval_time * rate)
        # print('#Complete reshape', use_sound_data.shape)
        return use_sound_data

    '''
    data[Time]
    '''

    @staticmethod
    def band_pass_filter(size, fs, start, data, freq_min, freq_max):
        d = 1.0 / fs
        hammingWindow = np.hamming(size)
        freq_list = np.fft.fftfreq(size, d)
        # print(freq_list)
        # print("filFft data :", data.shape)
        # print('size', size)
        windowedData = hammingWindow + data[start:start + size]  # 切り出し波形データ(窓関数)
        data = np.fft.fft(windowedData)
        id_min = np.abs(freq_list - freq_min).argmin()
        id_max = np.abs(freq_list - freq_max).argmin()
        bpf_distribution = np.ones((size,), dtype=np.float)
        bpf_distribution[int(-1 * id_min):] = 0
        bpf_distribution[id_max:int(-1 * id_max)] = 0
        bpf_distribution[:id_min] = 0
        # plt.plot(freq_list, bpf_distribution)
        # plt.show()
        fft_bpf_data = data * bpf_distribution
        ifft_bpf_data = np.fft.ifft(fft_bpf_data)
        return fft_bpf_data, ifft_bpf_data

    @staticmethod
    # root mean square
    def rms(data):
        ss_data = [x ** 2 for x in data]
        rms_list = math.sqrt(sum(ss_data) / len(ss_data))
        # print(rms)
        return rms_list

    @staticmethod
    def data_plot(x, data, title='test', xlabel='test', ylabel='test'):
        plt.plot(x, data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    @staticmethod
    def nextpow2(n):
        m_f = np.log2(n)
        m_i = np.ceil(m_f)
        return int(np.log2(2 ** m_i))

    @staticmethod
    def wave_save(data, channels=1, width=2, sampling=44100, wave_file='./out_put.wav'):
        wf = wave.Wave_write(wave_file)
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(sampling)
        wf.writeframes(b''.join(data))
        wf.close()
        print('saved', wave_file)

    def make_dir_path(self, array=False, img=False, exp=False, photo=False):
        path = '/20' + datetime.today().strftime("%m%d") + '/'
        if array:
            path = '../_array' + path
        elif img:
            path = '../_img' + path
        elif exp:
            path = '../_exp' + path
        elif photo:
            path = '../_photo' + path

        # print(path)
        self.my_makedirs(path)
        return path

    @staticmethod
    def zero_cross(data, step, sampling, size, need_frames, up=False):
        start = 0
        count = 0
        zero_cross = []
        frame_list = []
        time_list = []
        while count < need_frames /step:
            sign = np.diff(np.sign(data[:, start: start + size]))
            zero_cross.append(np.where(sign)[1].shape[0])
            frame_list.append(start)
            time_list.append(start/sampling)
            start = start + step
            count += 1
        # plt.figure()
        # plt.plot(time_list, zero_cross)
        # plt.ylim(0, 1000)
        # plt.show()
        if up:
            peak = signal.argrelmax(np.array(zero_cross), order=5)
            # print(peak)
            time_id = peak[0][np.argmax(np.array(zero_cross)[peak])]
        else:
            peak = signal.argrelmax(np.array(zero_cross), order=10)
            time_id = peak[0][np.argmax(np.array(zero_cross)[peak])]
        START_TIME = frame_list[int(time_id)] - need_frames
        return START_TIME

    @staticmethod
    def data_search(date, sound_kind, geometric, app, plane_wave=True, calibration=False):
        if plane_wave:
            speaker = 'P'
        else:
            speaker = 'S'

        dir_name = str(date) + '_' + speaker + sound_kind + geometric

        if calibration:
            if plane_wave:
                return 'D_' + dir_name + '/10.wav'
            else:
                return 'C_' + dir_name + '/10.wav'
        else:
            if app is not None:
                return dir_name + app + '/'
            else:
                return dir_name + '/'

    @staticmethod
    def freq_ids(freq_list, freq):
        freq_id = np.abs(freq_list - freq).argmin()
        return freq_id