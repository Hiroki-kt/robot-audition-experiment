# coding:utf-8
import numpy as np
from matplotlib import pyplot as plt
# from scipy import signal
# import math
from _function import MyFunc


class DataCheck(MyFunc):
    def __init__(self, data_set_file_path, data_name, mic, data_id):
        super().__init__()
        self.data_set = np.load(self.onedrive_path + data_set_file_path + data_name + '.npy')
        self.data_name = data_name
        print(data_set_file_path, ': data set Shape ', self.data_set.shape)
        self.DIRECTIONS = np.arange(self.data_set.shape[0]/2 * (-1), self.data_set.shape[0]/2)
        self.mic = mic
        self.data_id = data_id
        origin_path = self.speaker_sound_path + '2_up_tsp_1num.wav'
        smooth_step = 50
        freq_list = np.fft.rfftfreq(self.get_frames(origin_path), 1/44100)
        freq_max_id = self.freq_ids(freq_list, 7000)
        freq_min_id = self.freq_ids(freq_list, 1000)
        self.freq_list = freq_list[freq_min_id + int(smooth_step/2) - 1:freq_max_id - int(smooth_step/2)]

    def check_frequency(self):
        output_path = self.make_dir_path(img=True, directory_name='/freq_amp/' + self.data_name + '/')
        for dir_id, direction in enumerate(self.DIRECTIONS):
            plt.figure()
            plt.plot(self.freq_list, self.data_set[dir_id, self.mic, self.data_id, :], 'g')
            plt.xlabel('Frequency [Hz]', fontsize=15)
            plt.ylabel('Amplitude Spectrum', fontsize=15)
            plt.title('Direction = ' + str(int(direction)) + ' [deg]', fontsize=15)
            plt.tick_params(labelsize=15)
            # plt.legend(fontsize=15)
            plt.ylim(np.min(self.data_set), np.max(self.data_set))
            plt.savefig(output_path + str(int(direction + self.data_set.shape[0]/2)).zfill(3) + '.png')

    def check_frequency_mic(self):
        output_path = self.make_dir_path(img=True, directory_name='/freq_amp/' + self.data_name + '/')
        for dir_id, direction in enumerate(self.DIRECTIONS):
            plt.figure()
            for mic_id in [0, 3]:
                plt.plot(self.freq_list, self.data_set[dir_id, mic_id, self.data_id, :], label=str(mic_id + 1))
            plt.xlabel('Frequency [Hz]', fontsize=15)
            plt.ylabel('Amplitude Spectrum', fontsize=15)
            plt.title('Direction = ' + str(int(direction)) + ' [deg]', fontsize=15)
            plt.tick_params(labelsize=15)
            plt.legend(fontsize=15)
            plt.ylim(np.min(self.data_set), np.max(self.data_set))
            plt.savefig(output_path + str(int(direction + self.data_set.shape[0]/2)).zfill(3) + '.png')


if __name__ == '__main__':
    data_path = '_array/200220/'
    data_name = '191015_PTs01'
    mic = 0
    data_id = 0
    # data_path = '../../../../OneDrive/Research/_array/191217/1205_glass_plate_0cross.npy'
    dc = DataCheck(data_path, data_name, mic, data_id)
    dc.check_frequency_mic()
