# coding:utf-8
import numpy as np
from matplotlib import pyplot as plt
# from scipy import signal
# import math
from _function import MyFunc


class DataCheckMulti(MyFunc):
    def __init__(self, data_set_file_path, data, distance, mic_id, use_data_id):
        super().__init__()
        self.data_set_path = data_set_file_path + data
        self.data_name = data
        self.mic = mic_id
        self.data_id = use_data_id
        self.distance = distance_list
        origin_path = self.speaker_sound_path + '2_up_tsp_1num.wav'
        smooth_step = 50
        freq_list = np.fft.rfftfreq(self.get_frames(origin_path), 1/44100)
        freq_max_id = self.freq_ids(freq_list, 7000)
        freq_min_id = self.freq_ids(freq_list, 1000)
        self.freq_list = freq_list[freq_min_id + int(smooth_step/2) - 1:freq_max_id - int(smooth_step/2)]

    def check_frequency(self):
        Y_LABEL = "Amplitude Spectrum"
        X_LABEL = "Frequency [Hz]"
        output_path = self.make_dir_path(img=True, directory_name='/freq_amp/' + self.data_name + '/')
        data_list = []
        for dis in self.distance:
            data_set = np.load(self.onedrive_path + self.data_set_path + '_' + str(dis) + '.npy')
            print(self.data_set_path + '_' + str(dis), ': data set Shape ', data_set.shape)
            data_list.append(data_set)
        DIRECTIONS = np.arange(data_list[0].shape[0]/2 * (-1), data_list[0].shape[0]/2)
        for dir_id, direction in enumerate(DIRECTIONS):
            plt.figure()
            for i, dis in enumerate(self.distance):
                plt.plot(self.freq_list, data_list[i][dir_id, self.mic, self.data_id, :], label=str(dis) + ' [mm]')
            plt.xlabel('Frequency [Hz]', fontsize=15)
            plt.ylabel('Amplitude Spectrum', fontsize=15)
            plt.title('Direction = ' + str(int(direction)) + ' [deg]', fontsize=15)
            plt.tick_params(labelsize=15)
            plt.legend(fontsize=15)
            plt.ylim(np.min(np.array(data_list)), np.max(np.array(data_list)))
            plt.savefig(output_path + str(int(direction + data_list[0].shape[0]/2)).zfill(3) + '.png')


if __name__ == '__main__':
    data_path = '_array/200216/'
    data_name = '200210_PTs09_kuka_distance'
    distance_list = [200, 400]
    mic = 0
    use_data = 0
    # data_path = '../../../../OneDrive/Research/_array/191217/1205_glass_plate_0cross.npy'
    dcm = DataCheckMulti(data_path, data_name, distance_list, mic, use_data)
    dcm.check_frequency()
