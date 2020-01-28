# coding:utf-8
from _recode_func import RecodeFunc
from datetime import datetime
import joblib
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from _turntable_controler import StageControl
import time


class Estimate(RecodeFunc):
    def __init__(self):
        self.mic_path = '../_exp/19' + datetime.today().strftime("%m%d") + '/recode_data/' + \
                        datetime.today().strftime("%H%M%S") + '/'
        self.my_makedirs(self.mic_path)
        # self.mic_index, self.mic_channels = self.get_index('ReSpeaker 4 Mic Array (UAC1.0) ')
        self.mic_name = 'ReSpeaker 4 Mic Array (UAC1.0) '
        self.speak_path = "../_exp/Speaker_Sound/2_up_tsp_1num.wav"
        self.recode_second = 1.2
        self.freq_min = 800
        self.freq_max = 2200
        self.smooth_step = 50
        self.mic_id = 0
        self.model = joblib.load('../../../../OneDrive/Research/Model/191125_anechonic/anechonic_svr_model.pkl')
        self.scon = StageControl()
        self.scon.calibrate()
        self.check_ready(200)

    def estimate(self):
        input_file = self.mic_path + 'test.wav'
        recode_data, sampling = self.play_rec(self.speak_path, self.recode_second, device_name=self.mic_name,
                                              input_file_name=input_file, need_data=True)
        recode_data = np.delete(recode_data, [0, 5], 0)
        fft_data = np.fft.rfft(recode_data)
        freq_list = np.fft.rfftfreq(recode_data.shape[1], 1/sampling)
        freq_min_id = self.freq_ids(freq_list, self.freq_min)
        freq_max_id = self.freq_ids(freq_list, self.freq_max)
        data_set_freq_len = freq_max_id - freq_min_id - (self.smooth_step - 1)
        data_set = np.zeros((recode_data.shape[0], data_set_freq_len), dtype=np.float)
        for mic in range(fft_data.shape[0]):
            smooth_data = np.convolve(np.abs(fft_data[mic, freq_min_id:freq_max_id]),
                                      np.ones(self.smooth_step) / float(self.smooth_step), mode='valid')
            smooth_data = np.real(np.reshape(smooth_data, (1, -1)))[0]
            normalize_data = stats.zscore(smooth_data, axis=0)
            # normalize_data = (smooth_data - smooth_data.mean()) / smooth_data.std()
            # normalize_data = (smooth_data - min(smooth_data))/(max(smooth_data) - min(smooth_data))
            data_set[mic, :] = normalize_data
        # plt.figure()
        # plt.plot(data_set[self.mic_id, :])
        # plt.show()
        # print(data_set[self.mic_id].shape)
        print(self.model.predict(data_set[:, 256:]))

    @staticmethod
    def freq_ids(freq_list, freq):
        freq_id = np.abs(freq_list - freq).argmin()
        return freq_id

    def turn_table(self, direction):
        self.scon.move_table(direction + 50)
        self.check_ready(200)
        time.sleep(2)
        print("move to ", direction)

    def check_ready(self, wait_count):
        count_num = 0
        print("Now Moving ...")
        while count_num < wait_count:
            count_num += 1
            if not self.scon.isReady():
                while True:
                    count_num += 1
                    if self.scon.isReady():
                        break
        print("Ready")


if __name__ == '__main__':
    es = Estimate()
    DIRECTIONS = [0, 10, 20, 30]
    for i in DIRECTIONS:
        es.turn_table(i)
        es.estimate()
