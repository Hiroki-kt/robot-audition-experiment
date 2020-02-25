# coding:utf-8
import numpy as np
import time
from datetime import datetime
from transitions import Machine
from _turntable_controler import StageControl
# from _recode_func import RecodeFunc
# from multiprocessing import Pool
# from _socket_multi_crient import SocketClient
from _socket_crient import SocketClient
from _function import MyFunc
from _capture_photo import TakePhoto
import sys
from scipy import stats
import joblib
from matplotlib import pyplot as plt


class Estimate(MyFunc):
    def __init__(self, name, path,  host_ip_1, port_1, send_size, interval, retry_times, outfile="2_up_tsp_1num.wav"):
        super().__init__()
        self.mic_path = self.make_dir_path(exp=True)
        self.speak_path = self.speaker_sound_path + outfile
        self.recode_second = 1.2
        freq_min = 1000
        freq_max = 7000
        self.smooth_step = 50
        self.mic_id = 0
        self.scon = StageControl()
        # self.socket = SocketClient(host_ip_1, host_ip_2, port_1, port_2, send_size, interval, retry_times)
        self.socket = SocketClient(host_ip_1, port_1, send_size, interval, retry_times)
        self.model = joblib.load( self.onedrive_path + '_array/' + path + name + '.pkl')
        self.name = name
        origin_path = self.speaker_sound_path + '2_up_tsp_1num.wav'
        self.origin_frames = self.get_frames(origin_path)
        freq_list = np.fft.rfftfreq(self.get_frames(origin_path), 1/44100)
        self.freq_max_id = self.freq_ids(freq_list, freq_max)
        self.freq_min_id = self.freq_ids(freq_list, freq_min)
        self.freq_list = freq_list[self.freq_min_id + int(self.smooth_step/2) - 1:
                                   self.freq_max_id - int(self.smooth_step/2)]
        self.scon.calibrate()
        self.check_ready(200)
        self.socket.connect()

    def estimate(self, dir_name):
        # recode_data, sampling = self.play_rec(self.speak_path, self.recode_second, device_name=self.mic_name,
        #                                       input_file_name=input_file, need_data=True)

        recode_data = self.socket.send_GO(self.mic_path + str(dir_name) + '.wav',
                                          return_data=True)
        print(recode_data.shape)
        start_time = self.zero_cross(recode_data, 128, 44100, 512, self.origin_frames, up=True)
        if start_time < 0:
            start_time = 0
        sound_data = recode_data[:, start_time: int(start_time + self.origin_frames)]
        # sound_data = np.reshape(sound_data, (8, 8, -1))
        print(sound_data.shape)
        plt.figure()
        plt.specgram(sound_data[0], Fs=44100)
        plt.show()
        fft_data = np.fft.rfft(sound_data)
        print(fft_data.shape)
        data_set = np.zeros((recode_data.shape[0], len(self.freq_list)), dtype=np.float)
        for mic in range(fft_data.shape[0]):
            smooth_data = np.convolve(np.abs(fft_data[mic, self.freq_min_id:self.freq_max_id]),
                                      np.ones(self.smooth_step) / float(self.smooth_step), mode='valid')
            smooth_data = np.real(np.reshape(smooth_data, (1, -1)))[0]
            normalize_data = stats.zscore(smooth_data, axis=0)
            # normalize_data = (smooth_data - smooth_data.mean()) / smooth_data.std()
            # normalize_data = (smooth_data - min(smooth_data))/(max(smooth_data) - min(smooth_data))
            data_set[mic, :] = normalize_data
        plt.figure()
        plt.plot(data_set[0])
        plt.show()
        print(self.model.predict(data_set))

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
    HOST_IP_1 = "163.221.44.239"  # 接続するサーバーのIPアドレス
    # HOST_IP_2 = '163.221.44.237'
    HOST_IP_2 = '163.221.44.222'
    PORT_1 = 12345  # 接続するサーバーのポート
    PORT_2 = 12345  # 接続するサーバーのポート
    DATE_SIZE = 1024  # 受信データバイト数
    INTERVAL = 3  # ソケット接続時のリトライ待ち時間
    RETRY_TIMES = 1  # ソケット接続時のリトライ回数
    NAME = 'svr_200210_PTs09_kuka_distance_200_mic'
    PATH = '200221/'
    # es = Estimate("State", HOST_IP_1, HOST_IP_2, PORT_1, PORT_2, DATE_SIZE, INTERVAL, RETRY_TIMES)
    es = Estimate(NAME, PATH, HOST_IP_1,  PORT_1, DATE_SIZE, INTERVAL, RETRY_TIMES)
    # repeat_num = 10
    DISTANCES = [0.4]
    '''Experiment'''
    # order = np.arange(-45, 51, 5)
    # DIRECTIONS = st.make_dir(order)
    # st.main(DIRECTIONS, DISTANCES)
    '''for test'''
    DIRECTIONS = [45]
    for i in DIRECTIONS:
        es.turn_table(i)
        es.estimate(i)
