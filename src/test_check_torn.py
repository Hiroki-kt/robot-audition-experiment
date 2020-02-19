import numpy as np
from _function import MyFunc
from matplotlib import pyplot as plt
from scipy import stats

if __name__ == '__main__':
    mf = MyFunc()
    DIRECTIONS = np.arange(-50, 50)
    # DIRECTIONS = [0, 1]
    mic_num = 4
    wave_path = mf.recode_data_path + '200214_PTo10_time_1s_2000/'
    origin_time = 1.05
    data_num = 8
    # print(wave_path)
    # data_set_freq_len =
    # data_set = np.zeros((len(DIRECTIONS), mic_num, data_num, data_set_freq_len), dtype=np.float)
    max_list = []
    for data_dir in DIRECTIONS:
        print(data_dir)
        sound_data, channel, sampling, frames = mf.wave_read_func(wave_path + str(data_dir) + '.wav')
        sound_data = np.delete(sound_data, [0, 5], 0)
        start_time = mf.zero_cross(sound_data, 50, sampling, 512, origin_time*sampling, torn=True)
        # print(start_time)
        cut_data = sound_data[:, start_time: int(start_time + origin_time * sampling * data_num)]
        # print(cut_data.shape)
        cut_data = np.reshape(cut_data, (mic_num, data_num, -1))
        fft_data = np.fft.rfft(cut_data)
        # print(fft_data.shape)
        amp_spec_data = abs(fft_data[0, 0, :])  # Amplitude spectrum
        freq_list = np.fft.rfftfreq(cut_data[0, 0, :].shape[0], 1/44100)
        max_list.append(max(amp_spec_data))

        # plt.figure()
        # plt.plot(freq_list, amp_spec_data)
        # plt.xlim(500, 3500)
        # plt.show()

    # max_list = stats.zscore(max_list)
    test_list = [(i - min(max_list))/(max(max_list) - min(max_list)) for i in max_list]
    plt.figure(figsize=(10, 7))
    plt.rcParams["font.size"] = 18
    plt.plot(DIRECTIONS, max_list)
    plt.ylabel('Amplitude spectrum', fontsize=18)
    plt.xlabel('Azimuth', fontsize=18)
    plt.show()
    # # plt.ylim(50000, 2300000)
    # mf.my_makedirs('../_img/200210/')
    # plt.savefig('../_img/200210/'  str(mic+1) + 'no_normalize.png')
