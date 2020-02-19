import numpy as np
from _function import MyFunc
from matplotlib import pyplot as plt
from scipy import stats

if __name__ == '__main__':
    mf = MyFunc()
    DIRECTIONS = np.arange(-50, 50)
    # DIRECTIONS = [0, 1]
    mic_num = 4
    use_freq = [800, 1000, 2000]
    wave_path = mf.recode_data_path + '191015_PTo01/'
    # print(wave_path)
    # data_set_freq_len =
    # data_set = np.zeros((len(DIRECTIONS), mic_num, data_num, data_set_freq_len), dtype=np.float)
    for freq_id, freq in enumerate(use_freq):
        for mic in range(mic_num):
            max_list = []
            for data_dir in DIRECTIONS:
                sound_data, channel, sampling, frames = mf.wave_read_func(wave_path + str(data_dir) + '.wav')
                sound_data = np.delete(sound_data, [0, 5], 0)
                cut_data = mf.reshape_sound_data(sound_data, sampling, 0.95, 0.1, 0, [800, 1000, 2000])
                cut_data = np.reshape(cut_data, (mic_num, 3, -1))
                fft_data = np.fft.rfft(cut_data)
                amp_spec_data = abs(fft_data[mic, freq_id, :])  # Amplitude spectrum
                freq_list = np.fft.rfftfreq(cut_data[0, 1, :].shape[0], 1/44100)
                max_list.append(max(amp_spec_data))

                # plt.figure()
                # plt.plot(freq_list, amp_spec_data)
                # plt.xlim(500, 1500)
                # plt.show()

            # max_list = stats.zscore(max_list)
            test_list = [(i - min(max_list))/(max(max_list) - min(max_list)) for i in max_list]
            plt.figure(figsize=(10, 7))
            plt.rcParams["font.size"] = 18
            plt.plot(DIRECTIONS, max_list)
            plt.title('Freq :' + str(freq) + 'Mic : ' + str(mic + 1))
            plt.ylabel('Amplitude spectrum', fontsize=18)
            plt.xlabel('Azimuth', fontsize=18)
            plt.ylim(50000, 2300000)
            mf.my_makedirs('../_img/200210/')
            plt.savefig('../_img/200210/' + str(freq) + '_' + str(mic+1) + 'no_normalize.png')
