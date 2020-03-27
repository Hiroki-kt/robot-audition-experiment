import numpy as np
from _function import MyFunc
from scipy import signal
from scipy.stats import stats
from matplotlib import pyplot as plt

mf = MyFunc()
data_name = '10.wav'
d_file = mf.recode_data_path + '191015_PTs01/'
origin_path = mf.speaker_sound_path + '2_up_tsp_1num.wav'
smooth_step = 50
real_num = 8
mic_num = 4
origin_frames = mf.get_frames(origin_path)
freq_list = np.fft.rfftfreq(mf.get_frames(origin_path), 1 / 44100)
freq_max_id = mf.freq_ids(freq_list, 6000)
freq_min_id = mf.freq_ids(freq_list, 1000)
freq_list = freq_list[freq_min_id + int(smooth_step / 2) - 1:freq_max_id - int(smooth_step / 2)]

d_data, d_channels, d_samlpling, d_frames = MyFunc().wave_read_func(d_file + data_name)
d_data = np.delete(d_data, [0, 5], 0)
start_time = mf.zero_cross(d_data, 128, d_samlpling, 512, int(origin_frames), up=True)
if start_time < 0:
    if real_num != 8:
        start_time = start_time + origin_frames
    else:
        start_time = 0
cut_data = d_data[:, start_time: int(start_time + origin_frames * real_num)]
cut_data = np.reshape(cut_data, (mic_num, real_num, -1))
fft_data = np.fft.rfft(cut_data)
# r2_data = TSP(CONFIG_PATH).cut_tsp_data(use_data=r_data)

'''normalize'''
# fft_d = stats.zscore(fft_d)
# fft_d_normal = (fft_d - np.min(fft_d))/ (np.max(fft_d) - np.min(fft_d))
# fft_r = stats.zscore(fft_r)
# fft_r_normal = (fft_r - np.min(fft_r))/ (np.max(fft_r) - np.min(fft_r))

'''use data'''
use_fft_d = fft_data[freq_min_id:freq_max_id]

'''smoothing'''
n = 50
v = np.ones(n) / float(n)

data_set = np.zeros((4, 8, len(freq_list)))
print(data_set.shape)
for data_id in range(cut_data.shape[1]):
    for mic in range(cut_data.shape[0]):
        smooth_data = np.convolve(np.abs(fft_data[mic, data_id, freq_min_id:freq_max_id]),
                                  np.ones(smooth_step) / float(smooth_step), mode='valid')
        smooth_data = np.real(np.reshape(smooth_data, (1, -1)))[0]
        normalize_data = stats.zscore(smooth_data, axis=0)
        print(normalize_data.shape)
        # normalize_data = (smooth_data - smooth_data.mean()) / smooth_data.std()
        # normalize_data = (smooth_data - min(smooth_data))/(max(smooth_data) - min(smooth_data))
        data_set[mic, data_id, :] = normalize_data
        # print(normalize_data)

'''extremun(極値)'''
# peak = signal.argrelmax(use_fft_d, order=10)
# peak_r = signal.argrelmax((use_fft_r/2), order=10)
# peak_s = signal.argrelmax(use_fft_s, order=10)
# print(len(peak[0]), len(peak_r[0]), len(peak_s[0]))
# peak_ica = signal.argrelmax(use_fft_ica, order=10)

'''fitting'''
# d = 30
# direct_func = np.poly1d(np.polyfit(use_freq_list[peak], use_fft_d[peak], d))
# d_s = 3
# synthetic_func = np.poly1d(np.polyfit(use_freq_list[peak_s], use_fft_s[peak_s], d_s))

'''plot'''
plt.figure()
# plt.plot(use_freq_list, use_fft_s, 'y.', label='synthetic')
# plt.plot(use_freq_list, np.abs(use_fft_s), 'y.', label='synthetic')
# plt.plot(use_freq_list[peak_s[0]], use_fft_s[peak_s], label='$\psi$ = ' + str(num))
# plt.plot(use_freq_list, s_fft_s, label='$\psi$ = ' + str(num))
# plt.plot(use_freq_list[peak_ica[0]], use_fft_ica[peak_ica], label='$\psi$ = ' + str(num))
# plt.plot(use_freq_list[peak_s[0]], use_fft_s[peak_s], 'y.', label='synthetic')
# plt.plot(use_freq_list[peak[0]], use_fft_d[peak], 'g-', label='direct')
# plt.plot(use_freq_list[peak[0]], use_fft_d[peak], 'g.', label='direct')
# plt.plot(use_freq_list, direct_func(use_freq_list), 'r-', label='fitting')
# plt.plot(use_freq_list, synthetic_func(use_freq_list), 'c-', label='d=' + str(d_s))
# plt.plot(use_freq_list[peak_r[0]], use_fft_r[peak_r], 'b.', label="reflection")
# plt.plot(use_freq_list[peak_r[0]], use_fft_r[peak_r], label="reflection")
# plt.plot(use_freq_list, use_fft_r[mic], '-', label="reflection")
plt.plot(freq_list, data_set[0, 0, :])
plt.show()
#
# plt.xlim(1000, 2000)
# plt.legend()
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Amplitude')
# plt.ylim(0, 180000)
# # plt.show()
# plt.savefig('../_img/191105_2/' + str(num) + '.png')

# plt.figure()
# # plt.plot(time2_list, ica_reflections_data)
# plt.plot(use_freq_list[peak_ica[0]], use_fft_ica[peak_ica])
# plt.title(str(num))
# plt.ylim(0, 3.0)
# plt.show()

# plt.figure()
# plt.plot(time_list, np.abs(s_data[mic]), '.')
# plt.plot(time_list, np.convolve(np.abs(s_data[mic]), v, mode='same'))
# plt.xlim(0, 1.0)
# plt.title("Synthetic wave [mic" + str(mic-1) + "]")
# plt.show()
#
# plt.figure()
# plt.plot(time2_list, d2_data[mic])
# plt.ylim(0, 700)
# plt.title("Direct wave [mic" + str(mic-1) + "]")
# plt.show()
#
# plt.figure()
# plt.plot(time_list, r_data[mic])
# plt.xlim(0, 1.0)
# plt.title("Reflection wave [mic" + str(mic-1) + "]")
# plt.show()

# test = np.abs(fft_r)/np.abs(fft_d)
# #
# plt.figure()
# plt.xlim(500, 2000)
# # plt.ylim(0, 1)
# plt.title("reflection")
# plt.show()
#
# plt.figure()
# plt.plot(fft_list, fft_r[mic-1])
# plt.xlim(1500, 2000)
# plt.title("reflection")
# plt.show()

# plt.figure()
# plt.plot(fft_list, fft_d[mic-1])
# plt.xlim(500, 2000)
# plt.title("direct")
# # plt.ylim(0, 1)
# plt.show()
#
# plt.figure()
# plt.plot(fft_list, fft_s[mic])
# plt.xlim(500, 2000)
# plt.title("synthetic")
# plt.show()

# plt.figure()
# plt.plot(fft_list, test[mic-1])
# plt.xlim(500, 2000)
# plt.ylim(0, 3)
# plt.title("test")
# plt.show()