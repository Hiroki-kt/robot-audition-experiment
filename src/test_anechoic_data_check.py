from _function import MyFunc
from matplotlib import pyplot as plt
import numpy as np
from _create_data_set import CreateDataSet


class CheckAnechoicData(MyFunc):
    def __init__(self, wave_path):
        super().__init__()
        self.data_set, channels, sampling, frames = self.wave_read_func(self.recode_data_path + wave_path)
        tsp_data, tsp_channels, tsp_sampling, self.tsp_frames = \
            self.wave_read_func(self.speaker_sound_path + '2_up_tsp_1num.wav')
        print(self.data_set.shape)

    def fast_fourier_transform(self, data_set=None, sampling=44100):
        if data_set is None:
            data_set = self.data_set
        fft_data = np.fft.rfft(data_set)
        fft_list = np.fft.rfftfreq(data_set.shape[-1], 1/sampling)
        return fft_data, fft_list

    def cut_tsp_signal(self, sound_data=None, sampling=44100, data_num=8):
        if sound_data is None:
            sound_data = self.data_set
        sound_data = np.delete(sound_data, [0, 5], 0)
        start_time = self.zero_cross(sound_data, 128, sampling, 512, self.tsp_frames, up=True)
        if start_time < 0:
            start_time = 0
        cut_data = sound_data[:, start_time: int(start_time + self.tsp_frames * data_num)]
        cut_data = np.reshape(cut_data, (sound_data.shape[0], data_num, -1))
        return cut_data

    def check_freq_amplitude(self, fft_data, fft_list, mic_id=0, output='test'):
        plt.figure()
        plt.plot(fft_list, abs(fft_data[mic_id, :]), '.')
        plt.xlim(1000, 7000)
        plt.ylim(0, 100000)
        # plt.show()
        img_path = self.make_dir_path(img=True)
        plt.savefig(img_path + output + '.png')

    def check_from_low_data(self, output, data_set=None):
        if data_set is None:
            data_set = self.data_set
        cut_data = self.cut_tsp_signal(sound_data=data_set)
        fft_data, fft_list = self.fast_fourier_transform(data_set=cut_data)
        fft_data = np.average(fft_data, axis=1)
        # print(fft_data.shape)
        self.check_freq_amplitude(fft_data, fft_list, output=output)



if __name__ == '__main__':
    output_name = 'C_191029_STs04'
    file_name = '/' + output_name + '/10.wav'
    ca = CheckAnechoicData(file_name)
    ca.check_from_low_data(output_name)
