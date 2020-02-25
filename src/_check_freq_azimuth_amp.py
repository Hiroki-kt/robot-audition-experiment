import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from _function import MyFunc
import sys

if __name__ == '__main__':
    mf = MyFunc()
    data_set_file_path = mf.onedrive_path + '_array/200220/'
    data_name = '191205_PTs05'
    origin_path = mf.speaker_sound_path + '2_up_tsp_1num.wav'
    mic = 0
    data_id = 0
    smooth_step = 50
    data_set_file = data_set_file_path + data_name + '.npy'
    data_set = np.load(data_set_file)
    
    print(data_set.shape)
    directions = np.arange(data_set.shape[0]/2 * (-1), data_set.shape[0]/2)
    freq_list = np.fft.rfftfreq(mf.get_frames(origin_path), 1/44100)
    freq_max_id = mf.freq_ids(freq_list, 7000)
    freq_min_id = mf.freq_ids(freq_list, 1000)
    
    X1, X2 = np.meshgrid(freq_list[freq_min_id + int(smooth_step/2) - 1:freq_max_id - int(smooth_step/2)], directions)
    print(X1.shape, X2.shape)
    
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1,
                    X2,
                    data_set[:, mic, data_id, :],
                    cmap='winter',
                    linewidth=0)
    ax.set_ylabel('Azimuth [deg]', fontsize=12)
    ax.set_xlabel('Frequency [Hz]', fontsize=12)
    ax.set_zlabel('Amplitude spectrum', fontsize=12)
    ax.tick_params(labelsize=10)
    # fig.colorbar(surf)
    path = mf.make_dir_path(img=True, directory_name='/' + data_name + '/')
    for angle in range(0, 360):
        ax.view_init(30, angle)
        # fig.draw()
        # plt.pause(.001)
        # fig.show()
        plt.savefig(path + str(angle).zfill(3) + '.png')
    # fig.show()
