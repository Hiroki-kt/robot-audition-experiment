import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from _function import MyFunc
import sys

if __name__ == '__main__':
    mf = MyFunc()
    data_set_file_path = mf.onedrive_path + '_array/200220/'
    
    data_name = '191205_PTs05'
    
    data_set_file = data_set_file_path + data_name + '.npy'
    data_set = np.load(data_set_file)
    
    print(data_set.shape)
    directions = np.arange(-50, 50)
    if len(data_set.shape) == 4:
        freq_list = np.fft.fftfreq(data_set.shape[3], 1/44100)
    elif len(data_set.shape) == 2:
        freq_list = np.fft.fftfreq(data_set.shape[1], 1/44100)
    else:
        sys.exit()
    freq_max_id = mf.freq_ids(freq_list, 7000)
    freq_min_id = mf.freq_ids(freq_list, 1000)
    
    X1, X2 = np.meshgrid(directions, freq_list)
    print(X1.shape, X2.shape)
    
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1[freq_min_id:freq_max_id, :],
                    X2[freq_min_id:freq_max_id, :],
                    data_set[:, 0, 0, freq_min_id:freq_max_id].T,
                    cmap='winter',
                    linewidth=0)
    ax.set_xlabel('Azimuth [deg]', fontsize=12)
    ax.set_ylabel('Frequency [Hz]', fontsize=12)
    ax.set_zlabel('Amplitude spectrum', fontsize=12)
    ax.tick_params(labelsize=10)
    # fig.colorbar(surf)
    path = mf.make_dir_path(img=True, directory_name='/' + data_name + '/')
    for angle in range(0, 360):
        ax.view_init(30, angle)
        # fig.draw()
        # plt.pause(.001)
        # fig.show()
        plt.savefig(path + str(angle) + '.png')
    # fig.show()
