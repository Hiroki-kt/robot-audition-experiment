# coding:utf-8
import numpy as np
from matplotlib import pyplot as plt
# from scipy import signal
# import math
from _function import MyFunc


class DataCheck(MyFunc):
    def __init__(self, data_set_file_path):
        super().__init__()
        self.data_set = np.load(data_set_file_path)
        print(data_set_file_path, ': data set Shape ', self.data_set.shape)
        self.DIRECTIONS = np.arange(self.data_set.shape[0]/2 * (-1), self.data_set.shape[0]/2)

    def check_frequency(self):
        TITLE = ""
        Y_LABEL = ""
        X_LABEL = "Azimuth [deg]"
        output_path = self.make_dir_path(img=True)
        for id, direction in enumerate(self.DIRECTIONS):
            plt.figure()
            self.data_plot(range(self.data_set.shape[3]),
                           self.data_set[id, 0, 0, :],
                           TITLE,
                           X_LABEL,
                           Y_LABEL)
            plt.savefig(output_path + str(int(direction)) + '.png')


if __name__ == '__main__':
    data_path = '../_array/200201/200201_PTs07_kuka_distance_200.npy'
    # data_path = '../../../../OneDrive/Research/_array/191217/1205_glass_plate_0cross.npy'
    dc = DataCheck(data_path)
    dc.check_frequency()
