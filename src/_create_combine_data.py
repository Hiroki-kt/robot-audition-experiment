import numpy as np
from _function import MyFunc
import sys


class DataCombine(MyFunc):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def load_data(file):
        data_set = np.load(file[0])
        for i in file[1:]:
            data_set = np.append(data_set, np.load(i), axis=2)
        return data_set

    @staticmethod
    def combine_distance(file):
        data_set = np.empty((0, file.shape[1], file.shape[3]))
        for i in range(file.shape[2]):
            data_set = np.append(data_set, file[:, :, i, :], axis=0)
        print(data_set.shape)
        return data_set

    @staticmethod
    def combine_mic(file):
        data_set = np.empty((0, file.shape[2], file.shape[3]))
        for i in range(file.shape[1]):
            data_set = np.append(data_set, file[:, i, :, :], axis=0)
        print(data_set.shape)
        return data_set

    def combine_dis_mic(self, file):
        data = self.combine_distance(file)
        data_set = np.empty((0, file.shape[3]))
        for i in range(file.shape[1]):
            data_set = np.append(data_set, data[:, i, :], axis=0)
        print(data_set.shape)
        return data_set

    def save_array(self, file, output):
        path = self.make_dir_path(array=True)
        np.save(path + output + '.npy', file)
        print('Saved: ', output)

    def select_element(self, file, axis):
        if axis == 1:
            data_set = self.combine_mic(file)
        elif axis == 2:
            data_set = self.combine_distance(file)
        elif axis == 12:
            data_set = self.combine_dis_mic(file)
        else:
            print("Error")
            sys.exit()
        print(data_set.shape)
        return data_set


if __name__ == '__main__':
    dc = DataCombine()
    date = ['200214_PTs10']
    distance_list = [0]
    data_path = 'C:/Users/robotics/OneDrive/Research/_array/200225/'
    name = '200214_PTs10_mic_combine'
    file_list = []
    for y, d in enumerate(date):
        for x, distance in enumerate(distance_list):
            # file_list.append(data_path + d + '_kuka_distance_' + str(distance) + '.npy')
            file_list.append(data_path + d + '.npy')
            print(file_list[x])
    combine_data = dc.load_data(file_list)
    m_data = dc.select_element(combine_data, 1)
    dc.save_array(m_data, name)
