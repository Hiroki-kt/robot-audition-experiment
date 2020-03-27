# coding:utf-8
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVR, SVC
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LinearRegression
import random
from _function import MyFunc
from sklearn.model_selection import train_test_split
import sys
from test_pillow import TextImage


class ExecuteSVR(MyFunc):
    def __init__(self,
                 data_set_path,
                 use_mic_id=0,
                 use_test_num=3,
                 use_model_file=None,
                 output_file_name='test',
                 label_max=45,
                 label_min=-45,
                 not_auto=False):
        super().__init__()
        if not_auto:
            self.data_set = np.load(data_set_path)
            print(data_set_path)
            self.DIRECTIONS = np.arange(label_min, label_max)
            self.output_name = output_file_name
        else:
            self.data_set = np.load(data_set_path)
            print(data_set_path)
            # self.data_name = data_set_path.strip('../_array/200205/' '.npy')  # + '_model_cardboard_200_400'
            self.DIRECTIONS = np.arange(label_min, label_max)
            self.output_name = output_file_name
            print("### Data Name: ", output_file_name)
            print("### Data Set Shape: ", self.data_set.shape)
            if len(self.data_set.shape) == 4:
                self.x, self.x_test, self.y, self.y_test = self.split_train_test(use_mic_id, use_test_num)
            elif len(self.data_set.shape) == 3:
                self.x, self.x_test, self.y, self.y_test = self.split_train_test(use_mic_id, use_test_num)
            elif len(self.data_set.shape) == 2:
                self.x, self.x_test, self.y, self.y_test = self.split()
            if use_model_file is None:
                model = self.svr(output_file_name + '.pkl')
            else:
                model = joblib.load(use_model_file)
            self.model_check(model)
            # self.pca_check()
            # self.model_check_histgram(model)

    def split(self, train_size=6/8):
        if self.data_set.shape[0] % len(self.DIRECTIONS) != 0:
            print("Error")
            sys.exit()

        Y = np.array(self.DIRECTIONS.tolist() * int(self.data_set.shape[0]/len(self.DIRECTIONS)))
        print(Y)
        x, x_test, y, y_test = train_test_split(self.data_set, Y, train_size=train_size, shuffle=False)
        print('### Test & Traing data shape: ', x.shape, y.shape, x_test.shape, y_test.shape)
        return x, x_test, y, y_test

    def split_train_test(self, mic, test_num):
        x = np.empty((0, self.data_set.shape[3]), dtype=np.float)
        x_test = np.empty_like(x)
        for i in range(int(self.data_set.shape[2] - test_num)):
            x = np.append(x, self.data_set[:, mic, i, :], axis=0)
        for i in range(test_num):
            x_test = np.append(x_test, self.data_set[:, mic, int(-1 * test_num), :], axis=0)
        y = np.array(self.DIRECTIONS.tolist() * int(self.data_set.shape[2] - test_num))
        y_test = np.array(self.DIRECTIONS.tolist() * test_num)
        print('### Test & Traing data shape OK: ', x.shape, y.shape, x_test.shape, y_test.shape)
        return x, x_test, y, y_test

    # def split_train_test_ver3(self, mic, test_num):
    #     x = np.empty((0, self.data_set.shape[3]), dtype=np.float)
    #     x_test = np.empty_like(x)
    #     for i in range(int(self.data_set.shape[2] - test_num)):
    #         x = np.append(x, self.data_set[:, mic, i, :], axis=0)
    #     for i in range(test_num):
    #         x_test = np.append(x_test, self.data_set[:, mic, int(-1 * test_num), :], axis=0)
    #     y = np.array(self.DIRECTIONS.tolist() * int(self.data_set.shape[2] - test_num))
    #     y_test = np.array(self.DIRECTIONS.tolist() * test_num)
    #     print('### Test & Traing data shape OK: ', x.shape, y.shape, x_test.shape, y_test.shape)
    #     return x, x_test, y, y_test
    
    def gen_cv(self):
        m_train = np.floor(len(self.y) * 0.75).astype(int)  # このキャストをintにしないと後にハマる
        train_indices = np.arange(m_train)
        test_indices = np.arange(m_train, len(self.y))
        yield (train_indices, test_indices)
    
    def svr(self, file_name):
        print()
        print("*** Now fitting ...  ***")
        print()
        params_cnt = 5
        params = {"C": np.logspace(0, 2, params_cnt), "epsilon": np.logspace(-1, 1, params_cnt)}
        gridsearch = GridSearchCV(SVR(), params, cv=self.gen_cv(), scoring="r2", return_train_score=True)
        gridsearch.fit(self.x, self.y)
        print("C, εのチューニング")
        print("最適なパラメーター =", gridsearch.best_params_)
        print("精度 =", gridsearch.best_score_)
        print()
        # svr_model = SVR(C=1.0, epsilon=0.3)
        svr_model = SVR(C=gridsearch.best_params_['C'], epsilon=gridsearch.best_params_['epsilon'])
        # train_indices = next(self.gen_cv())[0]
        # valid_indices = next(self.gen_cv())[1]
        svr_model.fit(self.x, self.y)
        path = self.make_dir_path(array=True)
        joblib.dump(svr_model, path + file_name)
        # テストデータの精度を計算
        # print("テストデータにフィット")
        # print("テストデータの精度 =", model.score(self.x_test, self.y_test))
        # print()
        # print("※参考")
        # print("訓練データの精度 =", model.score(self.x[train_indices, :], self.y[train_indices]))
        # print("交差検証データの精度 =", model.score(self.x[valid_indices, :], self.y[valid_indices]))

        # print()
        # print("結果")
        # print(model.predict(self.x[0:90]))
        return svr_model
    
    def model_check(self, model, num=100):
        num = self.DIRECTIONS.shape[0]
        fig = plt.figure()
        fig.subplots_adjust(bottom=0.2, left=0.2)
        ax = fig.add_subplot(111)
        predict = model.predict(self.x_test[:num])
        # for i in range(9):
        #     print(predict[i*5:(i+1)*5])
        ax.plot(self.y_test[:num], predict, 'o', label="Estimated (SVR)")
        ax.plot(self.y_test[:num], self.y_test[:num], label="True")
        plt.xlabel('True azimuth angle [deg]', fontsize=15)
        plt.ylabel('Estimate azimuth angle [deg]', fontsize=15)
        plt.xlim(-45, 45)
        plt.ylim(-45, 45)
        plt.tick_params(labelsize=15)
        plt.legend(fontsize=15)
        # plt.show()
        img_path = self.make_dir_path(img=True)
        plt.savefig(img_path + self.output_name + '.png')

        fig = plt.figure()
        fig.subplots_adjust(bottom=0.2)
        ax = fig.add_subplot(111)
        # test_array = np.empty_like(self.DIRECTIONS)
        # for k in range(int(self.y_test.shape[0]/num)):
        #     test_array = np.append(test_array, abs(model.predict(self.x_test[k * num:(k+1) * num]) - self.y_test[k * num:(k+1) * num]))
        # a = abs(model.predict(self.x_test[:num]) - self.y_test[:num])
        # b = abs(model.predict(self.x_test[num:]) - self.y_test[num:])

        ax.plot(self.DIRECTIONS, abs(predict - self.y_test[:num]), lw=3)
        # ax.plot(self.DIRECTIONS, np.ones_like(self.DIRECTIONS)*5)
        # plt.plot(self.DIRECTIONS, y_data1[:num], label="True")
        plt.xlabel('Azimuth angle [deg]', fontsize=15)
        plt.ylabel('Error from true angle [deg]', fontsize=15)
        plt.ylim(0, 45)
        plt.xlim(-45, 45)
        plt.tick_params(labelsize=15)
        # plt.legend(fontsize=15)
        # plt.show()
        img_path = self.make_dir_path(img=True)
        plt.savefig(img_path + self.output_name + '_error.png')
        
        # test_num = random.randint(-45, 45)
        # print()
        # print("3つのデータの平均を出力")
        # print(test_num)
        # print(np.average(model.predict(self.x[test_num + 49:test_num + 52])))
        
        # print('### RMSE', np.sqrt(mean_squared_error(self.y_test[0:num], model.predict(self.x_test[0:num]))))
        print('### RMSE', np.sqrt(mean_squared_error(self.y_test[:num], predict)))
        # print("### R2 score", model.score(self.x_test, self.y_test))
        # print(predict.shape)
        # test_pre = np.reshape(predict, (9, -1)).T
        # test_y = np.reshape(self.y[:90],  (9, -1)).T
        # # print(test_pre)
        # test_pre = np.reshape(test_pre[::-1], (90,))
        # test_y = np.reshape(test_y[::-1], (90,))
        # plt.figure()
        # path = self.make_dir_path(img=True, directory_name='/plot/')
        # for i, estimate in enumerate(test_pre):
        #     TextImage().main(test_y[i], round(estimate, 2), out_put='{0:03}'.format(i))
        #     plt.plot(test_y[i], estimate, 'o', c='b', label='Estimate (SVR)')
        #     plt.plot(self.y_test[:num], self.y_test[:num], c='orange', label="True")
        #     plt.xlabel('True azimuth angle [deg]', fontsize=15)
        #     plt.ylabel('Estimate azimuth angle [deg]', fontsize=15)
        #     plt.tick_params(labelsize=15)
        #     plt.savefig(path + '{0:03}'.format(i) + '.png')

    def model_check_histgram(self, model):
        predict = model.predict(self.x_test)
        print(predict.shape)
        error = np.abs(predict - self.y_test)
        print(error.shape)
        plt.figure()
        plt.subplots_adjust(bottom=0.2, left=0.2)
        plt.hist(error, bins=40, normed=True, range=(0, 20))
        plt.xlabel('Estimation error [deg]', fontsize=15)
        plt.ylabel('Existence ratio', fontsize=15)
        plt.tick_params(labelsize=15)
        plt.show()

    def estimate_azimuth(self, model, test_num=random.randint(-45, 45)):
        train_indices = next(self.gen_cv())[0]
        model.fit(self.x[train_indices, :], self.y[train_indices])
        print()
        print("3つのデータの平均を出力")
        print(test_num)
        print(np.average(model.predict(self.x[test_num + 49:test_num + 52])))

    def model_check_multi(self, model1, model2, x_data1, x_data2, y_data1, y_data2):
        num = self.DIRECTIONS.shape[0]
        plt.figure()
        plt.plot(self.DIRECTIONS, model1.predict(x_data1[:num]), '.', label="Plane")
        plt.plot(self.DIRECTIONS, model2.predict(x_data2[:num]), 'g.', label="Spherical")
        plt.plot(self.DIRECTIONS, y_data1[:num], label="True")
        plt.xlabel('True azimuth angle [deg]', fontsize=15)
        plt.ylabel('Estimate azimuth angle [deg]', fontsize=15)
        plt.tick_params(labelsize=15)
        plt.legend(fontsize=15)
        # plt.show()
        img_path = self.make_dir_path(img=True)
        plt.savefig(img_path + self.output_name + '.png')

        fig = plt.figure()
        fig.subplots_adjust(bottom=0.2)
        ax = fig.add_subplot(111)
        ax.plot(self.DIRECTIONS, abs(model1.predict(x_data1[:num]) - y_data1[:num]),  label="Plane", lw=3)
        ax.plot(self.DIRECTIONS, abs(model2.predict(x_data2[:num]) - y_data2[:num]), 'y', label="Spherical", lw=3)
        # plt.plot(self.DIRECTIONS, y_data1[:num], label="True")
        plt.xlabel('Azimuth angle [deg]', fontsize=15)
        plt.ylabel('Error from true angle [deg]', fontsize=15)
        plt.ylim(0, 5)
        plt.tick_params(labelsize=15)
        plt.legend(fontsize=15)
        # plt.show()
        img_path = self.make_dir_path(img=True)
        plt.savefig(img_path + self.output_name + '_error.png')

        print('### RMSE', np.sqrt(mean_squared_error(y_data1[num:], model1.predict(x_data1[num:]))))
        print('### RMSE', np.sqrt(mean_squared_error(y_data2[num:], model2.predict(x_data2[num:]))))
        # print("### R2 score", model.score(self.x_test, self.y_test))


if __name__ == '__main__':
    onedrive_path = 'C:/Users/robotics/OneDrive/Research/'
    data_set_file_path = onedrive_path + '_array/200311/'
    model_file_path = onedrive_path + '_array/200229/'

    data_name = '200214_PTs10_mic_combine_2'
    model_name = '191015_PTs01_mic_combine_2'

    data_set_file = data_set_file_path + data_name + '.npy'
    print(data_set_file)
    output_file = 'svr_' + data_name
    model_file = model_file_path + 'svr_' + model_name + '.pkl'
    print(model_file)

    es = ExecuteSVR(data_set_file,
                    use_mic_id=0,
                    use_test_num=2,
                    use_model_file=model_file,
                    output_file_name=output_file,
                    # label_max=45,
                    # label_min=-45,
                    # not_auto=True
                    )
