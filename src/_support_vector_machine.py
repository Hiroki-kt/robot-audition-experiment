# coding:utf-8
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVC
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
import random
import sys
from _function import MyFunc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd


class ExecuteSVR(MyFunc):
    def __init__(self,
                 data_set,
                 label_list,
                 use_mic_id=0,
                 use_test_num=3,
                 use_model=None,
                 output='test',
                 label_min=-45,
                 label_max=45):
        super().__init__()

        self.data_set = np.load(data_set)
        self.output_name = output
        self.DIRECTIONS = np.arange(label_min, label_max)
        print("### Data Name: ", self.output_name)
        print("### Data Set Shape: ", self.data_set.shape)

        if len(self.data_set.shape) == 4:
            self.x, self.x_test, self.y, self.y_test = self.split_train_test(use_mic_id, use_test_num, label_list)
        elif len(self.data_set.shape) == 2:
            self.x, self.x_test, self.y, self.y_test = self.split(use_mic_id, use_test_num, label_list)
        
        if use_model is None:
            model = self.svm(output + '.pkl')
        else:
            model = joblib.load(use_model)
        self.model_check(model, label_list)

    def split(self, mic, test_num, label_list):
        if self.data_set.shape[0] % len(self.DIRECTIONS) != 0:
            print("Error")
            sys.exit()

        labeling_directions = self.labeling(label_list)
        Y = np.array(labeling_directions.tolist() * int(self.data_set.shape[0]/len(self.DIRECTIONS)))
        x, x_test, y, y_test = train_test_split(self.data_set, Y, train_size=0.8)
        print('### Test & Traing data shape: ', x.shape, y.shape, x_test.shape, y_test.shape)
        return x, x_test, y, y_test

    def split_train_test(self, mic, test_num, label_list):
        x = np.empty((0, self.data_set.shape[3]), dtype=np.float)
        x_test = np.empty_like(x)
        for i in range(int(self.data_set.shape[2] - test_num)):
            x = np.append(x, self.data_set[:, mic, i, :], axis=0)
        for i in range(test_num):
            x_test = np.append(x_test, self.data_set[:, mic, int(-1 * test_num), :], axis=0)
        labeling_directions = self.labeling(label_list)
        y = np.array(labeling_directions.tolist() * int(self.data_set.shape[2] - test_num))
        y_test = np.array(labeling_directions.tolist() * int(test_num))
        print('### Test & Traing data shape: ', x.shape, y.shape, x_test.shape, y_test.shape)
        return x, x_test, y, y_test
    
    def labeling(self, label_list):
        if len(self.DIRECTIONS) % len(label_list) != 0:
            print('Error con not split label, Now directions is :', self.DIRECTIONS.shape)
            sys.exit()
        else:
            label = np.zeros_like(self.DIRECTIONS)
            range_num = int(len(self.DIRECTIONS)/len(label_list)/2)
            for i in label_list:
                target_id = np.abs(self.DIRECTIONS - i).argmin()
                label[target_id-range_num:target_id+range_num] = i
            # if you see the label
            # for i in range(int(range_num * 2) - 1):
            #     print(label[int(i*10):int(i*10)+10])
            return label
        
    def gen_cv(self):
        m_train = np.floor(len(self.y) * 0.75).astype(int)  # このキャストをintにしないと後にハマる
        train_indices = np.arange(m_train)
        test_indices = np.arange(m_train, len(self.y))
        yield (train_indices, test_indices)
    
    def svm(self, file_name):
        print()
        print("*** Now fitting ...  ***")
        print()
        # tuned_parameters = [
        #     {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        #     {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
        #     {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
        #     {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
        # ]
        # gridsearch = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_weighted' % "f1")
        # gridsearch.fit(self.x, self.y)
        # print("Cのチューニング")
        # print("最適なパラメーター =", gridsearch.best_params_)
        # print("精度 =", gridsearch.best_score_)
        # print()
        # svm_model = SVC(C=gridsearch.best_params_["C"],
        #                 kernel=gridsearch.best_params_["kernel"],
        #                 # degree=gridsearch.best_params_['degree'],
        #                 gamma=gridsearch.best_params_['gamma'])
        svm_model = SVC(C=1,
                        kernel='rbf',
                        # degree=gridsearch.best_params_['degree'],
                        gamma=0.001)

        # train_indices = next(self.gen_cv())[0]
        svm_model.fit(self.x, self.y)
        path = self.make_dir_path(array=True)
        joblib.dump(svm_model, path + file_name)
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
        return svm_model
    
    def model_check(self, model, label_list):
        cm = confusion_matrix(self.y_test, model.predict(self.x_test))
        label = self.labeling(label_list)
        fig, ax = plt.subplots(figsize=(6, 6))
        df = pd.DataFrame(data=cm, index=label_list, columns=label_list)
        df = df.pivot('True', 'Estimated')
        sns.heatmap(df, annot=True, cmap='Blues', fmt="d", linewidths=.5, ax=ax, square=True)
        ax.set_ylim(len(df), 0)
        fig.show()

        print('### Accuracy rate', accuracy_score(self.y_test, model.predict(self.x_test)))
        print("### R2 score", model.score(self.x_test, self.y_test))

    def estimate_azimuth(self, model, test_num=random.randint(-45, 45)):
        train_indices = next(self.gen_cv())[0]
        model.fit(self.x[train_indices, :], self.y[train_indices])
        print()
        print("3つのデータの平均を出力")
        print(test_num)
        print(np.average(model.predict(self.x[test_num + 49:test_num + 52])))


if __name__ == '__main__':
    onedrive_path = 'C:/Users/robotics/OneDrive/Research/'
    data_set_file_path = onedrive_path + '_array/200216/'
    config_path = '../config_'
    model_file = onedrive_path + '_array/200216/svm_kuka_freq_1000_7000.pkl'
    
    data_name = 'kuka_freq_1000_7000'

    data_set_file = data_set_file_path + data_name + '.npy'
    output_file = 'svm_' + data_name
    
    LABEL = np.arange(-40, 41, 10)
    print(LABEL)
    
    # else select 0 ~ 8, you can make data set using id's mic
    # if use beamforming data use_mic_id is direction of data
    es = ExecuteSVR(data_set_file,
                    LABEL,
                    use_mic_id=0,
                    use_test_num=2,
                    use_model=model_file,
                    output=output_file,)
