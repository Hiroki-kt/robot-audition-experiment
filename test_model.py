import numpy as np
import joblib
from matplotlib import pyplot as plt


def test_estimate(data_file, model_file, target):
    data_set = np.load(data_file)
    model = joblib.load(model_file)
    print(data_file, data_set.shape)
    print(target, model.predict(data_set[target-2:target+3, 0, 0, :]))

    plt.figure()
    plt.plot(model.predict(data_set[:, 0, 0, :]))
    plt.show()


if __name__ == '__main__':
    model_file_path = '../../../../OneDrive/Research/_array/200116/191015_PTo_svr.pkl'
    data_file_path = '../../../../OneDrive/Research/_array/191217/1205_glass_plate.npy'
    target = 0
    test_estimate(data_file_path, model_file_path, target+50)
