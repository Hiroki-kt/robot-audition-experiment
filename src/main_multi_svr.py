from _support_vector_regression import ExecuteSVR
import joblib

if __name__ == '__main__':
    onedrive_path = 'C:/Users/robotics/OneDrive/Research/'
    data_set_file_path = onedrive_path + '_array/200212/'
    model_file1 = onedrive_path + '_array/200212/svr_191015_PTs01.pkl'
    model_file2 = onedrive_path + '_array/200212/svr_191015_STs01.pkl'

    data_name1 = '191015_PTs01'
    data_name2 = '191015_STs01'

    data_set_file1 = data_set_file_path + data_name1 + '.npy'
    data_set_file2 = data_set_file_path + data_name2 + '.npy'

    output_file = 'svr_' + data_name1 + '_' + data_name2

    es1 = ExecuteSVR(data_set_file1, not_auto=True, label_max=50, label_min=-50, output_file_name=output_file)
    es2 = ExecuteSVR(data_set_file2, not_auto=True)

    x_train1, x_data1, y_train1, y_data1 = es1.split_train_test(0, 2)
    x_train2, x_data2, y_train2, y_data2 = es2.split_train_test(0, 2)

    model1 = joblib.load(model_file1)
    model2 = joblib.load(model_file2)

    es1.model_check_multi(model1, model2, x_data1, x_data2, y_data1, y_data2)
