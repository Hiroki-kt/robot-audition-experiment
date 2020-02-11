from _support_vector_regression import ExecuteSVR

data_set_file_path = '../_array/200211/'
config_path = '../config_'
model_file = '../_array/200129/svr_200128_PTs07_kuka_distance_450.pkl'

for i in [200, 300, 400]:
    data_name = '200210_PTs09_kuka_distance_' + str(i)

    data_set_file = data_set_file_path + data_name + '.npy'
    output_file = 'svr_' + data_name
    config_file = config_path + data_name + '.ini'

    # if use mic id is '9-11' make test data for torn using mic data, use test num 800 = '9', 1000 = '10', 2000 = '11'
    # if use mic id is '-1' make test data for torn useing three data
    # else select 0 ~ 8, you can make data set using id's mic
    # if use beamforming data use_mic_id is direction of data
    es = ExecuteSVR(data_set_file,
                    use_mic_id=0,
                    use_test_num=2,
                    # use_model_file=model_file,
                    output_file_name=output_file
                    )