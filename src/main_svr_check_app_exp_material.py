from _support_vector_regression import ExecuteSVR
# from parametric_eigenspace import PrametricEigenspace
import numpy as np

# if use mic id is '9-11' make test data for torn using mic data, use test num 800 = '9', 1000 = '10', 2000 = '11'
# if use mic id is '-1' make test data for torn useing three data
# else select 0 ~ 8, you can make data set using id's mic

'''
How to use ExecuteSVR

input parametor

data_set_file                           : string (file path)
use_mic_id(default = 0)                 : int (mic id 0 ~ 3, or 0 ~  7)
use_test_num(default = 3)               : int (test data num)
use_model_file(default = None)          : string (file path) when you use svr model file
out_put_file_name(default = test.pkl)   : string (file path) when you make new svr model, the out put file name
'''

onedrive_path = 'C:/Users/robotics/OneDrive/Research/'
data_set_file_path = onedrive_path + '_array/200123/'
model_file_path = onedrive_path + '_array/200123/svr_'

data_name = '200121_PTs06'

material_list = ['glass', 'cardboard']
size_list = ['50_100', '100_200', '200_400']

mic = 0

print()
print("*******************************")
for i in size_list:
    for j in material_list:
        data_set_file = data_set_file_path + data_name + '_' + j + '_' + i + '.npy'
        model_file = model_file_path + data_name + '_' + j + '_' + i + '.pkl'
        es = ExecuteSVR(data_set_file,
                        use_mic_id=0,
                        use_test_num=2,
                        output_file_name=data_name + '_' + j + '_' + i,
                        # use_model_file=model_file,
                        # config_name=config_file
                        )
    # pe = PrametricEigenspace(config_file_glass)
    # pe.pca(x)
    print("*******************************")
