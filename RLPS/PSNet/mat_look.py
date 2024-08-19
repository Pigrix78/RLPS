import scipy.io as sio
import numpy as np

# 加载 .mat 文件
mat_contents = sio.loadmat("E:\\Datasets\\Botswana\\Botswana_gt.mat")

# 访问特定变量
variable_data = mat_contents['Botswana_gt']

# 手动转换数据类型为 uint8
variable_data = variable_data.astype(np.uint8)

print(variable_data)


# import h5py
#
# # 打开 MATLAB v7.3 格式的文件
# with h5py.File("E:\Datasets\Houston\Houston_gt.mat", 'r') as file:
#     # 获取特定变量的数据
#     variable_data = file['map'][:]
#     print(variable_data)
#     print(variable_data.shape)
