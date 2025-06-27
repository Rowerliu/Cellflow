import os.path
from PIL import Image
import numpy as np
import imageio
import cv2
import pathlib

home_folder = r'F:\BUAA\02_Code\02_ZhangLab\08_process-learning\01_results\translation_0_1\target_np'
file_name = 'target_3_0'
loadData = np.load(os.path.join(home_folder, file_name + r'.npy'))

pathlib.Path(home_folder).mkdir(parents=True, exist_ok=True)


# 双重循环
# numpy未转置

data = loadData[0]
im_array_source = data.transpose(1, 2, 0)

# source / latent后处理，target无需处理
# im_array_source = ((im_array_source + 1) * 127.5).astype('uint8')
imageio.imwrite(os.path.join(home_folder, file_name + r'.jpeg'), im_array_source)

print('transfer complete')

