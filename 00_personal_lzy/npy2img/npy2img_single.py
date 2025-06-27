import os.path
import numpy as np
import imageio
import pathlib

file_dir = r'E:\LZY\02_ddib_lzy\01_results_2\translation_HE_latent\source.npy'

loadData = np.load(file_dir)


# 未转置

n = len(loadData)


for i in range(n):

    data = loadData[i]
    im_array = data.transpose(1, 2, 0)
    # im_array = ((im_array + 1) * 127.5).astype('uint8')
    imageio.imwrite(rf'E:\LZY\02_ddib_lzy\01_results_3\translation_1_2\target_{i}.jpeg', im_array)
    print('target_i: ', i)
