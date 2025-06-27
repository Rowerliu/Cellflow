import os.path
import numpy as np
import imageio


#
file_dir = r'E:\LZY\02_ddib_lzy\02_results_saved\20230801_BCI_HE2IHC_val_crop256_he_latent\translation_source_latent\source_np'

# 列出文件夹中的所有文件
npy_files = [file for file in os.listdir(file_dir) if file.endswith('.npy')]

for npy_filename in npy_files:
    # 读取每个npy文件
    filename = npy_filename.split('.')[0]
    npy_filepath = os.path.join(file_dir, npy_filename)
    loadData = np.load(npy_filepath)
    n = len(loadData)
    for i in range(n):
        data = loadData[i]
        im_array = data.transpose(1, 2, 0)
        im_array = ((im_array + 1) * 127.5).astype('uint8')
        imageio.imwrite(rf'E:\LZY\02_ddib_lzy\02_results_saved\20230801_BCI_HE2IHC_val_crop256_he_latent\translation_source_latent\source\{filename}_{i}.jpeg', im_array)
        print('latent_i: ', i)