import os.path
import numpy as np
import imageio
import pathlib

home_folder = r'F:\BUAA\02_Code\02_ZhangLab\02_ddib_resizer_hyf\01_results\20230822\HE_F_H2L_F_KI67_T_L2KI67_T\N32_high-999_low-1_A'

# output_folder_source = os.path.join(home_folder + r'/source_img')
output_folder_latent = os.path.join(home_folder + r'/latent_img')
output_folder_target = os.path.join(home_folder + r'/target_img')

# loadData_source = np.load(os.path.join(home_folder + r'/source.npy'))
loadData_latent = np.load(os.path.join(home_folder + r'/latent.npy'))
loadData_target = np.load(os.path.join(home_folder + r'/target.npy'))

# pathlib.Path(output_folder_source).mkdir(parents=True, exist_ok=True)
pathlib.Path(output_folder_latent).mkdir(parents=True, exist_ok=True)
pathlib.Path(output_folder_target).mkdir(parents=True, exist_ok=True)


# 单重循环


# 未转置

n = len(loadData_latent)


# for i in range(n):
#
#     data = loadData_source[i]
#     im_array_source = data.transpose(1, 2, 0)
#     im_array_source = ((im_array_source + 1) * 127.5).astype('uint8')
#     imageio.imwrite(os.path.join(home_folder + rf'/source_img/source_{i}.jpeg'), im_array_source)
#     print('source_i: ', i)

for i in range(n):

    data = loadData_latent[i]
    im_array_latent = data.transpose(1, 2, 0)
    im_array_latent = ((im_array_latent + 1) * 127.5).astype('uint8')
    imageio.imwrite(os.path.join(output_folder_latent + rf'/latent_{i:04d}.png'), im_array_latent)
    print('latent_i: ', i)

for i in range(n):

    data = loadData_target[i]
    im_array_target = data.transpose(1, 2, 0)
    # im_array_target = ((im_array_target + 1) * 127.5).astype('uint8')
    imageio.imwrite(os.path.join(output_folder_target + rf'/target_{i:04d}.png'), im_array_target)
    print('target_i: ', i)
