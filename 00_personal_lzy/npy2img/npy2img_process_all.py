import os.path
import numpy as np
import imageio
import pathlib

home_folder = r'E:\01_LZY\02_Code\04_Cellflow\01_results\20250528_MHIST_HP-SSA_FFTT_amount10\translation_HP_SSA'


output_folder_source = os.path.join(home_folder + r'/source_img')
loadData_source = np.load(os.path.join(home_folder + r'/source.npy'))
pathlib.Path(output_folder_source).mkdir(parents=True, exist_ok=True)

output_folder_latent = os.path.join(home_folder + r'/latent_img')
loadData_latent = np.load(os.path.join(home_folder + r'/latent.npy'))
pathlib.Path(output_folder_latent).mkdir(parents=True, exist_ok=True)

output_folder_target = os.path.join(home_folder + r'/target_img')
loadData_target = np.load(os.path.join(home_folder + r'/target.npy'))
pathlib.Path(output_folder_target).mkdir(parents=True, exist_ok=True)

# diffusion_steps = 1000
amount = 10

def np_reshape(data):
    data = data.reshape(-1, 3, 256, 256)
    data = data.reshape(-1, amount, 3, 256, 256)
    return data

loadData_source = loadData_source.reshape(-1, 3, 256, 256)
loadData_latent = np_reshape(loadData_latent)
loadData_target = np_reshape(loadData_target)

n = loadData_latent.shape[0]
m = loadData_latent.shape[1]


# 双重循环
for i in range(n):

    data = loadData_source[i]
    im_array_source = data.transpose(1, 2, 0)
    im_array_source = ((im_array_source + 1) * 127.5).astype('uint8')
    imageio.imwrite(os.path.join(home_folder + rf'/source_img/source_{(i+1):04d}.png'), im_array_source)

    print('source: ', i)

for i in range(n):

    for j in range(m):
        data = loadData_latent[i, j]
        im_array_latent = data.transpose(1, 2, 0)
        im_array_latent = ((im_array_latent + 1) * 127.5).astype('uint8')
        imageio.imwrite(os.path.join(home_folder + rf'/latent_img/latent_{(i+1):04d}_{j+1:04d}.png'), im_array_latent)

        print('latent: ', i, j)

for i in range(n):
    for j in range(m):
        data = loadData_target[i, j]
        im_array_target = data.transpose(1, 2, 0)
        # im_array_target = ((im_array_target + 1) * 127.5).astype('uint8')
        imageio.imwrite(os.path.join(home_folder + rf'/target_img/target_{(i+1):04d}_{j+1:04d}.png'), im_array_target)
        print('target: ', i, j)