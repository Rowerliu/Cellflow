import os.path
import numpy as np
import imageio
import pathlib

home_folder = r'F:\01_BaiduNetdiskDownload\20240303_ROSE_Neg2Pos_FFFF_amount100\translation_0_1'

# output_folder_source = os.path.join(home_folder + r'/source_img')
# loadData_source = np.load(os.path.join(home_folder + r'/source.npy'))
# pathlib.Path(output_folder_source).mkdir(parents=True, exist_ok=True)

output_folder_latent = os.path.join(home_folder + r'/latent_img')
loadData_latent = np.load(os.path.join(home_folder + r'/latent.npy'))
pathlib.Path(output_folder_latent).mkdir(parents=True, exist_ok=True)

# output_folder_target = os.path.join(home_folder + r'/target_img')
# loadData_target = np.load(os.path.join(home_folder + r'/target.npy'))
# pathlib.Path(output_folder_target).mkdir(parents=True, exist_ok=True)


# 双重循环
# for i in range(10):
#
#     data = loadData[i]
#
#     for j in range(4):
#
#         im_array_0 = data[j].transpose(1, 2, 0)
#
#         # source
#         # im_array = ((im_array_0 + 1) * 127.5).astype('uint8')
#
#         # target
#         im_array = ((im_array_0 + 1) * 127.5).astype('uint8')
#
#         imageio.imwrite(rf'F:\BUAA\02_Code\21_Diffusion\12_ddib\03_samples\20230306_he2ihc_10\target\target_{i}_{j}.jpeg', im_array)
#
#         print('i: ', i, 'j: ', j)

# 单重循环


# 未转置

n = len(loadData_latent)


# for i in range(n):
#
#     data = loadData_source[i]
#     im_array_source = data.transpose(1, 2, 0)
#     im_array_source = ((im_array_source + 1) * 127.5).astype('uint8')
#     imageio.imwrite(os.path.join(home_folder + rf'/source_img/source_{i:04d}.png'), im_array_source)
#     print('source_i: ', i)

for i in range(n):

    data = loadData_latent[i]
    im_array_latent = data.transpose(1, 2, 0)
    im_array_latent = ((im_array_latent + 1) * 127.5).astype('uint8')
    imageio.imwrite(os.path.join(home_folder + rf'/latent_img/latent_{i:04d}.png'), im_array_latent)
    print('latent_i: ', i)

# for i in range(n):
#
#     data = loadData_target[i]
#     im_array_target = data.transpose(1, 2, 0)
#     im_array_target = ((im_array_target + 1) * 127.5).astype('uint8')
#     imageio.imwrite(os.path.join(home_folder + rf'/target_img/target_{i:04d}.png'), im_array_target)
#     print('target_i: ', i)


'''

# image已转置
n = len(loadData_source)
for i in range(n):

    data = loadData_source[i]
    im_array_source = data
    # im_array_source = ((im_array_source + 1) * 127.5).astype('uint8')

    imageio.imwrite(os.path.join(home_folder + rf'/source/source_{i}.jpeg'), im_array_source)

    print('source_i: ', i)

for i in range(n):

    data = loadData_latent[i]
    im_array_latent = data
    # im_array_latent = ((im_array_latent + 1) * 127.5).astype('uint8')
    imageio.imwrite(os.path.join(home_folder + rf'/latent/latent_{i}.jpeg'), im_array_latent)


    print('latent_i: ', i)

for i in range(n):

    data = loadData_target[i]
    im_array_target = data.astype('uint8')
    # im_array_target = ((im_array_target + 1) * 127.5).astype('uint8')
    imageio.imwrite(os.path.join(home_folder + rf'/target/target_{i}.jpeg'), im_array_target)


    print('target_i: ', i)

'''