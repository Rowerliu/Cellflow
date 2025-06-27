import os.path
import numpy as np
import imageio
import pathlib

def np_reshape(data, amount, image_size):
    data = data.reshape(-1, 3, image_size, image_size)
    data = data.reshape(-1, amount, 3, image_size, image_size)
    return data


def main():

    # home_folder = r'F:\BUAA\02_Code\02_ZhangLab\08_process-learning-V2\02_results_saved\20240103_Gleason_transition_PTL\TTFF_ddim100_a10_classifier\transition_benign_gleason5'
    home_folder = r'E:\BUAA\02_Code\02_ZhangLab\08_ADD\08_process-learning-V2\02_results_saved\20240516_ROSE_transition\neg2pos_2_a1\translation_0_1'
    image_size = 256

    output_folder_source = os.path.join(home_folder + r'/source_img')
    output_folder_latent = os.path.join(home_folder + r'/latent_img')
    output_folder_target = os.path.join(home_folder + r'/target_img')

    # loadData_source = np.load(os.path.join(home_folder + r'/source.npy'))
    # loadData_latent = np.load(os.path.join(home_folder + r'/latent.npy'))
    loadData_target = np.load(os.path.join(home_folder + r'/target.npy'))
    diffway_all = np.load(os.path.join(home_folder + r'/diffway.npy'))

    pathlib.Path(output_folder_source).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_folder_latent).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_folder_target).mkdir(parents=True, exist_ok=True)

    # diffusion_steps = 1000
    amount = 1

    # loadData_source = loadData_source.reshape(-1, 3, image_size, image_size)
    # loadData_latent = np_reshape(loadData_latent, amount, image_size)
    loadData_target = np_reshape(loadData_target, amount, image_size)

    n = loadData_target.shape[0]
    m = loadData_target.shape[1]


    # 双重循环
    # for i in range(n):
    #
    #     data = loadData_source[i]
    #     im_array_source = data.transpose(1, 2, 0)
    #     im_array_source = ((im_array_source + 1) * 127.5).astype('uint8')
    #     imageio.imwrite(os.path.join(home_folder + rf'/source_img/source_s{(i+1):04d}.png'), im_array_source)
    #
    #     print('source: ', i)
    #
    # for i in range(n):
    #
    #     for j in range(m):
    #         ratio = int((j+1) * 1e2 / amount)
    #         data = loadData_latent[i, j]
    #         diffway = diffway_all[i, j]
    #         im_array_latent = data.transpose(1, 2, 0)
    #         # im_array_latent = ((im_array_latent + 1) * 127.5).astype('uint8')
    #         imageio.imwrite(os.path.join(home_folder + rf'/latent_img/latent_s{(i+1):04d}_r{ratio:03d}_t{diffway:04d}.png'), im_array_latent)
    #
    #         print('latent: ', i, j)

    target = os.path.basename(home_folder).split('_')[-1]
    source = os.path.basename(home_folder).split('_')[-2]

    for i in range(n):
        for j in range(m):
            ratio = int((j+1) * 1e2 / amount)
            data = loadData_target[i, j]
            diffway = diffway_all[i, j]
            im_array_target = data.transpose(1, 2, 0)
            # im_array_target = ((im_array_target + 1) * 127.5).astype('uint8')
            imageio.imwrite(os.path.join(home_folder + rf'/target_img/{target}_{source}_s{(i+1):06d}_r{ratio:03d}_t{diffway:04d}.png'), im_array_target)
            print('target: ', i, j)

if __name__ == "__main__":
    main()