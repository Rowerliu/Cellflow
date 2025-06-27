import os
import pathlib

import cv2
import scipy
import numpy as np
from matplotlib import pyplot as plt


def create_filter_mask(img, type, threshold, scale):
    # 创建滤波器掩码
    rows, cols, _ = img.shape
    crow, ccol = rows // 2, cols // 2
    if type == 'HP':
        mask = np.ones((rows, cols), np.float32)
        mask[crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    elif type == 'LP':
        mask = np.ones((rows, cols), np.float32) * scale
        mask[crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 1.0
    else:
        mask = np.ones((rows, cols), np.float32)
    return mask


def image_gray_fft(img, filter=None, threshold=10, scale=0):

    # 2. 转换为灰度图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img_gray = img_gray / 255
    rows, cols = img_gray.shape

    # 3. 执行二维傅里叶变换
    dft = np.fft.fft2(img_gray)
    dft_shift = np.fft.fftshift(dft)

    # 创建高通和低通滤波器掩码
    mask_h = create_filter_mask(img, 'HP', threshold, scale)  # 高通滤波
    mask_l = create_filter_mask(img, 'LP', threshold, scale)  # 低通滤波

    # 3.3 滤波
    if filter == 'HP':
        mask = mask_h
    elif filter == 'LP':
        mask = mask_l
    else:
        mask = np.ones((rows, cols), np.float32)

    dft_shift = dft_shift * mask
    magnitude = np.log(np.abs(dft_shift))
    magnitude = np.abs(dft_shift)

    dft_shift_1 = np.fft.ifftshift(dft_shift)
    img_back = np.fft.ifft2(dft_shift_1)
    img_back = np.abs(img_back)

    return magnitude, img_back


def image_RGB_fft(img, filter=None, threshold=10, scale=0):

    img = img / 255

    rows, cols = img[:, :, 0].shape
    # 1 分割通道
    R, G, B = cv2.split(img)

    # 创建高通和低通滤波器掩码
    mask_h = create_filter_mask(img, 'HP', threshold, scale)  # 高通滤波
    mask_l = create_filter_mask(img, 'LP', threshold, scale)  # 低通滤波

    # 3 傅里叶变换
    # 3.1 正变换
    dftR = np.fft.fft2(R)
    dftG = np.fft.fft2(G)
    dftB = np.fft.fft2(B)

    # 3.2 频谱中心化
    dft_shiftR = np.fft.fftshift(dftR)
    dft_shiftG = np.fft.fftshift(dftG)
    dft_shiftB = np.fft.fftshift(dftB)

    # 3.3 滤波
    if filter == 'HP':
        mask = mask_h
    elif filter == 'LP':
        mask = mask_l
    else:
        mask = np.ones((rows, cols), np.float32)

    dft_shiftR = dft_shiftR * mask
    dft_shiftG = dft_shiftG * mask
    dft_shiftB = dft_shiftB * mask

    # 3.4 fourier结果
    magnitude_R = np.log(np.abs(dft_shiftR))
    magnitude_G = np.log(np.abs(dft_shiftG))
    magnitude_B = np.log(np.abs(dft_shiftB))
    magnitude = np.dstack((magnitude_R, magnitude_G, magnitude_B))

    # 3.5 频谱去中心化
    dft_shiftR2 = np.fft.ifftshift(dft_shiftR)
    dft_shiftG2 = np.fft.ifftshift(dft_shiftG)
    dft_shiftB2 = np.fft.ifftshift(dft_shiftB)


    # 4 傅里叶逆变换
    # 4.1 反变换
    img_backR3 = np.fft.ifft2(dft_shiftR2)
    img_backG3 = np.fft.ifft2(dft_shiftG2)
    img_backB3 = np.fft.ifft2(dft_shiftB2)

    # 4.2 计算灰度值
    img_backR3 = np.abs(img_backR3)
    img_backG3 = np.abs(img_backG3)
    img_backB3 = np.abs(img_backB3)


    # 5.1合并3个通道
    img = np.dstack((img_backR3, img_backG3, img_backB3))
    return magnitude, img


#  绘制频域幅度的一维曲线
def plot_magnitudes_1d(magnitudes, titles, save_path, save_filenames):
    fig, axs = plt.subplots(1, len(magnitudes), figsize=(16, 4))
    x_values = np.arange(-128, 128)
    for i, (magnitude, title) in enumerate(zip(magnitudes, titles)):
        center_row = magnitude[magnitude.shape[0] // 2, :]
        axs[i].plot(x_values, center_row)
        axs[i].set_title(title, fontsize=18)
        axs[i].set_xlabel("Frequency", fontsize=16)
        axs[i].set_ylabel("Magnitude", fontsize=16)
        axs[i].grid(True)
        if save_filenames is not None and len(save_filenames) == len(magnitudes):
            plt.savefig(os.path.join(save_path, save_filenames[i]), dpi=600)  # 分别保存为PNG图片，PPI=600

    plt.show()

def plot_magnitudes_1d_single(magnitudes, titles, save_path, save_filenames=None):
    for i, (magnitude, title) in enumerate(zip(magnitudes, titles)):
        fig, ax = plt.subplots(figsize=(4, 3))
        x_values = np.arange(-128, 128)  # 调整横坐标范围为-128到128
        center_row = magnitude[magnitude.shape[0] // 2, :]
        # center_row = scipy.signal.savgol_filter(center_row, 3, 3)  # 数据平滑
        ax.plot(x_values, center_row)  # 使用调整后的横坐标范围
        plt.ylim(-0.1e7, 1e7)  # 设置坐标范围
        # ax.set_title(title, fontsize=16)  # 设置标题字号
        ax.set_xlabel("Frequency", fontsize=14)  # 设置x轴标签字号
        ax.set_ylabel("Magnitude", fontsize=14)  # 设置y轴标签字号
        ax.grid(False)
        plt.tight_layout()
        if save_filenames is not None and len(save_filenames) == len(magnitudes):
            plt.savefig(os.path.join(save_path, save_filenames[i]), dpi=600, bbox_inches='tight')  # 保存为PNG图片，PPI=1200
        # plt.show()


def cal_high_freq(magnitude, threshold):

    center = int(magnitude.shape[0] / 2)
    magnitude_left = magnitude[center, int(center-10*threshold):int(center-threshold)]
    magnitude_right = magnitude[center, int(center+1*threshold):int(center+10*threshold)]
    magnitude_all = np.concatenate((magnitude_left, magnitude_right))
    magnitude_average = np.average(magnitude_all)

    return magnitude_average


def main():

    # 输入输出路径
    # dir_path = r'F:\12_Data\03_Pathology\a04_PTL\01_ROSE\02_test\tep'
    dir_path = r'E:\BUAA\b_Pathology\18_PPL\02_Results\ROSE\transition_data_select_2'
    save_path = r"E:\BUAA\b_Pathology\18_PPL\03_Figure\fourier_results\20240303_3"
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    magnitude_list = []

    for image in os.listdir(dir_path):

        image_path = os.path.join(dir_path, image)
        image_name = image.split('.')[0]

        # 1 读取图像
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_original = img

        magnitude, iimg = image_RGB_fft(img, filter='')
        saved_img = cv2.cvtColor((iimg*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, f'01_original_{image_name}.png'), saved_img)  # 保存文件

        magnitude_h, img_h = image_gray_fft(img, filter='HP', threshold=80, scale=0.990)
        # magnitude_h, img_h = image_gray_fft(img, filter='HP', threshold=120, scale=0.992)
        saved_img = cv2.cvtColor((img_h*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, f'02_High_filtered_{image_name}.png'), saved_img)  # 保存文件

        magnitude_l, img_l = image_RGB_fft(img, filter='LP', threshold=5, scale=0.1)
        saved_img = cv2.cvtColor((img_l*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, f'03_Low_filtered_{image_name}.png'), saved_img)  # 保存文件

        # magnitudes = [magnitude, magnitude_h, magnitude_l]
        # titles = ["Original", "High-pass", "Low-pass"]
        # save_filenames = [f"{image_name}_original", f"{image_name}_High-pass", f"{image_name}_Low-pass"]
        # plot_magnitudes_1d_single(magnitudes, titles, save_path, save_filenames)

        # Plot the High-pass magnitude curve
        magnitudes = [magnitude_h]
        titles = ["High Pass"]
        titles = ["High-pass Magnitude Curve"]
        save_filenames = [f"04_High-pass_{image_name}"]
        plot_magnitudes_1d_single(magnitudes, titles, save_path, save_filenames)


        # plt.subplot(331), plt.imshow(img_original, 'gray'), plt.title('Original Image')
        # plt.axis('off')
        # plt.subplot(332), plt.imshow(magnitude, 'gray'), plt.title('Fourier')
        # plt.axis('off')
        # plt.subplot(333), plt.imshow(iimg, 'gray'), plt.title('Inverse Image')
        # plt.axis('off')
        # plt.subplot(335), plt.imshow(magnitude_h, 'gray'), plt.title('Fourier_H')
        # plt.axis('off')
        # plt.subplot(336), plt.imshow(img_h, 'gray'), plt.title('Inverse Image_H')
        # plt.axis('off')
        # plt.subplot(338), plt.imshow(magnitude_l, 'gray'), plt.title('Fourier_L')
        # plt.axis('off')
        # plt.subplot(339), plt.imshow(img_l, 'gray'), plt.title('Inverse Image_L')
        # plt.axis('off')
        # plt.show()

        magnitude_average = cal_high_freq(magnitude_h, threshold=1)
        magnitude_list.append(magnitude_average)

    magnitude_list = magnitude_list[::-1]
    plt.plot(magnitude_list)
    plt.title('Plot of magnitude_list')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.savefig(os.path.join(save_path, 'Plot of magnitude_list.png'), dpi=1200, bbox_inches='tight')  # 保存为PNG图片，PPI=1200
    print("magnitude_list: ", magnitude_list)

if __name__ == '__main__':
    main()