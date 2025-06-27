import os
import pathlib
import shutil

def copy_files_by_pattern(source_folder, target_folder, pattern, destination_subfolder):
    for filename in os.listdir(source_folder):
        if pattern in filename:
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(target_folder, destination_subfolder, "gleason3")  # define output dir "0" / "1"
            pathlib.Path(destination_path).mkdir(parents=True, exist_ok=True)
            destination = os.path.join(destination_path, filename)
            shutil.copy2(source_path, destination)
            print(f"File '{filename}' copied to '{destination}'")

# 设置源文件夹和目标文件夹
source_folder = r"F:\BUAA\02_Code\02_ZhangLab\08_process-learning-V2\02_results_saved\20240103_Gleason_transition_PTL\TTFF_ddim100_a10_classifier\transition_benign_gleason3\target_img"
target_folder = r"G:\12_Data\03_Pathology\08_PTL\03_Gleason\Gleason_transition_CLS\train"

# 复制满足条件的文件到相应的子文件夹

copy_files_by_pattern(source_folder, target_folder, "_r010", "0.1")
copy_files_by_pattern(source_folder, target_folder, "_r020", "0.2")
copy_files_by_pattern(source_folder, target_folder, "_r030", "0.3")
copy_files_by_pattern(source_folder, target_folder, "_r040", "0.4")
copy_files_by_pattern(source_folder, target_folder, "_r050", "0.5")
copy_files_by_pattern(source_folder, target_folder, "_r060", "0.6")
copy_files_by_pattern(source_folder, target_folder, "_r070", "0.7")
copy_files_by_pattern(source_folder, target_folder, "_r080", "0.8")
copy_files_by_pattern(source_folder, target_folder, "_r090", "0.9")
copy_files_by_pattern(source_folder, target_folder, "_r100", "1.0")




