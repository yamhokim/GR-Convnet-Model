import os
import glob
from tqdm import tqdm
import shutil

def clean_pipeline(dataset_path: str):
    ### Makign a separate directory for each training point
    print("Separating training examples into individual directories...")
    dirs = os.listdir(dataset_path)
    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")
    for d in dirs:
        c = 0
        dir_path = os.path.join(dataset_path, d)
        example_dirs = os.listdir(dir_path)
        if ".DS_Store" in example_dirs:
            example_dirs.remove(".DS_Store")

        for example_dir in tqdm(example_dirs, leave=True, position=0):
            example_path = os.path.join(dir_path, example_dir)
            file_paths = os.listdir(example_path)
            if ".DS_Store" in file_paths:
                file_paths.remove(".DS_Store")

            i = 0
            moved_files = []
            while len(moved_files) != len(file_paths):
                new_dir = os.path.join(dir_path, str(c))
                os.mkdir(new_dir)
                for fn in file_paths:
                    if fn.startswith(str(i)):
                        full_fp = os.path.join(example_path, fn)
                        file_save_path = os.path.join(new_dir, fn)
                        shutil.copyfile(full_fp, file_save_path)
                        moved_files.append(fn)   

                i += 1
                c += 1

            shutil.rmtree(example_path)

    ### Renaming files
    print("Renaming files...")
    fps = glob.glob(os.path.join(dataset_path, "*/*/*"))
    for path in fps:
        splitted = path.split("/")
        if "grasps" in path:
            e = "grasps.txt"
        elif "RGB" in path:
            e = "RGB.png"
        elif "mask" in path:
            e = "mask.png"
        elif "stereo_depth" in path:
            e = "stereo_depth.tiff"
        elif "perfect_depth" in path:
            e = "perfect_depth.tiff"

        new_fname = splitted[-3] + "_" + splitted[-2] + "_" + e
        new_name = os.path.join(os.path.dirname(path), new_fname)
        os.rename(path, new_name)

    ### Removing incomplete directories
    print("Removing incomplete directories...")
    paths = glob.glob(os.path.join(dataset_path, "*/*"))
    num_removed = 0
    for path in paths:
        files = os.listdir(path)
        if ".DS_Store" in files:
            files.remove(".DS_Store")
        files_combined = "".join(files)
        for word in ["grasps", "RGB", "mask", "perfect_depth"]:
            if word not in files_combined:
                num_removed += 1
                shutil.rmtree(path)
                break
    print(f"{num_removed} incomplete directories deleted.")