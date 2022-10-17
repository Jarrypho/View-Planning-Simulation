import os
import sys
import shutil

out_path = "/media/dominik/Backup Plus/Masterarbeit/abc_dataset/stl/"

def rename_and_move_stls():
    """moves files from downloaded abc_dataset into one folder
    """
    for i in range(0,8):
        print("###################################")
        print(i)
        print("###################################")
        data_path = os.path.join(f"/media/dominik/Backup Plus/Masterarbeit/meta_2", "abc_000{i}_stl2_v00")
        for j in range(0, 10000):
            name = str(j + i*10000)
            cur_path = os.path.join(data_path, name.zfill(8))
            dirs = os.listdir(cur_path)
            print(str(name).zfill(8))
            for file in dirs:
                shutil.copy(os.path.join(cur_path, file), os.path.join(out_path, name.zfill(8) + ".stl"))

if __name__ == '__main__':
    rename_and_move_stls()

    

