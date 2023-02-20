'''
Copyright (C) 2022  Jan-Philipp Kaiser

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
'''

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

    

