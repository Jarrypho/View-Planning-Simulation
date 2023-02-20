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


#import pymeshlab
import os.path
import trimesh
BASE_PATH = os.path.abspath(os.getcwd())
data_dir = os.path.join(BASE_PATH, 'Data', 'abc')

in_path = os.path.join(data_dir, "stl")

#in_files = [fn for fn in os.listdir(in_path) if fn.endswith('.stl')]
#for filename in in_files:
for i in range(100):
    filename = str(i).zfill(8)
    filename = filename + ".stl"
    mesh = trimesh.load(os.path.join(in_path, filename))
    mesh.show()
    print(filename)
