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

# converts all obj files of 'in_folder' to stl files in 'out_folder'

#import pymeshlab
import os.path

in_folder = '/home/jonas/MA_Software/ma-jonas/Data/Motors/obj'
out_folder = '/home/jonas/MA_Software/ma-jonas/Data/Motors/stl'
filenames = [fn for fn in os.listdir(os.path.join(in_folder)) if fn.endswith('.obj')]

for filename in filenames:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.join(in_folder, filename))
    new_name = filename.strip('.obj') + '.stl'
    ms.save_current_mesh(os.path.join(out_folder, new_name))

"""
for filename in in_files:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.join(in_path, filename))
    new_name = filename.strip('.obj') + '.stl'
    ms. save_current_mesh(os.path.join(out_path, new_name))
"""
