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