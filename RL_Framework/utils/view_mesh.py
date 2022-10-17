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