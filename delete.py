import os
import shutil
import glob

paths=glob.glob('./train/*.png')

print sorted(paths[-1000:])

def split_path(path):
    dir_path , filename = os.path.split(path)
    filename , extension=os.path.splitext(filename)
    return dir_path , filename , extension

idx_test=40000
test_folder_path='./test'
for path in paths:
    dir_path, filename, extension=split_path(path)
    if idx_test <= int(filename):
        shutil.move(path , os.path.join(test_folder_path , filename+extension))


#shutil.move()
#os.mv