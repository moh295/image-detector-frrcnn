import glob
import os

dirpath = 'data/images/'
dirname = os.path.basename(dirpath)

filepath_list = glob.glob(os.path.join(dirpath, '*.png'))
pad = len(str(len(filepath_list)))
for n, filepath in enumerate(filepath_list, 1):
    os.rename(
        filepath,
        os.path.join(dirpath, '{:>0{}}.png'.format(n, pad)))
