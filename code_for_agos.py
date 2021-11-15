import h5py
import scipy.io

def hdf5_to_dict(grp):
    print(grp[()])
    assert(True==False)
    if isinstance(grp[()],h5py.Group):
        print('it is a group')
        dicti = {}
        for key in grp[()]:
            dicti[key] = hdf5_to_dict(grp[()][key])
    else:
        dicti = grp[:]
    return dicti

def convert_hdf5_to_mat(source_filename,target_filename):
    with h5py.File(filename,mode=‘r’) as ds:
        session_ids = list(ds.keys())
        for session_id in session_ids:
            sc0 = ds[session_id]['size_contrast_0']
    	scipy.io.savemat(target_filename,x
