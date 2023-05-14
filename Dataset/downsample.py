''' Downsample the 2DGS dataset '''

import numpy as np
import scipy.io as scio
import pyrtools as pt

if __name__ == '__main__':

    # downsample factors
    s_ds, t_ds = 4, 4 
    n_levels = 2
    noise = 0.05
    
    print('Downsample the high-res data of 2D GS...')
    print('Downsample size in time and space are: %d, %d' % (t_ds, s_ds))
    
    N = 128
    N_records = 3001
    
    for IC in range(1, 21):
    
        # load data
        data = scio.loadmat(f'2DGS/2DGS_IC{IC}_2x{N_records}x{N}x{N}.mat')
        data = data['uv'] # [2,3001,256,256]       
        print('hres shape: ', data.shape)
        
        ds = np.zeros([2, N_records // t_ds + 1, N // s_ds, N // s_ds])
        for ii in range(ds.shape[1]):
            ds[0,ii,:,:] = pt.blurDn(data[0,t_ds*ii,:,:], n_levels, filt='binom5')
            ds[1,ii,:,:] = pt.blurDn(data[1,t_ds*ii,:,:], n_levels, filt='binom5')

        print('lres shape: ', ds.shape)
        scio.savemat(f'2DGS_IC{IC}_2x{N_records // t_ds + 1}x{N // s_ds}x{N // s_ds}.mat', {'uv': ds})  
            
