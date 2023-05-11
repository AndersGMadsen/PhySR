import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(1) 
np.random.seed(1) 

lapl_op = [[[[    0,   0, -1/12,   0,     0],
             [    0,   0,   4/3,   0,     0],
             [-1/12, 4/3,    -5, 4/3, -1/12],
             [    0,   0,   4/3,   0,     0],
             [    0,   0, -1/12,   0,     0]]]]

# ============ define relevant functions =============
# https://github.com/benmaier/reaction-diffusion/blob/master/gray_scott.ipynb

def apply_laplacian(mat, dx = 0.01):
    # dx is inversely proportional to N
    """This function applies a discretized Laplacian
    in periodic boundary conditions to a matrix
    For more information see
    https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
    """

    # the cell appears 4 times in the formula to compute
    # the total difference
    neigh_mat = -5*mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [
                    ( 4/3,  (-1, 0) ),
                    ( 4/3,  ( 0,-1) ),
                    ( 4/3,  ( 0, 1) ),
                    ( 4/3,  ( 1, 0) ),
                    (-1/12,  (-2, 0)),
                    (-1/12,  (0, -2)),
                    (-1/12,  (0, 2)),
                    (-1/12,  (2, 0)),
                ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0,1))

    return neigh_mat/dx**2

# Define the update formula for chemicals A and B
def update(A, B, DA, DB, f, k, delta_t):
    """Apply the Gray-Scott update formula"""

    # compute the diffusion part of the update
    diff_A = DA * apply_laplacian(A)
    diff_B = DB * apply_laplacian(B)
    
    # Apply chemical reaction
    reaction = A*B**2
    diff_A -= reaction
    diff_B += reaction

    # Apply birth/death
    diff_A += f * (1-A)
    diff_B -= (k+f) * B

    A += diff_A * delta_t
    B += diff_B * delta_t

    return A, B


def GetEachTerm(A, B, DA, DB, f, k, delta_t, dx):
    lap_A = DA * apply_laplacian(A,dx)
    lap_B = DB * apply_laplacian(B,dx)
    # Apply chemical reaction
    reaction = A * B ** 2

    # Apply birth/death, linear term
    lin_A = f * (1 - A)
    lin_B = (k + f) * B
    return lap_A, lap_B, reaction, lin_A, lin_B

def update_rk4(A0, B0, DA, DB, f, k, delta_t, dx):
    """Update with Runge-kutta-4 method
       See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """
    ############# Stage 1 ##############
    # compute the diffusion part of the update
    diff_A = DA * apply_laplacian(A0, dx)
    diff_B = DB * apply_laplacian(B0, dx)

    # Apply chemical reaction
    reaction = A0 * B0 ** 2
    diff_A -= reaction
    diff_B += reaction

    # Apply birth/death
    diff_A += f * (1 - A0)
    diff_B -= (k + f) * B0

    K1_a = diff_A
    K1_b = diff_B

    ############# Stage 1 ##############
    A1 = A0 +  K1_a * delta_t/2.0
    B1 = B0 +  K1_b * delta_t/2.0

    diff_A = DA * apply_laplacian(A1, dx)
    diff_B = DB * apply_laplacian(B1, dx)

    # Apply chemical reaction
    reaction = A1 * B1 ** 2
    diff_A -= reaction
    diff_B += reaction

    # Apply birth/death
    diff_A += f * (1 - A1)
    diff_B -= (k + f) * B1

    K2_a = diff_A
    K2_b = diff_B

    ############# Stage 2 ##############

    A2 = A0 + K2_a * delta_t/2.0
    B2 = B0 + K2_b * delta_t/2.0

    diff_A = DA * apply_laplacian(A2, dx)
    diff_B = DB * apply_laplacian(B2, dx)

    # Apply chemical reaction
    reaction = A2 * B2 ** 2
    diff_A -= reaction
    diff_B += reaction

    # Apply birth/death
    diff_A += f * (1 - A2)
    diff_B -= (k + f) * B2

    K3_a = diff_A
    K3_b = diff_B

    ############# Stage 3 ##############
    A3 = A0 + K3_a * delta_t
    B3 = B0 + K3_b * delta_t

    diff_A = DA * apply_laplacian(A3, dx)
    diff_B = DB * apply_laplacian(B3, dx)

    # Apply chemical reaction
    reaction = A3 * B3 ** 2
    diff_A -= reaction
    diff_B += reaction

    # Apply birth/death
    diff_A += f * (1 - A3)
    diff_B -= (k + f) * B3

    K4_a = diff_A
    K4_b = diff_B

    # Final solution
    A = A0 + delta_t*(K1_a+2*K2_a+2*K3_a+K4_a)/6.0
    B = B0 + delta_t*(K1_b+2*K2_b+2*K3_b+K4_b)/6.0

    return A, B


def get_initial_A_and_B(N, init_dis, random_influence = 0.2):
    """get the initial chemical concentrations"""
    # get initial homogeneous concentrations
    A = (1-random_influence) * np.ones((N,N))
    B = np.zeros((N,N))

    # put some noise on there
    A += random_influence * np.random.random((N,N))
    B += random_influence * np.random.random((N,N))

    # initial disturbance 
    N1, N2, N3 = N//4-4, N//2, 3*N//4
    r = int(N/10.0)
    
    if 1 in init_dis:
        # initial disturbance 1  
        A[N1-r:N1+r, N1-r:N1+r] = 0.50
        B[N1-r:N1+r, N1-r:N1+r] = 0.25

    if 2 in init_dis:
        # initial disturbance 2
        A[N1-r:N1+r, N3-r:N3+r] = 0.50
        B[N1-r:N1+r, N3-r:N3+r] = 0.25

    if 3 in init_dis:
        # initial disturbance 3
        A[N3-r:N3+r, N3-r:N3+r] = 0.50
        B[N3-r:N3+r, N3-r:N3+r] = 0.25

    if 4 in init_dis:
        # initial disturbance 4
        A[N3-r:N3+r, N1-r:N1+r] = 0.50
        B[N3-r:N3+r, N1-r:N1+r] = 0.25

    if 5 in init_dis:
        # initial disturbance 5
        A[N2-r:N2+r, N2-r:N2+r] = 0.50
        B[N2-r:N2+r, N2-r:N2+r] = 0.25

    if 6 in init_dis:
        # initial disturbance 6
        A[N2-r:N2+r, N3-r:N3+r] = 0.50
        B[N2-r:N2+r, N3-r:N3+r] = 0.25

    return A, B


def postProcess(output, N, xmin, xmax, ymin, ymax, num, batch, save_path):
    ''' num: Number of time step
    '''
    x = np.linspace(xmin, xmax, N+1)[:-1]
    y = np.linspace(ymin, ymax, N+1)[:-1]
    x_star, y_star = np.meshgrid(x, y)
    u_pred = output[num, 0, :, :]

    # v_star = true[num+25, 1, 1:-1, 1:-1]
    v_pred = output[num, 1, :, :]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    cf = ax[0].scatter(x_star, y_star, c=u_pred, alpha=0.95, edgecolors='none', cmap='hot', marker='s', s=2)
    ax[0].axis('square')
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    cf.cmap.set_under('black')
    cf.cmap.set_over('white')
    ax[0].set_title('u-FDM')
    fig.colorbar(cf, ax=ax[0], extend='both')

    cf = ax[1].scatter(x_star, y_star, c=v_pred, alpha=0.95, edgecolors='none', cmap='hot', marker='s', s=2) #
    ax[1].axis('square')
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    cf.cmap.set_under('black')
    cf.cmap.set_over('white')
    ax[1].set_title('v-FDM')
    fig.colorbar(cf, ax=ax[1], extend='both')

    # plt.draw()
    plt.savefig(save_path + 'uv_[b=%d][t=%d].png'%(batch, num))
    plt.close('all')

if __name__ == '__main__':
    #################### generate data #####################
    # =========== define model parameters ==========
    # dt should be 1/2 of dx
        
    # Diffusion coefficients
    DA = 0.16
    DB = 0.08

    # define birth/death rates
    f = 0.06
    k = 0.062

    # grid size
    N = 128

    # update in time
    delta_t = 1.0
    # spatial step
    dx = 1.0

    N_simulation_steps = 15000
    record_interval = 5
    N_records = N_simulation_steps // record_interval + 1
    
    # Generate 20 different initial conditions where each is a subset of [1,2,3,4,5,6]
    init_dis_map = {
        1: [1, 5], 2: [1, 3, 5], 3: [2, 3, 5], 4: [3, 5], 5: [4, 5], 6: [5],
        7: [6, 4, 5], 8: [1, 4, 5], 9: [2, 4, 5], 10: [3, 4, 5], 11: [4, 5],
        12: [6, 3, 5], 13: [1, 3, 4, 5], 14: [2, 3, 4, 5], 15: [3, 4, 5],
        16: [6, 2, 5], 17: [1, 2, 5], 18: [2, 5], 19: [3, 2, 5], 20: [4, 2, 5]
    }    

    for IC, init_dis in init_dis_map.items():
        A, B = get_initial_A_and_B(N, init_dis=init_dis,  random_influence=0.0)

        save_path = f'2DGS_IC{IC}_2x{N_records}x{N}x{N}.mat'
        
        A_record = np.empty((N_records, N, N))
        B_record = np.empty((N_records, N, N))
        A_record[0] = A.copy()
        B_record[0] = B.copy()

        with tqdm(total=N_simulation_steps) as pbar:
            for step in range(N_simulation_steps):
                # Runge-kutta scheme
                A, B = update_rk4(A, B, DA, DB, f, k, delta_t, dx)

                if step % record_interval == 0:
                    record_index = step // record_interval + 1
                    A_record[record_index] = A.copy()
                    B_record[record_index] = B.copy()

                    pbar.set_description("record_index: %s" % (str(record_index)))

                pbar.update(1)

        A_record_shape = A_record.shape
        B_record_shape = B_record.shape
        
        #np.save('A_record.npy', A_record)
        #np.save('B_record.npy', B_record)
        
        #del A_record
        #del B_record
        
        assert A_record_shape == B_record_shape

        # Create a memory-mapped file
        output_shape = (2, *A_record_shape)
        #UV = np.memmap('output.npy', dtype=np.float32, mode='w+', shape=output_shape)
        UV = np.empty(output_shape, dtype=np.float32)

        # Load and copy A_record
        #A_record = np.load('A_record.npy', mmap_mode='r')
        UV[0] = A_record
        #del A_record

        # Load and copy B_record
        #B_record = np.load('B_record.npy', mmap_mode='r')
        UV[1] = B_record
        #del B_record   
        
        scio.savemat(save_path, {'uv': UV})
        
        del UV
        #os.remove('A_record.npy')
        #os.remove('B_record.npy')
        #os.remove('output.npy')
        
    # else:
    #     data = scio.loadmat(save_path)
    #     UV = data['uv']
    #     del data
    
    #Plot the result
    # output = np.transpose(UV, [1, 0, 2, 3])
    # fig_save_path = ''
    # for i in range(21):
    #     postProcess(output, N, 0, N*dx, 0, N*dx, num=150*i, batch=1,save_path=fig_save_path)

    # plt.figure()
    # plt.plot(UV[0, :, 50, 50], alpha=0.6, label='rk4, dt=1.0')
    # plt.legend()
    # # plt.show()
    # plt.savefig(fig_save_path + 'fig[x=50,y=50].png')

