import os
import numpy as np
import np_tif
from stack_registration import stack_registration
import matplotlib.pyplot as plt

def main():

    # the data to be plotted by this program is generated from raw tifs
    # and repetition_average_expt_and_control.py

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_4'):
        os.mkdir('./../images/figure_4')
    if not os.path.isdir('./../images/figure_5'):
        os.mkdir('./../images/figure_5')

    num_reps = 200 # number of times a power/delay stack was taken
    num_delays = 5
    image_h = 128
    image_w = 380

    # power calibration
    # red max power is 300 mW
    # green max power is 1450 mW
    # green powers calibrated using camera
    green_bg = 113.9
    green_max_mW = 1450
    green_powers = np.array(
        (113.9,119.6,124.5,135,145.5,159.5,175.3,193.1,234.5,272.2,334.1,385.7,446.1))
    green_powers -= green_bg
    green_powers = green_powers * green_max_mW / max(green_powers)
    green_powers = np.around(green_powers).astype(int)
##    sparse_green_nums = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    sparse_green_nums = [0,7,10,12]
    green_powers_sparse = green_powers[sparse_green_nums]

    # red powers calibrated using camera
    red_bg = 26.6
    red_max_mW = 300
    red_powers = np.array(
        (26.6, 113, 198, 276, 353, 438, 537))
    red_powers -= red_bg
    red_powers = red_powers * red_max_mW / max(red_powers)
    red_powers = np.around(red_powers).astype(int)
    sparse_red_nums = [0,3,6]
    red_powers_sparse = red_powers[sparse_red_nums]

    filename = './../../stimulated_emission_data/figure_4_5/dataset.tif'
    filename_ctrl = (
        './../../stimulated_emission_data/figure_4_5/dataset_green_blocked.tif')
    data = np_tif.tif_to_array(filename).astype(np.float64)
    data_ctrl = np_tif.tif_to_array(filename_ctrl).astype(np.float64)

    # get rid of overexposed rows at top and bottom of images
    less_rows = 3
    data = data[:,0+less_rows:data.shape[1]-less_rows,:]
    data_ctrl = data_ctrl[:,0+less_rows:data_ctrl.shape[1]-less_rows,:]

    # combine experiment and control images
    data_combined = np.zeros((2,data.shape[0],data.shape[1],data.shape[2]))
    data_combined[0] = data
    data_combined[1] = data_ctrl

    # register each control slice with the corresponding experimental slice
    fmm = 0.02 #fourier mask magnitude is a carefully tuned parameter
    for which_slice in range(data.shape[0]):
        stack_registration(
            data_combined[:,which_slice,:,:],
            fourier_mask_magnitude = fmm,
            )

    # reshape to hyperstack
    data = data_combined[0]
    data_ctrl = data_combined[1]
    data = data.reshape((
        len(red_powers),
        len(green_powers),
        num_delays,
        data.shape[1],
        data.shape[2],
        ))
    data_ctrl = data_ctrl.reshape((
        len(red_powers),
        len(green_powers),
        num_delays,
        data_ctrl.shape[1],
        data_ctrl.shape[2],
        ))
    # from the image where red/green are simultaneous, subtract the
    # average of images taken when the delay magnitude is greatest
    STE_stack = (
        data[:,:,2,:,:] - # zero red/green delay
        0.5 * (data[:,:,0,:,:] + data[:,:,4,:,:]) # max red/green delay
        )
    crosstalk_stack = (
        data_ctrl[:,:,2,:,:] - # zero red/green delay
        0.5 * (data_ctrl[:,:,0,:,:] + data_ctrl[:,:,4,:,:]) # max red/green delay
        )
    # darkfield image (no STE) stack
    darkfield_stack = 0.5 * (data[:,:,0,:,:] + data[:,:,4,:,:])
    darkfield_stack_ctrl = 0.5 * (data_ctrl[:,:,0,:,:] + data[:,:,4,:,:])

    # save processed stacks
##    tif_shape = (
##        len(red_powers)*len(green_powers),
##        STE_stack.shape[2],
##        STE_stack.shape[3],
##        )
##    np_tif.array_to_tif(
##        STE_stack.reshape(tif_shape),'STE_stack.tif')
##    np_tif.array_to_tif(
##        crosstalk_stack.reshape(tif_shape),'crosstalk_stack.tif')
##    np_tif.array_to_tif(
##        darkfield_stack.reshape(tif_shape),'darkfield_stack.tif')
##    np_tif.array_to_tif(
##        darkfield_stack_ctrl.reshape(tif_shape),'darkfield_stack_ctrl.tif')

    # plot darkfield and stim emission signal
    top = 20
    bot = 121
    left = 140
    right = 267
    darkfield_cropped = darkfield_stack[-1,-1,top:bot,left:right]
    STE_cropped = (STE_stack[-1,-1,top:bot,left:right] -
                   crosstalk_stack[-1,-1,top:bot,left:right])
    STE_cropped = STE_cropped.reshape(
        1,STE_cropped.shape[0],STE_cropped.shape[1])
    STE_cropped = annular_filter(STE_cropped,r1=0,r2=0.03)
    STE_cropped = STE_cropped[0,:,:]
    STE_cropped[91:97,5:34] = -82 # scale bar
    darkfield_cropped[91:97,5:34] = 36000 # scale bar

    fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2,figsize=(19,5))

    cax0 = ax0.imshow(darkfield_cropped, cmap=plt.cm.gray)
    ax0.axis('off')
    cbar0 = fig.colorbar(cax0, ax = ax0)
    ax0.set_title('A', fontsize = 30)

    cax1 = ax1.imshow(STE_cropped, cmap=plt.cm.gray)
    cbar1 = fig.colorbar(cax1, ax = ax1)
    ax1.set_title('B', fontsize = 30)
    ax1.axis('off')
    plt.savefig('./../images/figure_4/darkfield_STE_image.svg')
    
    # average points around center lobe of the nanodiamond image to get
    # "average signal level" for darkfield and STE images
    top = 45
    bot = 102
    left = 177
    right = 228
    STE_signal = (
        STE_stack[:,:,top:bot,left:right].mean(axis=3).mean(axis=2))
    crosstalk_signal = (
        crosstalk_stack[:,:,top:bot,left:right].mean(axis=3).mean(axis=2))
    darkfield_signal = (
        darkfield_stack[:,:,top:bot,left:right].mean(axis=3).mean(axis=2))
##    np_tif.array_to_tif(STE_signal,'STE_signal_array.tif')
##    np_tif.array_to_tif(crosstalk_signal,'crosstalk_signal_array.tif')
##    np_tif.array_to_tif(darkfield_signal,'darkfield_signal_array.tif')
    
    # plot STE signal v power
    true_signal = STE_signal - crosstalk_signal
    plt.figure()
    for (pow_num,gr_pow) in enumerate(green_powers):
        if pow_num in sparse_green_nums:
            plt.plot(
                red_powers,true_signal[:,pow_num],
                '.-',label=('Green power = '+str(gr_pow)+' mW'))
    plt.title('Change in signal in main scattered light lobe')
    plt.xlabel('Red power (mW)')
    plt.ylabel('Change in scattered light signal (CMOS pixel counts)')
    plt.legend(loc='lower left')
    plt.grid()
    plt.savefig('./../images/figure_5/STE_v_red_power_sparse.svg')
    plt.figure()
    for (pow_num,rd_pow) in enumerate(red_powers):
        if pow_num in sparse_red_nums:
            plt.plot(
                green_powers,true_signal[pow_num,:],
                '.-',label=('Red power = '+str(rd_pow)+' mW'))
    plt.title('Change in signal in main scattered light lobe')
    plt.xlabel('Green power (mW)')
    plt.ylabel('Change in scattered light signal (CMOS pixel counts)')
    plt.legend(loc='lower left')
    plt.ylim(-47, 8)
    plt.grid()
    plt.savefig('./../images/figure_5/STE_v_green_power_sparse.svg')
    plt.show()
    

    return None




def annular_filter(x, r1, r2):
    assert r2 > r1 >= 0

    x_ft = np.fft.fftn(x)
    n_y, n_x = x.shape[-2:]
    kx = np.fft.fftfreq(n_x).reshape(1, 1, n_x)
    ky = np.fft.fftfreq(n_y).reshape(1, n_y, 1)

    x_ft[kx**2 + ky**2 > r2**2] = 0
    x_ft[kx**2 + ky**2 < r1**2] = 0

    x_filtered = np.fft.ifftn(x_ft).real

    return x_filtered

if __name__ == '__main__':
    main()
