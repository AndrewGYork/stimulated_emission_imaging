import os
import numpy as np
import np_tif
from stack_registration import stack_registration
import matplotlib.pyplot as plt

def main():

    # the data to be plotted by this program is generated from raw tifs
    # and repetition_average_expt_and_control.py

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_A4'):
        os.mkdir('./../images/figure_A4')

    filename = (
        './../../stimulated_emission_data/figure_A4/dataset_red_300mW.tif')
    filename_ctrl = (
        './../../stimulated_emission_data/figure_A4/dataset_red_0mW.tif')
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
    num_delays = 5
    data = data.reshape((
        data.shape[0]/num_delays,
        num_delays,
        data.shape[1],
        data.shape[2],
        ))
    data_ctrl = data_ctrl.reshape((
        data_ctrl.shape[0]/num_delays,
        num_delays,
        data_ctrl.shape[1],
        data_ctrl.shape[2],
        ))
    # from the image where red/green are simultaneous, subtract the
    # average of images taken when the delay magnitude is greatest
    depletion_stack = (
        data[:,2,:,:] - # zero red/green delay
        0.5 * (data[:,0,:,:] + data[:,4,:,:]) # max red/green delay
        )
    crosstalk_stack = (
        data_ctrl[:,2,:,:] - # zero red/green delay
        0.5 * (data_ctrl[:,0,:,:] + data_ctrl[:,4,:,:]) # max red/green delay
        )
    # darkfield image (no STE) stack
    fluorescence_stack = 0.5 * (data[:,0,:,:] + data[:,4,:,:])

    # save processed stacks
##    np_tif.array_to_tif(STE_stack,'STE_stack.tif')
##    np_tif.array_to_tif(crosstalk_stack,'crosstalk_stack.tif')
##    np_tif.array_to_tif(darkfield_stack,'darkfield_stack.tif')

    # plot darkfield and stim emission signal
    top = 0
    bot = 106
    left = 92
    right = 240
    fluorescence_cropped = fluorescence_stack[:,top:bot,left:right] - 102
    depletion_cropped = (depletion_stack[:,top:bot,left:right] -
                   crosstalk_stack[:,top:bot,left:right])

    depletion_y_z = np.zeros((depletion_cropped.shape[0]*23,depletion_cropped.shape[1]))
    fluorescence_y_z = np.zeros((depletion_cropped.shape[0]*23,depletion_cropped.shape[1]))
    depletion_x_z = np.zeros((depletion_cropped.shape[0]*23,depletion_cropped.shape[2]))
    fluorescence_x_z = np.zeros((depletion_cropped.shape[0]*23,depletion_cropped.shape[2]))
    

    for z_num in range(depletion_cropped.shape[0]):
        # filter fluorescence and depletion images
        depletion_image = depletion_cropped[z_num,:,:]
        depletion_image = depletion_image.reshape(
            1,depletion_image.shape[0],depletion_image.shape[1])
        depletion_image = annular_filter(depletion_image,r1=0,r2=0.03)
        depletion_image = depletion_image[0,:,:]
        depletion_y_z[z_num*23:(z_num+1)*23,:] = depletion_image.mean(axis=1)
        depletion_x_z[z_num*23:(z_num+1)*23,:] = depletion_image.mean(axis=0)
        fluorescence_image = fluorescence_cropped[z_num,:,:]
        fluorescence_image = fluorescence_image.reshape(
            1,fluorescence_image.shape[0],fluorescence_image.shape[1])
        fluorescence_image = annular_filter(fluorescence_image,r1=0,r2=0.03)
        fluorescence_image = fluorescence_image[0,:,:]
        fluorescence_y_z[z_num*23:(z_num+1)*23,:] = fluorescence_image.mean(axis=1)
        fluorescence_x_z[z_num*23:(z_num+1)*23,:] = fluorescence_image.mean(axis=0)

        # generate and save plot
        depletion_image[0,0] = -38 # cheap way to conserve colorbar
        depletion_image[0,1] = 1 # cheap way to conserve colorbar
        fluorescence_image[0,0] = 380 #cheap way to conserve colorbar
        fluorescence_image[0,1] = 0 #cheap way to conserve colorbar
        fluorescence_image[103:109,5:34] =  np.max(fluorescence_image,(0,1))# scale bar
        depletion_image[103:109,5:34] =  np.min(depletion_image,(0,1))# scale bar
        
        fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2,figsize=(19,5))

        cax0 = ax0.imshow(fluorescence_image, cmap=plt.cm.gray)
        ax0.axis('off')
        cbar0 = fig.colorbar(cax0,ax=ax0)
        ax0.set_title('Fluorescence image of nanodiamond')

        cax1 = ax1.imshow(depletion_image, cmap=plt.cm.gray)
        cbar1 = fig.colorbar(cax1, ax = ax1)
        ax1.set_title('Fluorescence intensity decreased due to stim. emission')
        ax1.axis('off')
##        plt.savefig('./../images/figure_A4/fluorescence_depletion_image_' +
##                    str(z_num)+'.svg')
        plt.close()

    # generate and save x projection of stack
    fluorescence_y_z[428:434,5:34] =  np.max(fluorescence_y_z,(0,1))# scale bar
    depletion_y_z[428:434,5:34] =  np.min(depletion_y_z,(0,1))# scale bar
##    np_tif.array_to_tif(darkfield_y_z,'darkfield_y_z.tif')
##    np_tif.array_to_tif(STE_y_z,'STE_y_z.tif')
    fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2,figsize=(14,14))
    cax0 = ax0.imshow(fluorescence_y_z, cmap=plt.cm.gray)
    ax0.axis('off')
    cbar0 = fig.colorbar(cax0,ax=ax0)
    ax0.set_title('A', fontsize=30)

    cax1 = ax1.imshow(depletion_y_z, cmap=plt.cm.gray)
    cbar1 = fig.colorbar(cax1, ax = ax1)
    ax1.set_title('B', fontsize=30)
    ax1.axis('off')
    plt.savefig('./../images/figure_A4/fluorescence_nd_image_yz.svg')
    plt.close()

    # generate and save y projection of stack
##    darkfield_x_z = darkfield_x_z[:,30:175-30]
##    STE_x_z = STE_x_z[:,30:175-30]
    fluorescence_x_z[428:434,5:34] =  np.max(fluorescence_x_z,(0,1))# scale bar
    depletion_x_z[428:434,5:34] =  np.min(depletion_x_z,(0,1))# scale bar
##    np_tif.array_to_tif(darkfield_x_z,'darkfield_x_z.tif')
##    np_tif.array_to_tif(STE_x_z,'STE_x_z.tif')
    fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2,figsize=(14,14))
    cax0 = ax0.imshow(fluorescence_x_z, cmap=plt.cm.gray)
    ax0.axis('off')
    cbar0 = fig.colorbar(cax0,ax=ax0)
    ax0.set_title('A', fontsize=30)

    cax1 = ax1.imshow(depletion_x_z, cmap=plt.cm.gray)
    cbar1 = fig.colorbar(cax1, ax = ax1)
    ax1.set_title('B', fontsize=30)
    ax1.axis('off')
    plt.savefig('./../images/figure_A4/fluorescence_nd_image_xz.svg')
    plt.close()
    
    # average points around center lobe of the nanodiamond image to get
    # "average signal level" for darkfield and STE images
    top = 18
    bot = 78
    left = 133
    right = 188
    depletion_signal = (
        depletion_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    crosstalk_signal = (
        crosstalk_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    fluorescence_signal = (
        fluorescence_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    
    # plot signal v z
    z_list = a = np.arange(-1400,2301,200)
    plt.figure()
    plt.plot(z_list,fluorescence_signal,'.-',color='black')
    plt.title('Fluorescence image main lobe brightness')
    plt.xlabel('Z (nm)')
    plt.ylabel('Average intensity (CMOS pixel counts)')
    plt.grid()

    plt.figure()
    plt.plot(z_list,depletion_signal,'.-',label='depletion signal',color='blue')
    plt.plot(z_list,crosstalk_signal,'.-',label='AOM crosstalk',color='green')
    plt.title('Depletion signal main lobe intensity')
    plt.xlabel('Z (nm)')
    plt.ylabel('Change in fluorescence signal (CMOS pixel counts)')
    plt.legend(loc='lower right')
    plt.grid()

##    plt.show()
    

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

main()
