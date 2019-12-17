import os
import numpy as np
import np_tif
from stack_registration import stack_registration
from stack_registration import bucket
import matplotlib.pyplot as plt

def main():


    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_A4'):
        os.mkdir('./../images/figure_A4')

    z_voltages = [
        '_34000mV',
        '_36000mV',
        '_38000mV',
        '_40000mV',
        '_42000mV',
        '_44000mV',
        '_46000mV',
        '_48000mV',
        '_50000mV',
        '_52000mV',
        '_54000mV',
        '_56000mV',
        '_58000mV',
        '_60000mV',
        '_62000mV',
        '_64000mV',
        '_66000mV',
        '_68000mV',
        '_70000mV',
        ]

    bg_level = 111.5

    

    # for each piezo voltage (axial location) we: 
    # 1. repetition average the images for each delay
    # 2. register the repetition averaged "green off" image to the
    # corresponding "green on" image

    num_delays = 5
    num_reps = 20
    num_z = len(z_voltages)
    width = 380
    height = 128
    less_rows = 3
    data = np.zeros((num_z, num_delays, height - less_rows * 2, width))
    data_ctrl = np.zeros((num_z, num_delays, height - less_rows * 2, width))
    
    for z_index, z_v in enumerate(z_voltages):
        print('Piezo voltage', z_v)
        filename = (
            './../../stimulated_emission_imaging-data' +
            '/2016_11_02_STE_z_stack_depletion' +
            '/STE_darkfield_113_green_1500mW_red_300mW' +
            z_v + '_many_delays.tif')
        filename_ctrl = (
            './../../stimulated_emission_imaging-data' +
            '/2016_11_02_STE_z_stack_depletion' +
            '/STE_darkfield_113_green_1500mW_red_0mW' +
            z_v + '_many_delays.tif')
        data_z = np_tif.tif_to_array(filename).astype(np.float64) - bg_level
        data_z_ctrl = np_tif.tif_to_array(filename_ctrl).astype(np.float64) - bg_level
        # reshape data arrays: 0th axis is rep number, 1st is delay number
        data_z = data_z.reshape(num_reps, num_delays, height, width)
        data_z_ctrl = data_z_ctrl.reshape(num_reps, num_delays, height, width)
        # crop to remove overexposed rows (not necessary for low signal
        # measurements like fluorescence but whatever)
        data_z = data_z[:, :, less_rows:height - less_rows, :]
        data_z_ctrl = data_z_ctrl[:, :, less_rows:height - less_rows, :]

        # repetition average both image sets
        data_z_rep_avg = data_z.mean(axis=0)
        data_z_ctrl_rep_avg = data_z_ctrl.mean(axis=0)

        # registration shift control data to match "green on" data
        align_to_this_slice = data_z_rep_avg[0, :, :]
        print("Computing registration shift (no red)...")
        shifts = stack_registration(
            data_z_ctrl_rep_avg,
            align_to_this_slice=align_to_this_slice,
            refinement='integer',
            register_in_place=True,
            background_subtraction='edge_mean')
        print("... done computing shifts.")

        # build arrays with repetition averaged images
        data[z_index, ...] = data_z_rep_avg
        data_ctrl[z_index, ...] = data_z_ctrl_rep_avg

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

    # plot darkfield and stim emission signal
    top = 0
    bot = 106
    left = 92
    right = 240
    fluorescence_cropped = fluorescence_stack[:,top:bot,left:right]
    depletion_cropped = (depletion_stack[:,top:bot,left:right] -
                   crosstalk_stack[:,top:bot,left:right])

    # Our pixels are tiny (8.7 nm/pixel) to give large dynamic range.
    # This is not great for viewing, because fluctuations can swamp the
    # signal. This step bins the pixels into a more typical size.
    bucket_width = 8 # bucket width in pixels
    fluorescence_cropped = bucket(
        fluorescence_cropped, (1, bucket_width, bucket_width)) / bucket_width ** 2
    depletion_cropped = bucket(
        depletion_cropped, (1, bucket_width, bucket_width)) / bucket_width ** 2

    depletion_y_z = np.zeros((
        depletion_cropped.shape[0]*3, depletion_cropped.shape[1]))
    fluorescence_y_z = np.zeros((
        fluorescence_cropped.shape[0]*3, fluorescence_cropped.shape[1]))
    depletion_x_z = np.zeros((
        depletion_cropped.shape[0]*3, depletion_cropped.shape[2]))
    fluorescence_x_z = np.zeros((
        fluorescence_cropped.shape[0]*3, fluorescence_cropped.shape[2]))
    

    for z_num in range(fluorescence_cropped.shape[0]):
        # get darkfield and STE images and create yz and xz views
        depletion_image = depletion_cropped[z_num,:,:]
        depletion_y_z[z_num*3:(z_num+1)*3,:] = depletion_image.mean(axis=1)
        depletion_x_z[z_num*3:(z_num+1)*3,:] = depletion_image.mean(axis=0)
        
        fluorescence_image = fluorescence_cropped[z_num,:,:]
        fluorescence_y_z[z_num*3:(z_num+1)*3,:] = fluorescence_image.mean(axis=1)
        fluorescence_x_z[z_num*3:(z_num+1)*3,:] = fluorescence_image.mean(axis=0)

        # generate and save plot

        # scale bar
        fluorescence_image[-2:-1, 1:6] = 353.11
        depletion_image[-2:-1, 1:6] = -34.01
        
        fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2,figsize=(14,5))

        cax0 = ax0.imshow(fluorescence_image, cmap=plt.cm.gray,
                          interpolation='nearest', vmax=353.11, vmin=0)
        ax0.axis('off')
        cbar0 = fig.colorbar(cax0,ax=ax0)
        ax0.set_title('A', fontsize=30)

        cax1 = ax1.imshow(depletion_image, cmap=plt.cm.gray,
                          interpolation='nearest', vmax=0.61, vmin=-34.01)
        cbar1 = fig.colorbar(cax1, ax = ax1)
        ax1.set_title('B', fontsize=30)
        ax1.axis('off')
        plt.close()

    # x and y projections of stack of fluorescence images
    fluorescence_y_z[-2:-1, 1:6] =  np.max(fluorescence_y_z,(0,1))# scale bar
    fluorescence_x_z[-2:-1, 1:6] =  np.max(fluorescence_x_z,(0,1))# scale bar
    fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, figsize=(14,14))
    cax0 = ax0.imshow(fluorescence_y_z, cmap=plt.cm.gray,
                      interpolation='nearest')
    ax0.axis('off')
    cbar0 = fig.colorbar(cax0,ax=ax0)
    ax0.set_title('A', fontsize=30)

    cax1 = ax1.imshow(fluorescence_x_z, cmap=plt.cm.gray,
                      interpolation='nearest')
    cbar1 = fig.colorbar(cax1, ax = ax1)
    ax1.set_title('B', fontsize=30)
    ax1.axis('off')
    plt.show()

    # x projection of stack of fluorescence and depletion images
    fluorescence_y_z[-2:-1, 1:6] =  np.max(fluorescence_y_z,(0,1))# scale bar
    depletion_y_z[-2:-1, 1:6] =  np.min(depletion_y_z,(0,1))# scale bar
    fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, figsize=(14,14))
    cax0 = ax0.imshow(fluorescence_y_z, cmap=plt.cm.gray,
                      interpolation='nearest')
    ax0.axis('off')
    cbar0 = fig.colorbar(cax0,ax=ax0)
    ax0.set_title('A', fontsize=30)

    cax1 = ax1.imshow(depletion_y_z, cmap=plt.cm.gray,
                      interpolation='nearest')
    cbar1 = fig.colorbar(cax1, ax = ax1)
    ax1.set_title('B', fontsize=30)
    ax1.axis('off')
    plt.show()

    # y projection of stack of fluorescence and depletion images
    fluorescence_x_z[-2:-1, 1:6] =  np.max(fluorescence_x_z,(0,1))# scale bar
    depletion_x_z[-2:-1, 1:6] =  np.min(depletion_x_z,(0,1))# scale bar

    fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2,figsize=(14,14))
    cax0 = ax0.imshow(fluorescence_x_z, cmap=plt.cm.gray,
                      interpolation='nearest')
    ax0.axis('off')
    cbar0 = fig.colorbar(cax0,ax=ax0)
    ax0.set_title('A', fontsize=30)

    cax1 = ax1.imshow(depletion_x_z, cmap=plt.cm.gray,
                      interpolation='nearest')
    cbar1 = fig.colorbar(cax1, ax = ax1)
    ax1.set_title('B', fontsize=30)
    ax1.axis('off')
    plt.savefig('./../images/figure_A4/fluorescence_nd_image_xz.svg')
    plt.show()
    
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
    plt.show()
    plt.figure()
    plt.plot(z_list,depletion_signal,'.-',label='depletion signal',color='blue')
    plt.plot(z_list,crosstalk_signal,'.-',label='AOM crosstalk',color='green')
    plt.title('Depletion signal main lobe intensity')
    plt.xlabel('Z (nm)')
    plt.ylabel('Change in fluorescence light signal (CMOS pixel counts)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    

    return None



main()
