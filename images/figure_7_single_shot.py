import os
import numpy as np
import np_tif
from stack_registration import stack_registration
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main():

    # the data to be plotted by this program is generated from raw tifs
    # and repetition_average_expt_and_control.py

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_7'):
        os.mkdir('./../images/figure_7')


    # for unifying the color bar
    max_phase = 5746
    min_phase = -3696
    max_ste = 225
    min_ste = -263

    

    
    

    #####################################################################
    # meltmount mix data
    filename = (
        './../../stimulated_emission_data/figure_7/n_mix/dataset_green_1010mW_single_shot.tif')
    filename_ctrl = (
        './../../stimulated_emission_data/figure_7/n_mix/dataset_green_0mW_single_shot.tif')
    data = np_tif.tif_to_array(filename).astype(np.float64)
    data_ctrl = np_tif.tif_to_array(filename_ctrl).astype(np.float64)

    # get rid of overexposed rows at top and bottom of images
    less_rows = 3
    data = data[:,0+less_rows:data.shape[1]-less_rows,:]
    data = data[:,::-1,:]
    data_ctrl = data_ctrl[:,0+less_rows:data_ctrl.shape[1]-less_rows,:]
    data_ctrl = data_ctrl[:,::-1,:]

    # combine experiment and control images
    data_combined = np.zeros((2,data.shape[0],data.shape[1],data.shape[2]))
    data_combined[0] = data
    data_combined[1] = data_ctrl

    # reshape to hyperstack
    data = data_combined[0]
    data_ctrl = data_combined[1]
    num_delays = 3
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
    STE_stack = (
        data[:,1,:,:] - # zero red/green delay
        0.5 * (data[:,0,:,:] + data[:,2,:,:]) # max red/green delay
        )
    crosstalk_stack = (
        data_ctrl[:,1,:,:] - # zero red/green delay
        0.5 * (data_ctrl[:,0,:,:] + data_ctrl[:,2,:,:]) # max red/green delay
        )
    # phase contrast image (no STE) stack
    phase_stack = data[:,0,:,:]#0.5 * (data[:,0,:,:] + data[:,2,:,:])
    phase_stack_ctrl = data_ctrl[:,0,:,:]#0.5 * (data_ctrl[:,0,:,:] + data_ctrl[:,2,:,:])
    
    # bg subtract no-contrast image
    bg_index = 11
    phase_stack = phase_stack - phase_stack[bg_index,:,:]
    phase_stack_ctrl = phase_stack_ctrl - phase_stack_ctrl[bg_index,:,:]

    # plot phase contrast image and stim emission signal
    top = 0
    bot = 122
    left = 59+50
    right = 311+50
    phase_cropped = phase_stack[:,top:bot,left:right]
##    STE_cropped = (STE_stack[:,top:bot,left:right] -
##                   crosstalk_stack[:,top:bot,left:right])
    STE_cropped = STE_stack[:,top:bot,left:right]

    for angle_num in range(STE_cropped.shape[0]):
        # filter darkfield and STE images
        STE_image = STE_cropped[angle_num,:,:]
        STE_image = STE_image.reshape(
            1,STE_image.shape[0],STE_image.shape[1])
        STE_image = annular_filter(STE_image,r1=0,r2=0.02)
        STE_image = STE_image[0,:,:]
        
        phase_image = phase_cropped[angle_num,:,:]
        phase_image = phase_image.reshape(
            1,phase_image.shape[0],phase_image.shape[1])
        phase_image = annular_filter(phase_image,r1=0,r2=0.02)
        phase_image = phase_image[0,:,:]
        

        STE_image[0,0] = max_ste # cheap way to conserve colorbar
        STE_image[1,0] = min_ste # cheap way to conserve colorbar
        phase_image[0,0] = max_phase #cheap way to conserve colorbar
        phase_image[1,0] = min_phase #cheap way to conserve colorbar
        STE_image[108:114,5:34] = max_ste # scale bar
        phase_image[108:114,5:34] = max_phase # scale bar
##        print(angle_num,np.min(STE_image))
##        print(angle_num,np.max(STE_image))
##        print(angle_num,np.min(phase_image))
##        print(angle_num,np.max(phase_image))

        STE_cropped[angle_num,:,:] = STE_image
        phase_cropped[angle_num,:,:] = phase_image


    zero_phase_angle = 8
    pi_phase_angle = 0
    n_mix_zero_phase_bead_image = phase_cropped[zero_phase_angle,:,:]
    n_mix_pi_phase_bead_image = phase_cropped[pi_phase_angle,:,:]
    n_mix_zero_phase_STE_image = STE_cropped[zero_phase_angle,:,:]
    n_mix_pi_phase_STE_image = STE_cropped[pi_phase_angle,:,:]
    #####################################################################
    #####################################################################
    # meltmount n = 1.53 data
    filename = (
        './../../stimulated_emission_data/figure_7/n_1_53/dataset_green_970mW_single_shot.tif')
    filename_ctrl = (
        './../../stimulated_emission_data/figure_7/n_1_53/dataset_green_0mW_single_shot.tif')
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

    # reshape to hyperstack
    data = data_combined[0]
    data_ctrl = data_combined[1]
    num_delays = 3
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
    STE_stack = (
        data[:,1,:,:] - # zero red/green delay
        0.5 * (data[:,0,:,:] + data[:,2,:,:]) # max red/green delay
        )
    crosstalk_stack = (
        data_ctrl[:,1,:,:] - # zero red/green delay
        0.5 * (data_ctrl[:,0,:,:] + data_ctrl[:,2,:,:]) # max red/green delay
        )
    # phase contrast image (no STE) stack
    phase_stack = data[:,0,:,:]#0.5 * (data[:,0,:,:] + data[:,2,:,:])
    phase_stack_ctrl = data_ctrl[:,0,:,:]#0.5 * (data_ctrl[:,0,:,:] + data_ctrl[:,2,:,:])
    
    # bg subtract no-contrast image
    bg_index = 5
    phase_stack = phase_stack - phase_stack[bg_index,:,:]
    phase_stack_ctrl = phase_stack_ctrl - phase_stack_ctrl[bg_index,:,:]

    # plot phase contrast image and stim emission signal
    top = 0
    bot = 122
    left = 35+9
    right = 35+252+9
    phase_cropped = phase_stack[:,top:bot,left:right]
##    STE_cropped = (STE_stack[:,top:bot,left:right] -
##                   crosstalk_stack[:,top:bot,left:right])
    STE_cropped = STE_stack[:,top:bot,left:right]

    for angle_num in range(STE_cropped.shape[0]):
        # filter darkfield and STE images
        STE_image = STE_cropped[angle_num,:,:]
        STE_image = STE_image.reshape(
            1,STE_image.shape[0],STE_image.shape[1])
        STE_image = annular_filter(STE_image,r1=0,r2=0.02)
        STE_image = STE_image[0,:,:]
        phase_image = phase_cropped[angle_num,:,:]
        phase_image = phase_image.reshape(
            1,phase_image.shape[0],phase_image.shape[1])
        phase_image = annular_filter(phase_image,r1=0,r2=0.02)
        phase_image = phase_image[0,:,:]

        STE_image[0,0] = max_ste # cheap way to conserve colorbar
        STE_image[1,0] = min_ste # cheap way to conserve colorbar
        phase_image[0,0] = max_phase #cheap way to conserve colorbar
        phase_image[1,0] = min_phase #cheap way to conserve colorbar
        STE_image[108:114,5:34] = max_ste # scale bar
        phase_image[108:114,5:34] = max_phase # scale bar
##        print(angle_num,np.min(STE_image))
##        print(angle_num,np.max(STE_image))
##        print(angle_num,np.min(phase_image))
##        print(angle_num,np.max(phase_image))
        

        STE_cropped[angle_num,:,:] = STE_image
        phase_cropped[angle_num,:,:] = phase_image


    zero_phase_angle = 10
    pi_phase_angle = 1
    n_1_53_zero_phase_bead_image = phase_cropped[zero_phase_angle,:,:]
    n_1_53_pi_phase_bead_image = phase_cropped[pi_phase_angle,:,:]
    n_1_53_zero_phase_STE_image = STE_cropped[zero_phase_angle,:,:]
    n_1_53_pi_phase_STE_image = STE_cropped[pi_phase_angle,:,:]
    #####################################################################
    #####################################################################
    # meltmount n = 1.61 data
    filename = (
        './../../stimulated_emission_data/figure_7/n_1_61/dataset_green_1060mW_single_shot.tif')
    filename_ctrl = (
        './../../stimulated_emission_data/figure_7/n_1_61/dataset_green_0mW_single_shot.tif')
    data = np_tif.tif_to_array(filename).astype(np.float64)
    data_ctrl = np_tif.tif_to_array(filename_ctrl).astype(np.float64)

    # get rid of overexposed rows at top and bottom of images
    less_rows = 3
    data = data[:,0+less_rows:data.shape[1]-less_rows,:]
    data = data[:,::-1,:]
    data_ctrl = data_ctrl[:,0+less_rows:data_ctrl.shape[1]-less_rows,:]
    data_ctrl = data_ctrl[:,::-1,:]

    # combine experiment and control images
    data_combined = np.zeros((2,data.shape[0],data.shape[1],data.shape[2]))
    data_combined[0] = data
    data_combined[1] = data_ctrl

    # reshape to hyperstack
    data = data_combined[0]
    data_ctrl = data_combined[1]
    num_delays = 3
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
    STE_stack = (
        data[:,1,:,:] - # zero red/green delay
        0.5 * (data[:,0,:,:] + data[:,2,:,:]) # max red/green delay
        )
    crosstalk_stack = (
        data_ctrl[:,1,:,:] - # zero red/green delay
        0.5 * (data_ctrl[:,0,:,:] + data_ctrl[:,2,:,:]) # max red/green delay
        )
    # phase contrast image (no STE) stack
    phase_stack = data[:,0,:,:]#0.5 * (data[:,0,:,:] + data[:,2,:,:])
    phase_stack_ctrl = data_ctrl[:,0,:,:]#0.5 * (data_ctrl[:,0,:,:] + data_ctrl[:,2,:,:])
    
    # bg subtract no-contrast image
    bg_index = 14
    phase_stack = phase_stack - phase_stack[bg_index,:,:]
    phase_stack_ctrl = phase_stack_ctrl - phase_stack_ctrl[bg_index,:,:]

    # plot phase contrast image and stim emission signal
    top = 0
    bot = 122
    left = 59
    right = 311
    phase_cropped = phase_stack[:,top:bot,left:right]
##    STE_cropped = (STE_stack[:,top:bot,left:right] -
##                   crosstalk_stack[:,top:bot,left:right])
    STE_cropped = STE_stack[:,top:bot,left:right]

    for angle_num in range(STE_cropped.shape[0]):
        # filter darkfield and STE images
        STE_image = STE_cropped[angle_num,:,:]
        STE_image = STE_image.reshape(
            1,STE_image.shape[0],STE_image.shape[1])
        STE_image = annular_filter(STE_image,r1=0,r2=0.02)
        STE_image = STE_image[0,:,:]
        phase_image = phase_cropped[angle_num,:,:]
        phase_image = phase_image.reshape(
            1,phase_image.shape[0],phase_image.shape[1])
        phase_image = annular_filter(phase_image,r1=0,r2=0.02)
        phase_image = phase_image[0,:,:]

        STE_image[0,0] = max_ste # cheap way to conserve colorbar
        STE_image[1,0] = min_ste # cheap way to conserve colorbar
        phase_image[0,0] = max_phase #cheap way to conserve colorbar
        phase_image[1,0] = min_phase #cheap way to conserve colorbar
        STE_image[108:114,5:34] = max_ste # scale bar
        phase_image[108:114,5:34] = max_phase # scale bar
##        print(angle_num,np.min(STE_image))
##        print(angle_num,np.max(STE_image))
##        print(angle_num,np.min(phase_image))
##        print(angle_num,np.max(phase_image))

        STE_cropped[angle_num,:,:] = STE_image
        phase_cropped[angle_num,:,:] = phase_image


    zero_phase_angle = 8
    pi_phase_angle = 13
    n_1_61_zero_phase_bead_image = phase_cropped[zero_phase_angle,:,:]
    n_1_61_pi_phase_bead_image = phase_cropped[pi_phase_angle,:,:]
    n_1_61_zero_phase_STE_image = STE_cropped[zero_phase_angle,:,:]
    n_1_61_pi_phase_STE_image = STE_cropped[pi_phase_angle,:,:]
    #####################################################################

    num_angles, height, width = STE_cropped.shape
    between_pics = 16
    big_width = width*3 + between_pics*2
    
    between_color = max_phase
    zero_phase_bead_image = np.zeros((height,big_width)) + between_color
    pi_phase_bead_image = np.zeros((height,big_width)) + between_color

    between_color = max_ste
    zero_phase_STE_image = np.zeros((height,big_width)) + between_color
    pi_phase_STE_image = np.zeros((height,big_width)) + between_color

    # n = 1.53 on left
    left = 0
    right = width
    zero_phase_bead_image[:,left:right] = n_1_53_zero_phase_bead_image
    pi_phase_bead_image[:,left:right] = n_1_53_pi_phase_bead_image
    zero_phase_STE_image[:,left:right] = n_1_53_zero_phase_STE_image
    pi_phase_STE_image[:,left:right] = n_1_53_pi_phase_STE_image

    # n = 1.53/1.61 mix in center
    left = width + between_pics
    right = width*2 + between_pics
    zero_phase_bead_image[:,left:right] = n_mix_zero_phase_bead_image
    pi_phase_bead_image[:,left:right] = n_mix_pi_phase_bead_image
    zero_phase_STE_image[:,left:right] = n_mix_zero_phase_STE_image
    pi_phase_STE_image[:,left:right] = n_mix_pi_phase_STE_image

    # n = 1.61 in center
    left = width*2 + between_pics*2
    right = big_width
    zero_phase_bead_image[:,left:right] = n_1_61_zero_phase_bead_image
    pi_phase_bead_image[:,left:right] = n_1_61_pi_phase_bead_image
    zero_phase_STE_image[:,left:right] = n_1_61_zero_phase_STE_image
    pi_phase_STE_image[:,left:right] = n_1_61_pi_phase_STE_image


    # generate and save plot

    fig, (ax0, ax1) = plt.subplots(nrows=2,ncols=1,figsize=(20,7))

    cax0 = ax0.imshow(pi_phase_bead_image, cmap=plt.cm.gray)
    ax0.axis('off')
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right",size="1%",pad=0.25)
    plt.colorbar(cax0, cax = cax)
    ax0.set_title('Phase contrast image of bead',fontsize=30,)#fontweight='bold')
    ax0.text(130,115,r'$\Delta n\approx 0.05$',fontsize=38,color='white',fontweight='bold')
    ax0.text(433,115,r'$\Delta n\approx 0$',fontsize=38,color='white',fontweight='bold')
    ax0.text(643,115,r'$\Delta n\approx -0.01$',fontsize=38,color='white',fontweight='bold')


    cax1 = ax1.imshow(pi_phase_STE_image, cmap=plt.cm.gray)
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right",size="1%",pad=0.25)
    plt.colorbar(cax1, cax = cax)
    ax1.text(130,115,r'$\Delta n\approx 0.05$',fontsize=38,color='white',fontweight='bold')
    ax1.text(433,115,r'$\Delta n\approx 0$',fontsize=38,color='white',fontweight='bold')
    ax1.text(643,115,r'$\Delta n\approx -0.01$',fontsize=38,color='white',fontweight='bold')
    ax1.set_title('Change due to excitation',fontsize=30,)#fontweight='bold')
    ax1.axis('off')
    plt.show()
    plt.savefig('./../images/figure_7/STE_crimson_bead_pi_phase.svg',
                bbox_inches='tight', pad_inches=0.1)
##    plt.close()

    fig, (ax0, ax1) = plt.subplots(nrows=2,ncols=1,figsize=(20,7))

    cax0 = ax0.imshow(zero_phase_bead_image, cmap=plt.cm.gray)
    ax0.axis('off')
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right",size="1%",pad=0.25)
    plt.colorbar(cax0, cax = cax)
    ax0.set_title('Phase contrast image of bead',fontsize=30,)#fontweight='bold')
    ax0.text(130,115,r'$\Delta n\approx 0.05$',fontsize=38,color='white',fontweight='bold')
    ax0.text(433,115,r'$\Delta n\approx 0$',fontsize=38,color='white',fontweight='bold')
    ax0.text(642,115,r'$\Delta n\approx -0.01$',fontsize=38,color='white',fontweight='bold')


    cax1 = ax1.imshow(zero_phase_STE_image, cmap=plt.cm.gray)
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right",size="1%",pad=0.25)
    plt.colorbar(cax1, cax = cax)
    ax1.text(130,115,r'$\Delta n\approx 0.05$',fontsize=38,color='white',fontweight='bold')
    ax1.text(433,115,r'$\Delta n\approx 0$',fontsize=38,color='white',fontweight='bold')
    ax1.text(642,115,r'$\Delta n\approx -0.01$',fontsize=38,color='white',fontweight='bold')
    ax1.set_title('Change due to excitation',fontsize=30,)#fontweight='bold')
    ax1.axis('off')
    plt.show()
    plt.savefig('./../images/figure_7/STE_crimson_bead_zero_phase.svg',
                bbox_inches='tight', pad_inches=0.1)
##    plt.close()


##    fig, (ax0, ax1) = plt.subplots(nrows=2,ncols=1,figsize=(10,7))
##
##    cax0 = ax0.imshow(zero_phase_bead_image, cmap=plt.cm.gray)
##    ax0.axis('off')
##    cbar0 = fig.colorbar(cax0,ax=ax0)
##    ax0.set_title('Phase contrast image of bead',fontsize=30,fontweight='bold')
##    ax0.text(155,115,r'$\Delta n\approx 0.06$',fontsize=24,color='white',fontweight='bold')
##    ax0.text(450,115,r'$\Delta n\approx 0$',fontsize=24,color='white',fontweight='bold')
##    ax0.text(675,115,r'$\Delta n\approx -0.02$',fontsize=24,color='white',fontweight='bold')
##
##
##    cax1 = ax1.imshow(zero_phase_STE_image, cmap=plt.cm.gray)
##    cbar1 = fig.colorbar(cax1, ax = ax1)
##    ax1.text(155,115,r'$\Delta n\approx 0.06$',fontsize=24,color='white',fontweight='bold')
##    ax1.text(450,115,r'$\Delta n\approx 0$',fontsize=24,color='white',fontweight='bold')
##    ax1.text(675,115,r'$\Delta n\approx -0.02$',fontsize=24,color='white',fontweight='bold')
##    ax1.set_title('Change due to excitation',fontsize=30,fontweight='bold')
##    ax1.axis('off')
##    plt.savefig('./../images/figure_7/STE_crimson_bead_zero_phase.svg')
##    plt.close()

##        if angle_num == 0 or angle_num == 7:
##
##            # generate and save plot
##            print(np.max(STE_image,(0,1)))
##            print(np.min(STE_image,(0,1)))
##            print(np.max(phase_image,(0,1)))
##            print(np.min(phase_image,(0,1)))
##        
##            fig, (ax0, ax1) = plt.subplots(nrows=2,ncols=1,figsize=(9,8))
##
##            cax0 = ax0.imshow(phase_image, cmap=plt.cm.gray)
##            ax0.axis('off')
##            cbar0 = fig.colorbar(cax0,ax=ax0)
##            ax0.set_title('Phase contrast image of crimson bead')
##
##            cax1 = ax1.imshow(STE_image, cmap=plt.cm.gray)
##            cbar1 = fig.colorbar(cax1, ax = ax1)
##            ax1.set_title('Change in phase contrast image due to stim. emission')
##            ax1.axis('off')
##            plt.savefig('phase_STE_image_' + str(angle_num)+'.svg')
##            plt.show()
##    
##    
##    # average points around center lobe of the nanodiamond image to get
##    # "average signal level" for darkfield and STE images
##    top = 9
##    bot = 84
##    left = 153
##    right = 232
##    STE_signal = (
##        STE_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
##    crosstalk_signal = (
##        crosstalk_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
##    phase_signal = (
##        phase_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
##    
##    # plot signal v phase
##    angles = range(15)
##    true_signal = STE_signal - crosstalk_signal
##    print(angles)
##    print(phase_signal.shape)
##    plt.figure()
##    plt.plot(angles,phase_signal,'.-',color='black')
##    plt.title('Phase contrast image main lobe brightness')
##    plt.xlabel('Relative phase (arb. units)')
##    plt.ylabel('Average intensity (CMOS pixel counts)')
##    plt.grid()
####    plt.savefig('darkfield_v_z.svg')
##    plt.figure()
##    plt.plot(angles,STE_signal,'.-',label='STE signal',color='blue')
##    plt.plot(angles,crosstalk_signal,'.-',label='AOM crosstalk',color='green')
##    plt.title('Stimulated emission signal main lobe intensity')
##    plt.xlabel('Relative phase (arb. units)')
##    plt.ylabel('Change in phase contrast signal (CMOS pixel counts)')
##    plt.legend(loc='lower right')
##    plt.grid()
####    plt.savefig('darkfield_STE_v_z.svg')
##    plt.figure()
##    plt.plot(angles,true_signal,'.-',label='STE signal',color='red')
##    plt.title('Corrected stimulated emission signal main lobe intensity')
##    plt.xlabel('Relative phase (arb. units)')
##    plt.ylabel('Change in phase contrast signal (CMOS pixel counts)')
##    plt.legend(loc='lower right')
##    plt.grid()
####    plt.savefig('darkfield_STE_v_z.svg')
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
