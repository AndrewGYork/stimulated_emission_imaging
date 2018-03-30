import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import np_tif
from stack_registration import bucket

def main():
    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_7'):
        os.mkdir('./../images/figure_7')

    #####################################################################
    # meltmount mix data
    data = np_tif.tif_to_array(
        './../../stimulated_emission_imaging-data' +
        '/2018_02_23_STE_phase_cr_bead_4' +
        '/dataset_green_1010mW_single_shot.tif').astype(np.float64)

    # get rid of overexposed rows at top and bottom of images
    less_rows = 3
    data = data[:, 0+less_rows:data.shape[1]-less_rows, :]
    data = data[:, ::-1, :] # flip up down

    # reshape to hyperstack
    num_delays = 3
    data = data.reshape((
        data.shape[0]/num_delays,# phase plate angle number
        num_delays,
        data.shape[1],
        data.shape[2],
        ))

    # Get the average pixel brightness in the background region of the
    # meltmount mix data. We'll use it to account for laser intensity
    # fluctuations
    avg_laser_brightness = get_bg_level(data.mean(axis=(0, 1)))
    
    # scale all images to have the same background brightness. This
    # amounts to a correction of roughly 1% or less
    local_laser_brightness = get_bg_level(data)
    data = data * (avg_laser_brightness / local_laser_brightness).reshape(
        data.shape[0], data.shape[1], 1, 1)

    # get zero delay images, max delay images and phase contrast images
    zero_delay_images = data[:, 1, :, :] # zero red/green delay
    max_delay_images = data[
        :, 0:3:2, :, :].mean(axis=1) # average max and min delay
    phase_contrast_images = data[:, 0, :, :] # red before green
    
    # from the image where red/green are simultaneous, subtract the
    # average of the max and min delay images
    STE_stack = zero_delay_images - max_delay_images
    
    # phase contrast image (no STE) stack
    # there is a large background variation that has nothing to do with
    # the sample; it's due to multiple reflections in the microscope.
    # Some of it moves when you move the phase plate, and some of it
    # doesn't. This step subtracts off the stationary component.
    min_contrast_index = 11 # phase plate angle that minimizes bead visibility
    phase_stack = phase_contrast_images - phase_contrast_images[
        min_contrast_index:min_contrast_index + 1, :, :]

    # Luckily the non-stationary component is comprised of stripes that
    # are completely outside of the microscope's spatial pass-band. The
    # smoothing step below strongly attenuates this striping artifact
    # with almost no effect on spatial frequencies due to the sample.
    sigma = 0 # tune this parameter to reject high spatial frequency noise
    STE_stack = gaussian_filter(STE_stack, sigma=(0, sigma, sigma))
    phase_stack = gaussian_filter(phase_stack, sigma=(0, sigma, sigma))

    # crop images to center bead and fit into figure
    top = 0
    bot = 122
    left = 109
    right = 361
    phase_cropped = phase_stack[:, top:bot, left:right]
    STE_cropped = STE_stack[:, top:bot, left:right]

    # Our pixels are tiny (8.7 nm/pixel) to give large dynamic range.
    # This is not great for viewing, because fluctuations can swamp the
    # signal. This step bins the pixels into a more typical size.
    bucket_width = 8 # bucket width in pixels
    phase_cropped = bucket(
        phase_cropped, (1, bucket_width, bucket_width)) / bucket_width**2
    STE_cropped = bucket(
        STE_cropped, (1, bucket_width, bucket_width)) / bucket_width**2

    # use images for phase plate angles that maximize bead brightness
    zero_phase_angle = 8
    pi_phase_angle = 0
    n_mix_zero_phase_bead_image = phase_cropped[zero_phase_angle, :, :]
    n_mix_pi_phase_bead_image = phase_cropped[pi_phase_angle, :, :]
    n_mix_zero_phase_STE_image = STE_cropped[zero_phase_angle, :, :]
    n_mix_pi_phase_STE_image = STE_cropped[pi_phase_angle, :, :]
    
    #####################################################################
    #####################################################################
    # meltmount n = 1.54 data
    data = np_tif.tif_to_array(
        './../../stimulated_emission_imaging-data' +
        '/2018_02_27_STE_phase_n_1_54_cr_bead_0' +
        '/dataset_green_970mW_single_shot.tif').astype(np.float64)

    # get rid of overexposed rows at top and bottom of images
    data = data[:, 0+less_rows:data.shape[1]-less_rows, :]
    data = data[:, ::-1, :] # flip up down


    # reshape to hyperstack
    data = data.reshape((
        data.shape[0]/num_delays,# phase plate angle number
        num_delays,
        data.shape[1],
        data.shape[2],
        ))
    
    # scale all images to have the same background brightness. This
    # amounts to a correction of roughly 1% or less
    local_laser_brightness = get_bg_level(data)
    data = data * (avg_laser_brightness / local_laser_brightness).reshape(
        data.shape[0], data.shape[1], 1, 1)

    # get zero delay images, max delay images and phase contrast images
    zero_delay_images = data[:, 1, :, :] # zero red/green delay
    max_delay_images = data[
        :, 0:3:2, :, :].mean(axis=1) # average max and min delay
    phase_contrast_images = data[:, 0, :, :] # red before green
    
    # from the image where red/green are simultaneous, subtract the
    # average of the max and min delay images
    STE_stack = zero_delay_images - max_delay_images
    
    # phase contrast image (no STE) stack
    # there is a large background variation that has nothing to do with
    # the sample; it's due to multiple reflections in the microscope.
    # Some of it moves when you move the phase plate, and some of it
    # doesn't. This step subtracts off the stationary component.
    min_contrast_index = 11
    phase_stack = phase_contrast_images - phase_contrast_images[
        min_contrast_index:min_contrast_index + 1, :, :]

    # Luckily the non-stationary component is comprised of stripes that
    # are completely outside of the microscope's spatial pass-band. The
    # smoothing step below strongly attenuates this striping artifact
    # with almost no effect on spatial frequencies due to the sample.
    STE_stack = gaussian_filter(STE_stack, sigma=(0, sigma, sigma))
    phase_stack = gaussian_filter(phase_stack, sigma=(0, sigma, sigma))

    # crop images to center bead and fit into figure
    top = 0
    bot = 122
    left = 44
    right = 296
    phase_cropped = phase_stack[:,top:bot,left:right]
    STE_cropped = STE_stack[:,top:bot,left:right]

    # Our pixels are tiny (8.7 nm/pixel) to give large dynamic range.
    # This is not great for viewing, because fluctuations can swamp the
    # signal. This step bins the pixels into a more typical size.
    phase_cropped = bucket(
        phase_cropped, (1, bucket_width, bucket_width)) / bucket_width**2
    STE_cropped = bucket(
        STE_cropped, (1, bucket_width, bucket_width)) / bucket_width**2

    # use images for phase plate angles that maximize bead brightness
    zero_phase_angle = 8#10
    pi_phase_angle = 13#1
    n_1_53_zero_phase_bead_image = phase_cropped[zero_phase_angle, :, :]
    n_1_53_pi_phase_bead_image = phase_cropped[pi_phase_angle, :, :]
    n_1_53_zero_phase_STE_image = STE_cropped[zero_phase_angle, :, :]
    n_1_53_pi_phase_STE_image = STE_cropped[pi_phase_angle, :, :]
    
    #####################################################################
    #####################################################################
    # meltmount n = 1.61 data
    data = np_tif.tif_to_array(
        './../../stimulated_emission_imaging-data' +
        '/2018_02_26_STE_phase_n_1_61_cr_bead_0' +
        '/dataset_green_1060mW_single_shot.tif').astype(np.float64)

    # get rid of overexposed rows at top and bottom of images
    data = data[:, 0+less_rows:data.shape[1]-less_rows, :]
    data = data[:, ::-1, :] # flip up down

    # reshape to hyperstack
    data = data.reshape((
        data.shape[0]/num_delays,# phase plate angle number
        num_delays,
        data.shape[1],
        data.shape[2],
        ))

    # scale all images to have the same background brightness. This
    # amounts to a correction of roughly 1% or less
    local_laser_brightness = get_bg_level(data)
    data = data * (avg_laser_brightness / local_laser_brightness).reshape(
        data.shape[0], data.shape[1], 1, 1)

    # get zero delay images, max delay images and phase contrast images
    zero_delay_images = data[:, 1, :, :] # zero red/green delay
    max_delay_images = data[
        :, 0:3:2, :, :].mean(axis=1) # average max and min delay
    phase_contrast_images = data[:, 0, :, :] # red before green
    
    # from the image where red/green are simultaneous, subtract the
    # average of the max and min delay images
    STE_stack = zero_delay_images - max_delay_images
    
    # phase contrast image (no STE) stack there is a large background
    # variation that has nothing to do with the sample; it's due to
    # multiple reflections in the microscope. Some of it moves when you
    # move the phase plate, and some of it doesn't. This step subtracts
    # off the stationary component. for this image, the background
    # variation is sufficient that different low-contrast images should
    # be subtracted from the first and second halves of the data
    min_contrast_index_1 = 5#11#14 # first half of phase angles
    min_contrast_index_2 = 11 # second half of phase angles
    phase_stack = phase_contrast_images
    phase_stack[0:8, ...] -= phase_contrast_images[
        min_contrast_index_1:min_contrast_index_1 + 1, :, :]
    phase_stack[8:15, ...] -= phase_contrast_images[
        min_contrast_index_2:min_contrast_index_2 + 1, :, :]

    # Luckily the non-stationary component is comprised of stripes that
    # are completely outside of the microscope's spatial pass-band. The
    # smoothing step below strongly attenuates this striping artifact
    # with almost no effect on spatial frequencies due to the sample.
    STE_stack = gaussian_filter(STE_stack, sigma=(0, sigma, sigma))
    phase_stack = gaussian_filter(phase_stack, sigma=(0, sigma, sigma))

    # crop images to center bead and fit into figure
    top = 0
    bot = 122
    left = 59
    right = 311
    phase_cropped = phase_stack[:,top:bot,left:right]
    STE_cropped = STE_stack[:,top:bot,left:right]

    # Our pixels are tiny (8.7 nm/pixel) to give large dynamic range.
    # This is not great for viewing, because fluctuations can swamp the
    # signal. This step bins the pixels into a more typical size.
    phase_cropped = bucket(
        phase_cropped, (1, bucket_width, bucket_width)) / bucket_width**2
    STE_cropped = bucket(
        STE_cropped, (1, bucket_width, bucket_width)) / bucket_width**2

    # use images for phase plate angles that maximize bead brightness
    zero_phase_angle = 8#8
    pi_phase_angle = 0#13
    n_1_61_zero_phase_bead_image = phase_cropped[zero_phase_angle, :, :]
    n_1_61_pi_phase_bead_image = phase_cropped[pi_phase_angle, :, :]
    n_1_61_zero_phase_STE_image = STE_cropped[zero_phase_angle, :, :]
    n_1_61_pi_phase_STE_image = STE_cropped[pi_phase_angle, :, :]
    #####################################################################
    #####################################################################
    
    # start plotting all the images

    # get max and min values to unify the colorbar
    all_phase = np.concatenate((
        n_mix_zero_phase_bead_image,
        n_1_53_zero_phase_bead_image,
        n_1_61_zero_phase_bead_image,
        n_mix_pi_phase_bead_image,
        n_1_53_pi_phase_bead_image,
        n_1_61_pi_phase_bead_image), axis=0)
    all_STE = np.concatenate((
        n_mix_zero_phase_STE_image,
        n_1_53_zero_phase_STE_image,
        n_1_61_zero_phase_STE_image,
        n_mix_pi_phase_STE_image,
        n_1_53_pi_phase_STE_image,
        n_1_61_pi_phase_STE_image), axis=0)
    max_phase = int(np.amax(all_phase)) + 1
    min_phase = int(np.amin(all_phase)) - 1
    max_ste = int(np.amax(all_STE)) + 1
    min_ste = int(np.amin(all_STE)) - 1

    print(max_phase, min_phase, max_ste, min_ste)
    

    num_angles, height, width = STE_cropped.shape
    between_pics = int(16 / bucket_width)
    big_width = width*3 + between_pics*2

    # make scale bar black to give lower limit on colorbar
    bar_left = 1
    bar_right = 6
    bar_vert = -2
    n_mix_zero_phase_bead_image[bar_vert, bar_left:bar_right] = min_phase
    n_1_53_zero_phase_bead_image[bar_vert, bar_left:bar_right] = min_phase
    n_1_61_zero_phase_bead_image[bar_vert, bar_left:bar_right] = min_phase
    n_mix_pi_phase_bead_image[bar_vert, bar_left:bar_right] = min_phase
    n_1_53_pi_phase_bead_image[bar_vert, bar_left:bar_right] = min_phase
    n_1_61_pi_phase_bead_image[bar_vert, bar_left:bar_right] = min_phase
    n_mix_zero_phase_STE_image[bar_vert, bar_left:bar_right] = min_ste
    n_1_53_zero_phase_STE_image[bar_vert, bar_left:bar_right] = min_ste
    n_1_61_zero_phase_STE_image[bar_vert, bar_left:bar_right] = min_ste
    n_mix_pi_phase_STE_image[bar_vert, bar_left:bar_right] = min_ste
    n_1_53_pi_phase_STE_image[bar_vert, bar_left:bar_right] = min_ste
    n_1_61_pi_phase_STE_image[bar_vert, bar_left:bar_right] = min_ste

    between_color = max_phase # makes it white and gives upper limit on colorbar
    zero_phase_bead_image = np.zeros((height,big_width)) + between_color
    pi_phase_bead_image = np.zeros((height,big_width)) + between_color

    between_color = max_ste # makes it white and gives upper limit on colorbar
    zero_phase_STE_image = np.zeros((height,big_width)) + between_color
    pi_phase_STE_image = np.zeros((height,big_width)) + between_color


    # n = 1.53 on left
    left = 0
    right = width
    zero_phase_bead_image[:,left:right] = n_1_53_zero_phase_bead_image
    pi_phase_bead_image[:,left:right] = n_1_53_pi_phase_bead_image
    zero_phase_STE_image[:,left:right] = n_1_53_zero_phase_STE_image
    pi_phase_STE_image[:,left:right] = n_1_53_pi_phase_STE_image

    # n = 1.58/1.61 mix in center
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

    cax0 = ax0.imshow(pi_phase_bead_image, cmap=plt.cm.gray,
                      interpolation='nearest')
    ax0.axis('off')
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right",size="1%",pad=0.25)
    plt.colorbar(cax0, cax = cax)
    ax0.set_title('Phase contrast image of bead',fontsize=30,)#fontweight='bold')
    ax0.text(130,115,r'$\Delta n\approx 0.05$',fontsize=38,color='white',fontweight='bold')
    ax0.text(433,115,r'$\Delta n\approx 0$',fontsize=38,color='white',fontweight='bold')
    ax0.text(643,115,r'$\Delta n\approx -0.01$',fontsize=38,color='white',fontweight='bold')


    cax1 = ax1.imshow(pi_phase_STE_image, cmap=plt.cm.gray,
                      interpolation='nearest')
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right",size="1%",pad=0.25)
    plt.colorbar(cax1, cax = cax)
    ax1.text(130,115,r'$\Delta n\approx 0.05$',fontsize=38,color='white',fontweight='bold')
    ax1.text(433,115,r'$\Delta n\approx 0$',fontsize=38,color='white',fontweight='bold')
    ax1.text(643,115,r'$\Delta n\approx -0.01$',fontsize=38,color='white',fontweight='bold')
    ax1.set_title('Change due to excitation',fontsize=30,)#fontweight='bold')
    ax1.axis('off')
##    plt.savefig('./../images/figure_7/STE_crimson_bead_pi_phase.svg',
##                bbox_inches='tight', pad_inches=0.1)
    plt.show()
##    plt.close()

    fig, (ax0, ax1) = plt.subplots(nrows=2,ncols=1,figsize=(20,7))

    cax0 = ax0.imshow(zero_phase_bead_image, cmap=plt.cm.gray,
                      interpolation='nearest')
    ax0.axis('off')
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right",size="1%",pad=0.25)
    plt.colorbar(cax0, cax = cax)
    ax0.set_title('Phase contrast image of bead',fontsize=30,)#fontweight='bold')
    ax0.text(130,115,r'$\Delta n\approx 0.05$',fontsize=38,color='white',fontweight='bold')
    ax0.text(433,115,r'$\Delta n\approx 0$',fontsize=38,color='white',fontweight='bold')
    ax0.text(642,115,r'$\Delta n\approx -0.01$',fontsize=38,color='white',fontweight='bold')


    cax1 = ax1.imshow(zero_phase_STE_image, cmap=plt.cm.gray,
                      interpolation='nearest')
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right",size="1%",pad=0.25)
    plt.colorbar(cax1, cax = cax)
    ax1.text(130,115,r'$\Delta n\approx 0.05$',fontsize=38,color='white',fontweight='bold')
    ax1.text(433,115,r'$\Delta n\approx 0$',fontsize=38,color='white',fontweight='bold')
    ax1.text(642,115,r'$\Delta n\approx -0.01$',fontsize=38,color='white',fontweight='bold')
    ax1.set_title('Change due to excitation',fontsize=30,)#fontweight='bold')
    ax1.axis('off')
##    plt.savefig('./../images/figure_7/STE_crimson_bead_zero_phase.svg',
##                bbox_inches='tight', pad_inches=0.1)
    plt.show()
##    plt.close()
##
##
##
##
##
##
##
######    fig, (ax0, ax1) = plt.subplots(nrows=2,ncols=1,figsize=(10,7))
######
######    cax0 = ax0.imshow(zero_phase_bead_image, cmap=plt.cm.gray)
######    ax0.axis('off')
######    cbar0 = fig.colorbar(cax0,ax=ax0)
######    ax0.set_title('Phase contrast image of bead',fontsize=30,fontweight='bold')
######    ax0.text(155,115,r'$\Delta n\approx 0.06$',fontsize=24,color='white',fontweight='bold')
######    ax0.text(450,115,r'$\Delta n\approx 0$',fontsize=24,color='white',fontweight='bold')
######    ax0.text(675,115,r'$\Delta n\approx -0.02$',fontsize=24,color='white',fontweight='bold')
######
######
######    cax1 = ax1.imshow(zero_phase_STE_image, cmap=plt.cm.gray)
######    cbar1 = fig.colorbar(cax1, ax = ax1)
######    ax1.text(155,115,r'$\Delta n\approx 0.06$',fontsize=24,color='white',fontweight='bold')
######    ax1.text(450,115,r'$\Delta n\approx 0$',fontsize=24,color='white',fontweight='bold')
######    ax1.text(675,115,r'$\Delta n\approx -0.02$',fontsize=24,color='white',fontweight='bold')
######    ax1.set_title('Change due to excitation',fontsize=30,fontweight='bold')
######    ax1.axis('off')
######    plt.savefig('./../images/figure_7/STE_crimson_bead_zero_phase.svg')
######    plt.close()
####
######        if angle_num == 0 or angle_num == 7:
######
######            # generate and save plot
######            print(np.max(STE_image,(0,1)))
######            print(np.min(STE_image,(0,1)))
######            print(np.max(phase_image,(0,1)))
######            print(np.min(phase_image,(0,1)))
######        
######            fig, (ax0, ax1) = plt.subplots(nrows=2,ncols=1,figsize=(9,8))
######
######            cax0 = ax0.imshow(phase_image, cmap=plt.cm.gray)
######            ax0.axis('off')
######            cbar0 = fig.colorbar(cax0,ax=ax0)
######            ax0.set_title('Phase contrast image of crimson bead')
######
######            cax1 = ax1.imshow(STE_image, cmap=plt.cm.gray)
######            cbar1 = fig.colorbar(cax1, ax = ax1)
######            ax1.set_title('Change in phase contrast image due to stim. emission')
######            ax1.axis('off')
######            plt.savefig('phase_STE_image_' + str(angle_num)+'.svg')
######            plt.show()
######    
######    
######    # average points around center lobe of the nanodiamond image to get
######    # "average signal level" for darkfield and STE images
######    top = 9
######    bot = 84
######    left = 153
######    right = 232
######    STE_signal = (
######        STE_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
######    crosstalk_signal = (
######        crosstalk_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
######    phase_signal = (
######        phase_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
######    
######    # plot signal v phase
######    angles = range(15)
######    true_signal = STE_signal - crosstalk_signal
######    print(angles)
######    print(phase_signal.shape)
######    plt.figure()
######    plt.plot(angles,phase_signal,'.-',color='black')
######    plt.title('Phase contrast image main lobe brightness')
######    plt.xlabel('Relative phase (arb. units)')
######    plt.ylabel('Average intensity (CMOS pixel counts)')
######    plt.grid()
########    plt.savefig('darkfield_v_z.svg')
######    plt.figure()
######    plt.plot(angles,STE_signal,'.-',label='STE signal',color='blue')
######    plt.plot(angles,crosstalk_signal,'.-',label='AOM crosstalk',color='green')
######    plt.title('Stimulated emission signal main lobe intensity')
######    plt.xlabel('Relative phase (arb. units)')
######    plt.ylabel('Change in phase contrast signal (CMOS pixel counts)')
######    plt.legend(loc='lower right')
######    plt.grid()
########    plt.savefig('darkfield_STE_v_z.svg')
######    plt.figure()
######    plt.plot(angles,true_signal,'.-',label='STE signal',color='red')
######    plt.title('Corrected stimulated emission signal main lobe intensity')
######    plt.xlabel('Relative phase (arb. units)')
######    plt.ylabel('Change in phase contrast signal (CMOS pixel counts)')
######    plt.legend(loc='lower right')
######    plt.grid()
########    plt.savefig('darkfield_STE_v_z.svg')
######    plt.show()
##    

    return None

def get_bg_level(data):
    num_regions = 2
    
    # region 1
    bg_up = 2
    bg_down = 120
    bg_left = 285
    bg_right = 379
    bg_level = data[..., bg_up:bg_down, bg_left:bg_right].mean(axis=(-2, -1))

    # region 2
    bg_up = 2
    bg_down = 120
    bg_left = 1
    bg_right = 81
    bg_level += data[..., bg_up:bg_down, bg_left:bg_right].mean(axis=(-2, -1))

    return(bg_level / num_regions)


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
