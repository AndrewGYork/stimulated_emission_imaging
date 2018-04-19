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
    phase_contrast_images = data[:, 0, :, :] # red before green (min delay)
    
    # from the image where red/green are simultaneous, subtract the
    # average of the max and min delay images
    STE_stack = zero_delay_images - max_delay_images

    # phase contrast image (no STE) stack: there is a large background
    # variation that has nothing to do with the sample; it's due to
    # multiple reflections in the microscope. Some of it moves when you
    # move the phase plate, and some of it doesn't. This step subtracts
    # off the stationary component. For each image we use in the figure,
    # we subtract the minimum contrast image with the closest phase plate angle.
    # minimum contrast phase plate angle closest to first 7 phase plate angles:
    min_contrast_index_1 = 5
    # minimum contrast phase plate angle closest to last 7 phase plate angles:
    min_contrast_index_2 = 11
    phase_stack = phase_contrast_images
    phase_stack[0:8, ...] = phase_stack[0:8, ...] - phase_contrast_images[
        min_contrast_index_1:min_contrast_index_1 + 1, :, :]
    phase_stack[8:15, ...] = phase_stack[8:15, ...] - phase_contrast_images[
        min_contrast_index_2:min_contrast_index_2 + 1, :, :]

    # Luckily the non-stationary component is comprised of stripes that
    # are completely outside of the microscope's spatial pass-band. The
    # smoothing step below strongly attenuates this striping artifact
    # with almost no effect on spatial frequencies due to the sample.
    sigma = 9 # tune this parameter to reject high spatial frequencies
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

    # display images from the two phase plate angles that maximize bead
    # contrast (+/- contrast)
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
    phase_contrast_images = data[:, 0, :, :] # red before green (min delay)
    
    # from the image where red/green are simultaneous, subtract the
    # average of the max and min delay images
    STE_stack = zero_delay_images - max_delay_images

    # phase contrast image (no STE) stack: there is a large background
    # variation that has nothing to do with the sample; it's due to
    # multiple reflections in the microscope. Some of it moves when you
    # move the phase plate, and some of it doesn't. This step subtracts
    # off the stationary component. For each image we use in the figure,
    # we subtract the minimum contrast image with the closest phase plate angle.
    # minimum contrast phase plate angle closest to first 7 phase plate angles:
    min_contrast_index_1 = 5
    # minimum contrast phase plate angle closest to last 7 phase plate angles:
    min_contrast_index_2 = 11
    phase_stack = phase_contrast_images
    phase_stack[0:8, ...] = phase_stack[0:8, ...] - phase_contrast_images[
        min_contrast_index_1:min_contrast_index_1 + 1, :, :]
    phase_stack[8:15, ...] = phase_stack[8:15, ...] - phase_contrast_images[
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

    # display images from the two phase plate angles that maximize bead
    # contrast (+/- contrast)
    zero_phase_angle = 8
    pi_phase_angle = 13
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
    phase_contrast_images = data[:, 0, :, :] # red before green (min delay)
    
    # from the image where red/green are simultaneous, subtract the
    # average of the max and min delay images
    STE_stack = zero_delay_images - max_delay_images
    
    # phase contrast image (no STE) stack: there is a large background
    # variation that has nothing to do with the sample; it's due to
    # multiple reflections in the microscope. Some of it moves when you
    # move the phase plate, and some of it doesn't. This step subtracts
    # off the stationary component. For each image we use in the figure,
    # we subtract the minimum contrast image with the closest phase plate angle.
    # minimum contrast phase plate angle closest to first 7 phase plate angles:
    min_contrast_index_1 = 5
    # minimum contrast phase plate angle closest to last 7 phase plate angles:
    min_contrast_index_2 = 11
    phase_stack = phase_contrast_images
    phase_stack[0:8, ...] = phase_stack[0:8, ...] - phase_contrast_images[
        min_contrast_index_1:min_contrast_index_1 + 1, :, :]
    phase_stack[8:15, ...] = phase_stack[8:15, ...] - phase_contrast_images[
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

    # display images from the two phase plate angles that maximize bead
    # contrast (+/- contrast)
    zero_phase_angle = 8
    pi_phase_angle = 0
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

    # create wider image comprised of three side-by-side images
    # get width of wider image
    num_angles, height, width = STE_cropped.shape
    between_pics = int(16 / bucket_width)
    big_width = width*3 + between_pics*2
    # initialize wide phase contrast image and make "between color" white
    between_color = max_phase # makes it white and gives upper limit on colorbar
    zero_phase_bead_image = np.zeros((height,big_width)) + between_color
    pi_phase_bead_image = np.zeros((height,big_width)) + between_color
    # initialize wide STE image and make "between color" white
    between_color = max_ste # makes it white and gives upper limit on colorbar
    zero_phase_STE_image = np.zeros((height,big_width)) + between_color
    pi_phase_STE_image = np.zeros((height,big_width)) + between_color
    # n = 1.53 images on left side of wide image
    left = 0
    right = width
    zero_phase_bead_image[:,left:right] = n_1_53_zero_phase_bead_image
    pi_phase_bead_image[:,left:right] = n_1_53_pi_phase_bead_image
    zero_phase_STE_image[:,left:right] = n_1_53_zero_phase_STE_image
    pi_phase_STE_image[:,left:right] = n_1_53_pi_phase_STE_image
    # n = 1.58/1.61 mix images in center of wide image
    left = width + between_pics
    right = width*2 + between_pics
    zero_phase_bead_image[:,left:right] = n_mix_zero_phase_bead_image
    pi_phase_bead_image[:,left:right] = n_mix_pi_phase_bead_image
    zero_phase_STE_image[:,left:right] = n_mix_zero_phase_STE_image
    pi_phase_STE_image[:,left:right] = n_mix_pi_phase_STE_image
    # n = 1.61 on right side of wide image
    left = width*2 + between_pics*2
    right = big_width
    zero_phase_bead_image[:,left:right] = n_1_61_zero_phase_bead_image
    pi_phase_bead_image[:,left:right] = n_1_61_pi_phase_bead_image
    zero_phase_STE_image[:,left:right] = n_1_61_zero_phase_STE_image
    pi_phase_STE_image[:,left:right] = n_1_61_pi_phase_STE_image


    # generate and save plot

    fig, (ax0, ax1) = plt.subplots(nrows=2,ncols=1,figsize=(20,7))

    cax0 = ax0.imshow(pi_phase_bead_image, cmap=plt.cm.gray,
                      interpolation='nearest', vmax=2500, vmin=-4200)
    ax0.axis('off')
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right",size="1%",pad=0.25)
    plt.colorbar(cax0, cax = cax)
    ax0.set_title('Phase contrast image of bead',fontsize=30)
    ax0.text(
        12, 14, r'$\Delta n\approx +0.05$',
        fontsize=38, color='black', fontweight='bold')
    ax0.text(
        53, 14, r'$\Delta n\approx 0$',
        fontsize=38, color='black', fontweight='bold')
    ax0.text(
        79, 14, r'$\Delta n\approx -0.01$',
        fontsize=38, color='black', fontweight='bold')


    cax1 = ax1.imshow(pi_phase_STE_image, cmap=plt.cm.gray,
                      interpolation='nearest')
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right",size="1%",pad=0.25)
    plt.colorbar(cax1, cax = cax)
    ax1.text(
        12, 14 ,r'$\Delta n\approx +0.05$',
        fontsize=38, color='black', fontweight='bold')
    ax1.text(
        53, 14, r'$\Delta n\approx 0$',
        fontsize=38, color='black', fontweight='bold')
    ax1.text(
        79, 14, r'$\Delta n\approx -0.01$',
        fontsize=38, color='black', fontweight='bold')
    ax1.set_title('Change due to excitation',fontsize=30,)
    ax1.axis('off')
    plt.savefig('./../images/figure_7/STE_crimson_bead_pi_phase.svg',
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    fig, (ax0, ax1) = plt.subplots(nrows=2,ncols=1,figsize=(20,7))

    cax0 = ax0.imshow(zero_phase_bead_image, cmap=plt.cm.gray,
                      interpolation='nearest', vmin=-2300)
    ax0.axis('off')
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right",size="1%",pad=0.25)
    plt.colorbar(cax0, cax = cax)
    ax0.set_title('Phase contrast image of bead',fontsize=30)
    ax0.text(
        12, 14, r'$\Delta n\approx +0.05$',
        fontsize=38, color='white', fontweight='bold')
    ax0.text(
        53, 14, r'$\Delta n\approx 0$',
        fontsize=38, color='white', fontweight='bold')
    ax0.text(
        79, 14, r'$\Delta n\approx -0.01$',
        fontsize=38, color='white', fontweight='bold')


    cax1 = ax1.imshow(zero_phase_STE_image, cmap=plt.cm.gray,
                      interpolation='nearest')
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right",size="1%",pad=0.25)
    plt.colorbar(cax1, cax = cax)
    ax1.text(
        12, 14, r'$\Delta n\approx +0.05$',
        fontsize=38, color='white', fontweight='bold')
    ax1.text(
        53, 14, r'$\Delta n\approx 0$',
        fontsize=38, color='white', fontweight='bold')
    ax1.text(
        79, 14, r'$\Delta n\approx -0.01$',
        fontsize=38, color='white', fontweight='bold')
    ax1.set_title('Change due to excitation',fontsize=30)
    ax1.axis('off')
    plt.savefig('./../images/figure_7/STE_crimson_bead_zero_phase.svg',
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

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

main()
