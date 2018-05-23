import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import np_tif
from stack_registration import bucket

def main():
    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_5'):
        os.mkdir('./../images/figure_5')

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
    control_images = data[:, 0, :, :] # red before green
    max_delay_images = data[:, 2, :, :] # red after green
    phase_contrast_images = data[:, 0, :, :] # red before green
    
    # from the image where red/green are simultaneous, subtract the
    # average of the max and min delay images
    STE_stack = zero_delay_images - control_images
    control_stack = control_images - control_images
    max_delay_stack = max_delay_images - control_images

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
    max_delay_stack = gaussian_filter(max_delay_stack, sigma=(0, sigma, sigma))
    control_stack = gaussian_filter(control_stack, sigma=(0, sigma, sigma))
    phase_stack = gaussian_filter(phase_stack, sigma=(0, sigma, sigma))

    # crop images to center bead and fit into figure
    top = 0
    bot = 122
    left = 109
    right = 361
    phase_cropped = phase_stack[:, top:bot, left:right]
    STE_cropped = STE_stack[:, top:bot, left:right]
    max_delay_cropped = max_delay_stack[:, top:bot, left:right]
    control_cropped = control_stack[:, top:bot, left:right]

    # Our pixels are tiny (8.7 nm/pixel) to give large dynamic range.
    # This is not great for viewing, because fluctuations can swamp the
    # signal. This step bins the pixels into a more typical size.
    bucket_width = 8 # bucket width in pixels
    phase_cropped = bucket(
        phase_cropped, (1, bucket_width, bucket_width)) / bucket_width**2
    STE_cropped = bucket(
        STE_cropped, (1, bucket_width, bucket_width)) / bucket_width**2
    max_delay_cropped = bucket(
        max_delay_cropped, (1, bucket_width, bucket_width)) / bucket_width**2
    control_cropped = bucket(
        control_cropped, (1, bucket_width, bucket_width)) / bucket_width**2

    # display images from the two phase plate angles that maximize bead
    # contrast (+/- contrast)
    zero_phase_angle = 8
    pi_phase_angle = 0
    zero_phase_bead_image = phase_cropped[zero_phase_angle, :, :]
    pi_phase_bead_image = phase_cropped[pi_phase_angle, :, :]
    zero_phase_STE_image = STE_cropped[zero_phase_angle, :, :]
    pi_phase_STE_image = STE_cropped[pi_phase_angle, :, :]
    zero_phase_max_delay_image = max_delay_cropped[zero_phase_angle, :, :]
    pi_phase_max_delay_image = max_delay_cropped[pi_phase_angle, :, :]
    zero_phase_control_image = control_cropped[zero_phase_angle, :, :]
    pi_phase_control_image = control_cropped[pi_phase_angle, :, :]
    
    # start plotting all the images

    # combine images into single array
    diff_imgs = np.concatenate((
        pi_phase_control_image.reshape(
            1, pi_phase_control_image.shape[0], pi_phase_control_image.shape[1]),
        pi_phase_STE_image.reshape(
            1, pi_phase_STE_image.shape[0], pi_phase_STE_image.shape[1]),
        pi_phase_max_delay_image.reshape(
            1, pi_phase_max_delay_image.shape[0], pi_phase_max_delay_image.shape[1])
        ), axis=0)

    # get max and min values to unify the color scale
    diff_max = np.amax(diff_imgs)
    diff_min = - diff_max #np.amin(diff_imgs)
    phase_max = np.amax(pi_phase_bead_image) * 3
    phase_min = np.amin(pi_phase_bead_image) * 3
    print(phase_max, phase_min)
    # scale bars
    diff_imgs[:, -2:-1, 1:6] = diff_min
    pi_phase_bead_image[-2:-1, 1:6] = phase_min

    # make delay scan single image
    spacing = 4 # pixels between images
    full_scan_img = np.zeros((
        diff_imgs.shape[1],
        diff_imgs.shape[2] * 3 + spacing * 2))
    full_scan_img = full_scan_img + diff_max # make sure spacing is white
    for i in range(diff_imgs.shape[0]):
        print(i)
        x_position = (diff_imgs.shape[2] + spacing) * i
        full_scan_img[:, x_position:x_position + diff_imgs.shape[2]] = (
            diff_imgs[i, :, :])
    fig = plt.figure()
    plt.imshow(full_scan_img, cmap=plt.cm.gray,
               interpolation='nearest', vmax=diff_max, vmin=diff_min)
    plt.axis('off')
    plt.savefig('./../images/figure_A5/phase_STE_delay_scan.svg',
                bbox_inches='tight')
    plt.show()
    fig = plt.figure()
    plt.imshow(pi_phase_bead_image, cmap=plt.cm.gray,
               interpolation='nearest')
    plt.axis('off')
    plt.savefig('./../images/figure_A5/phase_image.svg',
                bbox_inches='tight')
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
