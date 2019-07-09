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

    z_voltage = '_46000mV'

    # 1. brightness-correct each delay scan by using the entire image
    # brightness at max/min delay as an estimate of the red laser
    # brightness over that particular delay scan
    # 2. repetition average the images for each delay
    # 3. register the repetition averaged "green off" image to the
    # corresponding "green on" image

    num_delays = 5
    num_reps = 200
    width = 380
    height = 128
    less_rows = 3

    filename = (
        './../../stimulated_emission_imaging-data' +
        '/2016_11_02_STE_z_stack_darkfield' +
        '/STE_darkfield_113_green_1500mW_red_300mW' +
        z_voltage + '_many_delays.tif')
    filename_ctrl = (
        './../../stimulated_emission_imaging-data' +
        '/2016_11_02_STE_z_stack_darkfield' +
        '/STE_darkfield_113_green_0mW_red_300mW' +
        z_voltage + '_many_delays.tif')
    data = np_tif.tif_to_array(filename).astype(np.float64)
    data_ctrl = np_tif.tif_to_array(filename_ctrl).astype(np.float64)
    # reshape data arrays: 0th axis is rep number, 1st is delay number
    data = data.reshape(num_reps, num_delays, height, width)
    data_ctrl = data_ctrl.reshape(num_reps, num_delays, height, width)
    # crop to remove overexposed rows
    data = data[:, :, less_rows:height - less_rows, :]
    data_ctrl = data_ctrl[:, :, less_rows:height - less_rows, :]
    # get slice for reference brightness
    ref_slice_1 = data[:, 0, :, :].mean(axis=0)
    ref_slice_2 = data[:, 4, :, :].mean(axis=0)
    ref_slice = (ref_slice_1 + ref_slice_2) / 2
    # red beam brightness correction
    red_avg_brightness = ref_slice.mean(axis=1).mean(axis=0)
    # for green on data
    local_laser_brightness = ((
        data[:, 0, :, :] + data[:, 4, :, :])/2
                              ).mean(axis=2).mean(axis=1)
    local_calibration_factor = red_avg_brightness / local_laser_brightness
    local_calibration_factor = local_calibration_factor.reshape(
        num_reps, 1, 1, 1)
    data = data * local_calibration_factor
    # for green off data
    local_laser_brightness = ((
        data_ctrl[:, 0, :, :] + data_ctrl[:, 4, :, :])/2
                              ).mean(axis=2).mean(axis=1)
    local_calibration_factor = red_avg_brightness / local_laser_brightness
    local_calibration_factor = local_calibration_factor.reshape(
        num_reps, 1, 1, 1)
    data_ctrl = data_ctrl * local_calibration_factor

    # repetition average both image sets
    data_rep_avg = data.mean(axis=0)
    data_ctrl_rep_avg = data_ctrl.mean(axis=0)

    # registration shift control data to match "green on" data
    align_to_this_slice = data_rep_avg[0, :, :]
    print("Computing registration shifts (no green)...")
    shifts = stack_registration(
        data_ctrl_rep_avg,
        align_to_this_slice=align_to_this_slice,
        refinement='integer',
        register_in_place=True,
        background_subtraction='edge_mean')
    print(shifts)
    print("... done computing shifts.")

    # replace arrays with repetition averaged images
    data = data_rep_avg
    data_ctrl = data_ctrl_rep_avg

    # from the image where red/green are simultaneous, subtract the
    # average of images taken when the delay magnitude is greatest
    diff_imgs = data - data[0, :, :].reshape(
        1, data.shape[1], data.shape[2])
    diff_imgs_ctrl = data_ctrl - data_ctrl[0, :, :].reshape(
        1, data_ctrl.shape[1], data_ctrl.shape[2])
    diff_imgs = diff_imgs - diff_imgs_ctrl #subtract crosstalk
    darkfield_img = data[0, :, :]


    # plot darkfield and stim emission signal
    top = 1
    bot = 116
    left = 104-20
    right = 219+20
    diff_imgs_cropped = diff_imgs[:, top:bot, left:right]
    darkfield_img_cropped = darkfield_img[top:bot, left:right]

    # Our pixels are tiny (8.7 nm/pixel) to give large dynamic range.
    # This is not great for viewing, because fluctuations can swamp the
    # signal. This step bins the pixels into a more typical size.
    bucket_width = 8 # bucket width in pixels
    diff_imgs_cropped = bucket(
        diff_imgs_cropped, (1, bucket_width, bucket_width)) / bucket_width ** 2
    darkfield_img_cropped = bucket(
        darkfield_img_cropped, (bucket_width, bucket_width)) / bucket_width ** 2

    # get max and min pixel values for plotting
    max_diff = np.amax(diff_imgs_cropped)
    min_diff = np.amin(diff_imgs_cropped)
    # make scale bars
    diff_imgs_cropped[:, -2:-1, 1:5] = min_diff
    darkfield_img_cropped[-2:-1, 1:5] = np.amax(darkfield_img_cropped)

    # make delay scan single image
    spacing = 4 #pixels between images
    full_scan_img = np.zeros((
        diff_imgs_cropped.shape[1],
        diff_imgs_cropped.shape[2] * 5 + spacing * 4))
    full_scan_img = full_scan_img + max_diff # make sure spacing is white

    for i in range(diff_imgs_cropped.shape[0]):
        print(i)
        x_position = (diff_imgs_cropped.shape[2] + spacing) * i
        full_scan_img[:, x_position:x_position + diff_imgs_cropped.shape[2]] = (
            diff_imgs_cropped[i, :, :])
    fig = plt.figure()
    plt.imshow(full_scan_img, cmap=plt.cm.gray,
               interpolation='nearest', vmax=max_diff, vmin=min_diff)
    plt.axis('off')
    plt.savefig('./../images/figure_A4/darkfield_STE_delay_scan.svg',
                bbox_inches='tight')
    plt.show()
    fig = plt.figure()
    plt.imshow(darkfield_img_cropped, cmap=plt.cm.gray,
               interpolation='nearest')
    plt.axis('off')
    plt.savefig('./../images/figure_A4/darkfield_image.svg',
                bbox_inches='tight')
    plt.show()

    return None



main()
