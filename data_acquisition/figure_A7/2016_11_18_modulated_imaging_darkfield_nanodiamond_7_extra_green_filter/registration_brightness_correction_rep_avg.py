import numpy as np
from scipy.ndimage import gaussian_filter
from stack_registration import stack_registration, apply_registration_shifts
import np_tif


def main():
    # each raw data stack has a full red and green power scan with red
    # varying slowly and green varying more quickly and green/red pulse
    # delay varying the quickest (5 delays, middle delay is 0 delay)

    num_reps = 200 # number power scans taken
    num_red_powers = 7
    num_green_powers = 13
    num_delays = 5
    image_h = 128
    image_w = 380
    less_rows = 3 # top/bottom 3 rows may contain leakage from outside pixels
    top = less_rows
    bot = image_h - less_rows

    # assume no sample motion during a single power scan
    # allocate hyperstack to carry power/delay-averaged images for registration
    data_rep = np.zeros((
        num_reps,
        image_h - less_rows * 2,
        image_w,
        ), dtype=np.float64)
    data_rep_bg = np.zeros((
        num_reps,
        image_h - less_rows * 2,
        image_w,
        ), dtype=np.float64)

    # allocate array to carry a number corresponding to the average red
    # beam brightness for each red power
    red_avg_brightness = np.zeros((num_red_powers))

    # populate hyperstack from data
    for rep_num in range(num_reps):
        filename = 'STE_darkfield_power_delay_scan_' + str(rep_num) + '.tif'
        print("Loading", filename)
        imported_power_scan = np_tif.tif_to_array(
            filename).astype(np.float64)[:, top:bot, :]
        red_avg_brightness += get_bg_level(
            imported_power_scan.reshape(
                num_red_powers,
                num_green_powers,
                num_delays,
                image_h - less_rows * 2,
                image_w).mean(axis=1).mean(axis=1)
            ) / (2 * num_reps)
        data_rep[rep_num, :, :] = imported_power_scan.mean(axis=0)
        filename_bg = (
            'STE_darkfield_power_delay_scan_' +
            str(rep_num) + '_green_blocked.tif')
        print("Loading", filename_bg)
        imported_power_scan_bg = np_tif.tif_to_array(
            filename_bg).astype(np.float64)[:, top:bot, :]
        red_avg_brightness += get_bg_level(
            imported_power_scan_bg.reshape(
                num_red_powers,
                num_green_powers,
                num_delays,
                image_h - less_rows * 2,
                image_w).mean(axis=1).mean(axis=1)
            ) / (2 * num_reps)
        data_rep_bg[rep_num, :, :] = imported_power_scan_bg.mean(axis=0)

    

    # reshape red_avg_brightness to add a dimension for multiplication
    # with a brightness array with dimensions num_red_powers X num_green
    # powers X num_delays
    red_avg_brightness = red_avg_brightness.reshape(num_red_powers, 1, 1)
    
    # pick image/slice for all stacks to align to
    representative_rep_num = 0
    align_slice = data_rep[representative_rep_num, :, :]

    # save pre-registered average data (all powers for each rep)
    np_tif.array_to_tif(data_rep,
                        'dataset_not_registered_power_avg.tif')
    np_tif.array_to_tif(data_rep_bg,
                        'dataset_green_blocked_not_registered_power_avg.tif')


    # compute registration shifts
    print("Computing registration shifts...")
    shifts = stack_registration(
        data_rep,
        align_to_this_slice=align_slice,
        refinement='integer',
        register_in_place=True,
        background_subtraction='edge_mean')
    print("Computing registration shifts (no green) ...")
    shifts_bg = stack_registration(
        data_rep_bg,
        align_to_this_slice=align_slice,
        refinement='integer',
        register_in_place=True,
        background_subtraction='edge_mean')

    # save registered average data (all powers for each rep) and shifts
    np_tif.array_to_tif(data_rep,
                        'dataset_registered_power_avg.tif')
    np_tif.array_to_tif(data_rep_bg,
                        'dataset_green_blocked_registered_power_avg.tif')
    np_tif.array_to_tif(shifts, 'shifts.tif')
    np_tif.array_to_tif(shifts_bg, 'shifts_bg.tif')


    # now apply shifts to raw data and compute space-averaged signal
    # and representative images

    # define box around main lobe for computing space-averaged signal
    rect_top = 44
    rect_bot = 102
    rect_left = 172
    rect_right = 228

    # initialize hyperstacks for signal (with/without green light)
    print('Applying shifts to raw data...')
    signal = np.zeros((
        num_reps,
        num_red_powers,
        num_green_powers,
        num_delays,
        ), dtype=np.float64)
    signal_bg = np.zeros((
        num_reps,
        num_red_powers,
        num_green_powers,
        num_delays,
        ), dtype=np.float64)
    data_hyper_shape = (
        num_red_powers, num_green_powers, num_delays, image_h, image_w)

    # get representative image cropping coordinates
    rep_top = 22
    rep_bot = 122
    rep_left = 136
    rep_right = 262

    # initialize representative images (with/without green light)
    darkfield_image = np.zeros((#num_reps,
        rep_bot-rep_top,
        rep_right-rep_left,
        ), dtype=np.float64)
    STE_image = np.zeros((#num_reps,
        rep_bot-rep_top,
        rep_right-rep_left,
        ), dtype=np.float64)
    darkfield_image_bg = np.zeros((#num_reps,
        rep_bot-rep_top,
        rep_right-rep_left,
        ), dtype=np.float64)
    STE_image_bg = np.zeros((#num_reps,
        rep_bot-rep_top,
        rep_right-rep_left,
        ), dtype=np.float64)

    # finally apply shifts and compute output data
    for rep_num in range(num_reps):
        filename = 'STE_darkfield_power_delay_scan_' + str(rep_num) + '.tif'
        data = np_tif.tif_to_array(
            filename).astype(np.float64)[:, top:bot, :]
        filename_bg = ('STE_darkfield_power_delay_scan_' + str(rep_num) +
                       '_green_blocked.tif')
        data_bg = np_tif.tif_to_array(filename_bg).astype(
            np.float64)[:, top:bot, :]
        print(filename)
        print(filename_bg)
        # apply registration shifts
        apply_registration_shifts(
            data,
            registration_shifts=[shifts[rep_num]]*data.shape[0],
            registration_type='nearest_integer',
            edges='sloppy')
        apply_registration_shifts(
            data_bg,
            registration_shifts=[shifts_bg[rep_num]]*data_bg.shape[0],
            registration_type='nearest_integer',
            edges='sloppy')
        # re-scale images to compensate for red beam brightness fluctuations
        # for regular data
        local_laser_brightness = get_bg_level(
            data.reshape(
                num_red_powers,
                num_green_powers,
                num_delays,
                data.shape[-2],
                data.shape[-1]))
        local_calibration_factor = red_avg_brightness / local_laser_brightness
        local_calibration_factor = local_calibration_factor.reshape(
            num_red_powers * num_green_powers * num_delays, 1, 1)
        data = data * local_calibration_factor
        # for green blocked data
        local_laser_brightness_bg = get_bg_level(
            data_bg.reshape(
                num_red_powers,
                num_green_powers,
                num_delays,
                data.shape[-2],
                data.shape[-1]))
        local_calibration_factor_bg = (
            red_avg_brightness / local_laser_brightness_bg)
        local_calibration_factor_bg = local_calibration_factor_bg.reshape(
            num_red_powers * num_green_powers * num_delays, 1, 1)
        data_bg = data_bg * local_calibration_factor_bg
        # draw rectangle around bright lobe and spatially average signal
        data_space_avg = data[:, rect_top:rect_bot,
                              rect_left:rect_right].mean(axis=2).mean(axis=1)
        data_bg_space_avg = data_bg[:, rect_top:rect_bot,
                                    rect_left:rect_right].mean(axis=2).mean(axis=1)
        # reshape 1D signal and place in output file
        signal[rep_num, :, :, :] = data_space_avg.reshape(
            num_red_powers, num_green_powers, num_delays)
        signal_bg[rep_num, :, :, :] = data_bg_space_avg.reshape(
            num_red_powers, num_green_powers, num_delays)
        # capture average images for max red/green power
        image_green_power = num_green_powers - 1
        image_red_power = num_red_powers - 1
        STE_image += data[
            -3, # Zero delay, max red power, max green power
            rep_top:rep_bot,
            rep_left:rep_right
            ]/num_reps
        darkfield_image += data[
            -1, # max red-green delay (2.5 us), max red power, max green power
            rep_top:rep_bot,
            rep_left:rep_right
            ]/num_reps/2 # one of two maximum absolute red/green delay values
        darkfield_image += data[
            -5, # min red-green delay (-2.5 us), max red power, max green power
            rep_top:rep_bot,
            rep_left:rep_right
            ]/num_reps/2 # one of two maximum absolute red/green delay values
        STE_image_bg += data_bg[
            -3, # Zero delay, max red power, max green power
            rep_top:rep_bot,
            rep_left:rep_right
            ]/num_reps
        darkfield_image_bg += data_bg[
            -1, # max red-green delay (2.5 us), max red power, max green power
            rep_top:rep_bot,
            rep_left:rep_right
            ]/num_reps/2 # one of two maximum absolute red/green delay values
        darkfield_image_bg += data_bg[
            -5, # min red-green delay (-2.5 us), max red power, max green power
            rep_top:rep_bot,
            rep_left:rep_right
            ]/num_reps/2 # one of two maximum absolute red/green delay values                        

    print('Done applying shifts')


    signal_tif_shape = (signal.shape[0] * signal.shape[1],
                        signal.shape[2],signal.shape[3])

    print("Saving...")
    np_tif.array_to_tif(signal.reshape(signal_tif_shape),
                        'signal_all_scaled.tif')
    np_tif.array_to_tif(signal_bg.reshape(signal_tif_shape),
                        'signal_green_blocked_all_scaled.tif')
    np_tif.array_to_tif(darkfield_image,
                        'darkfield_image_avg.tif')
    np_tif.array_to_tif(darkfield_image_bg,
                        'darkfield_image_bg_avg.tif')
    np_tif.array_to_tif(STE_image,
                        'STE_image_avg.tif')
    np_tif.array_to_tif(STE_image_bg,
                        'STE_image_bg_avg.tif')
    print("... done.")

    return None

def get_bg_level(data):
    num_regions = 2
    
    # region 1
    bg_up = 9
    bg_down = 112
    bg_left = 325#270#325
    bg_right = 366
    bg_level = data[..., bg_up:bg_down, bg_left:bg_right].mean(axis=(-2, -1))

    # region 2
    bg_up = 9
    bg_down = 112
    bg_left = 8
    bg_right = 64#130#64
    bg_level += data[..., bg_up:bg_down, bg_left:bg_right].mean(axis=(-2, -1))

    return(bg_level / num_regions)

main()
