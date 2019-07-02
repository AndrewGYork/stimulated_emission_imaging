import numpy as np
from scipy.ndimage import gaussian_filter
from stack_registration import stack_registration, apply_registration_shifts
import np_tif

# each raw data stack has a full red and green power scan with red
# varying slowly and green varying more quickly and green/red pulse
# delay varying the quickest (5 delays, middle delay is 0 delay)

num_reps = 10 # number of times a power/delay stack was taken
num_red_powers = 7
num_green_powers = 13
num_delays = 5
image_h = 128
image_w = 380

# allocate hyperstack
data_rep = np.zeros((
    num_reps,
    num_red_powers,
    num_green_powers,
    num_delays,
    image_h,
    image_w,
    ), dtype=np.float64)

# populate hyperstack
for rep_num in range(num_reps):
    filename = 'STE_darkfield_power_delay_scan_' + str(rep_num) + '.tif'
    print("Loading", filename)
    data_rep[rep_num,...] = np_tif.tif_to_array(filename
                                                ).reshape(data_rep.shape[1:])

stack_shape = (
    data_rep.shape[0]*data_rep.shape[1]*data_rep.shape[2]*data_rep.shape[3],
    data_rep.shape[4],
    data_rep.shape[5],
    )

print("Smoothing in time...")
register_me = gaussian_filter(data_rep.reshape(stack_shape),
                              sigma=(num_green_powers*num_delays/2, 0, 0))

print("Computing registration shifts...")
shifts = stack_registration(
    register_me,
    align_to_this_slice=1508,
    register_in_place=False,
    background_subtraction='edge_mean')

print("Applying shifts to raw data...")
apply_registration_shifts(data_rep.reshape(stack_shape),
                          registration_shifts=shifts,
                          edges='sloppy')

# crop data (rectangle around bright fluorescent lobe)
data_rep_cropped = data_rep[:, :, :, :, 49:97, 147:195]
data_rep_signal = data_rep_cropped.mean(axis=5).mean(axis=4)

# get data background
data_rep_bg = data_rep[:, :, :, :, 20:30, 20:30]
data_rep_bg = data_rep_bg.mean(axis=5).mean(axis=4)

# compute repetition average of data after image registration
data_avg = data_rep.mean(axis=0)
##representative_image_avg = (data_avg[0,-1,-1,16:128,108:238] -
##                            data_rep_bg[:,0,-1,-1].mean(axis=0))
rep_image_single_shot = (data_rep[0,-1,-1,-1,16:128,108:238] -
                         data_rep_bg[0,-1,-1,-1])

data_avg_tif_shape = (
    data_avg.shape[0] * data_avg.shape[1] * data_avg.shape[2],
    data_avg.shape[3],
    data_avg.shape[4])

point_data_tif_shape = (
    data_rep_signal.shape[0] * data_rep_signal.shape[1],
    data_rep_signal.shape[2],
    data_rep_signal.shape[3])

print("Saving...")
np_tif.array_to_tif(
    data_rep_signal.reshape(point_data_tif_shape),'data_point_signal.tif')
np_tif.array_to_tif(
    data_rep_bg.reshape(point_data_tif_shape),'data_point_bg.tif')
##np_tif.array_to_tif(
##    representative_image_avg,'representative_image_avg.tif')
np_tif.array_to_tif(
    rep_image_single_shot,'rep_image_single_shot.tif')
print("... done.")
