import os
import numpy as np
import np_tif

# each raw data stack is a red/green delay scan (five different delays,
# middle one 0 delay) over many repetitions

def main():

    num_angles = 32
    angles = range(num_angles)
    num_reps_original = 1000
    num_delays = 5
    image_h = 128
    image_w = 380
    less_rows = 3


    # define repetitions with dust particles crossing field of view or
    # extreme red power fluctuations
    gr_on_remove = {
        1: [[432,435],[694,719],[860,865]],
        5: [[54,57],[505,508],[901,906]],
        10: [[214,218],[421,428]],
        14: [[356,360],[391,394],[711,713],[774,802]],
        18: [[208,210],[661,667],[989,992]],
        21: [[181,187]],
        22: [[63,70],[328,333],[440,451],[544,557],[897,902],[935,964]],
        24: [[287,306],[922,924]],
        25: [[69,73],[639,675],[880,898]],
        26: [[667,677]],
        27: [[9,16],[557,560],[664,669]],
        29: [[219,221],[366,369],[452,458],[871,875]],
        }
    gr_off_remove = {
        1: [[706,708]],
        5: [[219,222],[505,510],[553,557]],
        7: [[158,165],[213,220],[310,316],[493,497],[950,961]],
        12: [[173,176],[432,434],[914,922]],
        13: [[494,527]],
        14: [[983,987]],
        15: [[451,458],[698,715],[873,883]],
        16: [[171,178]],
        17: [[100,104],[323,327]],
        21: [[51,56],[293,295],[385,390],[858,864]],
        22: [[106,109],[279,285],[565,580],[829,834],[904,924]],
        }

    green_powers = [
        '_0mW',
        '_1500mW',
        ]

    remove_dict_list = [gr_off_remove, gr_on_remove]# same order as green_powers

    red_powers = [
        '_300mW',
        ]
    rd_pow = red_powers[0]

    data_mean = np.zeros((
        len(green_powers),
        num_angles,
        num_delays,
        image_h - 2 * less_rows,
        image_w,
        ))

    for gr_pow_num, gr_pow in enumerate(green_powers):
        remove_range_dict = remove_dict_list[gr_pow_num]
        for ang in angles:
            filename = (
                'STE_' +
                'darkfield_' + str(ang) +
                '_green' + gr_pow +
                '_red' + rd_pow +
                '_many_delays.tif')
            assert os.path.exists(filename)
            print("Loading", filename)
            data = np_tif.tif_to_array(filename).astype(np.float64)
            assert data.shape == (num_delays*num_reps_original, image_h, image_w)
            # Stack to hyperstack
            data = data.reshape(num_reps_original, num_delays, image_h, image_w)
            # crop data to remove over-exposed stuff
            data = data[:, :, less_rows:image_h - less_rows, :]

            # delete repetitions with dust particles crossing field of view
            if ang in remove_range_dict:
                remove_range_list = remove_range_dict[ang]
                for my_range in reversed(remove_range_list):
                    first = my_range[0]
                    last = my_range[1]
                    delete_length = last - first + 1
                    data = np.delete(data, first + np.arange(delete_length), 0)
            print(ang, data.shape)

            # Get the average pixel brightness in the background region of the
            # phase contrast data. We'll use it to account for laser intensity
            # fluctuations
            avg_laser_brightness = get_bg_level(data.mean(axis=(0, 1)))

            # scale all images to have the same background brightness. This
            # amounts to a correction of roughly 1% or less
            local_laser_brightness = get_bg_level(data)
            data = data * (
                avg_laser_brightness / local_laser_brightness).reshape(
                    data.shape[0], data.shape[1], 1, 1)

            # Average data over repetitions
            data = data.mean(axis=0)
            # Put data in file to be saved
            data_mean[gr_pow_num, ang, ...] = data

        # save data for a particular green power
        print("Saving...")
        tif_shape = (num_angles * num_delays, image_h - 2 * less_rows, image_w)
        np_tif.array_to_tif(
            data_mean[gr_pow_num, :, :, :, :].reshape(tif_shape),
            ('dataset_green' + gr_pow + '.tif'))
        print("Done saving.")

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
