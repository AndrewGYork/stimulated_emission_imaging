import time
import numpy as np
import image_data_pipeline
import ni
##import thorlabs
from pco import pco_edge_camera_child_process
import pickle



def main():
        
    # This incantation is forced on us so the IDP won't print everything twice:
    import logging
    import multiprocessing as mp
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)

    # Set parameters for IDP (Image Data Pipeline)
    set_num_buffers = 7
    image_height_pixels = 128
    image_width_pixels = 380

    # Set parameters for DAQ (analog out card)
    num_daq_channels = 3
    daq_rate = 8e5

    ##############################################################
    # Set exposure parameters for camera and laser illumination: #
    ##############################################################
    
    green_AOM_mV = 300#calibrated
    green_power = '130mW'
    red_AOM_mV = 269#calibrated
    red_power = '230mW'

    # Set laser pulse duration VERY SHORT
    green_pulse_duration_pixels = 1
    red_pulse_duration_pixels = 1

    # Set green pulse train repetition time short enough to
    # thermally stabilize the sample
    green_rep_time_us = 600
    green_rep_time_pixels = int(np.ceil(
        green_rep_time_us * 1e-6 * daq_rate))

    # how many red laser shots in an exposure?
    pulses_per_exposure = 8
    # you don't want red light leaking into next exposure so set this to
    # 1 if you're imaging 720 nm.
    # set to zero if you're looking for depletion, because you need
    # every green pulse matched with a red for that measurement
    less_red_pulses = 0
    
    desired_effective_exposure_time_pixels = (green_rep_time_pixels *
                                              pulses_per_exposure)
    assert desired_effective_exposure_time_pixels > 0

    #define red/green pulse delays
    red_start_pixel_array = np.array([-2, 0, 2])
    num_delays = red_start_pixel_array.shape[0]
    print('Red/green delay (us) =', red_start_pixel_array / daq_rate * 1e6)
    # number of exposures should be the first dimension of the idp buffer
    num_delay_scan_repetitions = 3001 # first delay scan is pre-frames
    num_exposures = num_delays * num_delay_scan_repetitions

    # actual roll time is 640 us, which should be a multiple of
    # green_rep_time_us, but may not always be
    # this only works for the current field of view height 128 pixels
    # 10 us per line, rolling is symmetrical around middle of chip
    rolling_time_us = 640 #experimentally determined for this field of view
    rolling_time_pixels = int(np.ceil(
        rolling_time_us * 1e-6 * daq_rate))
    extra_time_after_roll_pixels = (green_rep_time_pixels -
                                    rolling_time_pixels %
                                    green_rep_time_pixels)
    effective_exposure_time_pixels = (extra_time_after_roll_pixels +
                                      desired_effective_exposure_time_pixels)
    # reminder: negative delay values (red before green) are only valid if the
    # camera roll finishes before the red pulse gets there
    assert extra_time_after_roll_pixels > -min(red_start_pixel_array)
    set_exposure_time_pixels = (rolling_time_pixels +
                                effective_exposure_time_pixels)
    # set exposure time must be an integer multiple of green rep time
    assert (set_exposure_time_pixels % green_rep_time_pixels) == 0
    set_exposure_time_us = int(np.ceil(
        set_exposure_time_pixels / daq_rate * 1e6))

    # Initialize the IDP:
    idp = image_data_pipeline.Image_Data_Pipeline(
        num_buffers=set_num_buffers,
        buffer_shape=(num_exposures, image_height_pixels, image_width_pixels),
        camera_child_process=pco_edge_camera_child_process)
    assert idp.buffer_shape[0] == num_exposures
    
    # Initialize the DAQ:
    daq = ni.PCI_6733(
        num_channels=num_daq_channels,
        rate=daq_rate,
        verbose=True)
    assert daq.rate == daq_rate

    try:
        # Apply camera settings:
        idp.display.set_intensity_scaling('median_filter_autoscale')
        idp.apply_camera_settings(
            trigger='external_trigger',
            exposure_time_microseconds = set_exposure_time_us,
            region_of_interest  ={'bottom': 1088,
                                  'top': 961,
                                  'left': 841,
                                  'right': 1220},
            preframes=0)
        # UNCOMMON COMMAND: the daq voltage string can get very long, so
        # Andy wrote a new part of pco.py that adjusts the set timeout
        # for waiting for the FIRST camera trigger (Oct 4, 2016)
        idp.camera.commands.send(('set_first_trigger_timeout_seconds',
                                  {'first_trigger_timeout_seconds': 3}))
        assert idp.camera.commands.recv() == 3 # clear command queue
        # Figure out some basic timing information: This is what the
        # camera thinks it's doing. Is it what we want it to do?
        exposure_time_us = idp.camera.get_setting('exposure_time_microseconds')
        print('I want exposure time to be (us)',set_exposure_time_us)
        print('Exposure time actually is (us)',exposure_time_us)
        assert exposure_time_us == set_exposure_time_us
        rolling_time_us = idp.camera.get_setting('rolling_time_microseconds')
        rolling_time_jitter_us = 15 #experimentally measured and also in spec
        rolling_time_us += rolling_time_jitter_us
        pulse_tail_us = 25 #experimentally measured response of buffer amp and AOM
        print("\nCamera exposure time:", exposure_time_us, "(us)\n")
        print("\nCamera rolling time:", rolling_time_us, "(us)\n")
        effective_exposure_us = exposure_time_us - rolling_time_us
        print("\nCamera effective exposure:", effective_exposure_us, "(us)\n")

        # Calculate DAQ voltages

        # Set voltages to play on analog out card
        green_voltage = green_AOM_mV/1000
        red_voltage = red_AOM_mV/1000
        trig_voltage = 3

        # time between exposures must be greater than camera trigger
        # jitter and a multiple of the green rep time
        # trigger jitter is about 10 us
        time_between_exposures_pixels = 2 * green_rep_time_pixels
        camera_rep_time_pixels = (set_exposure_time_pixels +
                                  time_between_exposures_pixels)
        camera_rep_time_us = camera_rep_time_pixels / daq_rate * 1e6

        voltages = np.zeros((camera_rep_time_pixels * num_exposures,
                             num_daq_channels))

        # camera trigger duration should be 3us or greater
        trigger_duration_us = 3
        trigger_duration_pixels = int(np.ceil(
            trigger_duration_us / 1e6 * daq_rate))

        # loop used to define camera trigger and red laser pulse
        # voltages
        for which_exposure in range(num_exposures):
            cursor = which_exposure * camera_rep_time_pixels

            # Camera triggers:
            voltages[cursor:cursor + trigger_duration_pixels, 0] = (
                trig_voltage)

            # Red laser pulses
            red_start_pixel = (
                red_start_pixel_array[which_exposure % num_delays])
            red_series_start = (cursor +
                                rolling_time_pixels +
                                extra_time_after_roll_pixels +
                                red_start_pixel)
            red_chunk = np.zeros(green_rep_time_pixels)
            red_chunk[0:red_pulse_duration_pixels] = red_voltage

            red_exposure_array = np.tile(red_chunk, (
                pulses_per_exposure - less_red_pulses))
            # only send red voltages if it's the fourth exposure or later
            if which_exposure > 2:
                voltages[
                    red_series_start:(
                        red_series_start + red_exposure_array.shape[0]),
                    2] = red_exposure_array

            # Green laser pulses
            green_start_pixel = 0
            green_series_start = (cursor +
                                rolling_time_pixels +
                                extra_time_after_roll_pixels +
                                green_start_pixel)
            green_chunk = np.zeros(green_rep_time_pixels)
            green_chunk[0:green_pulse_duration_pixels] = green_voltage

            green_exposure_array = np.tile(green_chunk, (
                pulses_per_exposure - less_red_pulses))
            # only send green voltages if it's the fourth exposure or later
            if which_exposure > 2:
                voltages[
                    green_series_start:(
                        green_series_start + green_exposure_array.shape[0]),
                    1] = green_exposure_array

##        # save voltages that will be sent to daq
##        with open('voltages_green_' + green_power +
##                  '_red_' + red_power +
##                  '_phase.pickle', 'wb') as f:
##            pickle.dump(voltages, f)

        time_course_letters = ['a', 'b', 'c', 'd', 'e']
        num_time_courses = len(time_course_letters)
        sleep_time = 0 # seconds
        for time_course in range(num_time_courses):
            time.sleep(sleep_time)
            # Put it all together
            idp.load_permission_slips(
                num_slips=1,
                file_saving_info=[
                    {'filename': (
                        'STE_depletion_' +
                        'green_' + green_power +
                        '_red_' + red_power +
                        '_' + time_course_letters[time_course] +
                        '.tif'),
                     'channels': num_delays,
                     'slices': num_delay_scan_repetitions,
                     }])
            daq.play_voltages(voltages, block=True)
    finally:
        # Shut everything down. This can be important!
        daq.close()
        idp.close()

if __name__ == '__main__':
    main()
