Note: original file folder name/path:
C:\Users\Admin\Desktop\stimulated_emission_imaging_data_store\2016_10_31_STE_phase_scan_re-center_phase_plate\nanodiamond_7_phase_scan

*************************************
These files helped me take the data
*************************************

(takes phase contrast images with fixed red/green powers, run once for 1500 mW, once for 0 mW (green blocked control))
modulated_imaging_z_stack_all.py

(delete bad data files, crop top and bottom three rows (overexposed), do brightness correction and finally repetition average)
time_average_expt_and_control_cull_brightness_correct.py

(dependencies)
arrayimage.py
image_data_pipeline.py
ni.py
np_tif.py
pco.py
SC2_Cam.dll
sc2_cl_me4.dll
stack_registration.py (used in data pre-processing code)
thorlabs.py

(not used by modulated_imaging_z_stack_all.py but used to determine proper axial range)
run_piezo.py



*************************************
This is the resulting data (deleted voltages file because it's very big)
*************************************
(from modulated_imaging_green_red_short_pulse_delay_red_green_power_scan.py)
STE_darkfield_0_green_0mW_red_300mW_many_delays.tif
STE_darkfield_0_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_10_green_0mW_red_300mW_many_delays.tif
STE_darkfield_10_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_11_green_0mW_red_300mW_many_delays.tif
STE_darkfield_11_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_12_green_0mW_red_300mW_many_delays.tif
STE_darkfield_12_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_13_green_0mW_red_300mW_many_delays.tif
STE_darkfield_13_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_14_green_0mW_red_300mW_many_delays.tif
STE_darkfield_14_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_15_green_0mW_red_300mW_many_delays.tif
STE_darkfield_15_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_16_green_0mW_red_300mW_many_delays.tif
STE_darkfield_16_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_17_green_0mW_red_300mW_many_delays.tif
STE_darkfield_17_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_18_green_0mW_red_300mW_many_delays.tif
STE_darkfield_18_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_19_green_0mW_red_300mW_many_delays.tif
STE_darkfield_19_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_1_green_0mW_red_300mW_many_delays.tif
STE_darkfield_1_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_20_green_0mW_red_300mW_many_delays.tif
STE_darkfield_20_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_21_green_0mW_red_300mW_many_delays.tif
STE_darkfield_21_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_22_green_0mW_red_300mW_many_delays.tif
STE_darkfield_22_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_23_green_0mW_red_300mW_many_delays.tif
STE_darkfield_23_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_24_green_0mW_red_300mW_many_delays.tif
STE_darkfield_24_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_25_green_0mW_red_300mW_many_delays.tif
STE_darkfield_25_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_26_green_0mW_red_300mW_many_delays.tif
STE_darkfield_26_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_27_green_0mW_red_300mW_many_delays.tif
STE_darkfield_27_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_28_green_0mW_red_300mW_many_delays.tif
STE_darkfield_28_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_29_green_0mW_red_300mW_many_delays.tif
STE_darkfield_29_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_2_green_0mW_red_300mW_many_delays.tif
STE_darkfield_2_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_30_green_0mW_red_300mW_many_delays.tif
STE_darkfield_30_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_31_green_0mW_red_300mW_many_delays.tif
STE_darkfield_31_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_3_green_0mW_red_300mW_many_delays.tif
STE_darkfield_3_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_4_green_0mW_red_300mW_many_delays.tif
STE_darkfield_4_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_5_green_0mW_red_300mW_many_delays.tif
STE_darkfield_5_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_6_green_0mW_red_300mW_many_delays.tif
STE_darkfield_6_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_7_green_0mW_red_300mW_many_delays.tif
STE_darkfield_7_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_8_green_0mW_red_300mW_many_delays.tif
STE_darkfield_8_green_1500mW_red_300mW_many_delays.tif
STE_darkfield_9_green_0mW_red_300mW_many_delays.tif
STE_darkfield_9_green_1500mW_red_300mW_many_delays.tif

(from time_average_expt_and_control_cull_brightness_correct.py)
dataset_green_0mW.tif
dataset_green_1500mW.tif