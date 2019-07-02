Note: Original file path:
C:\Users\Admin\Desktop\stimulated_emission_imaging_data_store\2016_10_24_STE_darkfield_depletion_pulse_length_constant_energy_scan\nanodiamond_7_4x_long_pulse\modulated_imaging_darkfield_less_1_red_pulse
*************************************
These files helped me take the data
*************************************

(takes phase contrast images with fixed red/green powers and many green/red delays, run once for 1500 mW, once for 0 mW (green blocked control))
modulated_imaging_green_red_short_pulse_delay_green_power_scan.py

(dependencies)
arrayimage.py
image_data_pipeline.py
ni.py
np_tif.py
pco.py
SC2_Cam.dll
sc2_cl_me4.dll


(not used by modulated_imaging_one_z_STE_phase_thermal.py but used to determine proper axial range)
run_piezo.py
additionally depends on thorlabs.py as well

Also, read_pickle_voltages.py is used to read voltages file


*************************************
This is the resulting data
*************************************
STE_phase_angle_10_green_0mW_red_300mW.tif
STE_phase_angle_10_green_1350mW_red_300mW.tif
voltages_green_0mW_red_300mW_phase.pickle
voltages_green_1350mW_red_300mW_phase.pickle