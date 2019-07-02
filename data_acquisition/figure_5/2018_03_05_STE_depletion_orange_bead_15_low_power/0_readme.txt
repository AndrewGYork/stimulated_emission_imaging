*************************************
These files helped me take the data
*************************************

(takes fluorescence depletion images with various red/green powers)
modulated_imaging_one_z_power_scan.py


(dependencies)
arrayimage.py
image_data_pipeline.py
ni.py
np_tif.py
pco.py
SC2_Cam.dll
sc2_cl_me4.dll

(not a dependency, but used along with Camware64 to position sample at focus)
run_piezo.py
thorlabs.py

(check voltages sent by modulated_imaging_*.py to analog out card)
read_pickle_voltages.py


*************************************
This is the resulting data
*************************************
(produced by modulated_imaging_one_z_power_scan.py)
fluorescence_green_0mW_red_0mW_up.tif
fluorescence_green_0mW_red_230mW_up.tif
fluorescence_green_130mW_red_0mW_up.tif
fluorescence_green_130mW_red_230mW_up.tif
fluorescence_green_13mW_red_0mW_up.tif
fluorescence_green_13mW_red_230mW_up.tif
fluorescence_green_26mW_red_0mW_up.tif
fluorescence_green_26mW_red_230mW_up.tif
fluorescence_green_4mW_red_0mW_up.tif
fluorescence_green_4mW_red_230mW_up.tif
fluorescence_green_53mW_red_0mW_up.tif
fluorescence_green_53mW_red_230mW_up.tif
fluorescence_green_96mW_red_0mW_up.tif
fluorescence_green_96mW_red_230mW_up.tif
voltages_green_0mW_red_0mW_fluorescence.pickle
voltages_green_0mW_red_230mW_fluorescence.pickle
voltages_green_130mW_red_0mW_fluorescence.pickle
voltages_green_130mW_red_230mW_fluorescence.pickle
voltages_green_13mW_red_0mW_fluorescence.pickle
voltages_green_13mW_red_230mW_fluorescence.pickle
voltages_green_26mW_red_0mW_fluorescence.pickle
voltages_green_26mW_red_230mW_fluorescence.pickle
voltages_green_4mW_red_0mW_fluorescence.pickle
voltages_green_4mW_red_230mW_fluorescence.pickle
voltages_green_53mW_red_0mW_fluorescence.pickle
voltages_green_53mW_red_230mW_fluorescence.pickle
voltages_green_96mW_red_0mW_fluorescence.pickle
voltages_green_96mW_red_230mW_fluorescence.pickle


*************************************
Pre-process data into a form readable by figure generation code
*************************************
time_average_expt_and_control_power_up.py

(resulting pre-processed data)
dataset_green_all_powers_up.tif