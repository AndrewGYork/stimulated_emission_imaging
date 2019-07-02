*************************************
These files helped me take the data
*************************************

(takes fluorescence depletion images to locate bead)
modulated_imaging_one_z_STE_depletion.py

(takes stimulated emission phase contrast images of bead)
modulated_imaging_one_z_STE_phase.py

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
(produced by modulated_imaging_one_z_STE_depletion.py)
STE_depletion_angle_14_green_970mW_red_0mW.tif
STE_depletion_angle_14_green_970mW_red_240mW.tif
voltages_green_970mW_red_0mW_depletion.pickle
voltages_green_970mW_red_240mW_depletion.pickle

(produced by modulated_imaging_one_z_STE_phase.py)
STE_phase_angle_0_green_0mW_red_240mW.tif
STE_phase_angle_0_green_970mW_red_240mW.tif
STE_phase_angle_10_green_0mW_red_240mW.tif
STE_phase_angle_10_green_970mW_red_240mW.tif
STE_phase_angle_11_green_0mW_red_240mW.tif
STE_phase_angle_11_green_970mW_red_240mW.tif
STE_phase_angle_12_green_0mW_red_240mW.tif
STE_phase_angle_12_green_970mW_red_240mW.tif
STE_phase_angle_13_green_0mW_red_240mW.tif
STE_phase_angle_13_green_970mW_red_240mW.tif
STE_phase_angle_14_green_0mW_red_240mW.tif
STE_phase_angle_14_green_970mW_red_240mW.tif
STE_phase_angle_1_green_0mW_red_240mW.tif
STE_phase_angle_1_green_970mW_red_240mW.tif
STE_phase_angle_2_green_0mW_red_240mW.tif
STE_phase_angle_2_green_970mW_red_240mW.tif
STE_phase_angle_3_green_0mW_red_240mW.tif
STE_phase_angle_3_green_970mW_red_240mW.tif
STE_phase_angle_4_green_0mW_red_240mW.tif
STE_phase_angle_4_green_970mW_red_240mW.tif
STE_phase_angle_5_green_0mW_red_240mW.tif
STE_phase_angle_5_green_970mW_red_240mW.tif
STE_phase_angle_6_green_0mW_red_240mW.tif
STE_phase_angle_6_green_970mW_red_240mW.tif
STE_phase_angle_7_green_0mW_red_240mW.tif
STE_phase_angle_7_green_970mW_red_240mW.tif
STE_phase_angle_8_green_0mW_red_240mW.tif
STE_phase_angle_8_green_970mW_red_240mW.tif
STE_phase_angle_9_green_0mW_red_240mW.tif
STE_phase_angle_9_green_970mW_red_240mW.tif
voltages_green_0mW_red_240mW_phase.pickle
voltages_green_970mW_red_240mW_phase.pickle


*************************************
Pre-process data into a form readable by figure generation code
*************************************
time_average_expt_and_control_single_shot.py

(resulting pre-processed data)
dataset_green_0mW_single_shot.tif
dataset_green_970mW_single_shot.tif