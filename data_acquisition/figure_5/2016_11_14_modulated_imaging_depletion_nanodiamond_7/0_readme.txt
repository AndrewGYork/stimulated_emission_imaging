*************************************
These files helped me take the data
*************************************

(takes fluorescence depletion images with various red/green powers)
modulated_imaging_reps_separate.py


(dependencies)
arrayimage.py
image_data_pipeline.py
ni.py
np_tif.py
pco.py
SC2_Cam.dll
sc2_cl_me4.dll
stack_registration.py

(not a dependency, but used along with Camware64 to position sample at focus)
run_piezo.py
thorlabs.py


*************************************
This is the resulting data
*************************************
(produced by modulated_imaging_one_z_power_scan.py)
STE_darkfield_power_delay_scan_0.tif
STE_darkfield_power_delay_scan_1.tif
STE_darkfield_power_delay_scan_2.tif
STE_darkfield_power_delay_scan_3.tif
STE_darkfield_power_delay_scan_4.tif
STE_darkfield_power_delay_scan_5.tif
STE_darkfield_power_delay_scan_6.tif
STE_darkfield_power_delay_scan_7.tif
STE_darkfield_power_delay_scan_8.tif
STE_darkfield_power_delay_scan_9.tif


*************************************
Pre-process data into a form readable by figure generation code
*************************************
register_rep_average_test_point_data.py

(resulting pre-processed data)
data_point_bg.tif
data_point_signal.tif
representative_image_avg.tif