import thorlabs
import time



piezo = thorlabs.MDT694B_piezo_controller()
piezo_center_voltage = 34
piezo_voltage_step = 1
num_z_steps = 8

##for which_z_step in range(num_z_steps):
##
##    piezo_voltage = (piezo_center_voltage + piezo_voltage_step *
##                     (which_z_step - int(num_z_steps / 2)))
##    piezo.set_voltage(piezo_voltage)
##    time.sleep(2)

piezo.set_voltage(piezo_center_voltage)

##input("Press enter to continue...")
piezo.close()
