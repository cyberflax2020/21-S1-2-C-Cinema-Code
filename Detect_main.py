import age_gender_detect
import Facial_exp_main
import Body_exp_main

'''
Run this script and enter the source video path in the console to start the calculation. 
The parameter of the function is a Boolean value. 
When the value is True, the real-time image is output during calculation. 
When the value is False (default), no image is output to improve the running speed.
'''

age_gender_detect.age_gender_detect(False)
Facial_exp_main.facial_exp_detect(False)
Body_exp_main.action_detect(True)