# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:53:01 2026

@author: JoanaCatarino
"""


import pandas as pd


filedlc= 'L:/dmclab/Joana/PFC-Str_behavior_project/Analysis/out_dlc/999770/day2/999770_day22025-11-12T12_38_33DLC_Resnet50_test_manul_wholevideoMar12shuffle1_snapshot_040.h5'


dlc= pd.read_hdf(filedlc)

dlc.columns= dlc.columns.droplevel(0)



# filedlc_csv= 'L:/dmclab/Joana/PFC-Str_behavior_project/Analysis/out_dlc/999770/day2/999770_day22025-11-12T12_38_33DLC_Resnet50_test_manul_wholevideoMar12shuffle1_snapshot_040.csv'

# dlc= pd.read_csv(filedlc_csv)

# dlc.columns= dlc.columns.droplevel(0)







