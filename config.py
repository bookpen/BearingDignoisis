fault_map = {
   '1_1': 1,
   '1_2': 1,
   '1_3': 1,
   '1_4': 2,
   '1_5': 3,
   '2_1': 3,
   '2_2': 1,
   '2_3': 2,
   '2_4': 1,
   '2_5': 1,
   '3_1': 1,
   '3_2': 4,
   '3_3': 3,
   '3_4': 3,
   '3_5': 1
}
label = {"Normal":0,
         "Bearing1_1":1,
         "Bearing1_2":1,
         "Bearing1_3":1,
         "Bearing1_4":2,
         "Bearing1_5":3,
         "Bearing2_1":3,
         "Bearing2_2":1,
         "Bearing2_3":2,
         "Bearing2_4":1,
         "Bearing2_5":1,
         "Bearing3_1":1,
         "Bearing3_2":3,
         "Bearing3_3":3,
         "Bearing3_4":3,
         "Bearing3_5":1,
         }

# 0:正常
# 1:外圈
# 2:保持架
# 3:内圈
# 4:滚动体

CLS2IDX = {
   0 : 'None fault',
   1 : 'fault',
   2 : 'fault',
   3 : 'fault',
   4 : 'fault'
}