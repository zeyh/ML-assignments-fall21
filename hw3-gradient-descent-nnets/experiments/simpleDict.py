
import numpy as np

regularizations = [1,0.1,0.01,0.001,0.0001]
currTestingHParam = regularizations
accuracy_dict = {}
for testing_val in currTestingHParam:
    accuracy_log = [1,2,3,4]
    accuracy_dict[testing_val] = accuracy_log
print(">>>>>>>",accuracy_dict)
for item in accuracy_dict.items():
    print("Hyperparameter Value: ",item[0])
    # print("Values: ",item[1])
    print("std: ", np.std(item[1]))
    print("mean: ", np.std(item[1]))
