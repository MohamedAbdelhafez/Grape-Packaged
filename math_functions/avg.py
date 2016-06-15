import numpy as np

def get_avg(av,average_path,iterations):
    #Averaging trajectory gradients
    a1 = iterations * average_path
    a2 = np.add(av,a1)
    return np.divide(a2,(iterations + 1))