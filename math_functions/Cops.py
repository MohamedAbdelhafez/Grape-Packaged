import numpy as np
from math_functions.Get_state_index import Get_State_index

# A function that builds all possible jump operators, i.e. noise/environment errors
def Build_c_ops(qubit_state_num,mode_state_num,v,dressed):

        T1_ge = 10.0
        T1_ef = 180.0
       
        
            
        repeat=mode_state_num**2
        if repeat == 0:
            repeat =1
        ge=np.identity(repeat*qubit_state_num)
        for ii in range (repeat):
            ge=ge+ np.outer(v[:,Get_State_index(ii,dressed)],np.conjugate(v[:,Get_State_index((ii+repeat),dressed)]))
            ge=ge-np.outer(v[:,Get_State_index((ii+repeat),dressed)],np.conjugate(v[:,Get_State_index((ii+repeat),dressed)]))
           
            
        
        
        ef=np.identity(repeat*qubit_state_num)
        for ii in range (repeat):
            ef=ef+ np.outer(v[:,Get_State_index(ii+2*repeat,dressed)],np.conjugate(v[:,Get_State_index(ii+repeat,dressed)]))
            ef=ef- np.outer(v[:,Get_State_index(ii+repeat,dressed)],np.conjugate(v[:,Get_State_index(ii+repeat,dressed)]))
        
        ge=ge*np.sqrt(1/T1_ge) #decay from excited to ground qubit states
        ef=ef*np.sqrt(1/T1_ef) #jump from excited to second excited state (f)
        

        c_ops=[]
      
        c_ops.append(ef)
        c_ops.append(ge)
        
        return c_ops
        