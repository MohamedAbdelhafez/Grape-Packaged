import numpy as np
from math_functions.Sort import Sort_ev

class Analysis:
    
    def __init__(self, sys_para, Evolution_states, tf_ops_weight,tf_xy_weight, tf_xy_nocos,  tf_inter_vecs):
        self.sys_para = sys_para
        self.tf_ops_weight = tf_ops_weight
        self.tf_xy_weight = tf_xy_weight
        self.tf_xy_nocos = tf_xy_nocos
        self.tf_inter_vecs = tf_inter_vecs
        self.Evolution_states = Evolution_states
    
    def RtoCMat(self,M):
        #convert back to complex matrices
        state_num = self.sys_para.state_num
        M_real = M[:state_num,:state_num]
        M_imag = M[state_num:2*state_num,:state_num]
        
        return (M_real+1j*M_imag)
        
        
    def get_ops_weight(self):        
        ops_weight = self.tf_ops_weight.eval()
        np.save("./data/GRAPE-ops-weight", np.array(ops_weight))
        return ops_weight
    
    def get_xy_weight(self):        
        xy_weight = self.tf_xy_weight.eval()
        np.save("./data/GRAPE-xy-weight", np.array(xy_weight))
        return xy_weight
    
    def get_nonmodulated_weight(self):        
        xy_nocos = self.tf_xy_nocos.eval()
        np.save("./data/GRAPE-nocos-weight", np.array(xy_nocos))
        return xy_nocos
    
    def get_final_vecs(self):
        
        state_num = self.sys_para.state_num
        final_vecs_mag_squared = []
        v_sorted=Sort_ev(self.sys_para.v_c,self.sys_para.dressed)
        for final_vec in self.Evolution_states:
            inter_vec = final_vec.eval()
            inter_vec_real = 0.5*(inter_vec[0:state_num,:]+inter_vec[state_num:2*state_num,:])
            inter_vec_imag = 0.5*(inter_vec[state_num:2*state_num,:] - inter_vec[0:state_num,:])
            inter_vec_c = inter_vec_real+1j*inter_vec_imag
            
            dressed_vec_c= np.dot(np.transpose(v_sorted),inter_vec_c)
            
            inter_vec_mag_squared = np.square(np.absolute(dressed_vec_c))
            inter_vecs_mag_squared.append(inter_vec_mag_squared)
        
        np.save("./data/GRAPE-final_vecs", np.array(inter_vecs_mag_squared))
        return inter_vecs_mag_squared
    
    def get_inter_vecs(self):
        state_num = self.sys_para.state_num
        inter_vecs_mag_squared = []
        v_sorted=Sort_ev(self.sys_para.v_c,self.sys_para.dressed)
        for tf_inter_vec in self.tf_inter_vecs:
            inter_vec = tf_inter_vec
            inter_vec_real = 0.5*(inter_vec[0:state_num]+inter_vec[state_num:2*state_num])
            inter_vec_imag = 0.5*(inter_vec[state_num:2*state_num] - inter_vec[0:state_num])
            inter_vec_c = inter_vec_real+1j*inter_vec_imag
         
            dressed_vec_c= np.dot(np.transpose(v_sorted),inter_vec_c)
           
            inter_vec_mag_squared = np.square(np.absolute(dressed_vec_c))
            inter_vecs_mag_squared.append(inter_vec_mag_squared)
   
        np.save("./data/GRAPE-inter_vecs", np.array(inter_vecs_mag_squared))
        return inter_vecs_mag_squared