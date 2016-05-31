import os

import numpy as np
import tensorflow as tf
from math_functions.c_to_r_mat import CtoRMat
from custom_kernels.gradients.matexp_grad import *

class TensorflowState:
    
    def __init__(self,sys_para):
        self.sys_para = sys_para
	user_ops_path = './custom_kernels/build'
	self.matrix_exp_module = tf.load_op_library(os.path.join(user_ops_path,'cuda_matexp.so'))
        
    def init_variables(self):
        self.tf_identity = tf.constant(self.sys_para.identity,dtype=tf.float32)
        self.tf_neg_i = tf.constant(CtoRMat(-1j*self.sys_para.identity_c),dtype=tf.float32)
        self.tf_one_minus_gaussian_evelop = tf.constant(self.sys_para.one_minus_gauss,dtype=tf.float32)
        
        
    def init_tf_vectors(self):
        self.tf_initial_vectors=[]
        for initial_vector in self.sys_para.initial_vectors:
            tf_initial_vector = tf.constant(initial_vector,dtype=tf.float32)
            self.tf_initial_vectors.append(tf_initial_vector)
    
    def init_tf_states(self):
        #tf initial and target states
        self.tf_initial_state = tf.constant(self.sys_para.initial_state,dtype=tf.float32)
        self.tf_target_state = tf.constant(self.sys_para.target_state,dtype=tf.float32)
        print "State initialized."
        
        
    def init_tf_ops(self):
        #flat operators for control Hamiltonian 
        i_array = np.eye(2*self.sys_para.state_num)
        op_matrix_I=i_array.tolist()
        self.I_flat = [item for sublist in op_matrix_I  for item in sublist]
        self.H0_flat = [item for sublist in self.sys_para.H0  for item in sublist]
        
        self.flat_ops = []
        for op in self.sys_para.ops:
            flat_op = [item for sublist in op for item in sublist]
            self.flat_ops.append(flat_op)
            
        print "Operators initialized."
        
    def get_j(self,l):
        dt=self.sys_para.dt
        Dt=self.sys_para.Dt
        jj=np.floor((l*dt-0.5*Dt)/Dt)
        return jj
    
    
            
    def transfer_fn(self,xy):
        
        indices=[]
        values=[]
        shape=[self.sys_para.steps,self.sys_para.control_steps]
        dt=self.sys_para.dt
        Dt=self.sys_para.Dt
    
    # Cubic Splines
        for ll in range (self.sys_para.steps):
            jj=self.get_j(ll)
            tao= ll*dt - jj*Dt - 0.5*Dt
            if jj >= 1:
                indices.append([int(ll),int(jj-1)])
                temp= -(tao/(2*Dt))*((tao/Dt)-1)**2
                values.append(temp)
                
            if jj >= 0:
                indices.append([int(ll),int(jj)])
                temp= 1+((3*tao**3)/(2*Dt**3))-((5*tao**2)/(2*Dt**2))
                values.append(temp)
                
            if jj+1 <= self.sys_para.control_steps-1:
                indices.append([int(ll),int(jj+1)])
                temp= ((tao)/(2*Dt))+((4*tao**2)/(2*Dt**2))-((3*tao**3)/(2*Dt**3))
                values.append(temp)
               
            if jj+2 <= self.sys_para.control_steps-1:
                indices.append([int(ll),int(jj+2)])
                temp= ((tao**3)/(2*Dt**3))-((tao**2)/(2*Dt**2))
                values.append(temp)
                
            
        T1=tf.SparseTensor(indices, values, shape)  
        T2=tf.sparse_reorder(T1)
        T=tf.sparse_tensor_to_dense(T2)
        temp1 = tf.matmul(T,tf.reshape(xy[0,:],[self.sys_para.control_steps,1]))
        temp2 = tf.matmul(T,tf.reshape(xy[1,:],[self.sys_para.control_steps,1]))
        xys=tf.concat(1,[temp1,temp2])
        return tf.transpose(xys)

        
    def init_tf_ops_weight(self):
        
        
        
        #tf weights of operators
        if not self.sys_para.initial_pulse:
            initial_guess = 0
            initial_xy_stddev = (0.1/np.sqrt(self.sys_para.control_steps))
            initial_z_stddev = (0.1/np.sqrt(self.sys_para.steps))
            self.xy_weight_base = tf.Variable(tf.truncated_normal([self.sys_para.ops_len,self.sys_para.control_steps],
                                                                   mean= initial_guess ,dtype=tf.float32,
                            stddev=initial_xy_stddev ),name="xy_weights")
            self.z_weight_base =  tf.Variable(tf.truncated_normal([1,self.sys_para.steps],
                                                                   mean= initial_guess ,dtype=tf.float32,
                            stddev=initial_z_stddev ),name="z_weights")
            self.xy_weight = tf.tanh(self.xy_weight_base)
            self.z_weight = tf.tanh(self.z_weight_base)
            if self.sys_para.Interpolation:
                self.xy_nocos = self.transfer_fn(self.xy_weight)
            else:
                self.xy_nocos = self.xy_weight


            if self.sys_para.Modulation:
                
                cosine= tf.cos(2*np.pi*self.sys_para.freq_ge*np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]))
                sine= tf.sin(2*np.pi*self.sys_para.freq_ge*np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]))
                temp1 = tf.mul(self.xy_nocos[0,:],tf.cast(cosine, tf.float32))
                temp2 = -tf.mul(self.xy_nocos[1,:],tf.cast(sine, tf.float32))
                self.xy_cos = tf.concat(0,[tf.reshape(temp1,[1,self.sys_para.steps]),tf.reshape(temp2,[1,self.sys_para.steps])],name="modulated")
                self.ops_weight = tf.concat(0,[self.xy_cos,self.z_weight],name="ops_weight")

            else:
                self.ops_weight = tf.concat(0,[self.xy_nocos,self.z_weight],name="ops_weight")
            
        else:
            if self.sys_para.prev_pulse:
                self.ops_weight_base = tf.Variable(tf.constant(self.sys_para.prev_ops_weight_base,dtype=tf.float32),name="ops_weight")
            else:
                self.ops_weight_base = tf.Variable(tf.constant(self.sys_para.manual_pulse,dtype=tf.float32),name="ops_weight")
        #self.ops_weight = tf.tanh(self.ops_weight_base)
        self.H0 = tf.Variable(tf.ones([self.sys_para.steps]), trainable=False)
        #self.Hx = tf.Variable(tf.zeros([self.sys_para.steps]))
        #self.Hz = tf.Variable(tf.zeros([self.sys_para.steps]))
        self.Hx = self.sys_para.ops_max_amp[0]*tf.add(self.ops_weight[0,:],self.ops_weight[1,:])
        self.Hz = self.sys_para.ops_max_amp[1]*self.z_weight
        print "Operators weight initialized."
                
    def init_tf_inter_states(self):
        #initialize intermediate states
        self.inter_states = []    
        for ii in range(self.sys_para.steps):
            self.inter_states.append(tf.zeros([2*self.sys_para.state_num,2*self.sys_para.state_num],
                                              dtype=tf.float32,name="inter_state_"+str(ii)))
        print "Intermediate states initialized."
            
    def get_inter_state_op(self,layer):
        # build opertor for intermediate state propagation
        # This function determines the nature of propagation
        propagator = self.matrix_exp_module.matrix_exp(self.H0[layer],self.Hx[layer],self.Hz[0,layer],size=2*self.sys_para.state_num,
                                      exp_num = self.sys_para.exp_terms
                                      ,matrix_0=self.H0_flat,
                                       matrix_1=self.flat_ops[0],matrix_2=self.flat_ops[1],
                                      matrix_I = self.I_flat)
        
        
        return propagator    
        
    def init_tf_propagator(self):
        # build propagator for all the intermediate states
        
        #first intermediate state
        self.inter_states[0] = tf.matmul(self.get_inter_state_op(0),self.tf_initial_state)
        #subsequent operation layers and intermediate states
        for ii in np.arange(1,self.sys_para.steps):
            self.inter_states[ii] = tf.matmul(self.get_inter_state_op(ii),self.inter_states[ii-1])
            
        #apply global phase operator to final state
        self.final_state = self.inter_states[self.sys_para.steps-1]
        
        self.unitary_scale = (0.5/self.sys_para.state_num)*tf.reduce_sum(tf.matmul(tf.transpose(self.final_state),self.final_state))
        
        print "Propagator initialized."
        
    def init_tf_inter_vectors(self):
        self.inter_vecs=[]
        
        for tf_initial_vector in self.tf_initial_vectors:
        
            inter_vec = tf.reshape(tf_initial_vector,[2*self.sys_para.state_num,1])
            for ii in np.arange(0,self.sys_para.steps):
                inter_vec_temp = tf.matmul(self.inter_states[ii],tf.reshape(tf_initial_vector,[2*self.sys_para.state_num,1]))
                inter_vec = tf.concat(1,[inter_vec,inter_vec_temp])
                
            self.inter_vecs.append(inter_vec)
            
        print "Vectors initialized."
        
        
    def init_training_loss(self):

        inner_product = tf.matmul(tf.transpose(self.tf_target_state),self.final_state)
        inner_product_trace_real = tf.reduce_sum(tf.pack([inner_product[ii,ii] for ii in self.sys_para.states_concerned_list]))\
        /float(len(self.sys_para.states_concerned_list))
        inner_product_trace_imag = tf.reduce_sum(tf.pack([inner_product[self.sys_para.state_num+ii,ii] for ii in self.sys_para.states_concerned_list]))\
        /float(len(self.sys_para.states_concerned_list))
        
        inner_product_trace_mag_squared = tf.square(inner_product_trace_real) + tf.square(inner_product_trace_imag)
        
        self.loss = tf.abs(1 - inner_product_trace_mag_squared)
    
    
        # Regulizer
        self.reg_loss = self.loss
        self.reg_alpha_coeff = tf.placeholder(tf.float32,shape=[])
        reg_alpha = self.reg_alpha_coeff/float(self.sys_para.steps)
        self.reg_loss = self.reg_loss + reg_alpha * tf.nn.l2_loss(tf.mul(self.tf_one_minus_gaussian_evelop,self.ops_weight))
        
        # Constrain Z to have no dc value
        self.z_reg_alpha_coeff = tf.placeholder(tf.float32,shape=[])
        z_reg_alpha = self.z_reg_alpha_coeff/float(self.sys_para.steps)
        self.reg_loss = self.reg_loss + z_reg_alpha*tf.square(tf.reduce_sum(self.ops_weight[2,:]))
        
        # Limiting the dwdt of control pulse
        self.dwdt_reg_alpha_coeff = tf.placeholder(tf.float32,shape=[])
        dwdt_reg_alpha = self.dwdt_reg_alpha_coeff/float(self.sys_para.steps)
        self.reg_loss = self.reg_loss + dwdt_reg_alpha*tf.nn.l2_loss((self.ops_weight[:,1:]-self.ops_weight[:,:self.sys_para.steps-1])/self.sys_para.dt)
        
        # Limiting the d2wdt2 of control pulse
        self.d2wdt2_reg_alpha_coeff = tf.placeholder(tf.float32,shape=[])
        d2wdt2_reg_alpha = self.d2wdt2_reg_alpha_coeff/float(self.sys_para.steps)
        self.reg_loss = self.reg_loss + d2wdt2_reg_alpha*tf.nn.l2_loss((self.ops_weight[:,2:] -\
                        2*self.ops_weight[:,1:self.sys_para.steps-1] +self.ops_weight[:,:self.sys_para.steps-2])/(self.sys_para.dt**2))
        
        # Limiting the access to forbidden states
        self.inter_reg_alpha_coeff = tf.placeholder(tf.float32,shape=[])
        inter_reg_alpha = self.inter_reg_alpha_coeff/float(self.sys_para.steps)
        
        for inter_vec in self.inter_vecs:
            for state in self.sys_para.states_forbidden_list:
                forbidden_state_pop = tf.square(0.5*(inter_vec[state,:] +\
                                                     inter_vec[self.sys_para.state_num + state,:])) +\
                                    tf.square(0.5*(inter_vec[state,:] -\
                                                     inter_vec[self.sys_para.state_num + state,:]))
                self.reg_loss = self.reg_loss + inter_reg_alpha * tf.nn.l2_loss(forbidden_state_pop)
            
        print "Training loss initialized."
            
    def init_optimizer(self):
        # Optimizer. Takes a variable learning rate.
        self.learning_rate = tf.placeholder(tf.float32,shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.reg_loss)
        
        print "Optimizer initialized."
    
    def init_utilities(self):
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()
        
        print "Utilities initialized."
        
      
            
    def build_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            
            print "Building graph:"
            
            self.init_variables()
            self.init_tf_vectors()
            self.init_tf_states()
            self.init_tf_ops()
            self.init_tf_ops_weight()
            self.init_tf_inter_states()
            self.init_tf_propagator()
            self.init_tf_inter_vectors()
            self.init_training_loss()
            self.init_optimizer()
            self.init_utilities()
            
            print "Graph built!"

        return graph
