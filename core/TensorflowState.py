import os
from tensorflow.python.ops import control_flow_ops
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
        #converting numpy parameters into tensorflow constants
        self.tf_identity = tf.constant(self.sys_para.identity,dtype=tf.float32)
        self.tf_neg_i = tf.constant(CtoRMat(-1j*self.sys_para.identity_c),dtype=tf.float32)
        self.tf_one_minus_gaussian_evelop = tf.constant(self.sys_para.one_minus_gauss,dtype=tf.float32)
        self.tf_dressed = tf.constant(self.sys_para.dressed,dtype=tf.float32)
        self.tf_H0 = tf.constant(self.sys_para.H0,dtype=tf.float32)
        self.tf_Heff = tf.constant(self.sys_para.Heff,dtype=tf.float32)
        self.tf_c_ops = tf.constant(np.reshape(self.sys_para.c_ops_new,[len(self.sys_para.c_ops),2*self.sys_para.state_num,2*self.sys_para.state_num]),dtype=tf.float32)
        self.tf_cdagger_c = tf.constant(np.reshape(self.sys_para.cdagger_c,[len(self.sys_para.c_ops),2*self.sys_para.state_num,2*self.sys_para.state_num]),dtype=tf.float32)
        self.norms=[]
        self.jumps=[]
        
        
       
        
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
        #tf operators for control Hamiltonian (Heff is the drift hamiltonian in the trajectories framework, and the flat ops are x and z for control)
        self.tf_H0 = tf.constant(self.sys_para.H0,dtype=tf.float32)
        
        self.tf_ops = []
        for op in self.sys_para.ops:
            self.tf_ops.append(tf.constant(op,dtype=tf.float32))
        
        
        i_array = np.eye(2*self.sys_para.state_num)
        op_matrix_I=i_array.tolist()
        self.I_flat = [item for sublist in op_matrix_I  for item in sublist]
        self.H0_flat = [item for sublist in self.sys_para.H0  for item in sublist]
        self.Heff_flat = [item for sublist in self.sys_para.Heff  for item in sublist]
        
        self.flat_ops = []
        for op in self.sys_para.ops:
            flat_op = [item for sublist in op for item in sublist]
            self.flat_ops.append(flat_op)
            
        print "Operators initialized."
        
    
        
    def get_j(self,l):
        #A function to help in interpolation 
        dt=self.sys_para.dt
        Dt=self.sys_para.Dt
        jj=np.floor((l*dt-0.5*Dt)/Dt)
        return jj
    
    
            
    def transfer_fn(self,xy):
        # A function that takes the control pulses with a smaller timestep and interpolate between them to generate the simulation weights
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
                #moudlating x and y by a carrier
                cosine= tf.cos(2*np.pi*self.sys_para.freq_ge*np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]))
                sine= tf.sin(2*np.pi*self.sys_para.freq_ge*np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]))
                temp1 = tf.mul(self.xy_nocos[0,:],tf.cast(cosine, tf.float32))
                temp2 = -tf.mul(self.xy_nocos[1,:],tf.cast(sine, tf.float32))
                self.xy_cos = tf.concat(0,[tf.reshape(temp1,[1,self.sys_para.steps]),tf.reshape(temp2,[1,self.sys_para.steps])],name="modulated")
                self.ops_weight = tf.concat(0,[self.xy_cos,self.z_weight],name="ops_weight")

            else:
                self.ops_weight = tf.concat(0,[self.xy_nocos,self.z_weight],name="ops_weight")
            
        #Defining the weights to enter into the optimizer
        self.H0 = tf.Variable(tf.ones([self.sys_para.steps]), trainable=False) #multiplying H0/Heff, just ones
        self.Hx = self.sys_para.ops_max_amp[0]*tf.add(self.ops_weight[0,:],self.ops_weight[1,:]) #The x and y weights multiply the x operator
        self.Hz = self.sys_para.ops_max_amp[1]*self.z_weight # The z weight multiplies the z operator
        print "Operators weight initialized."
                
    def init_tf_inter_states(self):
        #initialize intermediate states
        self.inter_states = []    
        for ii in range(self.sys_para.steps):
            self.inter_states.append(tf.zeros([2*self.sys_para.state_num,2*self.sys_para.state_num],
                                              dtype=tf.float32,name="inter_state_"+str(ii)))
        print "Intermediate states initialized."
        
    
        

    
    def init_one_trajectory(self):
        # Create a trajectory for each initial state
        self.Evolution_states=[]
        self.inter_lst = []
        for tf_initial_vector in self.tf_initial_vectors:
            self.Evolution_states.append(self.One_Trajectory(tf_initial_vector)) #returns the final state of the trajectory
        self.packed = tf.pack(self.inter_lst)
        print "One Trajectory done"
        
    
    def get_random(self,start,end):
        #Returns a random number between 0 & 1 to tell jumps when to occur
        rand=tf.random_uniform([1],start,end)
        return rand
    
    def evolve_Heff(self,psi,layer):
        #The evolution with no jumps just follows exp(iHeff dt). We always evolve first by Heff then check for the norm of the state to see if we should have a jump next
        #The exponential is done via a self written kernel 
        propagator = self.matrix_exp_module.matrix_exp(self.H0[layer],self.Hx[layer],self.Hz[0,layer],size=2*self.sys_para.state_num,
                                      exp_num = self.sys_para.exp_terms
                                      ,matrix_0=self.Heff_flat,
                                       matrix_1=self.flat_ops[0],matrix_2=self.flat_ops[1],
                                      matrix_I = self.I_flat) #this is evolution operator for this (layer) timestep
        inter_vec_temp = tf.matmul(propagator,tf.reshape(psi,[2*self.sys_para.state_num,1])) #we evolve the state by multiplying the operator by the previous state
        return propagator, inter_vec_temp
    
    def get_norm(self,psi):
        #Take a state psi, calculate its norm. Not trivial since psi is 2n instead of n to avoid dealing with complex numbers
        state_num=self.sys_para.state_num
        psi_real = 0.5*(psi[0:state_num,:]+psi[state_num:2*state_num,:])
        psi_imag = 0.5*(psi[state_num:2*state_num,:] - psi[0:state_num,:])
        norm = tf.reduce_sum(tf.add(tf.square(psi_real),tf.square(psi_imag)))
        return norm
    def get_inner_product(self,psi1,psi2):
        #Take 2 states psi1,psi2, calculate their overlap.
        state_num=self.sys_para.state_num
        psi_1_real = 0.5*(psi1[0:state_num,:]+psi1[state_num:2*state_num,:])
        psi_1_imag = 0.5*(psi1[state_num:2*state_num,:] - psi1[0:state_num,:])
        psi_2_real = 0.5*(psi2[0:state_num,:]+psi2[state_num:2*state_num,:])
        psi_2_imag = 0.5*(psi2[state_num:2*state_num,:] - psi2[0:state_num,:])
        # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude
        ac = tf.mul(psi_1_real,psi_2_real)
        bd = tf.mul(psi_1_imag,psi_2_imag)
        bc = tf.mul(psi_1_imag,psi_2_real)
        ad = tf.mul(psi_1_real,psi_2_imag)
        reals = tf.square(tf.add(tf.reduce_sum(ac),tf.reduce_sum(bd)))
        imags = tf.square(tf.sub(tf.reduce_sum(bc),tf.reduce_sum(ad)))
        norm = tf.add(reals,imags)
        return norm
    
    def jump (self,psi):
        #This is the alternate evolution at one timestep if a jump occurs
        weights=[]
        sums=[]
        s=0
        state_num=self.sys_para.state_num
        #We have different jump operators that can occur. We pick a new random number and calculate the probabilities of each jump 
        #(by the relative c dagger c expectation value) and then decide which jump to do according to that uniform random number
        for ii in range (len(self.sys_para.c_ops_new)):
            
            temp=tf.matmul(tf.transpose(tf.reshape(psi,[2*state_num,1])),self.tf_cdagger_c[ii,:,:])
            temp2=tf.matmul(temp,tf.reshape(psi,[2*state_num,1])) #get the jump expectation value
            weights=tf.concat(0,[weights,tf.reshape(temp2,[1])])
        weights=tf.abs(weights/tf.reduce_sum(tf.abs(weights))) #convert them to probabilities
        
        for jj in range (len(self.sys_para.c_ops_new)):
            #create a list of their summed probabilities
            s=s+weights[jj]
            sums=tf.concat(0,[sums,tf.reshape(s,[1])])
            
        r2 = self.get_random(0,1)
        #tensorflow conditional graphing, checks for the first time a summed probability exceeds the random number
        rvector=r2 * tf.ones_like(sums)
        cond= tf.greater_equal(sums,rvector)
        a=tf.where(cond)
        final =tf.reshape(a[0,:],[])
        
        #apply the chosen jump operator
        propagator2 = tf.gather(self.tf_c_ops,final)
        inter_vec_temp2 = tf.matmul(propagator2,tf.reshape(psi,[2*self.sys_para.state_num,1]))
        norm2 = self.get_norm(inter_vec_temp2)
        inter_vec_temp2 = inter_vec_temp2 / tf.sqrt(norm2)
        propagator2 = propagator2 / tf.sqrt(norm2)
        
        
        return propagator2, inter_vec_temp2
        
            
        
    def One_Trajectory(self,psi0):
        #Creates a trajectory for psi0
       
        self.inter_vecs=[]
        self.inter_state_op=[]
        norm = 1
        self.r=self.get_random(0,1)
        jumps=tf.constant(0)
        
        inter_vec = tf.reshape(psi0,[2*self.sys_para.state_num,1])
        for ii in np.arange(0,self.sys_para.steps):
            prob, inter_vec_temp = self.evolve_Heff(psi0,ii) #first evolve the timestep using Heff
            
            new_norm= self.get_inner_product(inter_vec_temp,inter_vec_temp) #then calculate the new norm of the state, which should be less than 1 because Heff is not hermitian
            norm= norm*new_norm
            
            
            def f1(): #The tensorflow path if no jump occurs
                vector= inter_vec_temp/tf.sqrt(new_norm)
                propa = prob / tf.sqrt(new_norm)
                #we already evolved by Heff so just normalize the state and move on with the same random number
                counter=tf.constant(0)
                t=self.r  
                return t,counter,norm,propa,vector
                        
            def f2(): #The tensorflow path if a jump occurs
               
                no = tf.constant(1,dtype=tf.float32) # the new norm is back to 1     
                k=self.get_random(0,1) #We need a new random number for later steps
                propa2, vector2 = self.jump(psi0) #do the jump
                counter= tf.constant(1) #count one jump
                return k,counter,no,propa2,vector2

        

            condition=tf.less(self.r,norm) # a tensorflow condition to check if the random number is less than the norm already, so do a jump
            ra,c,n,p,v = tf.cond(tf.reshape(condition,[]), f1, f2) #perform f1 if cond is true, f2 if cond is false
            norm = n
            jumps = tf.add(jumps,c)
            self.r = ra
            
            inter_vec = tf.concat(1,[inter_vec,v]) #save the intermediate states
            self.inter_state_op.append(p)
            psi0=v
            self.norms= tf.concat(0,[self.norms,tf.reshape(norm,[1])])
            
       
        self.inter_lst.append(inter_vec)
        self.inter_vecs = inter_vec
        self.jumps= tf.concat(0,[self.jumps,tf.reshape(jumps,[1])])
        
        
        return psi0
    
    
    
    
    def init_training_loss(self):
        #defining the cost function
        ii=0
        err=[]
        for tf_initial_vector in self.tf_initial_vectors:
            #for every initial vector, get the overlap betwen its final state of the trajectory with the desired state, and average the error
            final_state=self.Evolution_states[ii]
            target_vector=tf.matmul(self.tf_target_state,tf.reshape(tf_initial_vector,[2*self.sys_para.state_num,1]))
            inner_product = self.get_inner_product(final_state,target_vector)
            err=tf.concat(0,[err,tf.reshape(tf.sub(tf.constant(1,dtype=tf.float32),inner_product),[1])])
            ii=ii+1
        

        self.loss = tf.reduce_mean(err)
    
    
        # Regulaizer to make it look like a gaussian
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
        
        #for inter_vec in self.inter_vecs:
        for state in self.sys_para.states_forbidden_list:
            forbidden_state_pop = tf.square(0.5*(self.inter_vecs[state,:] +\
                                                     self.inter_vecs[self.sys_para.state_num + state,:])) +\
                                    tf.square(0.5*(self.inter_vecs[state,:] -\
                                                     self.inter_vecs[self.sys_para.state_num + state,:]))
            self.reg_loss = self.reg_loss + inter_reg_alpha * tf.nn.l2_loss(forbidden_state_pop)
            
        print "Training loss initialized."
   
            
    def init_optimizer(self):
        # Optimizer. Takes a variable learning rate.
        self.learning_rate = tf.placeholder(tf.float32,shape=[])
        self.opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        
        #Here we extract the gradients of the xy and z pulses
        self.grad = self.opt.compute_gradients(self.reg_loss)
        self.grads =[g for g, _ in self.grad]
        self.var = [v for _,v in self.grad]
        self.xy_grads = self.grads[0]
        self.z_grads = self.grads[1]
        
        #We leave a placeholder for the averaged gradients to be passed at runtime after averaging many trajectories
        self.avg_xy = tf.placeholder(tf.float32, shape = [2,self.sys_para.control_steps])
        self.avg_z = tf.placeholder(tf.float32, shape = [1,self.sys_para.steps])
        self.avg = []
        self.avg.append(self.avg_xy)
        self.avg.append(self.avg_z)
        self.new_grad = zip(self.avg,self.var)
        #apply the averaged gradient
        self.optimizer = self.opt.apply_gradients(self.new_grad)
        
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
            self.init_one_trajectory()
            self.init_training_loss()
            self.init_optimizer()
            self.init_utilities()
            
            print "Graph built!"

        return graph