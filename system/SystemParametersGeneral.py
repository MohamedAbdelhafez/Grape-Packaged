import numpy as np
from math_functions.c_to_r_mat import CtoRMat
import scipy.linalg as la
from math_functions.Get_state_index import Get_State_index
from math_functions.Cops import Build_c_ops


class SystemParametersGeneral:

    def __init__(self,total_time, Modulation, Interpolation, D):
        # Input variable
        
        self.total_time = total_time
        self.Modulation = Modulation
        self.Interpolation = Interpolation
        self.D = D
        self.init_system()
        self.init_operators()
        self.init_one_minus_gaussian_envelop()
        self.init_pulse_operator()
        self.prev_ops_weight()

    def init_system(self):
        self.initial_pulse = False
        self.prev_pulse = False

        self.qubit_state_num = 4
        self.alpha = 0.224574
        self.freq_ge = 3.9225#GHz
        self.ens = np.array([ 2*np.pi*ii*(self.freq_ge - 0.5*(ii-1)*self.alpha) for ii in np.arange(self.qubit_state_num)])

        self.mode_state_num = 0

        self.qm_g1 = 2*np.pi*0.1 #GHz
        
        


        self.qm_g2 = 2*np.pi*0.1 #GHz
        
        

        self.state_num = self.qubit_state_num 

        
        

        self.states_forbidden_list = [3]

        self.pts_per_period = 10
        self.exp_terms = 20
        self.subpixels = 50
        

        self.dt = (1./6.5)/self.pts_per_period
        if self.Interpolation:
            self.Dt = self.dt*self.subpixels
        else:
            self.Dt = self.dt
        self.steps = int(self.total_time/self.dt)+1 
        self.control_steps = int(self.total_time/self.Dt)+1
        
        
    def init_vectors(self):
        self.initial_vectors=[]

        for state in self.states_concerned_list:
            if self.D:
                self.initial_vector_c= self.v_c[:,Get_State_index(state,self.dressed)]
            else:
                self.initial_vector_c=np.zeros(self.state_num)
                self.initial_vector_c[state]=1
            self.initial_vector = np.append(self.initial_vector_c,self.initial_vector_c)

            self.initial_vectors.append(self.initial_vector)


    def init_operators(self):
        # Create operator matrix in numpy array

        H_q = np.diag(self.ens)
        

        Q_x   = np.diag(np.sqrt(np.arange(1,self.qubit_state_num)),1)+np.diag(np.sqrt(np.arange(1,self.qubit_state_num)),-1)
        Q_y   = (0+1j) *(np.diag(np.sqrt(np.arange(1,self.qubit_state_num)),1)-np.diag(np.sqrt(np.arange(1,self.qubit_state_num)),-1))
        Q_z   = np.diag(np.arange(0,self.qubit_state_num))

        

        
        x_op = CtoRMat(-1j*self.dt*Q_x)
        y_op = CtoRMat(Q_y)
        z_op = CtoRMat(-1j*self.dt*Q_z)
        
             
        self.ops = [x_op,z_op]
            
        self.ops_max_amp = [4.0,2*np.pi*2.0]

        self.ops_len = len(self.ops)

        self.H0_c = H_q 
        self.w_c, self.v_c = la.eig(self.H0_c)
        
        self.dressed=[]
        if self.D:
            for ii in range (len(self.v_c)):
                index=np.argmax(np.abs(self.v_c[:,ii]))
                if index not in self.dressed:
                    self.dressed.append(index)
                else:
                    temp= (np.abs(self.v_c[:,ii])).tolist()
                    while index in self.dressed:

                        temp.remove(max(temp))
                        index2= np.argmax(np.array(temp))

                        if index2<index:
                            #dressed.append(index2)
                            index=index2
                        else:
                            #dressed.append(index2-1)
                            index=index2+1
                    self.dressed.append(index)
        
        self.H0_diag=np.diag(self.w_c) 
        self.c_ops = Build_c_ops(self.qubit_state_num,self.mode_state_num,self.v_c,self.dressed)
        self.Heff_c=self.H0_c
        self.cdagger_c=[]
        self.c_ops_new=[]
        #ceating the effective hamiltonian that describes the evolution of states if no jumps occur
        for ii in range (len(self.c_ops)):
            temp = np.dot(np.transpose(np.conjugate(self.c_ops[ii])),self.c_ops[ii])
            self.c_ops_new.append(CtoRMat(self.c_ops[ii]))
            self.cdagger_c.append(CtoRMat(temp))
            self.Heff_c= self.Heff_c + ((0-1j)/2)* ( temp)
        #making the effective hamiltonian ready for tensorflow later
        self.H0 = CtoRMat(-1j*self.dt*self.H0_c)
        self.Heff = CtoRMat(-1j*self.dt*self.Heff_c)
      
        
        self.identity_c = np.identity(self.qubit_state_num)
        
        self.identity = CtoRMat(self.identity_c)
        
    def init_one_minus_gaussian_envelop(self):
        # This is used for weighting the weight so the final pulse can have more or less gaussian like
        one_minus_gauss = []
        offset = 0.0
        overall_offset = 0.01
        for ii in range(self.ops_len+1):
            constraint_shape = np.ones(self.steps)- self.gaussian(np.linspace(-2,2,self.steps)) - offset
            constraint_shape = constraint_shape * (constraint_shape>0)
            constraint_shape = constraint_shape + overall_offset* np.ones(self.steps)
            one_minus_gauss.append(constraint_shape)


        self.one_minus_gauss = np.array(one_minus_gauss)


    def gaussian(self,x, mu = 0. , sig = 1. ):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def init_pulse_operator(self):

        #functions
        def sin(t, a, f):
            return a*np.sin(2*np.pi*f*t)

        def cos(t, a, f):
            return a*np.cos(2*np.pi*f*t)

        # gaussian envelop
        gaussian_envelop = self.gaussian(np.linspace(-2,2,self.steps))

        # This is to generate a manual pulse
        manual_pulse = []

        a=0.00

        manual_pulse.append(gaussian_envelop * cos(np.linspace(0,self.total_time,self.steps),a,self.freq_ge))
        manual_pulse.append(gaussian_envelop * sin(np.linspace(0,self.total_time,self.steps),a,self.freq_ge))
        manual_pulse.append(np.zeros(self.steps))


        self.manual_pulse = np.array(manual_pulse)

    def prev_ops_weight(self):
        if self.initial_pulse and self.prev_pulse:
            prev_ops_weight = np.load("/home/nelson/Simulations/GRAPE-GPU/data/g00-g11/GRAPE-control.npy")
            prev_ops_weight_base = np.arctanh(prev_ops_weight)
            temp_ops_weight_base = np.zeros([self.ops_len,self.steps])
            temp_ops_weight_base[:,:len(prev_ops_weight_base[0])] +=prev_ops_weight_base
            self.prev_ops_weight_base = temp_ops_weight_base
            
