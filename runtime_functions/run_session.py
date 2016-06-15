import numpy as np
import tensorflow as tf
from runtime_functions.Analysis import Analysis
from math_functions.avg import get_avg


def run_session(tfs,graph,conv,sys_para,single_simulation = False):
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        
        
        print "Initialized"
        trajectories = 0
        iterations=0
        traj_num=200
       
        
        while True:
            if (single_simulation == False):
                max_iterations = conv.max_iterations
            else:
                max_iterations = 0
            learning_rate = 0.02
            xy_i= np.zeros((2,sys_para.control_steps))
            z_i= np.zeros((1,sys_para.steps))
            
            #passing parameteres for the graph
            feed_dict = {tfs.learning_rate : learning_rate, tfs.z_reg_alpha_coeff: conv.z_reg_alpha_coeff,
                        tfs.reg_alpha_coeff: conv.reg_alpha_coeff, 
                         tfs.dwdt_reg_alpha_coeff: conv.dwdt_reg_alpha_coeff,
                         tfs.d2wdt2_reg_alpha_coeff: conv.d2wdt2_reg_alpha_coeff,
                         tfs.inter_reg_alpha_coeff:conv.inter_reg_alpha_coeff,
                        tfs.avg_xy:xy_i, tfs.avg_z:z_i}
            #Run the trajectory, get the gradients and number of jumps
            xy,z,j,lo,rl = session.run([ tfs.xy_grads,tfs.z_grads,tfs.jumps,tfs.loss,tfs.reg_loss], feed_dict=feed_dict)
            if trajectories ==0:
              
                j_plot=[]
                lo_plot=[]
                rl_plot=[]
                xy_av=xy
                z_av=z
            else:
                #average gradients
                xy_av=get_avg(xy,xy_av,trajectories)
                z_av=get_avg(z,z_av,trajectories)
        
            
            j_plot.append(j)
            rl_plot.append(rl)
            lo_plot.append(lo)
      
            trajectories+=1
            if (trajectories == traj_num):
                
               
                trajectories =0
                #pass the averaged gradients to tensorflow optimizer
                feed_dict = {tfs.learning_rate : learning_rate, tfs.z_reg_alpha_coeff: conv.z_reg_alpha_coeff,
                        tfs.reg_alpha_coeff: conv.reg_alpha_coeff, 
                         tfs.dwdt_reg_alpha_coeff: conv.dwdt_reg_alpha_coeff,
                         tfs.d2wdt2_reg_alpha_coeff: conv.d2wdt2_reg_alpha_coeff,
                         tfs.inter_reg_alpha_coeff:conv.inter_reg_alpha_coeff,
                        tfs.avg_xy:xy_av, tfs.avg_z:z_av}
                _ ,pc= session.run([tfs.optimizer,tfs.packed], feed_dict=feed_dict) #optimize
                
                if (iterations % conv.update_step == 0):    
                
                # Plot convergence
                    l=np.mean(lo_plot)
                    rl=np.mean(rl_plot)
                    j=np.reshape(np.mean(j_plot,axis=0),[len(sys_para.states_concerned_list),])
                    
                    anly = Analysis(sys_para, tfs.Evolution_states,tfs.ops_weight,tfs.xy_weight, tfs.xy_nocos,pc)
                    conv.update_convergence(l,rl,j,anly)
                
                # Save the variables to disk.
                    save_path = tfs.saver.save(session, "./tmp/grape.ckpt")
                    if (iterations >= max_iterations): #(l<conv.conv_target) or (iterations>=conv.max_iterations):
                        anly.get_ops_weight()
                        anly.get_xy_weight()
                        if sys_para.Modulation:
                            anly.get_nonmodulated_weight() 
                        break
                iterations+=1