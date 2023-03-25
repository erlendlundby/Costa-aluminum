import casadi as ca
import numpy as np
import torch
def ablated_PBM_RK4():
    # Physics model - f_pbm
    ksl = 2.5 #1.07
    kwall = 6.5 #4.5
    xwall = 0.1 #0.05

    h_bsl = 1200
    h_w0 = 25 #15

    k0 = 2*10**-5 
    k1 = 2*ksl*17.92/119495    #3.1*10**-4  #2*ksl*Asl/delta_fus_H_cry = (2*1.07*17.92)/119495 = 3.2*10**-4
    k2 = h_bsl*17.92/119495    #0.18        #h_b-sl*A_sl/delta_fus_H_cry = 1200*17.92/119495 = 0.17996
    k3 = 1.7*10**-7            #0.002*101.96*0.95/(96485.33*12) = 1.673*10**-7
    k4 = 0.02
    k5 = 0.03
    k6 = 4.43*10**-8             #CE*0.002*26.98/(12*96485) = 4.4275*10**-8
    k7 = k2*1881                 #k2*cp_cry_liq = 0.17996*1881 = 338.5
    k8 = k1*1881                 #k1*cp_cry_liq = 3.2*10**-4*1881 = 0.602
    k9 = 17.92                   #A_sl =17.92
    k10 = 1/h_bsl                #0.0008      #1/h_b_sl = 1/1200 = 8.33*10**-4
    k11 = 1/(2*ksl)              #0.47            # 1/2*ksl =0.4673
    k12 = k2*1320                #k2*cp_cry_s = 0.18*1320 = 237.6
    k13 = k1*1320                #k1*cp_cry_s = 3.2*10**-4*1320=0.4224 
    k14 = xwall/(2*kwall)        #5.556*10**-3           #xwall/2*k_wall = 0.05/2*4.5 = 5.556*10**-3
    k15 = 1/(2*ksl)              #0.4673                 #1/2ksl  =1/2*1.07 = 0.4673
    k16 = 38.4     #NOT USED  2*ksl*Asl = 38.35 
    k17 = 6*10**-7       
    k18 = k9       #NOT USED     k9=k18
    k19 = 35                      #T0 = 35deg
    k20 = 1/h_w0                 #0.0667 #1/h_w_0 = 1/15 = 0.0667
    alpha = 5.66*10**-4          #1/cp_bath_l = 1/1767 = 5.66*10**-4
    beta = 7.58*10**-4           #1/cp_cry_s = 7.58*10**-4

    x = ca.SX.sym('x',8)
    u = ca.SX.sym('u',5)

    c_x1 =x[1]/(x[1]+x[2]+x[3])
    c_x2 = x[2]/(x[1]+x[2]+x[3])


    pr_x1_crit = 2.2
    pr_x1 = 100*c_x1
    pr_x2 = 100*c_x2
    #-----------------------------------------------------------------------
    #Choose some terms to ignore:




    #------------------------------------------------------------------------------
    #----------------------------------------------------------------------------
    #Nonlinear help functions
    #----------------------------------------------------------------------------
    g1 = 991.2 + 1.12*pr_x2 - 0.13*pr_x2**2.2 + 0.061*pr_x2**1.5\
    - 7.93*pr_x1/(1 + 0.0936*pr_x2 - 0.0017*pr_x2**2 - 0.0023*pr_x2*pr_x1)




    g2 = np.exp(2.496 - 2068.4/(273+x[5]) - 2.07*c_x1)




    g3 = 0.531 + 6.958*10**-7*u[1] - 2.51*10**-12*u[1]**2 + 3.06*10**-18*u[1]**3\
    + (0.431 - 0.1437*(pr_x1 - pr_x1_crit))/(1 + 7.353*(pr_x1 - pr_x1_crit))



    g4 = (0.5517 + 3.8168*10**-6*u[1])/(1 + 8.271*10**-6*u[1])




    g5 = (3.8168*10**-6*g3*g4*u[1])/(g2*(1-g3))

    g1 = 968
    #g2 = 2.18
    #g3 = 0.99
    #g4 = 0.5
    #g5 = 12
    #----------------------------------------------------------------------------
    h1 = u[1]*(g5 + u[1]*u[4]/(2620*g2))
    #h1 = 1851580.75


    #----------------------------------------------------------------------------
    #Set of ODE's
    #----------------------------------------------------------------------------
    xdot_1 = k1*(g1 - x[6])/(k0*x[0]) - k2*(x[5] - g1)
    xdot_2 = u[0] - k3*u[1]
    xdot_3 = u[2] - k4*u[0]
    xdot_4 = -(k1*(g1 - x[6])/(k0*x[0]) - k2*(x[5] - g1)) + k5*u[0]
    xdot_5 = k6*u[1] - u[3]

    xdot_6 = alpha/(x[1] + x[2] + x[3])*(h1 - k9*(x[5] - x[6])/(k10 + k11*k0*x[0])\
                                        -(k7*(x[5] - g1)**2 - k8*(x[5] - g1)*(g1 - x[6])/(x[0]*k0))) #w_fus*cp_cry_l*(T_b - T_liq)
                                        
                                        
                                        #- ca.fmax(0,k7*(x[5]*g1 - g1**2 - x[5]*x[6] + g1*x[6]) - k8*(g1-x[6])**2/(k0*x[0])))




    #Etc_freeze = k1(g1x6 - x6x7 + x7g1 -g1^2)/(x1k0) - k2(x6 - g1)^2
    #k1 = cp_cry_l*2ksl*Asl/delta_H_cry    =  1881*2*1.07*17.92/119495 = 0.6036
    #k2 = cp_cry_l*h_b_sl*A_sl/delta_H_cry = 1881*1200*17.92/119495 = 338.5


    xdot_7 = beta/x[0]*(k9*(g1 - x[6])/(k15*k0*x[0]) - k9*(x[6] - x[7])/(k14 + k15*k0*x[0])\
                        - (k12*(x[5] - g1)*(g1 - x[6]) - k13*(g1 - x[6])**2/(x[0]*k0))) #w_fus*cp_cry_s*(T_liq - T_sl)                
                        
                        #ca.fmax(0,k12*(g1*x[5] - g1**2 - x[5]*x[6] + x[6]*g1)/(k0*x[0]) - k13*(x[5] - g1)**2)
                                        
                                
                                        
                                        
    xdot_8 = k17*(k9*(x[6] - x[7])/(k14 + k15*k0*x[0]) - k9*(x[7] - k19)/(k14 + k20))


    xdot = ca.vcat([xdot_1, xdot_2, xdot_3, xdot_4, xdot_5, xdot_6, xdot_7,xdot_8])

    f_pbm = ca.Function('ca_nonlin_fun', [x,u],[xdot])
    #---------------------------------------------------
    U_cell = g5 + u[1]*u[4]/(2620*g2)
    T_liq = g1

    U_cell_T_liq = ca.vcat([U_cell, T_liq])

    #Variables with physical interpretation:
    volt_T_liq_pbm = ca.Function('V_T_liq',[x,u],[U_cell_T_liq])


    M=1   
    X = ca.MX.sym('X', 8)
    Y = ca.MX.sym('Y', 8)
    U = ca.MX.sym('U',5)
    #DT = 30
    Delta_T = ca.MX.sym('DT',1)

    for j in range(M):
        c1 = f_pbm(X, U)
        c2= f_pbm(X + Delta_T/2 * c1, U)
        c3 = f_pbm(X + Delta_T/2 * c2, U)
        c4 = f_pbm(X + Delta_T * c3, U)
        X_out=Y + Delta_T/6*(c1 +2*c2 +2*c3 +c4)
        #Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)

    RK4_pbm = ca.Function('RK4', [X, U,Y,Delta_T], [X_out],['x','u', 'y', 'DT'],['xf'])


    return RK4_pbm, volt_T_liq_pbm, f_pbm


def X_pbm(data,RK4_pbm,DT):
    no_traj = data.shape[0]
    no_sim_steps = data.shape[1]
    no_var = data.shape[2]

    X_out = torch.empty(no_traj, no_sim_steps, no_var)

    for i in range(no_traj):
        X_out[i,0,:] = data[i,0,:]

        for j in range(1,no_sim_steps):
            X_out[i,j,0:8] = torch.from_numpy(RK4_pbm(data[i,j-1,0:8].detach().numpy(),data[i,j-1,8:].detach().numpy(),X_out[i,j-1,0:8].detach().numpy(),DT).full()[:,0])
            X_out[i,j,8:] = data[i,j,8:]
    return X_out

def dXdt_pbm(data, f_pbm):
    #Used to calculate time derivative of states in 
    no_traj = data.shape[0]
    no_sim_steps = data.shape[1] 
    

    dXdt_out = torch.empty(no_traj, no_sim_steps-1, 8)

    for i in range(no_traj):
        for j in range(no_sim_steps-1):
            dXdt_out[i,j,0:8] = torch.from_numpy(f_pbm(data[i,j,0:8].detach().numpy(),data[i,j,8:].detach().numpy()).full()[:,0])
            
    return dXdt_out

def RollingForecast_pbm(test,RK4, DT):
    no_traj = test.shape[0]
    no_sim_steps = test.shape[1]
    no_var = test.shape[2]

    X_out = torch.empty(no_traj, no_sim_steps, no_var)

    for i in range(no_traj):
        X_out[i,0,:] = test[i,0,:]

        for j in range(1,no_sim_steps):
            X_out[i,j,0:8] = torch.from_numpy(RK4(X_out[i,j-1,0:8].detach().numpy(),test[i,j-1,8:].detach().numpy(),X_out[i,j-1,0:8].detach().numpy(),DT).full()[:,0])
            X_out[i,j,8:] = test[i,j,8:]
    return X_out

def Voltage_liquidus_temp_pbm(X_pbm_RF, volt_T_liq_pbm):
    no_sim = X_pbm_RF.shape[0]
    sim_steps = X_pbm_RF.shape[1]

    V_cell_T_liq = torch.zeros(no_sim,sim_steps,2)

    for i in range(no_sim):
        for j in range(sim_steps):
            
            V_cell_T_liq[i,j,:] = torch.from_numpy(volt_T_liq_pbm(X_pbm_RF[i,j,0:8].numpy(),X_pbm_RF[i,j,8:].numpy()).full()[:])[:,0]
        
    return V_cell_T_liq


