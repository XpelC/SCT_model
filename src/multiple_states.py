# Xinpu Chen
# This program helps to simulate the exciton dynamic on core/surface states of quantum dots.
# Simulate surface trapping and detrapping process basing on the semi-classical Marcus-Jortner electron transfer theory
# Estimate the quantum yield of quantum dots.
import numpy as np
import numpy.matlib
from scipy.integrate import odeint
from scipy.integrate import solve_bvp
from scipy.special import factorial
import matplotlib.pyplot as plt


#Input parameter control region

#Please enter how many core state you want to talk about
#currently it has to be no larger than 5
#If want to deal with higher core state number, put new initial guess at the end of (kc_rxl_ini) array （line 22)
core_state_number=5

#manually set the relaxation rate of each core state, in the increasing order from state 1 to the highest core state
# kc_rxl_ini=[k10,k21,k32,k43,k54]
kc_rlx_ini = [4.5*(10**7), 1 / (3000* (10 ** (-14))), 1/(300* (10 ** (-14))), 1/(200*(10 ** (-14))),1 /(11*(10** (-14)))]
ksg =5.5*(10**7) #exciton decay rate from surface to ground state

t_point=1000 #generate 1000 point on the graph, control the pace size
t_span1=6*(10**(-12)) #control the time region of the graph on the top
t_span2=7*(10**(-8))  #control the time region of the graph on the bottom


# code region

#dG_abs: the absolute value of the gibbs energy change
#because we want to study the energy gap between core state and surface state
dG_abs=5*(10**(-2))
H=0.0614
T=300
kc_rlx=[]
kc_rlx[:]=kc_rlx_ini[0:core_state_number]
E=np.arange(core_state_number)
#the energy gap between each core state and surface state
dE=dG_abs+0.05*E #the energy levels are assumed evenly spaced by 0.05 eV

# to create multiple surface by using matrix method
#surface trapping rate
def trap_rate (H,T,dE):
    # give values to the constants
    h= 4.135667662*(10**(-15)) #h: plank constant in eV
    niu= 208                   #frequancy of ground state electron in cm^-1
    kb= 8.6173324*(10**(-5))
    lambda_out= 0.2412         # the unit of lambda_out is eV
    phonon_energy= h*niu*3*(10**10) #change the niu unit from cm^-1 to Hz
    S=22                       #lambda_in/phonon_energy
    h_bar=6.582*(10**(-16))    #h/2pi

    temperature_change=kb*lambda_out*T
    free_energy_change=0
    for i in range (40):       #use a large number instead of infinity 40, talk in the project
        energy=(S**(i)/factorial(i))*(np.exp(-((dE-i*phonon_energy-lambda_out)**2)/(4*temperature_change)))
        free_energy_change+=energy
    k=(2*(np.pi)*(H**2)/h_bar)*(1/np.sqrt(2*(np.pi)*temperature_change))*(np.exp(-S))*free_energy_change
    return k


#surface detrapping rate
def detrap_rate (ks,dG_abs,T):
    kdt=ks*np.exp(-dG_abs/(8.6173303*(10**(-5))*T))
    return kdt


#nc: core electron number
#t: time
#kc: core emission rate to ground state, ks: surface trapping rate, ks_dtr: surface detrapping rate
def derivative (nc_ini,t,ks,ks_dtr,kc_rlx,ksg):
    kc_rlx=np.asarray(kc_rlx)

    # generate the matrix
    # initialize the population of electron on each core state
    P = np.matlib.zeros((core_state_number, 4))
    row_p = np.size(P, 0)
    ns = nc_ini[0]
    if ns<0:
        ns=0
    ng = nc_ini[1]
    if ng>1:
        ng=1

    nc_fund = np.zeros((core_state_number + 2))
    nc_fund = list(nc_fund)
    nc_fund[:] = nc_ini[:]

    for i in range (core_state_number+2):
        if nc_fund[i] < 0:
            nc_fund[i] = 0

    nc_matrix= np.matlib.zeros((core_state_number+2))
    nc_matrix[:]=nc_fund[:]

    #initiailize the matrix of population
    nc_matrix_tr=np.transpose(nc_matrix)
    P[0:(row_p - 1), 0] = nc_matrix_tr[3:]  # P column 1:electron absorbed from higher energy state
    P[:, 1] = nc_matrix_tr[2:]              # P column 2:electron relaxed to lower energy state
    P[:, 2] = nc_matrix_tr[2:]              # P column 3:electron relaxed to the surface state
    P[0, 3] = ns                            # P column 4:electron gained from the surface state
    
    # initialize the matrix of rate
    R = np.matlib.zeros((4, core_state_number))
    column_r = np.size(R, 1)
    R[0, 0:(column_r - 1)] = kc_rlx[1:]  # R column 1:rate of getting electron from higher energy state
    R[1, :] = -kc_rlx                    # R row 2:rate of electron relax to lower energy state
    R[2, :] = -ks                        # R row 3:rate of electron relax to the surface state
    R[3, 0] = ks_dtr                     # R column 4:rate of surface detrapping (back to core state)

    dncdt=np.zeros(core_state_number)
    for j in range(core_state_number):
        temp = np.matmul(P[j, :], R[:, j])
        if nc_fund[j+2]<=0 and temp[:]<0:
            temp[:]=0      # Prevent the function from having further decay
        dncdt[j]=temp[:]   # change data type from [[]] to []

    dncdt=list(dncdt) # dncdt is an array now

    nc=nc_matrix_tr[2:(core_state_number+2),0]
    n1 = nc_fund[2]         # electron on the first core state
    dnsdt=np.matmul(ks,nc)-ns*ks_dtr-ns*ksg #dnsdt is a matrix
    dnsdt_arr=dnsdt.getA1().tolist() #dnsdt is an array now
    if ns<=0 and dnsdt_arr[0]<0:
        dnsdt[0]=0
    dngdt=n1*kc_rlx[0]+ns*ksg  #dngdt is a single number,plus [] convert to array
    if ng>=1 and dngdt>0:
        dngdt=0


    solution=dnsdt_arr+[dngdt]+dncdt
    return solution



#initiate the electron number on the core state
#nc_ini[0]=ns surface state population
#nc_ini[1]=n0 ground state population
#nc_ini[2:"end"] denote the core state population
nc_ini=np.zeros((core_state_number+2)) #7 elements vector, if core_state_num=5
nc_ini[(core_state_number+1)]=1 #denote the electron population on highest energy core state
nc_ini=list(nc_ini)

ks_array = trap_rate(H, T, dE)
ks= np.matlib.zeros(core_state_number)
ks[:]=ks_array[:]

ks_dtr = detrap_rate(ks_array[0], dG_abs, T)

t1=np.linspace(0,t_span1,t_point)
t2=np.linspace(0,t_span2,t_point)

#Get the electron population on each state
#y_short stands for the changing of electron population within a shorter period of time
    #Aim to show the behavior of higher core state electron relaxation (Xn to X2)
#y_long stands for the changing of electron population within a longer period of time
    #Aim to show the electron relaxing process between X1,Xs, and X0, and QY calculation
y_short=odeint(derivative,nc_ini,t1,args=(ks,ks_dtr,kc_rlx,ksg))
y_long=odeint(derivative,nc_ini,t2,args=(ks,ks_dtr,kc_rlx,ksg))

#Correct the negative population which violate the true physical meaning
for i in range (core_state_number+2):
    for j in range (t_point):
        if y_long[j,i]<0:
            y_long[j,i]=0

#figure 1: print the electron population decay on each state
def create_plots (y,t,figure):
    for j in range (core_state_number):
        plt.plot(t,y[:,j+2],label='X{}'.format(j+1))
        j+=1

    figure=plt.plot(t,y[:,0],label='surface')
    figure=plt.plot(t,y[:,1],label='ground state')
    plt.legend(loc=0, prop={'size': 6})
    return figure

figure1=plt.figure(1)
plt.subplot(211)
create_plots(y_short,t1,figure1)
plt.ylabel('Exciton population\n on core states', fontsize=9)
plt.subplot(212)
create_plots(y_long,t2,figure1)
figure1.suptitle('The exciton population on each state', fontsize=16)
plt.xlabel('time (s)', fontsize=10)
plt.ylabel('Exciton population\n on core states', fontsize=9)


#quantum yield calculation
y1 = y_long[:, 2]  # X1 population
y2 = y_long[:, 0]  # surface
a1 = np.trapz(y1)  # exciton population on core state 1
a2 = np.trapz(y2)  # exciton population on surface
QY = a1 / (a1 + a2)

print ('QY = ', QY)

plt.show()