# Xinpu Chen
# This program helps to simulate the exciton dynamic on core/surface states of quantum dots.
# Simulate surface trapping and detrapping process basing on the semi-classical Marcus-Jortner electron transfer theory
# Generate the quantum yield of quantum dots basing on their time resolved spectra.

import numpy as np
import numpy.matlib
from scipy.integrate import odeint
from scipy.integrate import solve_bvp
from scipy.special import factorial
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Parameter control region

#import your data here
data=np.genfromtxt('Decay.txt',delimiter=',')
# Set the number of core states you want to correlate in the model
#currently it has to be no larger than 5
#If want to deal with higher core state number
    # put new initial guess at the end of (kc_rxl_ini) array （line 26)
    # Also change parameter number on line 184, 187, 203, 209 follow the instructions
core_state_number =3
# manually set the relaxation rate of each core state, in the increasing order from state 1 to the highest core state
# The initial guess kc_rxl_ini=[k10,k21,k32,k43,k54] (1*5 matrix)
kc_rlx_ini = [4.5*(10**8), 1 / (3000* (10 ** (-14))), 1/(400* (10 ** (-14))), 1/(200*(10 ** (-14))),1 /(11*(10** (-14)))]
ksg =5.5*(10**8)


#code region

t1=data[:,0]
t1=(10**-9)*t1  #the unit of time is ns
time_length=np.amax(t1)
y1=data[:,1]
max_intensity=np.amax(y1) #get the maximum count in data

#remove points with "negative time"
point_num=np.size(y1,0)
index_max=np.where(y1==max_intensity)

m=0
for i in range(point_num):
    if y1[i]<1000 and t1[i]<t1[index_max]:
        m+=1
t1=t1[m:point_num]
y1=y1[m:point_num]

#normalize the intensity
y1=y1/max_intensity


# dG_abs: the absolute value of the gibbs energy change
# because we want to study the energy gap between core state and surface state
dG_abs = 5 * (10 ** (-2))
H = 0.0614
T = 300


kc_rlx = []
kc_rlx[:] = kc_rlx_ini[0:core_state_number]
E = np.arange(core_state_number)
# the energy gap between each core state and surface state
dE = dG_abs + 0.05 * E



# to create multiple surface by using matrix method
# surface trapping rate
def trap_rate(H, T, dE):
    # give values to the constants
    h = 4.135667662 * (10 ** (-15))  # h: plank constant in eV
    niu = 208  # frequancy of ground state electron in cm^-1
    kb = 8.6173324 * (10 ** (-5))
    lambda_out = 0.2412  # the unit of lambda_out is eV
    phonon_energy = h * niu * 3 * (10 ** 10)  # change the niu unit from cm^-1 to Hz
    S = 22  # lambda_in/phonon_energy
    h_bar = 6.582 * (10 ** (-16))  # h/2pi

    temperature_change = kb * lambda_out * T
    free_energy_change = 0
    for i in range(40):  # use a large number instead of infinity 40, talk in the project
        energy = (S ** (i) / factorial(i)) * (
            np.exp(-((-dE + i * phonon_energy + lambda_out) ** 2) / (4 * temperature_change)))
        free_energy_change += energy
    k = (2 * (np.pi) * (H ** 2) / h_bar) * (1 / np.sqrt(2 * (np.pi) * temperature_change)) * (
        np.exp(-S)) * free_energy_change
    return k


# surface detrapping rate
def detrap_rate(ks, dG_abs, T):
    kdt = ks * np.exp(-dG_abs / (8.6173303 * (10 ** (-5)) * T))
    return kdt


ks_array = trap_rate(H, T, dE)
ks = np.matlib.zeros(core_state_number)
ks[:] = ks_array[:]
ks_dtr = detrap_rate(ks_array[0], dG_abs, T)

# nc: core electron number
# t: time
# kc: core emission rate to ground state, ks: surface trapping rate, ks_dtr: surface detrapping rate
def derivative(nc_ini, t, kc_rlx, ksg):
    kc_rlx=np.asarray(kc_rlx)
    # generate the matrix
    # initiate the population of electron on each core state

    # initiate the matrix of electron population on each state for calculation (P)

    ns = nc_ini[0]
    if ns < 0:
        ns = 0
    ng = nc_ini[1]
    if ng > 1:
        ng = 1
    # nc_matrix= np.matlib.zeros((core_state_number+2))
    nc_fund = np.zeros((core_state_number + 2))
    nc_fund = list(nc_fund)
    nc_fund[:] = nc_ini[:]

    #stop decay when molecule number<0
    for i in range(core_state_number + 2):
        if nc_fund[i] < 0:
            nc_fund[i] = 0

    nc_matrix = np.matlib.zeros((core_state_number + 2))
    nc_matrix[:] = nc_fund[:]

    nc_matrix_tr = np.transpose(nc_matrix)
    #synthesize the matrix of population
    P = np.matlib.zeros((core_state_number, 4))
    row_p = np.size(P, 0)
    P[0:(row_p - 1), 0] = nc_matrix_tr[3:]  # P column 1:electron absorbed from higher energy state
    P[:, 1] = nc_matrix_tr[2:]              # P column 2:electron relaxed to lower energy state
    P[:, 2] = nc_matrix_tr[2:]              # P column 3:electron relaxed to the surface state
    P[0, 3] = ns                            # P column 4:electron gained from the surface state

    # initialize the matrix of rate
    R = np.matlib.zeros((4, core_state_number))
    column_r = np.size(R, 1)
    R[0, 0:(column_r - 1)] = kc_rlx[1:]  # R row 1:rate of getting electron from higher energy state
    R[1, :] = -kc_rlx                    # R row 2:rate of electron relax to lower energy state
    R[2, :] = -ks                        # R row 3:rate of electron relax to the surface state
    R[3, 0] = ks_dtr                     # R row 4:rate of surface detrapping (back to core state)

    dncdt = np.zeros(core_state_number)
    for j in range(core_state_number):
        temp = np.matmul(P[j, :], R[:, j])
        if nc_fund[j + 2] <= 0 and temp[:] < 0:
            temp[:] = 0  # Prevent the function from having further decay
        dncdt[j] = temp[:]  # change data type from [[]] to []

    dncdt = list(dncdt)  # dncdt is an array now

    nc = nc_matrix_tr[2:(core_state_number + 2), 0]
    n1 = nc_fund[2]  # electron on the first core state
    dnsdt = np.matmul(ks, nc) - ns * ks_dtr - ns * ksg  # dnsdt is a matrix
    dnsdt_arr = dnsdt.getA1().tolist()  # dnsdt is an array now

    if ns <= 0 and dnsdt_arr[0] < 0:
        dnsdt[0] = 0

    dngdt = n1 * kc_rlx[0] + ns * ksg  # dngdt is a single number,plus [] convert to array
    if ng >= 1 and dngdt > 0:
        dngdt = 0
    solution = dnsdt_arr + [dngdt] + dncdt
    return solution


# initiate the electron number on the core state
# nc_ini[0]=ns surface state population
# nc_ini[1]=n0 ground state population
# nc_ini[2:"end"] denote the core state population
nc_ini = np.zeros((core_state_number + 2))  # 7 elements vector, if core_state_num=5
nc_ini[(core_state_number + 1)] = 1  # denote the electron population on highest energy core state
nc_ini = list(nc_ini)


#if want to work with more core state, add parameter here
    #ex: If have 6 core states, add k65 behind "k54"
    #change "range (7)" to "range(8)"
def model (t,ksg,k10,k21,k32,k43,k54):
    count=core_state_number
    # re-synthesize proper kc parameter for odeint
    temp_kc=[k10,k21,k32,k43,k54] #add k65 behind k54 if want to work with 6 core states
    #get non zero kc
    kc=temp_kc[0:count]
    y = odeint(derivative, nc_ini, t, args=(kc, ksg))
    state1= y[:,2] #extract core state 1
    #normalized the function of core state 1
    max=np.amax(state1)
    state1=np.asarray(state1)
    state1=state1/max
    return state1


#here we assume the highest core state number is 5
#if want to use 6 core states change "6" to "7" (line 203)
#if want to use 9 core states change "6" to "12" etc. (line 203)
kc_rlx=np.asarray(kc_rlx)
initial_guess= np.matlib.zeros((1,6))
initial_guess[0,0]=ksg
initial_guess[0,1:(1+core_state_number)]=kc_rlx
initial_guess= initial_guess.getA1().tolist()

fit,cov=curve_fit(model,t1,y1,p0=initial_guess,bounds=(0,np.inf)) #return the fitted parameters in turn
function_x1=model(t1,fit[0],fit[1],fit[2],fit[3],fit[4],fit[5]) # add fit[6],fit[7] behind fit [5] if necessary

#unit of PL lifetime in "s"
print('tao(sg):', 1/fit[0]) #PL lifetime of surface to ground
for j in range (core_state_number): #PL lifetime of core states
    print('tao{}:'.format(j+1),1/fit[j+1])

figure1=plt.figure(1)
plt.plot(t1,y1,'yo',label='CdSe QD')
plt.plot(t1,function_x1, label='the kinetic model')
plt.xlabel('time (s)', fontsize=10)
plt.ylabel('normalized intensity', fontsize=9)
plt.title('A fluorescence decay of CdSe quantum dot',fontsize=16)
plt.legend(loc='best')

#compute the chi square
chisq=sum(((y1[0:1000]-function_x1[0:1000])**2)/y1[0:1000])
plt.figtext(0.2,0.8,'chi-square: %.2f'%chisq,fontweight='bold')

# plot the residuals
figure2=plt.figure(2)
plt.errorbar(t1,y1-function_x1)
plt.hlines(0,0,time_length)
plt.ylabel('residuals')
plt.xlabel('time (s)')

#get the quantum yield
kc_optimize=fit[1:core_state_number+1]
ksg_optimize=fit[0]
kc_optimize=kc_optimize.tolist()
decay_final = odeint(derivative, nc_ini, t1, args=(kc_optimize, ksg))
y1 = decay_final[:, 2]  # X1 population
y2 = decay_final[:, 0]  # surface
a1 = np.trapz(y1)  # exciton population on core state 1
a2 = np.trapz(y2)  # exciton population on surface
QY = a1 / (a1 + a2)
print ('QY = ',QY)

plt.show()