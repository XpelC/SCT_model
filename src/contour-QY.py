# Xinpu Chen
# Simulate surface trapping and detrapping process basing on the semi-classical Marcus-Jortner electron transfer theory
# Estimate the effect of temperature, and X1-XS energy gap on the quantum yield of QDs
import numpy as np
import numpy.matlib
from scipy.integrate import odeint
from scipy.special import factorial
import matplotlib.pyplot as plt
from matplotlib import cm

# Parameter control region

# Please enter how many core state you want to use
# currently the core state number shouldn't be larger than 5
# If want to deal with higher core state number, initiate relaxation rate of the state (kc_rxl_ini) on line 19
core_state_number = 3
# manually set the relaxation rate of each core state, in the increasing order from state 1 to the highest core state
# kc_rxl_ini=[k10,k21,k32,k43,k54] (1*5 matrix)
kc_rlx_ini = [1.7229*(10**8), 2.3759*(10**11),6.7937*(10 ** 10),1/(200*(10 ** (-15))),1 /(10*(10 ** (-15)))]
ksg = 3.1793*(10**7)



# code region

kc_rlx = np.matlib.zeros(core_state_number)
kc_rlx[:] = kc_rlx_ini[0:core_state_number]
#dG_abs: the absolute value of the gibbs energy change
#because we want to study the energy gap between core state and surface state
dG_abs = np.arange(0, 10 * (10 ** (-2)), 0.1 * (10 ** (-2)))
T = np.arange(10, 100, 20) # Set the temperature range
H = 0.0614
E = np.arange(core_state_number)
row_dE= np.size(dG_abs)
length_T=np.size(T)
dE = np.matlib.zeros((row_dE, core_state_number))
# the energy gap between each core state and surface state
for i in range (row_dE):
    dE[i,:]= dG_abs[i] + 0.2 * E


# to create multiple surface by using matrix method
# surface trapping rate
def trap_rate(H, T, dE):
    # give values to the constants
    h = 4.135667662 * (10 ** (-15))  # h: plank constant in eV
    niu = 208  # frequancy of ground state electron in cm^-1
    kb = 8.6173324 * (10 ** (-5))
    lambda_out = 0.2412  # The outter shell contribution of the reorganization energy. Unit in eV
    phonon_energy = h * niu * 3 * (10 ** 10)  # change the niu unit from cm^-1 to Hz
    S = 22  # lambda_in/phonon_energy, unit is cm^-1. lambda_in: inner shell contribution of the reorganization energy
    h_bar = 6.582 * (10 ** (-16))  # h/2pi

    temperature_change = kb * lambda_out * T
    free_energy_change = 0
    for i in range(40):  # use a large number instead of infinity 40, talk in the project
        energy = (S ** (i) / factorial(i)) * (
            np.exp(-((dE - i * phonon_energy - lambda_out) ** 2) / (4 * temperature_change)))
        free_energy_change += energy
    k = (2 * (np.pi) * (H ** 2) / h_bar) * (1 / np.sqrt(2 * (np.pi) * temperature_change)) * (
        np.exp(-S)) * free_energy_change
    return k


# surface detrapping rate
def detrap_rate(ks, dG_abs, T):
    kdt = ks * np.exp(-dG_abs / (8.6173303 * (10 ** (-5)) * T))
    return kdt


# nc: core electron number
# t: time
# kc: core emission rate to ground state, ks: surface trapping rate, ks_dtr: surface detrapping rate
def derivative(nc_ini, t, ks, ks_dtr, kc, ksg):

    # initiate the matrix of electron population on each state for calculation (P)
    P = np.matlib.zeros((core_state_number, 4))
    row_p = np.size(P, 0)
    ns = nc_ini[0]
    if ns < 0:
        ns = 0
    ng = nc_ini[1]
    if ng > 1:
        ng = 1

    nc_fund = np.zeros((core_state_number + 2))
    nc_fund = list(nc_fund)
    nc_fund[:] = nc_ini[:]

    for i in range(core_state_number + 2):
        if nc_fund[i] < 0:
            nc_fund[i] = 0

    nc_matrix = np.matlib.zeros((core_state_number + 2))
    nc_matrix[:] = nc_fund[:]
    nc_matrix_tr = np.transpose(nc_matrix)

    # initialize the population matrix
    P[0:(row_p - 1), 0] = nc_matrix_tr[3:]  # P column 1:electron absorbed from higher energy state
    P[:, 1] = nc_matrix_tr[2:]              # P column 2:electron relaxed to lower energy state
    P[:, 2] = nc_matrix_tr[2:]              # P column 3:electron relaxed to the surface state
    P[0, 3] = ns                            # P column 4:electron gained from the surface state

    # initialize the matrix of rate
    R = np.matlib.zeros((4, core_state_number))
    column_r = np.size(R, 1)
    R[0, 0:(column_r - 1)] = kc_rlx[0, 1:]  # R column 1:rate of getting electron from higher energy state
    R[1, :] = -kc_rlx                       # R row 2:rate of electron relax to lower energy state
    R[2, :] = -ks                           # R row 3:rate of electron relax to the surface state
    R[3, 0] = ks_dtr                        # R column 4:rate of surface detrapping (back to core state)

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
    dngdt = n1 * kc + ns * ksg  # dngdt is a single number,plus [] convert to array
    if ng >= 1 and dngdt > 0:
        dngdt = 0

    solution = dnsdt_arr + [dngdt] + dncdt
    return solution


# initialize the electron number on the core state
# nc_ini[0]=ns surface state population
# nc_ini[1]=n0 ground state population
# nc_ini[2:"end"] denote the core state population
nc_ini = np.zeros((core_state_number + 2))  # 7 elements vector, if core_state_num=5
nc_ini[(core_state_number + 1)] = 1  # denote the electron population on highest energy core state
nc_ini = list(nc_ini)


ks = np.matlib.zeros(core_state_number)


kc = kc_rlx[0,0]

t_point = 1000
t_span1 = 1 * (10 ** (-8))
t = np.linspace(0, t_span1, t_point)
# The goal of this function is to extract the sufficient part of X1 population (i.e the number of e is still changing)
# Provide a better function model for trapz during the process of integration
def model_improver(y):
    y1 = y[:, 2]  # extract X1 population
    y1 = y1.tolist()
    n = 0
    # extract the time period from the Xn relaxation beginning to all of the X1 electron relaxing to the ground state
    for i in range(t_point):
        if i == 0 and y1[i] == 0:
            n += 1
        elif (y1[i] == 0 and y1[i - 1] == 0) or y1[i] > 0:
            n += 1
        elif y1[i] == 0:
            n += 1
            break
        else:
            print('Oops! Electron population on X1<0. Good luck :(')
    return n

def quantum_yield (y):
# generate the average life time (effective rate)
    sufficient_point = model_improver(y)

    y1 = y[:, 2]  # X1 population
    y1 = y1[:sufficient_point]
    y2 = y[:, 0]  # surface
    y2 = y2[:sufficient_point]
    a1 = np.trapz(y1)  # exciton population on core state 1
    a2 = np.trapz(y2)  # exciton population on surface
    Q = a1 / (a1 + a2) # quantum yield
    return Q


QY=np.matlib.zeros((row_dE, length_T))
temp=np.zeros(core_state_number)
for k in range (row_dE):
    for l in range (length_T):
        temp[:]=dE[k,:]
        ks_array = trap_rate(H, T[l], temp)
        ks[:] = ks_array[:]
        ks_dtr = detrap_rate(ks_array[0], dG_abs[k], T[l])
        # Get the electron population on each state
        # Aim to show the electron relaxing process between X1,Xs, and X0, and QY calculation
        y = odeint(derivative, nc_ini, t, args=(ks, ks_dtr, kc, ksg))
        # Correct the negative population which violate the true physical meaning
        for i in range(core_state_number + 2):
            for j in range(t_point):
                if y[j, i] < 0:
                    y[j, i] = 0
        QY[k,l]=quantum_yield(y)

figure1=plt.figure(1)
energy_gap=dG_abs.tolist()
temperature= T.tolist()

#plot the contour figure which reflect the relationship between energy gap and temperature
norm=cm.colors.Normalize(vmax=1,vmin=0)
levels=np.linspace(0,0.6,10) # Control the gradient of QY you want to use
cmap=cm.PRGn
figure1=plt.contourf(temperature,energy_gap,QY,levels, extend='both', norm=norm,cmap=cmap)
plt.title('The effect of energy gap and temperature\n on quantum yield')
plt.xlabel('temperature (K)')
plt.ylabel('energy gap \n delta G')
plt.colorbar()
plt.show()
