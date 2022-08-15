import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

df = pd.read_csv(
    'C:/Users/Ionap/OneDrive/Documents/Edinburgh/Dissertation/elec_led_mismatch.csv')
#defining constants
k_a = 2.7 #W/mK
k_c = 3.8 #W/mK
#C_a = 3e6 #J/m^3/K
#C_c = 2e6 #J/m^3/K
C_a = 830 #Wh/m^3/K
C_c = 550 #Wh/m^3/K
C_w = 1166.7
T_i = 60
T0= 10
#E = 866529123.8#W
E = 767832413.5

#E = 437.8e6#J
rho = 2500

#temporal discretisation
dt = 0.05

ti = 38900

# spatial discretisation
M = 100
Z = 20
H = 20
R = (E/(np.pi*H*C_a*(T_i-T0)))**(1/2)
Qi= C_a*np.pi*H*(R**2)/(C_w*ti)
print(Qi)
Vw= (C_a*np.pi*H*R**2)/C_w
print(Vw)
RR = 200
print(R)

r=[]
for m in range(RR):
    rm= R*((m/M)**(1/2))
    r+=[rm]

z = np.linspace(0, 20, Z)
rr, zz = np.meshgrid(r,z, indexing='ij')

T_old = 10*np.ones((RR,Z))
T_new =np.empty((RR, Z))

#defining layer conditions
k=np.empty((RR,Z))
for m in range(RR):
    for n in range(Z):
        if z[n]>H/2:
            k[m,n]= k_c
        else:
            k[m,n] = k_a
        k[m,n] = k[m,n]

C=np.empty((RR,Z))
for m in range(RR):
    for n in range(Z):
        if z[n]>H/2:
            C[m,n]= C_a
        else:
            C[m,n] = C_a
        C[m,n] = C[m,n]

V = np.empty((RR, Z))
for m in range(RR-1):
    for n in range(Z-1):
       V[m,n] = (z[n+1]-z[n])*np.pi*((r[m+1]**2)-(r[m]**2))


qr= np.empty((RR, Z))
qz = np.empty((RR, Z))

#Boundary conditions:
T_old[0:RR,10] = (k_c*T_old[0:RR,11]+k_c*T_old[0:RR,9])/(k_c+k_a)

#time stepping
shift = 336 #ti/M

for i in range(M):
    T_old[1:RR,0:10] = T_old [0:RR-1,0:10]
    #T_old[m, n] = T_old [m-1,n]
    #print (T_old)
    T_old[0,0:10] = T_i


    for j in range(shift):
        for m in range(RR-1):
            for n in range(Z-1):
                if n == 0:
                    qz[m, n] = (T_old[m, n + 1] - T_old[m, n]) / (z[1] / (k[m, n]))
                #elif n==51:
                 #   qz[m, n] = (k_a/k_c)*(T_old[m, n - 2] - T_old[m, n-1]) / (
                  #          ((z[n] - z[n - 1]) / (2 * k[m, n - 1])) + ((z[n + 1] - z[n]) / (2 * k[m, n])))
                #elif n==50:
                 #   qz[m, n] = (k_c / k_a) * (T_old[m, n] - T_old[m, n +1]) / (
                  #          ((z[n] - z[n - 1]) / (2 * k[m, n - 1])) + ((z[n + 1] - z[n]) / (2 * k[m, n])))

                else:
                    qz[m, n] = (T_old[m, n - 1] - T_old[m, n]) / (
                                ((z[n] - z[n - 1]) / (2 * k[m, n - 1])) + ((z[n + 1] - z[n]) / (2 * k[m, n])))

                if m == 0:
                    qr[m, n] = (T_old[m + 1, n] - T_old[m, n]) / (r[1] / (k[m, n]))
                else:
                    qr[m, n] = (T_old[m - 1, n] - T_old[m, n]) / (
                                ((r[m] - r[m - 1]) / (2 * k[m - 1, n])) + ((r[m + 1] - r[m]) / (2 * k[m, n])))



                f =  (2*(z[n+1]-z[n])*((r[m]*qr[m,n])-(r[m+1]*qr[m+1,n])))+(((r[m+1]**2)-(r[m]**2))*(qz[m,n]-qz[m,n+1]))

                T_new[m,n] = T_old[m,n] + ((dt*np.pi)/(C[m,n]*V[m,n]))*f


                #print(T_new)


        T_old = T_new

    print (T_old[0:M,0])
fig, ax = plt.subplots()
im= ax.pcolormesh(rr, zz, T_old)
fig.colorbar(im, ax=ax)
plt.xlabel('Distance (m)')
plt.ylabel('Depth (m)')

plt.show()
#print (T_new[0:M,0])

shift = 144

for i in range(M):
    for j in range(shift):

        for m in range(RR-1):
            for n in range(Z-1):
                if n == 0:
                    qz[m, n] = (T_old[m, n + 1] - T_old[m, n]) / (z[1] / (k[m, n]))
               # elif n==51:
                #    qz[m, n] = (k_a/k_c)*(T_old[m, n - 2] - T_old[m, n-1]) / (
                 #           ((z[n] - z[n - 1]) / (2 * k[m, n - 1])) + ((z[n + 1] - z[n]) / (2 * k[m, n])))
                #elif n==50:
                 #   qz[m, n] = (k_c / k_a) * (T_old[m, n] - T_old[m, n +1]) / (
                  #          ((z[n] - z[n - 1]) / (2 * k[m, n - 1])) + ((z[n + 1] - z[n]) / (2 * k[m, n])))

                else:
                    qz[m, n] = (T_old[m, n - 1] - T_old[m, n]) / (
                                ((z[n] - z[n - 1]) / (2 * k[m, n - 1])) + ((z[n + 1] - z[n]) / (2 * k[m, n])))

                if m == 0:
                    qr[m, n] = (T_old[m + 1, n] - T_old[m, n]) / (r[1] / (k[m, n]))
                else:
                    qr[m, n] = (T_old[m - 1, n] - T_old[m, n]) / (
                                ((r[m] - r[m - 1]) / (2 * k[m - 1, n])) + ((r[m + 1] - r[m]) / (2 * k[m, n])))


                f =  (2*(z[n+1]-z[n])*((r[m]*qr[m,n])-(r[m+1]*qr[m+1,n])))+(((r[m+1]**2)-(r[m]**2))*(qz[m,n]-qz[m,n+1]))

                T_new[m,n] = T_old[m,n] + ((dt*np.pi)/(C[m,n]*V[m,n]))*f

                #print(T_new)



        T_old = T_new

        print (T_old[0:M,0])
fig, ax = plt.subplots()
im= ax.pcolormesh(rr, zz, T_old)
fig.colorbar(im, ax=ax)
plt.xlabel('Distance (m)')
plt.ylabel('Depth (m)')

plt.show()


t_p =59
shift = 432

T_p = []
for i in range(M):
    T_old[0:RR-1,0:10] = T_old [1:RR,0:10]
    #T_old[m, n] = T_old [m-1,n]
    #print (T_old)
    #T_old[0,0:50] = T_i


    for j in range(shift):
        for m in range(RR-1):
            for n in range(Z-1):
                if n == 0:
                    qz[m, n] = (T_old[m, n + 1] - T_old[m, n]) / (z[1] / (k[m, n]))
               # elif n==51:
                #    qz[m, n] = (k_a/k_c)*(T_old[m, n - 2] - T_old[m, n-1]) / (
                 #           ((z[n] - z[n - 1]) / (2 * k[m, n - 1])) + ((z[n + 1] - z[n]) / (2 * k[m, n])))
                #elif n==50:
                 #   qz[m, n] = (k_c / k_a) * (T_old[m, n] - T_old[m, n +1]) / (
                  #          ((z[n] - z[n - 1]) / (2 * k[m, n - 1])) + ((z[n + 1] - z[n]) / (2 * k[m, n])))

                else:
                    qz[m, n] = (T_old[m, n - 1] - T_old[m, n]) / (
                                ((z[n] - z[n - 1]) / (2 * k[m, n - 1])) + ((z[n + 1] - z[n]) / (2 * k[m, n])))

                if m == 0:
                    qr[m, n] = (T_old[m + 1, n] - T_old[m, n]) / (r[1] / (k[m, n]))
                else:
                    qr[m, n] = (T_old[m - 1, n] - T_old[m, n]) / (
                                ((r[m] - r[m - 1]) / (2 * k[m - 1, n])) + ((r[m + 1] - r[m]) / (2 * k[m, n])))



                f =  (2*(z[n+1]-z[n])*((r[m]*qr[m,n])-(r[m+1]*qr[m+1,n])))+(((r[m+1]**2)-(r[m]**2))*(qz[m,n]-qz[m,n+1]))

                T_new[m,n] = T_old[m,n] + ((dt*np.pi)/(C[m,n]*V[m,n]))*f


                #print(T_new)


        T_old = T_new

        #print (T_old[0,0:Z])
        Tp = np.mean(T_old[0,0:10])
        print(Tp )
        T_p += [Tp]

T_p_av= np.mean(T_p)
print(T_p_av)

df_t = pd.DataFrame({'T_p': T_p})
df_t.to_csv('store_out_temp_pv3.csv')


fig, ax = plt.subplots()
im= ax.pcolormesh(rr, zz, T_old)
fig.colorbar(im, ax=ax)
plt.xlabel('Distance (m)')
plt.ylabel('Depth (m)')

plt.show()


