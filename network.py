from tespy.components import Source, Sink, HeatExchangerSimple, Pipe
from tespy.components import (
    Sink, Source, Valve, Turbine, Splitter, Merge, Condenser, Pump,
    HeatExchangerSimple, CycleCloser
)
from tespy.connections import Connection, Bus, Ref
from tespy.networks import Network

from tespy.tools import CharLine
from tespy.tools import document_model

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sub_consumer import (LinConsumClosed as lc,
                          LinConsumOpen as lo,
                          Fork as fo, Building as bld, CHP)

# %% network

nw = Network(fluids=['water'], T_unit='C', p_unit='bar', h_unit='kJ / kg')

# %% components

so = Source('source')
si = Sink('sink')

#Consumers
LIAC = HeatExchangerSimple('LIAC')
Roslin = HeatExchangerSimple('Roslin')
Greenwood = HeatExchangerSimple('Greenwood')
Centre = HeatExchangerSimple('Centre')

valve_L = Valve('LIAC valve')
valve_R = Valve('Roslin valve')
valve_G = Valve('Greenwood valve')
valve_C = Valve('Centre valve')

# %% construction part

# pipe_feed
cc = CycleCloser('cycle closer')
pif1 = Pipe('pipe1_feed', ks=7e-5, L=275, D=0.25,kA = 270)
pif2 = Pipe('pipe2_feed', ks=7e-5, L=10, D=0.2, kA = 8.2)

pif3 = Pipe('pipe3_feed', ks=7e-5, L=430, D=0.2, kA = 351.3)
pif4 = Pipe('pipe4_feed', ks=7e-5, L=80, D=0.065, kA = 32.6)

pif5 = Pipe('pipe5_feed', ks=7e-5, L=30, D=0.2, kA = 24.4)
pif6 = Pipe('pipe6_feed', ks=7e-5, L=180, D=0.125, kA = 113.1)

pif7 = Pipe('pipe7_feed', ks=7e-5, L=35, D=0.2, kA = 28.6)
pif8 = Pipe('pipe8_feed', ks=7e-5, L=30, D=0.2, kA = 24.4)

pif9 = Pipe('pipe9_feed', ks=7e-5, L=30, D=0.1, kA = 16)

# pipe_back

pib1 = Pipe('pipe1_back', ks=7e-5, L=275, D=0.25, kA = 270)
pib2 = Pipe('pipe2_back', ks=7e-5, L=10, D=0.2, kA = 8.2)

pib3 = Pipe('pipe3_back', ks=7e-5, L=430, D=0.2, kA = 351.3)
pib4 = Pipe('pipe4_back', ks=7e-5, L=80, D=0.065, kA = 32.6)

pib5 = Pipe('pipe5_back', ks=7e-5, L=30, D=0.2, kA = 28.6)
pib6 = Pipe('pipe6_back', ks=7e-5, L=180, D=0.125,kA = 113.1)

pib7 = Pipe('pipe7_back', ks=7e-5, L=35, D=0.2, kA = 28.6)
pib8 = Pipe('pipe8_back', ks=7e-5, L=30, D=0.2, kA = 24.4)

pib9 = Pipe('pipe9_back', ks=7e-5, L=30, D=0.1, kA = 16)


# %% subsystems for forks

#k1 = fo('K1', 2)
k2 = fo('K2', 2)
k3 = fo('K3', 2)
k4 = fo('K4',2)

nw.add_subsys(k4, k3, k2)

#consumer parameters

LIAC.set_attr(Q=-0.3e6, pr=0.99)

Roslin.set_attr(Q=-1.409e6, pr=0.99)

Greenwood.set_attr(Q=-0.4e6, pr=0.99)

Centre.set_attr(Q=-0.708e6, pr=0.99)

# temperature difference factor for pipes:

dT_feed = 300
dT_return = 600


# %% LIAC

# feed
so_pif3 = Connection(so, 'out1', pif3, 'in1', T=90, p=15, fluid={'water': 1})
pif3_k2 = Connection(pif3, 'out1', k2.comps['splitter'], 'in1')
k2f_pif4 = Connection(k2.comps['splitter'], 'out1', pif4, 'in1')
pif4_LIAC = Connection(pif4, 'out1', LIAC, 'in1')


# back
LIAC_va = Connection(LIAC, 'out1', valve_L, 'in1', T=45)
LIAC_pib4 = Connection(valve_L, 'out1', pib4, 'in1',T=45)
pib4_k2 = Connection(pib4, 'out1', k2.comps['valve_0'], 'in1')
k2_pib3 = Connection(k2.comps['merge'], 'out1', pib3, 'in1', p=11)
pib3_si = Connection(pib3, 'out1', si, 'in1')


nw.add_conns(so_pif3, pif3_k2, k2f_pif4, pif4_LIAC)
nw.add_conns(LIAC_va,LIAC_pib4,pib4_k2,k2_pib3, pib3_si)

# %% center

# feed
k2_pif5 = Connection(k2.comps['splitter'], 'out2', pif5, 'in1')
pif5_k3 = Connection(pif5, 'out1', k3.comps['splitter'], 'in1')
k3f_pif6 = Connection(k3.comps['splitter'], 'out1', pif6, 'in1')
pif6_Centre = Connection(pif6, 'out1', Centre, 'in1')


# back
Centre_va = Connection(Centre, 'out1', valve_C, 'in1', T=45)
Centre_pib6 = Connection(valve_C, 'out1', pib6, 'in1',T=45)
pib6_k3 = Connection(pib6, 'out1', k3.comps['valve_0'], 'in1')
k3_pib5 = Connection(k3.comps['merge'], 'out1', pib5, 'in1', p=12)
pib5_k2 = Connection(pib5, 'out1', k2.comps['valve_1'], 'in1')


nw.add_conns(k2_pif5, pif5_k3, k3f_pif6, pif6_Centre)
nw.add_conns(Centre_va,Centre_pib6,pib6_k3,k3_pib5, pib5_k2)

# %% Roslin

# feed
k3_pif7 = Connection(k3.comps['splitter'], 'out2',  pif7, 'in1')
pif7_k4 = Connection(pif7, 'out1', k4.comps['splitter'], 'in1')
k4_pif8 = Connection(k4.comps['splitter'], 'out1', pif8, 'in1')
pif8_Roslin = Connection(pif8, 'out1', Roslin, 'in1')

# back
Roslin_va = Connection(Roslin, 'out1', valve_R, 'in1', T=45)
Roslin_pib8 = Connection(valve_R, 'out1', pib8, 'in1',T=45)
pib8_k4 = Connection(pib8, 'out1', k4.comps['valve_0'], 'in1')
k4_pib7 = Connection(k4.comps['merge'], 'out1', pib7, 'in1', p=13)
pib7_k3 = Connection(pib7, 'out1', k3.comps['valve_1'], 'in1')
#
nw.add_conns(k3_pif7,pif7_k4,k4_pif8,pif8_Roslin)
nw.add_conns(Roslin_va, Roslin_pib8,pib8_k4,k4_pib7,pib7_k3)

# %% Greenwood

# feed
k4_pif9 = Connection(k4.comps['splitter'], 'out2', pif9, 'in1',)
pif9_Greenwood = Connection(pif9, 'out1', Greenwood, 'in1')

# back
Greenwood_va = Connection(Greenwood, 'out1', valve_G, 'in1', T=45)
Greenwood_pib9 = Connection(valve_G, 'out1', pib9, 'in1',T=45)
pib9_k4 = Connection(pib9, 'out1', k4.comps['valve_1'], 'in1')

nw.add_conns(k4_pif9, pif9_Greenwood)
nw.add_conns(Greenwood_va, Greenwood_pib9, pib9_k4)



#original buses
heat_losses = Bus('network losses')
heat_consumer = Bus('network consumer')

nw.check_network()

for comp in nw.comps['object']:
    if isinstance(comp, Pipe):
        comp.set_attr(Tamb=0)

        heat_losses.add_comps({'comp': comp})

    if (isinstance(comp, HeatExchangerSimple) and
            not isinstance(comp, Pipe)):
        heat_consumer.add_comps({'comp': comp})

nw.add_busses(heat_losses, heat_consumer)



# design case: 0 °C ambient temperature
nw.solve('design')
nw.save('grid')
document_model(nw)
# no documentation of offedesign state added, as report creation takes
# quite long with all characteristics applied, try it out yourself :)

print('Heat demand consumer:', heat_consumer.P.val)
print('network losses at 0 °C outside temperature (design):', heat_losses.P.val)

# offdesign case: 10 °C ambient temperature

for comp in nw.comps['object']:
    if isinstance(comp, Pipe):
        comp.set_attr(Tamb=10)

nw.solve('offdesign', design_path='grid')
print('Heat demand consumer:', heat_consumer.P.val)
print('network losses at 10 °C outside temperature:', heat_losses.P.val)

for comp in nw.comps['object']:
    if isinstance(comp, Pipe):
        comp.set_attr(Tamb=20)

nw.solve('offdesign', design_path='grid')
print('Heat demand consumer:', heat_consumer.P.val)
print('network losses at 20 °C outside temperature:', heat_losses.P.val)

# offdesign case: -10 °C ambient temperature

for comp in nw.comps['object']:
    if isinstance(comp, Pipe):
        comp.set_attr(Tamb=-10)

nw.solve('offdesign', design_path='grid')

print('Heat demand consumer:', heat_consumer.P.val)
print('network losses at -10 °C outside temperature:', heat_losses.P.val)

df = pd.read_csv(
    'C:/Users/Ionap/OneDrive/Documents/Edinburgh/Dissertation/network test.csv')

# offdesign case: -20 °C ambient temperature

for comp in nw.comps['object']:
    if isinstance(comp, Pipe):
        comp.set_attr(Tamb=-20)

nw.solve('offdesign', design_path='grid')
print('Heat demand consumer:', heat_consumer.P.val)
print('network losses at -20 °C outside temperature:', heat_losses.P.val)

df = pd.read_csv(
    'C:/Users/Ionap/OneDrive/Documents/Edinburgh/Dissertation/network test.csv')

losses = []

for index, row in df.iterrows():
    T = row['AirTemp']
    Q = row['Dem']
    print(T)
    print(Q)

    LIAC.set_attr(Q=-0.11e3*Q, offdesign =['Q'])
    Roslin.set_attr(Q=-0.5e3*Q, offdesign =['Q'])
    Greenwood.set_attr(Q=-0.14e3*Q, offdesign =['Q'])
    Centre.set_attr(Q=-0.25e3*Q, offdesign =['Q'])

    for comp in nw.comps['object']:
        if isinstance(comp, Pipe):
            comp.set_attr(Tamb=T)

    nw.solve(mode = 'offdesign',init_path='grid', design_path='grid')
    print(heat_losses.P.val)
    losses += [heat_losses.P.val]
    print(losses)

df_L = pd.DataFrame({'losses': losses})
df_L.to_csv('network losses 90.csv')
