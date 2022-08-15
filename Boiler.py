from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import pandas as pd

from tespy.networks import Network
from tespy.components import (
    Sink, Source, Compressor, Turbine, Condenser, CombustionChamber, Pump,
    HeatExchanger, Drum, CycleCloser)
from tespy.connections import Connection, Bus, Ref
from tespy.tools import document_model

from tespy.tools import CharLine
import numpy as np

# %% network
fluid_list = ['Ar', 'N2', 'O2', 'CO2', 'CH4', 'H2O']

nw = Network(fluids=fluid_list, p_unit='bar', T_unit='C', h_unit='kJ / kg')

# %% components

gt_out = Source('gas flue')

# waste heat recovery
suph = HeatExchanger('superheater')
evap = HeatExchanger('evaporator')
dr = Drum('drum')
eco = HeatExchanger('economizer')
dh_whr = HeatExchanger('waste heat recovery')
ch = Sink('chimney')

# steam turbine part
turb = Turbine('steam turbine')
cond = Condenser('condenser')
pu = Pump('feed water pump')
cc = CycleCloser('ls cycle closer')

# district heating
dh_in = Source('district heating backflow')
dh_out = Sink('district heating feedflow')

# %% connections

gt_eco = Connection(gt_out, 'out1', eco,'in1')

nw.add_conns(gt_eco)

# waste heat recovery (flue gas side)

eco_dh = Connection(eco, 'out1', dh_whr, 'in1')
dh_ch = Connection(dh_whr, 'out1', ch, 'in1')

nw.add_conns( eco_dh, dh_ch)


# steam turbine
eco_ls = Connection(eco, 'out2', cc, 'in1')
ls = Connection(cc, 'out1', cond, 'in1')
c_p = Connection(cond, 'out1', pu, 'in1')
fw = Connection(pu, 'out1', eco, 'in2')

nw.add_conns(eco_ls, ls, c_p, fw)

# district heating
dh_c = Connection(dh_in, 'out1', cond, 'in2')
dh_i = Connection(cond, 'out2', dh_whr, 'in2')
dh_w = Connection(dh_whr, 'out2', dh_out, 'in1')

nw.add_conns(dh_c, dh_i, dh_w)


# characteristic function for generator efficiency
x = np.array([ 0,0.2, 0.4, 0.6, 0.8, 1, 1.2])
y = np.array([ 0.5, 0.86, 0.9, 0.93, 0.95, 0.96, 0.95])

char = CharLine(x=x, y=y)

# %% busses
#elec = Bus('power output')
#elec.add_comps({'comp': g_turb, 'char': char}, {'comp': comp, 'char': char, 'base': 'bus'}, {'comp': turb, 'char': char}, {'comp': pu, 'char': char, 'base': 'bus'})

heat_out = Bus('heat output')
heat_out.add_comps({'comp': cond, 'char':char}, {'comp': dh_whr, 'char':char})

#heat_in = Bus('heat input')
#heat_in.add_comps({'comp': c_c})

nw.add_busses(heat_out)

# %% component parameters


# steam turbine
#suph.set_attr(pr1=0.99, pr2=0.834, design=['pr1', 'pr2'], offdesign=['zeta1', 'zeta2', 'kA_char'])
eco.set_attr(pr1=0.99, pr2=1, design=['pr1'], offdesign=['zeta1', 'zeta2', 'kA_char'])
#evap.set_attr(pr1=0.99, pr2 = ttd_l=25, design=['pr1', 'ttd_l'], offdesign=['zeta1', 'kA_char'])
dh_whr.set_attr(pr1=0.99, pr2=0.98, design=['pr1', 'pr2'], offdesign=['zeta1', 'zeta2', 'kA_char'])
cond.set_attr(pr1=0.99, pr2=0.98, design=['pr2'], offdesign=['pr1','zeta2', 'kA_char'])
pu.set_attr(eta_s=0.75, design=['eta_s'], offdesign=['eta_s_char'])

# %% connection parameters

# gas turbine
gt_eco.set_attr(
    T= 300, m =4.1,h0=1000, fluid={
        'CO2': 0.016497, 'Ar': 0.012822, 'N2': 0.750752, 'O2': 0.206946, 'H2O': 0.012983, 'CH4': 0
    }, design =['m']
)

# waste heat recovery
eco_dh.set_attr(T=200, design=['T'], p0=1)
dh_ch.set_attr(T=100, design=['T'], p=1)

eco_ls.set_attr(
    p=10, T=290, fluid={
        'CO2': 0, 'Ar': 0, 'N2': 0, 'O2': 0, 'H2O': 1, 'CH4': 0
    }, design=['p', 'T']
)

# district heating
dh_c.set_attr(T=60, p=5, fluid={'CO2': 0, 'Ar': 0, 'N2': 0, 'O2': 0, 'H2O': 1, 'CH4': 0 })
dh_w.set_attr(T=90)

# %%
nw.solve(mode='design')
#nw.print_results()
#print(fuel.m.val)
nw.save('design_point')
#document_model(nw, filename='report_design.tex')
print(heat_out.P.val)


#heat_out.set_attr( P = -811.4043036e3)

#nw.solve(mode='offdesign', init_path='design_point',
     #   design_path='design_point')
#print(gt_eco.m.val)

#document_model(nw, filename='report_offdesign.tex')


mode = 'design'
gt_eco.set_attr(m=5.2)
nw.set_attr(iterinfo=False)
nw.solve(mode=mode)
nw.save('boiler_1.04')

gt_eco.set_attr(m=5)
nw.solve(mode=mode)
nw.save('boiler_1')

gt_eco.set_attr(m=4.8)
nw.solve(mode=mode)
nw.save('boiler_0.96')

gt_eco.set_attr(m=4.5)
nw.solve(mode=mode)
nw.save('boiler_0.9')

gt_eco.set_attr(m=4.4)
nw.solve(mode=mode)
nw.save('boiler_0.878')

gt_eco.set_attr(m=4.3)
nw.solve(mode=mode)
nw.save('boiler_0.86')

gt_eco.set_attr(m=4)
nw.solve(mode=mode)
nw.save('boiler_0.8')

gt_eco.set_attr(m=3.7)
nw.solve(mode=mode)
nw.save('boiler_0.738')

gt_eco.set_attr(m=3.5)
nw.solve(mode=mode)
nw.save('boiler_0.7')

gt_eco.set_attr(m=3.4)
nw.solve(mode=mode)
nw.save('boiler_0.68')

gt_eco.set_attr(m=3.3)
nw.solve(mode=mode)
nw.save('boiler_0.66')

gt_eco.set_attr(m=3.2)
nw.solve(mode=mode)
nw.save('boiler_0.64')

gt_eco.set_attr(m=3.1)
nw.solve(mode=mode)
nw.save('boiler_0.62')

gt_eco.set_attr(m=3)
nw.solve(mode=mode)
nw.save('boiler_0.6')

gt_eco.set_attr(m=2.9)
nw.solve(mode=mode)
nw.save('boiler_0.578')

gt_eco.set_attr(m=2.8)
nw.solve(mode=mode)
nw.save('boiler_0.558')

gt_eco.set_attr(m=2.7)
nw.solve(mode=mode)
nw.save('boiler_0.538')

gt_eco.set_attr(m=2.6)
nw.solve(mode=mode)
nw.save('boiler_0.52')

gt_eco.set_attr(m=2.5)
nw.solve(mode=mode)
nw.save('boiler_0.5')

gt_eco.set_attr(m=2.4)
nw.solve(mode=mode)
nw.save('boiler_0.48')

gt_eco.set_attr(m=2.3)
nw.solve(mode=mode)
nw.save('boiler_0.46')

gt_eco.set_attr(m=2.2)
nw.solve(mode=mode)
nw.save('boiler_0.439')

gt_eco.set_attr(m=2.1)
nw.solve(mode=mode)
nw.save('boiler_0.42')

gt_eco.set_attr(m=2)
nw.solve(mode=mode)
nw.save('boiler_0.4')

gt_eco.set_attr(m=1.9)
nw.solve(mode=mode)
nw.save('boiler_0.38')

gt_eco.set_attr(m=1.8)
nw.solve(mode=mode)
nw.save('boiler_0.36')

gt_eco.set_attr(m=1.7)
nw.solve(mode=mode)
nw.save('boiler_0.34')

gt_eco.set_attr(m=1.6)
nw.solve(mode=mode)
nw.save('boiler_0.32')

gt_eco.set_attr(m=1.5)
nw.solve(mode=mode)
nw.save('boiler_0.3')

gt_eco.set_attr(m=1.4)
nw.solve(mode=mode)
nw.save('boiler_0.28')

gt_eco.set_attr(m=1.3)
nw.solve(mode=mode)
nw.save('boiler_0.26')

gt_eco.set_attr(m=1.2)
nw.solve(mode=mode)
nw.save('boiler_0.24')

gt_eco.set_attr(m=1.1)
nw.solve(mode=mode)
nw.save('boiler_0.22')

gt_eco.set_attr(m=1)
nw.solve(mode=mode)
nw.save('boiler_0.205')


mode = 'offdesign'

df = pd.read_csv('CHP_after_store.csv')

mass = []

for i in np.array(df['0'].tolist()) * -1:
    heat_out.set_attr(P=i)
    if  i < -1.03e6:
        nw.solve(mode=mode, init_path='boiler_1.04', design_path='boiler_1.04')

        m = gt_eco.m.val
        print(m)
    elif -0.996e6 > i >= -1.03e6:
        nw.solve(mode=mode, init_path='boiler_1', design_path='boiler_1')

        m = gt_eco.m.val
        print(m)
    elif -0.93e6 > i >= -0.996e6:
        nw.solve(mode=mode, init_path='boiler_0.96', design_path='boiler_0.96')

        m = gt_eco.m.val
        print(m)
    elif -0.89e6 > i >= -0.93e6:
        nw.solve(mode=mode, init_path='boiler_0.9', design_path='boiler_0.9')

        m = gt_eco.m.val
        print(m)
    elif -0.876e6 > i >= -0.89e6:
        nw.solve(mode=mode, init_path='boiler_0.878', design_path='boiler_0.878')

        m = gt_eco.m.val
        print(m)
    elif -0.83e6 > i >= -0.876e6:
        nw.solve(mode=mode, init_path='boiler_0.86', design_path='boiler_0.86')

        m = gt_eco.m.val
        print(m)

    elif -0.76e6 > i >= -0.83e6:
        nw.solve(mode=mode, init_path='boiler_0.8', design_path='boiler_0.8')

        m = gt_eco.m.val
        print(m)
    elif -0.72e6 > i >= -0.76e6:
        nw.solve(mode=mode, init_path='boiler_0.738', design_path='boiler_0.738')

        m = gt_eco.m.val
        print(m)
    elif -0.69e6 > i >= -0.72e6:
        nw.solve(mode=mode, init_path='boiler_0.7', design_path='boiler_0.7')

        m = gt_eco.m.val
        print(m)
    elif -0.67e6 > i >= -0.69e6:
        nw.solve(mode=mode, init_path='boiler_0.68', design_path='boiler_0.68')

        m = gt_eco.m.val
        print(m)
    elif -0.65e6 > i >= -0.67e6:
        nw.solve(mode=mode, init_path='boiler_0.66', design_path='boiler_0.66')

        m = gt_eco.m.val
        print(m)
    elif -0.63e6 > i >= -0.65e6:
        nw.solve(mode=mode, init_path='boiler_0.64', design_path='boiler_0.64')

        m = gt_eco.m.val
        print(m)
    elif -0.61e6 > i >= -0.63e6:
        nw.solve(mode=mode, init_path='boiler_0.62', design_path='boiler_0.62')

        m = gt_eco.m.val
        print(m)
    elif -0.59e6 > i >= -0.61e6:
        nw.solve(mode=mode, init_path='boiler_0.6', design_path='boiler_0.6')

        m = gt_eco.m.val
        print(m)
    elif -0.573e6 > i >= -0.59e6:
        nw.solve(mode=mode, init_path='boiler_0.578', design_path='boiler_0.578')

        m = gt_eco.m.val
        print(m)
    elif -0.55e6 > i >= -0.573e6:
        nw.solve(mode=mode, init_path='boiler_0.558', design_path='boiler_0.558')

        m = gt_eco.m.val
        print(m)
    elif -0.53e6 > i >= -0.55e6:
        nw.solve(mode=mode, init_path='boiler_0.538', design_path='boiler_0.538')

        m = gt_eco.m.val
        print(m)
    elif -0.52e6 > i >= -0.53e6:
        nw.solve(mode=mode, init_path='boiler_0.52', design_path='boiler_0.52')

        m = gt_eco.m.val
        print(m)
    elif -0.49e6 > i >= -0.51e6:
        nw.solve(mode=mode, init_path='boiler_0.5', design_path='boiler_0.5')

        m = gt_eco.m.val
        print(m)
    elif -0.473e6 > i >= -0.49e6:
        nw.solve(mode=mode, init_path='boiler_0.48', design_path='boiler_0.48')

        m = gt_eco.m.val
        print(m)
    elif -0.45e6 > i >= -0.473e6:
        nw.solve(mode=mode, init_path='boiler_0.46', design_path='boiler_0.46')

        m = gt_eco.m.val
        print(m)
    elif -0.43e6 > i >= -0.45e6:
        nw.solve(mode=mode, init_path='boiler_0.439', design_path='boiler_0.439')

        m = gt_eco.m.val
        print(m)
    elif -0.41e6 > i >= -0.43e6:
        nw.solve(mode=mode, init_path='boiler_0.42', design_path='boiler_0.42')

        m = gt_eco.m.val
        print(m)
    elif -0.391e6 > i >= -0.41e6:
        nw.solve(mode=mode, init_path='boiler_0.4', design_path='boiler_0.4')

        m = gt_eco.m.val
        print(m)
    elif -0.37e6 > i >= -0.391e6:
        nw.solve(mode=mode, init_path='boiler_0.38', design_path='boiler_0.38')

        m = gt_eco.m.val
        print(m)

    elif -0.35e6 > i >= -0.37e6:
        nw.solve(mode=mode, init_path='boiler_0.36', design_path='boiler_0.36')

        m = gt_eco.m.val
        print(m)
    elif -0.332e6 > i >= -0.35e6:
        nw.solve(mode=mode, init_path='boiler_0.34', design_path='boiler_0.34')

        m = gt_eco.m.val
        print(m)
    elif -0.318e6 > i >= -0.332e6:
        nw.solve(mode=mode, init_path='boiler_0.32', design_path='boiler_0.32')

        m = gt_eco.m.val
        print(m)
    elif -0.29e6 > i >= -0.318e6:
        nw.solve(mode=mode, init_path='boiler_0.3', design_path='boiler_0.3')

        m = gt_eco.m.val
        print(m)
    elif -0.27e6 > i >= -0.29e6:
        nw.solve(mode=mode, init_path='boiler_0.28', design_path='boiler_0.28')

        m = gt_eco.m.val
        print(m)
    elif -0.25e6 > i >= -0.27e6:
        nw.solve(mode=mode, init_path='boiler_0.26', design_path='boiler_0.26')

        m = gt_eco.m.val
        print(m)
    elif -0.23e6 > i >= -0.25e6:
        nw.solve(mode=mode, init_path='boiler_0.24', design_path='boiler_0.24')

        m = gt_eco.m.val
        print(m)
    elif -0.21e6 > i >= -0.23e6:
        nw.solve(mode=mode, init_path='boiler_0.22', design_path='boiler_0.22')

        m = gt_eco.m.val
        print(m)
    elif -0.2e6 > i >= -0.21e6:
        nw.solve(mode=mode, init_path='boiler_0.205', design_path='boiler_0.205')

        m = gt_eco.m.val
        print(m)
    elif i>=-0.2e6:
        m = 0
        print(m)
    mass += [m]

df_m = pd.DataFrame({'mass': mass})
df_m.to_csv('mass_store1.csv')


