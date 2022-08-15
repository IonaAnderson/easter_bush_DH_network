# -*- coding: utf-8 -*-
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
# gas turbine part
comp = Compressor('compressor')
c_c = CombustionChamber('combustion')
g_turb = Turbine('gas turbine')

CH4 = Source('fuel source')
air = Source('ambient air')

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
# gas turbine part
c_in = Connection(air, 'out1', comp, 'in1')
c_out = Connection(comp, 'out1', c_c, 'in1')
fuel = Connection(CH4, 'out1', c_c, 'in2')
gt_in = Connection(c_c, 'out1', g_turb, 'in1')
gt_out = Connection(g_turb, 'out1', suph, 'in1')

nw.add_conns(c_in, c_out, fuel, gt_in, gt_out)

# waste heat recovery (flue gas side)
suph_evap = Connection(suph, 'out1', evap, 'in1')
evap_eco = Connection(evap, 'out1', eco, 'in1')
eco_dh = Connection(eco, 'out1', dh_whr, 'in1')
dh_ch = Connection(dh_whr, 'out1', ch, 'in1')

nw.add_conns(suph_evap, evap_eco, eco_dh, dh_ch)

# waste heat recovery (water side)
eco_drum = Connection(eco, 'out2', dr, 'in1')
drum_evap = Connection(dr, 'out1', evap, 'in2')
evap_drum = Connection(evap, 'out2', dr, 'in2')
drum_suph = Connection(dr, 'out2', suph, 'in2')

nw.add_conns(eco_drum, drum_evap, evap_drum, drum_suph)

# steam turbine
suph_ls = Connection(suph, 'out2', cc, 'in1')
ls = Connection(cc, 'out1', turb, 'in1')
ws = Connection(turb, 'out1', cond, 'in1')
c_p = Connection(cond, 'out1', pu, 'in1')
fw = Connection(pu, 'out1', eco, 'in2')

nw.add_conns(suph_ls, ls, ws, c_p, fw)

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
elec = Bus('power output')
elec.add_comps({'comp': g_turb, 'char': char}, {'comp': comp, 'char': char, 'base': 'bus'}, {'comp': turb, 'char': char}, {'comp': pu, 'char': char, 'base': 'bus'})

heat_out = Bus('heat output')
heat_out.add_comps({'comp': cond, 'char':char}, {'comp': dh_whr, 'char':char})

heat_in = Bus('heat input')
heat_in.add_comps({'comp': c_c})

nw.add_busses(elec, heat_out, heat_in)

# %% component parameters
# gas turbine
comp.set_attr(pr=14, eta_s=0.91, design=['pr', 'eta_s'], offdesign=['eta_s_char'])
g_turb.set_attr(eta_s=0.9, design=['eta_s'], offdesign=['eta_s_char', 'cone'])

# steam turbine
suph.set_attr(pr1=0.99, pr2=0.834, design=['pr1', 'pr2'], offdesign=['zeta1', 'zeta2', 'kA_char'])
eco.set_attr(pr1=0.99, pr2=1, design=['pr1', 'pr2'], offdesign=['zeta1', 'zeta2', 'kA_char'])
evap.set_attr(pr1=0.99, ttd_l=25, design=['pr1', 'ttd_l'], offdesign=['zeta1', 'kA_char'])
dh_whr.set_attr(pr1=0.99, pr2=0.98, design=['pr1', 'pr2'], offdesign=['zeta1', 'zeta2', 'kA_char'])
turb.set_attr(eta_s=0.9, design=['eta_s'], offdesign=['eta_s_char', 'cone'])
cond.set_attr(pr1=0.99, pr2=0.98, design=['pr2'], offdesign=['zeta2', 'kA_char'])
pu.set_attr(eta_s=0.75, design=['eta_s'], offdesign=['eta_s_char'])

#c_c.set_attr(lamb=17)

# %% connection parameters

# gas turbine
c_in.set_attr(
    T=20, p=1,m=3.327, fluid={
        'Ar': 0.0129, 'N2': 0.7553, 'H2O': 0, 'CH4': 0, 'CO2': 0.0004,
        'O2': 0.2314
    }, design=['m']
)
gt_in.set_attr(T=1200)
gt_out.set_attr(p0=1)
fuel.set_attr(
    T=Ref(c_in, 1, 0), h0=800, fluid={
        'CO2': 0.04, 'Ar': 0, 'N2': 0, 'O2': 0, 'H2O': 0, 'CH4': 0.96
    }
)

# waste heat recovery
eco_dh.set_attr(T=290, design=['T'], p0=1)
dh_ch.set_attr(T=100, design=['T'], p=1)

# steam turbine
evap_drum.set_attr(m=Ref(drum_suph, 4, 0))
suph_ls.set_attr(
    p=100, T=550, fluid={
        'CO2': 0, 'Ar': 0, 'N2': 0, 'O2': 0, 'H2O': 1, 'CH4': 0
    }, design=['p', 'T']
)
ws.set_attr(p=0.8, design=['p'])

# district heating
dh_c.set_attr(T=60, p=5, fluid={'CO2': 0, 'Ar': 0, 'N2': 0, 'O2': 0, 'H2O': 1, 'CH4': 0 })
dh_w.set_attr(T=90)

# %%
#nw.solve(mode='design')
#nw.print_results()
#print(fuel.m.val)
#nw.save('design_point')
#document_model(nw, filename='report_design.tex')


#elec.set_attr( P = -1.3e6)

#nw.solve(mode='offdesign', init_path='design_point',
 #        design_path='design_point')
#print((list(suph_evap.fluid.val.items())[2])[1])
#print(abs(suph_evap.m.val*(list(suph_evap.fluid.val.items())[2])[1]))

#document_model(nw, filename='report_offdesign.tex')

# %% solving
mode = 'design'
nw.set_attr(iterinfo=False)
nw.solve(mode=mode)
nw.save('cc_bp_1.5')
document_model(nw, filename='report_design.tex')

c_in.set_attr(m=3)
nw.solve(mode=mode)
nw.save('cc_bp_1.41')

c_in.set_attr(m=2.75)
nw.solve(mode=mode)
nw.save('cc_bp_1.27')

c_in.set_attr(m=2.5)
nw.solve(mode=mode)
nw.save('cc_bp_1.16')

c_in.set_attr(m=2)
nw.solve(mode=mode)
nw.save('cc_bp_0.93')

c_in.set_attr(m=1.9)
nw.solve(mode=mode)
nw.save('cc_bp_0.88')

c_in.set_attr(m=1.8)
nw.solve(mode=mode)
nw.save('cc_bp_0.83')

c_in.set_attr(m=1.75)
nw.solve(mode=mode)
nw.save('cc_bp_0.81')

c_in.set_attr(m=1.7)
nw.solve(mode=mode)
nw.save('cc_bp_0.79')

c_in.set_attr(m=1.6)
nw.solve(mode=mode)
nw.save('cc_bp_0.74')

c_in.set_attr(m=1.5)
nw.solve(mode=mode)
nw.save('cc_bp_0.7')

c_in.set_attr(m=1.4)
nw.solve(mode=mode)
nw.save('cc_bp_0.65')

c_in.set_attr(m=1.3)
nw.solve(mode=mode)
nw.save('cc_bp_0.60')

c_in.set_attr(m=1.25)
nw.solve(mode=mode)
nw.save('cc_bp_0.58')

c_in.set_attr(m=1.2)
nw.solve(mode=mode)
nw.save('cc_bp_0.56')

c_in.set_attr(m=1.1)
nw.solve(mode=mode)
nw.save('cc_bp_0.51')

c_in.set_attr(m=1)
nw.solve(mode=mode)
nw.save('cc_bp_0.46')

c_in.set_attr(m=0.95)
nw.solve(mode=mode)
nw.save('cc_bp_0.44')

c_in.set_attr(m=0.9)
nw.solve(mode=mode)
nw.save('cc_bp_0.42')

c_in.set_attr(m=0.85)
nw.solve(mode=mode)
nw.save('cc_bp_0.39')

c_in.set_attr(m=0.8)
nw.solve(mode=mode)
nw.save('cc_bp_0.37')

c_in.set_attr(m=0.75)
nw.solve(mode=mode)
nw.save('cc_bp_0.35')

c_in.set_attr(m=0.7)
nw.solve(mode=mode)
nw.save('cc_bp_0.32')

c_in.set_attr(m=0.65)
nw.solve(mode=mode)
nw.save('cc_bp_0.3')

c_in.set_attr(m=0.6)
nw.solve(mode=mode)
nw.save('cc_bp_0.28')

c_in.set_attr(m=0.55)
nw.solve(mode=mode)
nw.save('cc_bp_0.25')

c_in.set_attr(m=0.5)
nw.solve(mode=mode)
nw.save('cc_bp_0.23')

c_in.set_attr(m=0.45)
nw.solve(mode=mode)
nw.save('cc_bp_0.208')

c_in.set_attr(m=0.4)
nw.solve(mode=mode)
nw.save('cc_bp_0.185')

c_in.set_attr(m=0.35)
nw.solve(mode=mode)
nw.save('cc_bp_0.162')

c_in.set_attr(m=0.3)
nw.solve(mode=mode)
nw.save('cc_bp_0.129')

c_in.set_attr(m=0.25)
nw.solve(mode=mode)
nw.save('cc_bp_0.115')

mode = 'offdesign'

df = pd.read_csv(
    'C:/Users/Ionap/OneDrive/Documents/Edinburgh/Dissertation/AMR_Data_for_meter_0795NE003V_Easter Bush Elec.csv')

Power = []
Heat = []
CO2 = []
gas = []

for i in np.array(df['Power'].tolist()) *-1:
    elec.set_attr(P=i)

    if i >= -0.1e6:
        power = 0
        print('Power generation: ' + '{:.3f}'.format(0))
        heat = 0
        # print(
        #  'Heat generation: ' +
        # '{:.3f}'.format(abs(0)))
        co2 = 0
        print('C02 emissions' + '{:.3f}'.format(abs(0)))
        Gas = 0

    elif i < -1.40e6:
        nw.solve(mode=mode, init_path='cc_bp_1.41', design_path='cc_bp_1.5')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #    'Heat generation: ' +
        #   '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))

        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val
    elif -1.33e6 > i >= -1.40e6:
        nw.solve(mode=mode, init_path='cc_bp_1.41', design_path='cc_bp_1.41')


        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #    'Heat generation: ' +
        #   '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))

        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val
    elif -1.19e6 > i >= -1.33e6:
        nw.solve(mode=mode, init_path='cc_bp_1.27', design_path='cc_bp_1.27')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #    'Heat generation: ' +
        #   '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))

        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.95e6 > i >= -1.19e6:
        nw.solve(mode=mode, init_path='cc_bp_1.16', design_path='cc_bp_1.16')

        power = i

        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #  'Heat generation: ' +
        #   '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.91e6 > i >= -0.95e6:
        nw.solve(mode=mode, init_path='cc_bp_0.93', design_path='cc_bp_0.93')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.85e6 > i >= -0.91e6:
        nw.solve(mode=mode, init_path='cc_bp_0.88', design_path='cc_bp_0.88')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.82e6 > i >= -0.85e6:
        nw.solve(mode=mode, init_path='cc_bp_0.83', design_path='cc_bp_0.83')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.80e6 > i >= -0.82e6:
        nw.solve(mode=mode, init_path='cc_bp_0.81', design_path='cc_bp_0.81')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val
    elif -0.76e6 > i >= -0.80e6:
        nw.solve(mode=mode, init_path='cc_bp_0.79', design_path='cc_bp_0.79')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.72e6 > i >= -0.76e6:
        nw.solve(mode=mode, init_path='cc_bp_0.74', design_path='cc_bp_0.74')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val
    elif -0.67e6 > i >= -0.72e6:
        nw.solve(mode=mode, init_path='cc_bp_0.7', design_path='cc_bp_0.7')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.62e6 > i >= -0.67e6:
        nw.solve(mode=mode, init_path='cc_bp_0.65', design_path='cc_bp_0.65')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val
    elif -0.59e6 > i >= -0.62e6:
        nw.solve(mode=mode, init_path='cc_bp_0.60', design_path='cc_bp_0.60')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.57e6 > i >= -0.59e6:
        nw.solve(mode=mode, init_path='cc_bp_0.58', design_path='cc_bp_0.58')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.53e6 > i >= -0.57e6:
        nw.solve(mode=mode, init_path='cc_bp_0.56', design_path='cc_bp_0.56')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val
    elif -0.48e6 > i >= -0.53e6:
        nw.solve(mode=mode, init_path='cc_bp_0.51', design_path='cc_bp_0.51')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val
    elif -0.45e6 > i >= -0.48e6:
        nw.solve(mode=mode, init_path='cc_bp_0.46', design_path='cc_bp_0.46')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.43e6 > i >= -045e6:
        nw.solve(mode=mode, init_path='cc_bp_0.44', design_path='cc_bp_0.44')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.405e6 > i >= -0.43e6:
        nw.solve(mode=mode, init_path='cc_bp_0.42', design_path='cc_bp_0.42')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.38e6 > i >= -0.405e6:
        nw.solve(mode=mode, init_path='cc_bp_0.39', design_path='cc_bp_0.39')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.36e6 > i >= -0.38e6:
        nw.solve(mode=mode, init_path='cc_bp_0.37', design_path='cc_bp_0.37')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.335e6 > i >= -0.36e6:
        nw.solve(mode=mode, init_path='cc_bp_0.35', design_path='cc_bp_0.35')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))

    elif -0.31e6 > i >= -0.335e6:
        nw.solve(mode=mode, init_path='cc_bp_0.32', design_path='cc_bp_0.32')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))

    elif -0.29e6 > i >= -0.31e6:
        nw.solve(mode=mode, init_path='cc_bp_0.3', design_path='cc_bp_0.3')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.265e6 > i >= -0.2648e6:
        nw.solve(mode=mode, init_path='cc_bp_0.28', design_path='cc_bp_0.28')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.24e6 > i >= -0.2648e6:
        nw.solve(mode=mode, init_path='cc_bp_0.25', design_path='cc_bp_0.25')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.22e6 > i >= -0.24e6:
        nw.solve(mode=mode, init_path='cc_bp_0.23', design_path='cc_bp_0.23')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.197e6 > i >= -0.22e6:
        nw.solve(mode=mode, init_path='cc_bp_0.208', design_path='cc_bp_0.208')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.17e6 > i >= -0.197e6:
        nw.solve(mode=mode, init_path='cc_bp_0.185', design_path='cc_bp_0.185')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    elif -0.14e6 > i >= -0.17e6:
        nw.solve(mode=mode, init_path='cc_bp_0.162', design_path='cc_bp_0.162')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val
    elif -0.122e6 > i >= -0.14e6:
        nw.solve(mode=mode, init_path='cc_bp_0.129', design_path='cc_bp_0.129')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val
    elif -1e6 > i >= -0.122e6:
        nw.solve(mode=mode, init_path='cc_bp_0.115', design_path='cc_bp_0.115')

        power = i
        print('Power generation: ' + '{:.3f}'.format(abs(i)))
        heat = abs(heat_out.P.val)
        # print(
        #   'Heat generation: ' +
        #  '{:.3f}'.format(abs((dh_whr.Q.val + cond.Q.val) / c_c.ti.val)))
        co2 = abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])
        print('C02 emissions' + '{:.3f}'.format(abs(dh_ch.m.val * (list(dh_ch.fluid.val.items())[2])[1])))
        Gas = fuel.m.val

    Power += [power]
    Heat += [heat]
    CO2 += [co2]
    gas += [Gas]

    df_P = pd.DataFrame({'Power': Power})
    df_H = pd.DataFrame({'Heat': Heat})
    df_C = pd.DataFrame({'CO2': CO2})
    gas += [Gas]

print(CO2)

df_P.to_csv('Power.csv')
df_H.to_csv('Heat.csv')
df_C.to_csv('CO2.csv')
df_g.to_csv('gas_chp.csv')




