import numpy as np
import pandas as pd

from tespy.components import (Subsystem,Sink, Source, Valve, Turbine,
                              Splitter, Merge, Condenser, Pipe, Pump,
                              HeatExchangerSimple, CycleCloser)

from tespy.connections import Connection, Bus, Ref
from tespy.tools import CharLine
from tespy.tools.helpers import TESPyComponentError

class LinConsumOpen(Subsystem):

    def __init__(self, label, num_consumer):

        if not isinstance(label, str):
            msg = 'Subsystem label must be of type str!'
            logging.error(msg)
            raise ValueError(msg)

        elif len([x for x in [';', ', ', '.'] if x in label]) > 0:
            msg = 'Can\'t use ' + str([';', ', ', '.']) + ' in label.'
            logging.error(msg)
            raise ValueError(msg)
        else:
            self.label = label

        if num_consumer <= 1:
            raise TESPyComponentError('Minimum number of consumers is 2.')
        else:
            self.num_consumer = num_consumer

        self.comps = {}
        self.conns = {}
        self.create_comps()
        self.create_conns()

    def create_comps(self):

        for i in range(self.num_consumer - 1):
            j = str(i)
            self.comps['feed_' + j] = Pipe(self.label + '_pipe feed_' + j)
            self.comps['return_' + j] = Pipe(self.label + '_pipe return_' + j)

        for i in range(self.num_consumer):
            j = str(i)
            self.comps['splitter_' + j] = Splitter(self.label + '_splitter_' + j)
            self.comps['merge_' + j] = Merge(self.label + '_merge_' + j)
            self.comps['consumer_' + j] = HeatExchangerSimple(self.label + '_consumer_' + j)
            self.comps['valve_' + j] = Valve(self.label + '_valve_' + j)

    def create_conns(self):

        for i in range(self.num_consumer):
            j = str(i)
            if i > 0:
                self.conns['fesp_' + j] = Connection(self.comps['feed_' + str(i - 1)], 'out1', self.comps['splitter_' + j], 'in1')
                self.conns['mere_' + j] = Connection(self.comps['merge_' + j], 'out1', self.comps['return_' + str(i - 1)], 'in1')

            self.conns['spco_' + j] = Connection(self.comps['splitter_' + j], 'out1', self.comps['consumer_' + j], 'in1')
            self.conns['cova_' + j] = Connection(self.comps['consumer_' + j], 'out1', self.comps['valve_' + j], 'in1')
            self.conns['vame_' + j] = Connection(self.comps['valve_' + j], 'out1', self.comps['merge_' + j], 'in2')

            if i < self.num_consumer - 1:
                self.conns['spfe_' + j] = Connection(self.comps['splitter_' + j], 'out2', self.comps['feed_' + j], 'in1')
                self.conns['reme_' + j] = Connection(self.comps['return_' + j], 'out1', self.comps['merge_' + j], 'in1')


class LinConsumClosed(Subsystem):

    def __init__(self, label, num_consumer):

        if not isinstance(label, str):
            msg = 'Subsystem label must be of type str!'
            logging.error(msg)
            raise ValueError(msg)

        elif len([x for x in [';', ', ', '.'] if x in label]) > 0:
            msg = 'Can\'t use ' + str([';', ', ', '.']) + ' in label.'
            logging.error(msg)
            raise ValueError(msg)
        else:
            self.label = label

        if num_consumer <= 1:
            raise TESPyComponentError('Minimum number of consumers is 2.')
        else:
            self.num_consumer = num_consumer

        self.comps = {}
        self.conns = {}
        self.create_comps()
        self.create_conns()

    def create_comps(self):

        for i in range(self.num_consumer - 1):
            j = str(i)
            self.comps['splitter_' + j] = Splitter(self.label + '_splitter_' + j)
            self.comps['merge_' + j] = Merge(self.label + '_merge_' + j)
            self.comps['consumer_' + j] = HeatExchangerSimple(self.label + '_consumer_' + j)
            self.comps['valve_' + j] = Valve(self.label + '_valve_' + j)
            self.comps['feed_' + j] = Pipe(self.label + '_pipe feed_' + j)
            self.comps['return_' + j] = Pipe(self.label + '_pipe return_' + j)

        j = str(i + 1)
        self.comps['consumer_' + j] = HeatExchangerSimple(self.label + '_consumer_' + j)

    def create_conns(self):

        for i in range(self.num_consumer - 1):
            j = str(i)

            if i > 0:
                self.conns['fesp_' + j] = Connection(self.comps['feed_' + str(i - 1)], 'out1', self.comps['splitter_' + j], 'in1')
                self.conns['mere_' + j] = Connection(self.comps['merge_' + j], 'out1', self.comps['return_' + str(i - 1)], 'in1')

            self.conns['spco_' + j] = Connection(self.comps['splitter_' + j], 'out1', self.comps['consumer_' + j], 'in1')
            self.conns['cova_' + j] = Connection(self.comps['consumer_' + j], 'out1', self.comps['valve_' + j], 'in1')
            self.conns['vame_' + j] = Connection(self.comps['valve_' + j], 'out1', self.comps['merge_' + j], 'in2')
            self.conns['spfe_' + j] = Connection(self.comps['splitter_' + j], 'out2', self.comps['feed_' + j], 'in1')
            self.conns['reme_' + j] = Connection(self.comps['return_' + j], 'out1', self.comps['merge_' + j], 'in1')

        self.conns['spco_' + str(i + 1)] = Connection(self.comps['feed_' + j], 'out1', self.comps['consumer_' + str(i + 1)], 'in1')
        self.conns['cova_' + str(i + 1)] = Connection(self.comps['consumer_' + str(i + 1)], 'out1', self.comps['return_' + j], 'in1')


class Fork(Subsystem):

    def __init__(self, label, num_branch):

        if not isinstance(label, str):
            msg = 'Subsystem label must be of type str!'
            logging.error(msg)
            raise ValueError(msg)

        elif len([x for x in [';', ', ', '.'] if x in label]) > 0:
            msg = 'Can\'t use ' + str([';', ', ', '.']) + ' in label.'
            logging.error(msg)
            raise ValueError(msg)
        else:
            self.label = label

        if num_branch <= 1:
            raise TESPyComponentError('Minimum number of branches is 2.')
        else:
            self.num_branch = num_branch

        self.comps = {}
        self.conns = {}
        self.create_comps()
        self.create_conns()

    def create_comps(self):

        self.comps['splitter'] = Splitter(self.label + '_splitter')
        self.comps['merge'] = Merge(self.label + '_merge')

        for i in range(self.num_branch):
            j = str(i)
            self.comps['valve_' + j] = Valve(self.label + '_valve_' + j)

    def create_conns(self):

        for i in range(self.num_branch):
            j = str(i)
            k = str(i + 1)
            self.conns['vame_' + j] = Connection(self.comps['valve_' + j], 'out1', self.comps['merge'], 'in' + k)


class Building(Subsystem):

    def __init__(self, label, num_consumer):

        if not isinstance(label, str):
            msg = 'Subsystem label must be of type str!'
            logging.error(msg)
            raise ValueError(msg)

        elif len([x for x in [';', ', ', '.'] if x in label]) > 0:
            msg = 'Can\'t use ' + str([';', ', ', '.']) + ' in label.'
            logging.error(msg)
            raise ValueError(msg)
        else:
            self.label = label

        self.num_consumer = num_consumer


        self.comps = {}
        self.conns = {}
        self.create_comps()
        self.create_conns()

    def create_comps(self):

            self.comps['consumer'] = HeatExchangerSimple(self.label + '_consumer')
            self.comps['valve'] = Valve(self.label + '_valve')
            #self.comps['feed'] = Pipe(self.label + '_pipe feed')
            #self.comps['return'] = Pipe(self.label + '_pipe return')


    def create_conns(self):

        self.conns['cova_0'] = Connection(self.comps['consumer'], 'out1', self.comps['valve'], 'in1')


class CHP(Subsystem):
    def __init__(self, label):

        if not isinstance(label, str):
            msg = 'Subsystem label must be of type str!'
            logging.error(msg)
            raise ValueError(msg)

        elif len([x for x in [';', ', ', '.'] if x in label]) > 0:
            msg = 'Can\'t use ' + str([';', ', ', '.']) + ' in label.'
            logging.error(msg)
            raise ValueError(msg)
        else:
            self.label = label


        self.comps = {}
        self.conns = {}
        self.create_comps()
        self.create_conns()

    def create_comps(self):
        #turbine
            self.comps['valve_turb'] = Valve('turbine inlet valve')
            self.comps['turbine_hp'] = Turbine('high pressure turbine')
            self.comps['split'] = Splitter('extraction splitter')
            self.comps['turbine_lp'] = Turbine('low pressure turbine')

        # condenser and preheater'
            self.comps['cond'] = Condenser('condenser')
            self.comps['preheater'] = Condenser('preheater')
            self.comps['merge_ws'] = Merge('waste steam merge')
            self.comps['valve_pre'] = Valve('preheater valve')

        # feed water
            self.comps['pump'] = Pump('pump')
            self.comps['steam_generator'] = HeatExchangerSimple('steam generator')

            self.comps['closer'] = CycleCloser('cycle closer')

        # source and sink for cooling water
            self.comps['source_cw'] = Source('source_cw')
            self.comps['sink_cw'] = Sink('sink_cw')

    def create_conns(self):
        #Turbine
        self.conns['fs_in'] = Connection(self.comps['closer'], 'out1', self.comps['valve_turb'], 'in1')
        self.conns['fs'] = Connection(self.comps['valve_turb'], 'out1', self.comps['turbine_hp'], 'in1')
        self.conns['ext'] = Connection(self.comps['turbine_hp'], 'out1', self.comps['split'], 'in1')
        self.conns['ext_v'] = Connection(self.comps['split'], 'out1', self.comps['preheater'], 'in1')
        self.conns['ext_turb'] = Connection(self.comps['split'], 'out2', self.comps['turbine_lp'], 'in1')

        # preheater and condenser
        self.conns['ext_cond'] = Connection(self.comps['preheater'], 'out1', self.comps['valve_pre'], 'in1')
        self.conns['cond_ws'] = Connection(self.comps['valve_pre'], 'out1', self.comps['merge_ws'], 'in2')
        self.conns['turb_ws'] = Connection(self.comps['turbine_lp'], 'out1', self.comps['merge_ws'], 'in1')
        self.conns['ws'] = Connection(self.comps['merge_ws'], 'out1', self.comps['cond'], 'in1')

        # feed water
        self.conns['con'] = Connection(self.comps['cond'], 'out1', self.comps['pump'], 'in1')
        self.conns['fw_c'] = Connection(self.comps['pump'], 'out1', self.comps['preheater'], 'in2')
        self.conns['fw_w'] = Connection(self.comps['preheater'], 'out2', self.comps['steam_generator'], 'in1')
        self.conns['fs_out'] = Connection(self.comps['steam_generator'], 'out1', self.comps['closer'], 'in1')

        # cooling water
        #self.conns['cw_in'] = Connection(self.comps['source_cw'], 'out1', self.comps['cond'], 'in2')
        #self.conns['cw_out'] = Connection(self.comps['cond'], 'out2', self.comps['sink_cw'], 'in1')


