import numpy as np
from neuron import h

def taper_diam(sec, zero_bound, one_bound):
    dx = 1.0 / sec.nseg
    for seg, x in zip(sec, np.arange(dx / 2, 1, dx)):
        seg.diam = (one_bound - zero_bound) * x + zero_bound


class ExcMorrisLecar(object):
    
    def __init__(self, i, synapse_mechanisms={}):
        super().__init__()
        self.synapse_mechanisms = synapse_mechanisms
        self.ndends = 4
        self.syndict = {}
        self.create_sections()
        self.define_geometry()
        self.build_topology()
        self.build_subsets()
        self.define_biophysics()
        self.create_synapses()
        self.position = 0
        
    def create_sections(self):
        """create a soma"""
        self.soma = h.Section(name="soma", cell=self)
        self.trunk = h.Section(name="trunk", cell=self)
        self.dends = []
        for n in range(self.ndends):
            self.dends.append(h.Section(name=f"dend{(n+1)}", cell=self))

    def build_topology(self):
        self.trunk.connect(self.soma)
        for sec in self.dends:
            sec.connect(self.trunk)

    def build_subsets(self):
        self.all = self.soma.wholetree()

    def define_geometry(self):
        self.soma.L = 10
        self.soma.diam = 10
        self.soma.nseg = 3
        self.trunk.L = 10
        self.trunk.diam = 5
        self.trunk.nseg = 3
        taper_diam(self.trunk, 10.0, 1.0)
        for sec in self.dends:
            sec.L = 20
            sec.diam = 1.0
            sec.nseg = 5

    def define_biophysics(self):
        Ra = 200
        for sec in self.all:
            sec.Ra = Ra
            sec.cm = 1

        self.soma.insert("ml")
        for seg in self.soma:
            seg.betaw_ml = 12.0
            seg.gammaw_ml = 17.0
            seg.phi_ml = 0.067
            seg.gcabar_ml = 0.005  # Ca conductance in S/cm2
            seg.gkbar_ml = 0.008  # K conductance in S/cm2
            seg.gl_ml = 1.0e-4  # Passive conductance in S/cm2
            seg.el_ml = -70  # Leak reversal potential mV
            seg.ek = -90  # K reversal potential mV
        self.trunk.insert("ml")
        for seg in self.trunk:
            seg.betaw_ml = 12.0
            seg.gammaw_ml = 17.0
            seg.phi_ml = 0.067
            seg.gcabar_ml = 0.001  # Ca conductance in S/cm2
            seg.gkbar_ml = 0.004  # K conductance in S/cm2
            seg.gl_ml = 1.0e-4  # Passive conductance in S/cm2
            seg.el_ml = -70  # Leak reversal potential mV
            seg.ek = -90  # K reversal potential mV
        for sec in self.dends:
            sec.insert("pas")
            for seg in sec:
                seg.pas.g = 1e-5  # Passive conductance in S/cm2
                seg.pas.e = -70  # Leak reversal potential mV

    def create_synapses(self):
        input_synlist = []
        GABA_syn_mech_name = self.synapse_mechanisms["GABA"]
        AMPA_syn_mech_name = self.synapse_mechanisms["AMPA"]
        NMDA_syn_mech_name = self.synapse_mechanisms["NMDA"]
        for sec in self.dends:
            syn_AMPA = getattr(h, AMPA_syn_mech_name)(sec(0.5))
            syn_NMDA = getattr(h, NMDA_syn_mech_name)(sec(0.5))
            input_synlist.append({ "AMPA": syn_AMPA,
                                   "NMDA": syn_NMDA })
        self.syndict["input excitatory"] = input_synlist

        excitatory_synlist = []
        for sec in self.dends:
            syn_AMPA = getattr(h, AMPA_syn_mech_name)(sec(0.5))
            syn_NMDA = getattr(h, NMDA_syn_mech_name)(sec(0.5))
            excitatory_synlist.append({"AMPA": syn_AMPA,
                                       "NMDA": syn_NMDA })
        self.syndict["recurrent excitatory"] = excitatory_synlist

        inhibitory_synlist = []
        for sec in [self.soma, self.trunk] + self.dends:
            syn_GABA = getattr(h, GABA_syn_mech_name)(sec(0.5))
            inhibitory_synlist.append({"GABA": syn_GABA})
        self.syndict["inhibitory"] = inhibitory_synlist

    def set_position(self, pos):
        self.position = pos

    def connect2target(self, target, thresh=0):
        """Make a new NetCon with this cell's membrane
        potential at the soma as the source (i.e. the spike detector)
        onto the target passed in (i.e. a synapse on a cell).
        Subclasses may override with other spike detectors."""
        nc = h.NetCon(self.soma(1)._ref_v, target, sec=self.soma)
        nc.threshold = thresh
        return nc



class InhMorrisLecar(object):
    def __init__(self, i, synapse_mechanisms={}):
        super().__init__()
        self.synapse_mechanisms = synapse_mechanisms
        self.ndends = 4
        self.syndict = {}
        self.create_sections()
        self.define_geometry()
        self.build_topology()
        self.build_subsets()
        self.define_biophysics()
        self.create_synapses()
        self.position = 0

    def create_sections(self):
        """create a soma"""
        self.soma = h.Section(name="soma", cell=self)
        self.trunk = h.Section(name="trunk", cell=self)
        self.dends = []
        for n in range(self.ndends):
            self.dends.append(h.Section(name=f"dend{(n+1)}", cell=self))

    def build_topology(self):
        self.trunk.connect(self.soma)
        for sec in self.dends:
            sec.connect(self.trunk)

    def build_subsets(self):
        self.all = self.soma.wholetree()

    def define_geometry(self):
        self.soma.L = 10
        self.soma.diam = 10
        self.soma.nseg = 3
        self.trunk.L = 10
        self.trunk.diam = 5
        self.trunk.nseg = 3
        taper_diam(self.trunk, 10.0, 1.0)
        for sec in self.dends:
            sec.L = 20
            sec.diam = 1.0
            sec.nseg = 5

    def define_biophysics(self):
        Ra = 200
        for sec in self.all:
            sec.Ra = Ra
            sec.cm = 1

        self.soma.insert("ml")
        for seg in self.soma:
            seg.betaw_ml = 12.0
            seg.gammaw_ml = 17.0
            seg.phi_ml = 0.067
            seg.gcabar_ml = 0.005  # Ca conductance in S/cm2
            seg.gkbar_ml = 0.008  # K conductance in S/cm2
            seg.gl_ml = 1.0e-4  # Passive conductance in S/cm2
            seg.el_ml = -70  # Leak reversal potential mV
            seg.ek = -90  # K reversal potential mV
        self.trunk.insert("ml")
        for seg in self.trunk:
            seg.betaw_ml = 12.0
            seg.gammaw_ml = 17.0
            seg.phi_ml = 0.067
            seg.gcabar_ml = 0.001  # Ca conductance in S/cm2
            seg.gkbar_ml = 0.004  # K conductance in S/cm2
            seg.gl_ml = 1.0e-4  # Passive conductance in S/cm2
            seg.el_ml = -70  # Leak reversal potential mV
            seg.ek = -90  # K reversal potential mV
        for sec in self.dends:
            sec.insert("pas")
            for seg in sec:
                seg.pas.g = 1e-5  # Passive conductance in S/cm2
                seg.pas.e = -70  # Leak reversal potential mV

    def create_synapses(self):
        GABA_syn_mech_name = self.synapse_mechanisms["GABA"]
        AMPA_syn_mech_name = self.synapse_mechanisms["AMPA"]
        excitatory_synlist = []
        for sec in self.dends:
            syn_AMPA = getattr(h, AMPA_syn_mech_name)(sec(0.5))
            excitatory_synlist.append({"AMPA": syn_AMPA})
        self.syndict["excitatory"] = excitatory_synlist

        inhibitory_synlist = []
        for sec in [self.soma, self.trunk] + self.dends:
            syn_GABA = getattr(h, GABA_syn_mech_name)(sec(0.5))
            inhibitory_synlist.append({"GABA": syn_GABA})
        self.syndict["inhibitory"] = inhibitory_synlist

    def set_position(self, pos):
        self.position = pos

    def connect2target(self, target, thresh=0):
        """Make a new NetCon with this cell's membrane
        potential at the soma as the source (i.e. the spike detector)
        onto the target passed in (i.e. a synapse on a cell).
        Subclasses may override with other spike detectors."""
        nc = h.NetCon(self.soma(1)._ref_v, target, sec=self.soma)
        nc.threshold = thresh
        return nc

