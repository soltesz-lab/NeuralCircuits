from neuron import h
import numpy as np
from itertools import chain

def save_spikes(pc, save_filepath, *spike_time_dicts):
    all_spike_dicts = pc.py_gather(spike_time_dicts, 0)

    if pc.id() == 0:
        spike_dict = {}
        for ds in all_spike_dicts:
            for d in ds:
                spike_dict.update([(str(k), v) for (k, v) in d.items()])
        np.savez(save_filepath, **spike_dict)

    pc.barrier()


def gather_spikes(pc, *spike_time_dicts):
    all_spike_dicts = pc.py_gather(spike_time_dicts, 0)
    spike_dict = None

    if pc.id() == 0:
        spike_dict = {}
        for ds in all_spike_dicts:
            for d in ds:
                spike_dict.update([(k, v.reshape((-1,))) for (k, v) in d.items()])

    pc.barrier()
    return spike_dict


class Network:
    def __init__(self, seed=0):
        self.gidlist = []
        self.gid2cell = {}
        self.netcons = {}
        self.cellnums = {}
        self.offsets = {}
        ## TODO: add separate random number generator for nc_append syn selection
        self.rng = np.random.RandomState(seed)

    def add_population(self, name, number, offset):
        self.cellnums[name] = number
        self.offsets[name] = offset
        
    def register_cell(self, gid, cellobject):
        self.gid2cell[gid] = cellobject

    def gid_exists(self, gid):
        return gid in self.gidlist

    def nc_reset(self):
        self.netcons = {}

    def spike_record(self, gid, thresh=0):
        raise NotImplementedError("spike_record is not implemented.")

    def make_connection(self, src_gid, target_gid, synapse_id, weight, delay, thresh):
        raise NotImplementedError("make_connection is not implemented.")

    def run(self):
        raise NotImplementedError("run is not implemented.")

    def want_all_spikes(self, thresh=0):
        for gid in self.gidlist:
            self.spike_record(gid, thresh=thresh)


class SerialNetwork(Network):
    def __init__(self):
        super().__init__()
        self.nc_reset()
        self.gidlist = range(N)
        self.spikes = {}

    def spike_record(self, gid, thresh=0):
        self.spikes[gid] = (h.APCount(self.gid2cell[gid].soma(0.5)), h.Vector())
        self.spikes[gid][0].thresh = thresh
        self.spikes[gid][0].record(self.spikes[gid][1])

    def make_connection(
        self,
        src_gid,
        target_gid,
        synapse_type,
        weight,
        delay,
        thresh=0,
        synapse_id=None,
    ):
        target_synlist = self.gid2cell[target_gid].syndict[synapse_type]
        if synapse_id is None:
            synapse_id = np.random.int(0, len(target_synlist), 1)
        syn = target_synlist[synapse_id]
        self.netcons[(src_gid, target_gid, synapse_id)] = self.gid2cell[
            src_gid
        ].connect2target(syn, thresh=thresh)
        self.netcons[(src_gid, target_gid, synapse_id)].weight[0] = weight
        self.netcons[(src_gid, target_gid, synapse_id)].delay = delay

    def run(self):
        h.run()
        self.gatherspikes()

    def gatherspikes(self):
        self.spikevec = []
        self.idvec = []
        for gid in self.spikes:
            sp1 = self.spikes[gid][1].to_python()
            self.idvec.extend([gid] * len(sp1))
            self.spikevec.extend(sp1)


class ParallelNetwork(Network):
    def __init__(self, seed=None, use_coreneuron=False, use_cvode=False):
        super().__init__()
        
        self.rng = np.random.RandomState(seed)

        self.pc = h.ParallelContext()
        self.use_coreneuron = use_coreneuron
        self.cvode = h.CVode()
        self.cvode.use_fast_imem(1)
        self.cvode.cache_efficient(1)
        if self.use_coreneuron:
            from neuron import coreneuron

            coreneuron.enable = True
            coreneuron.verbose = 1  # if env.verbose else 0
            self.cvode.active(False)
        else:
            if use_cvode:
                self.cvode.active(True)

        self.spikevec = h.Vector()
        self.idvec = h.Vector()

    def add_population(self, name, number, offset):
        super().add_population(name, number, offset)

        #### Round-robin counting.
        #### Each host as an id from 0 to pc.nhost() - 1.
        for i in range(int(self.pc.id()), number, int(self.pc.nhost())):
            self.gidlist.append(i + offset)

    def register_cell(self, gid, cellobject):
        if self.gid_exists(gid):
            super().register_cell(gid, cellobject)
            self.pc.set_gid2node(gid, int(self.pc.id()))
            nc = None
            if hasattr(cellobject, "connect2target"):
                nc = cellobject.connect2target(None)
            else:
                nc = h.NetCon(cellobject, None)
            # nc.threshold = thresh
            self.pc.cell(gid, nc)

    def spike_record(self, gid, thresh=0):
        if self.pc.gid_exists(gid):
            self.pc.spike_record(gid, self.spikevec, self.idvec)
        else:
            raise RuntimeError(
                "Cell {} does not exist in the node {}.".format(gid, self.pc.id())
            )

    def set_maxstep(self, n):
        self.pc.set_maxstep(n)
        
        
    def make_connection(
        self,
        src_gid,
        target_gid,
        synapse_sections,
        delay,
        netcon_params={},
        mech_params={},
        synapse_mechanisms={},
        thresh=0,
        synapse_id=None,
    ):
        if self.pc.gid_exists(target_gid):
            target = self.pc.gid2cell(target_gid)
            synlist = list(chain.from_iterable([target.syndict[synapse_section]
                                                for synapse_section in synapse_sections]))
            if synapse_id is None:
                synapse_id = self.rng.randint(0, len(synlist), 1)[0]
            synloc = None # TODO
            synobjects = synlist[synapse_id]
            for mech_type, syn in synobjects.items():
                nc = self.pc.gid_connect(src_gid, syn)
                mech_name = synapse_mechanisms[mech_type]
                for mech_param_name, mech_param_value in mech_params[mech_name].items():
                    setattr(syn, mech_param_name, mech_param_value)
                for netcon_param_index, netcon_param_value in netcon_params[mech_name].items():
                    nc.weight[netcon_param_index] = netcon_param_value
                nc.delay = delay
                nc.threshold = thresh
                self.netcons[(src_gid, target_gid, synapse_id, mech_name, synloc)] = nc
            return synapse_id

    def run(self, tstop=100):
        h.finitialize(-65)
        self.pc.psolve(tstop)

    def done(self):
        self.pc.runworker()
        self.pc.done()

    def get_cell_spikes(self, celltype):
        offset = self.offsets[celltype]
        num = self.cellnums[celltype]
        idvec = np.asarray(self.idvec.to_python())
        spikevec = np.asarray(self.spikevec.to_python())
        spike_times = {}
        for gid in range(offset, offset + num):
            if gid in self.gidlist:
                spikeidxs = np.argwhere(idvec == gid)
                spike_times[gid] = spikevec[spikeidxs]
        return spike_times
