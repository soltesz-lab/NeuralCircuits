from mpi4py import MPI
import uuid
import numpy as np
import matplotlib.pyplot as plt
from neuron import h
import argparse
from NeuralCircuits.model import NetworkModel
from NeuralCircuits.simtime import SimTimeEvent
from NeuralCircuits.config import read_from_yaml, IncludeLoader

soma_v = None
dend_v = None
t = None

h.nrnmpi_init()

def main(args):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    model_config_path = args.model_config_path
    tstop = args.tstop

    run_id = args.run_id

    if run_id is None:
        if rank == 0:
            run_id = uuid.uuid4()
        run_id = comm.bcast(run_id, 0)
    
    model_config = read_from_yaml(model_config_path, include_loader=IncludeLoader)
    
    net = NetworkModel(model_config)
    net.generate_connections()

    # Attach a stimulus to a synapse in the middle of the dendrite
    # of the first cell in the network.
    ncstim = None
    stim = None
    if rank == 0:
        cell = net.gid2cell(0)
        syn_ = cell.syndict["input excitatory"][0]

        stim = h.NetStim()  # Make a new stimulator
        stim.number = 3
        stim.start = 9

        syn_AMPA = syn_['AMPA']
        syn_AMPA.e = 0
        syn_AMPA.tau_rise = 0.1
        syn_AMPA.tau_decay = 5.0
        ncstim_AMPA = h.NetCon(stim, syn_AMPA)
        ncstim_AMPA.delay = 1
        ncstim_AMPA.weight[0] = 1.0
        ncstim_AMPA.weight[1] = 0.002

        syn_NMDA = syn_['NMDA']
        syn_NMDA.e = 0
        ncstim_NMDA = h.NetCon(stim, syn_NMDA)
        ncstim_NMDA.delay = 1
        ncstim_NMDA.weight[0] = 1.0
        ncstim_NMDA.weight[1] = 0.001

        print(cell.soma.psection())
        print(cell.dends[0].psection())

        cell = net.gid2cell(177)
        soma_v = h.Vector().record(cell.soma(0.5)._ref_v)
        dend_v = h.Vector().record(cell.dends[0](0.5)._ref_v)
        t = h.Vector().record(h._ref_t)

    simtime = None
    if not net.use_coreneuron:
        simtime = SimTimeEvent(net.pnm.pc, tstop, 8.0, 10, 0)

    net.run(tstop)

    if rank == 0:
        all_spikes = net.gather_spikes()
        
        spikes = []
        for gid in range(net.Ncells):
            spikes.append(all_spikes[gid].reshape((-1,)))

        fig, axs = plt.subplots(2, 1)
                
        axs[0].eventplot(spikes, lineoffsets=np.arange(net.Ncells), orientation="horizontal")
        axs[0].set(xlabel="Time (ms)", ylabel="Neuron")
        axs[1].plot(np.asarray(t.as_numpy()), np.asarray(soma_v.as_numpy()), label="soma")
        axs[1].plot(np.asarray(t.as_numpy()), np.asarray(dend_v.as_numpy()), label="dend")
        plt.savefig(f"{run_id}.png")

    net.done()


    
if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='This code runs a single-pulse network stimulation protocol')

    parser.add_argument('-c', '--model-config-path', type=str,
        action="store",
        required=True,
        help='The path to the model configuration YAML file')

    parser.add_argument('-i', '--run-id', type=str,
        action="store",
        required=False,
        help='Run id string')
    
    parser.add_argument('-t', '--tstop', type=float,
        action='store',
        required=False,
        default=100.0, 
        help='The desired simulation physical time.'
    )

    args = parser.parse_args()

    main(args)
