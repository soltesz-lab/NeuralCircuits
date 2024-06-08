import sys
from mpi4py import MPI
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from neuron import h
from network import ParallelNetwork, save_spikes, gather_spikes
from model import (
    Ncells,
    Nout,
    encoding_dim,
    make_cells,
    make_stims,
    init_inputs,
    generate_connections,
    network_params,
    set_plasticity,
    transfer_plastic_weights,
)
from simtime import SimTimeEvent
from mnist_data import generate_inputs
from decoding import predict_logistic, fit_logistic_decoder, MeanISISpikeTransform
from sklearn.metrics import accuracy_score

matplotlib.use("Agg")

soma_v = None
dend_v = None
t = None

h.nrnmpi_init()


def mpi_excepthook(type, value, traceback):
    """

    :param type:
    :param value:
    :param traceback:
    :return:
    """
    sys_excepthook(type, value, traceback)
    sys.stderr.flush()
    sys.stdout.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook


presentation_time = 0.1

train_size = 200
test_size = 10

seed = 999

Ninputs = encoding_dim

pnm = ParallelNetwork(Ncells + Ninputs, use_coreneuron=True)

train_image_array, train_labels = None, None
test_image_array, test_labels = None, None
if pnm.pc.id() == 0:
    train_image_array, train_labels = generate_inputs(
        plot=False,
        train_size=train_size,
        dataset="train",
        data_prefix="./datasets/mnist",
        seed=seed,
    )
    test_image_array, test_labels = generate_inputs(
        plot=False,
        test_size=test_size,
        dataset="test",
        data_prefix="./datasets/mnist",
        seed=seed,
    )
pnm.pc.barrier()
train_image_array, train_labels = pnm.pc.py_broadcast(
    (train_image_array, train_labels), 0
)
test_image_array, test_labels = pnm.pc.py_broadcast((test_image_array, test_labels), 0)

input_dim = np.prod(train_image_array.shape[1:])

make_cells(pnm)
make_stims(pnm, Ninputs)
generate_connections(pnm, network_params)

init_inputs(
    pnm, train_image_array, presentation_time=presentation_time, encoder_rf="gaussian"
)


if pnm.pc.id() == 0:
    print(f"train_image_array.shape = {train_image_array.shape}")
    print(f"test_image_array.shape = {test_image_array.shape}")
    gid = 0
    if pnm.gid_exists(gid):
        cell = pnm.gid2cell[gid]

        print(cell.soma.psection())
        print(cell.dends[0].psection())

        soma_v = h.Vector().record(cell.soma(0.5)._ref_v)
        dend_v = h.Vector().record(cell.dends[0](0.5)._ref_v)
        t = h.Vector().record(h._ref_t)


pnm.set_maxstep(10)
pnm.want_all_spikes()

tstop_train = train_image_array.shape[0] * presentation_time * 1000
tstop_test = test_image_array.shape[0] * presentation_time * 1000

if not pnm.use_coreneuron:
    simtime = SimTimeEvent(pnm.pc, tstop, 8.0, 10, 0)

# if pnm.pc.id() == 0:
#    for (src_gid, target_gid, synapse_id, mech_name), nc in pnm.netcons.items():
#        if mech_name == 'AMPA' and nc.wcnt() > 1:
#            print(f' {src_gid} -> {target_gid}: initial weights: {nc.weight[0]} {nc.weight[1]}')

pnm.run(tstop_train)

# if pnm.pc.id() == 0:
#    for (src_gid, target_gid, synapse_id, mech_name), nc in pnm.netcons.items():
#        if mech_name == 'AMPA' and nc.wcnt() > 1:
#            print(f' {src_gid} -> {target_gid}: weights after training {nc.weight[0]} {nc.weight[1]}')

cell_spikes_bp_train = pnm.get_cell_spikes("bp")
cell_spikes_out_train = pnm.get_cell_spikes("output")
cell_spikes_exc_train = pnm.get_cell_spikes("excitatory")
cell_spikes_fb_train = pnm.get_cell_spikes("fb")
cell_spikes_input_train = pnm.get_cell_spikes("input")
save_spikes(
    pnm.pc,
    f"cell_spikes_train_mnist",
    cell_spikes_exc_train,
    cell_spikes_out_train,
    cell_spikes_bp_train,
    cell_spikes_fb_train,
    cell_spikes_input_train,
)
all_spikes = gather_spikes(
    pnm.pc,
    cell_spikes_exc_train,
    cell_spikes_out_train,
    cell_spikes_fb_train,
    cell_spikes_bp_train,
    cell_spikes_input_train,
)


if pnm.pc.id() == 0:
    # plt.show()
    spikes = []
    for gid in range(Ncells):
        spikes.append(all_spikes[gid].reshape((-1,)))

    fig, axs = plt.subplots(2, 1)

    axs[0].eventplot(spikes, lineoffsets=np.arange(Ncells), orientation="horizontal")
    axs[0].set(xlabel="Time (ms)", ylabel="Neuron")
    axs[1].plot(np.asarray(t.as_numpy()), np.asarray(soma_v.as_numpy()), label="soma")
    axs[1].plot(np.asarray(t.as_numpy()), np.asarray(dend_v.as_numpy()), label="dend")
    plt.savefig("cell_spikes_train_mnist.png")

pnm.pc.barrier()

set_plasticity(pnm, False)
transfer_plastic_weights(pnm)
init_inputs(pnm, test_image_array, presentation_time=presentation_time)
pnm.run(tstop_test)

# if pnm.pc.id() == 0:
#    for (src_gid, target_gid, synapse_id, mech_name), nc in pnm.netcons.items():
#        if mech_name == 'AMPA' and nc.wcnt() > 1:
#            print(f' {src_gid} -> {target_gid}: weights after testing: {nc.weight[0]} {nc.weight[1]}')

cell_spikes_out_test = pnm.get_cell_spikes("output")
cell_spikes_bp_test = pnm.get_cell_spikes("bp")
cell_spikes_exc_test = pnm.get_cell_spikes("excitatory")
cell_spikes_fb_test = pnm.get_cell_spikes("fb")
cell_spikes_input_test = pnm.get_cell_spikes("input")
save_spikes(
    pnm.pc,
    f"cell_spikes_test_mnist",
    cell_spikes_exc_test,
    cell_spikes_out_test,
    cell_spikes_fb_test,
    cell_spikes_bp_test,
    cell_spikes_input_test,
)
all_spikes = gather_spikes(
    pnm.pc,
    cell_spikes_exc_test,
    cell_spikes_out_test,
    cell_spikes_fb_test,
    cell_spikes_bp_test,
    cell_spikes_input_test,
)

pnm.pc.barrier()

n_train_labels = len(np.unique(train_labels))
n_out = pnm.cellnums["output"]
offset_out = pnm.offsets["output"]
n_input = pnm.cellnums["input"]
offset_input = pnm.offsets["input"]
if pnm.pc.id() == 0:
    spike_transform = MeanISISpikeTransform()
    logistic_decoder_input = fit_logistic_decoder(
        cell_spikes_input_train,
        n_units=n_input,
        unit_offset=offset_input,
        label_dur=presentation_time * 1000.0,
        labels=train_labels,
        spike_transform=spike_transform,
    )
    input_predictions_train = predict_logistic(
        cell_spikes_input_train,
        logistic_decoder_input,
        label_count=len(train_labels),
        label_dur=presentation_time * 1000.0,
        n_units=n_input,
        unit_offset=offset_input,
        spike_transform=spike_transform,
    )
    input_train_score = accuracy_score(train_labels, input_predictions_train)

    input_predictions_test = predict_logistic(
        cell_spikes_input_test,
        logistic_decoder_input,
        label_count=len(test_labels),
        label_dur=presentation_time * 1000.0,
        n_units=n_input,
        unit_offset=offset_input,
        spike_transform=spike_transform,
    )
    input_test_score = accuracy_score(test_labels, input_predictions_test)

    logistic_decoder = fit_logistic_decoder(
        cell_spikes_out_train,
        n_units=n_out,
        unit_offset=offset_out,
        label_dur=presentation_time * 1000.0,
        labels=train_labels,
        spike_transform=spike_transform,
    )
    output_predictions_train = predict_logistic(
        cell_spikes_out_train,
        logistic_decoder,
        label_count=len(train_labels),
        label_dur=presentation_time * 1000.0,
        n_units=n_out,
        unit_offset=offset_out,
        spike_transform=spike_transform,
    )
    output_train_score = accuracy_score(train_labels, output_predictions_train)

    output_predictions_test = predict_logistic(
        cell_spikes_out_test,
        logistic_decoder,
        label_count=len(test_labels),
        label_dur=presentation_time * 1000.0,
        n_units=n_out,
        unit_offset=offset_out,
        spike_transform=spike_transform,
    )
    output_test_score = accuracy_score(test_labels, output_predictions_test)

    print(
        f"input score (train): {input_train_score}\n"
        f"input score (test): {input_test_score}\n"
    )
    print(
        f"output score (train): {output_train_score}\n"
        f"output score (test): {output_test_score}\n"
    )

    spikes = []
    for gid in range(Ncells):
        spikes.append(all_spikes[gid].reshape((-1,)))

    fig, axs = plt.subplots(2, 1)

    axs[0].eventplot(spikes, lineoffsets=np.arange(Ncells), orientation="horizontal")
    axs[0].set(xlabel="Time (ms)", ylabel="Neuron")
    axs[1].plot(np.asarray(t.as_numpy()), np.asarray(soma_v.as_numpy()), label="soma")
    axs[1].plot(np.asarray(t.as_numpy()), np.asarray(dend_v.as_numpy()), label="dend")

    plt.savefig("cell_spikes_test_mnist.png")


pnm.pc.barrier()
pnm.done()
# h.quit()
