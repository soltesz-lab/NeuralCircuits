from mpi4py import MPI
import copy
import numpy as np
from neuron import h
from network import ParallelNetwork, save_spikes, gather_spikes
import scipy
from scipy.spatial.distance import cdist
from collections import defaultdict
from encoding import poisson_rate_generator, transfer_gaussian_rf, transfer_linear_rf
from NeuralCircuits import config


def get_param_val_from_distribution(config_dict, rng):
    distribution = config_dict["distribution"]
    distribution_params = config_dict["distribution params"]
    rsamp = getattr(rng, distribution)
    val = np.clip(rsamp(**distribution_params), 1.0e-6, None)
    return val


def distance_probs(dist, sigma):
    weights = np.exp(-dist / sigma**2)
    prob = weights / weights.sum(axis=0)
    return prob


def convergent_topo_transform(
    rng,
    n_pre,
    n_post,
    coords_pre,
    coords_post,
    p_initial,
    sigma_scale,
    exclude_self=True,
):
    transform = {}
    for i in range(n_post):
        dist = cdist(coords_post[i, :].reshape((1, -1)), coords_pre).flatten()
        sigma = sigma_scale * p_initial * n_pre
        prob = distance_probs(dist, sigma)
        if exclude_self:
            source_choices = np.asarray([j for j in range(n_pre) if i != j])
            dist = dist[source_choices]
            prob = prob[source_choices]
            prob = prob / prob.sum(axis=0)
        else:
            source_choices = np.asarray(range(n_pre))
        chosen_inds = rng.choice(
            len(source_choices), round(p_initial * n_pre), replace=False, p=prob
        )

        sources = np.asarray(source_choices[chosen_inds], dtype=np.int32)
        transform[i] = (
            sources,
            np.clip(np.asarray(dist[chosen_inds], dtype=np.float32), 0.1, None),
        )

    return transform


def divergent_topo_transform(
    rng,
    n_pre,
    n_post,
    coords_pre,
    coords_post,
    p_initial,
    sigma_scale,
    exclude_self=True,
):
    post_transform = defaultdict(list)
    post_dists = defaultdict(list)
    for i in range(n_pre):
        if exclude_self:
            target_choices = np.asarray([j for j in range(n_post) if i != j])
        else:
            target_choices = np.asarray(range(n_post))
        dist = cdist(
            coords_pre[i, :].reshape((1, -1)), coords_post[target_choices, :]
        ).flatten()
        sigma = sigma_scale * p_initial * n_post
        prob = distance_probs(dist, sigma)
        chosen_inds = rng.choice(
            len(target_choices), round(p_initial * n_post), replace=False, p=prob
        )
        targets = np.asarray(target_choices[chosen_inds], dtype=np.int32)
        target_dists = dist[chosen_inds]

        for target_index, j in enumerate(targets):
            post_transform[j].append(i)
            post_dists[j].append(target_dists[target_index])

    transform = dict(
        {
            j: (
                np.asarray(sources, dtype=np.int32),
                np.clip(np.asarray(post_dists[j], dtype=np.float32), 0.1, None),
            )
            for j, sources in post_transform.items()
        }
    )
    return transform


class NetworkModel:
    def __init__(self, toplevel_config):
        self.toplevel_config = toplevel_config
        self.circuit_config = toplevel_config["Circuit"]
        self.runtime_config = toplevel_config["Runtime"]
        self.celltypes = {}
        self.populations = {}
        self.Ncells = 0
        self.Ninputs = 0
        self.synapse_parameter_rules = toplevel_config["Synapse Parameter Rules"]
        self.synapse_mechanisms = toplevel_config["Synapse Mechanisms"]
        self.random_seeds = toplevel_config.get("Random seeds", {})
        self.rng_coordinates = np.random.RandomState(
            int(self.random_seeds.get("Coordinates", 0))
        )
        self.rng_connections = np.random.RandomState(
            int(self.random_seeds.get("Distance-Dependent Connectivity", 0))
        )
        self.rng_weights = np.random.RandomState(
            int(self.random_seeds.get("Synaptic Weights", 0))
        )
        self.use_coreneuron = self.runtime_config.get("use coreneuron", False)
        self.use_cvode = self.runtime_config.get("cvode", False)
        self.dt = self.runtime_config.get("dt", 0.025)
        h.dt = self.dt
        self.pnm = ParallelNetwork(
            int(self.random_seeds.get("Network instantiation", 0)),
            use_cvode=self.use_cvode,
            use_coreneuron=self.use_coreneuron,
        )
        self._init_celltypes()
        self._init_populations()
        self._make_cells()

    def _init_celltypes(self):
        for k, cell_config in self.circuit_config["cell types"].items():
            template_class = cell_config["template class"]
            template_obj = config.import_object_by_path(template_class)
            self.celltypes[k] = {"template": template_obj}

    def _init_populations(self):
        offset = 0
        for k, population_config in self.circuit_config["populations"].items():
            number = population_config["number"]
            cell_type_name = population_config["cell type"]
            coordinates = self.rng_coordinates.random(size=(number, 2))
            self.populations[k] = {
                "offset": offset,
                "number": number,
                "cell class": self.celltypes[cell_type_name],
                "coordinates": coordinates,
            }
            self.pnm.add_population(k, number=number, offset=offset)
            offset = offset + number
            self.Ncells = self.Ncells + number

    def _make_cells(self):
        for population in sorted(self.populations):
            config = self.populations[population]
            N = config["number"]
            offset = config["offset"]
            coordinates = config["coordinates"]
            cell_class = config["cell class"]["template"]
            for i in range(N):
                gid = i + offset
                if self.pnm.gid_exists(gid):
                    cell = cell_class(gid, synapse_mechanisms=self.synapse_mechanisms)
                    self.pnm.register_cell(gid, cell)
                    cell.set_position(coordinates[i])

    def make_inputs(self, Ninputs):
        self.Ninputs = Ninputs

        stim_coordinates = self.rng_coordinates.uniform(size=((Ninputs, 2)))

        self.coordinates["input"] = stim_coordinates
        offset_Ninputs = self.Ncells

        self.pnm.add_population("input", number=Ninputs, offset=offset_Ninputs)

        for i in range(Ninputs):
            gid = i + offset_Ninputs
            if pnm.gid_exists(gid):
                cell = h.VecStim(gid)
                pnm.register_cell(gid, cell)

    def init_inputs(
        self,
        input_array,
        presentation_time=0.01,
        dt=0.001,
        input_encoder=poisson_rate_generator,
        encoder_rf="linear",
    ):
        pnm = self.pnm
        encoding_dim = self.Ninputs
        input_dim = np.prod(input_array.shape[1:])
        input_array = input_array.reshape(-1, input_dim)
        encoding_dim = pnm.cellnums["input"]
        input_offset = pnm.offsets["input"]
        assert encoding_dim >= input_dim
        encoding_n_fields = encoding_dim // input_dim
        assert encoding_n_fields >= 1
        input_range = (np.min(input_array), np.max(input_array))
        encoder_params = {}
        if encoder_rf == "linear":
            encoder_params["transfer_function"] = transfer_linear_rf
        elif encoder_rf == "gaussian":
            encoder_params["transfer_function"] = transfer_gaussian_rf
        else:
            raise RuntimeError(f"Unknown encoder receptive field type {encoder_rf}")
        for i in range(encoding_dim):
            gid = i + input_offset
            input_i = i // encoding_n_fields
            encoding_mod = i % encoding_n_fields
            if pnm.gid_exists(gid):
                inp = input_array[:, input_i].reshape((-1, 1))
                encoder_params_i = copy.deepcopy(encoder_params)
                encoder_params_i["transfer_kwargs"] = {
                    "unit_no": i,
                    "module_index": encoding_mod,
                    "n_fields": encoding_n_fields,
                }
                gen = input_encoder(
                    inp,
                    time_window=presentation_time,
                    dt=dt,
                    input_range=input_range,
                    **encoder_params_i,
                )
                spike_list = []
                for spike_times in gen:
                    spike_list.append(spike_times)
                spike_array = np.concatenate(spike_list) * 1000.0
                input_cell = pnm.gid2cell[gid]
                input_cell.play(h.Vector().from_python(spike_array))

    def set_plasticity(
        self,
        populations,
        enable,
        syn_types=["input excitatory", "recurrent excitatory"],
        syn_mechs=["AMPA"],
    ):
        pnm = self.pnm
        for p in populations:
            n = pnm.cellnums[p]
            offset = pnm.offsets[p]
        for gid in range(offset, offset + n):
            if pnm.gid_exists(gid):
                cell = pnm.gid2cell[gid]
                for syn_type in syn_types:
                    synlist = cell.syndict[syn_type]
                    for synobjs in input_synlist:
                        for syn_mech in syn_mechs:
                            synobjs[syn_mech].on = 1 if enable else 0

    def transfer_trained_weights(self, populations, syn_mech_name="AMPA"):
        pnm = self.pnm
        for p in populations:
            n = pnm.cellnums[p]
            offset = pnm.offsets[p]
            for gid in range(offset, offset + n):
                if pnm.gid_exists(gid):
                    for (
                        src_gid,
                        target_gid,
                        synapse_id,
                        mech_name,
                    ), nc in pnm.netcons.items():
                        if (target_gid == gid) and (mech_name == syn_mech_name):
                            nc.weight[0] = nc.weight[1]

    def generate_connections(self):
        pnm = self.pnm
        connectivity_params = self.circuit_config["connectivity"]

        for postsyn_pop in sorted(connectivity_params):
            Npost = self.populations[postsyn_pop]["number"]
            offset_post = self.pnm.offsets[postsyn_pop]
            coords_post = self.populations[postsyn_pop]["coordinates"]

            for presyn_pop in sorted(connectivity_params[postsyn_pop]):
                Npre = self.populations[presyn_pop]["number"]
                offset_pre = self.pnm.offsets[presyn_pop]
                coords_pre = self.populations[presyn_pop]["coordinates"]

                projection_params = connectivity_params[postsyn_pop][presyn_pop]
                projection_pattern = projection_params["pattern"]
                prob = projection_params["probability"]
                sigma_scale = projection_params["scale"]
                synapse_params = projection_params["synapse"]
                synapse_sections = synapse_params["sections"]
                synapse_mechs = synapse_params["mechanism"]
                projection_operator = convergent_topo_transform
                if projection_pattern == "convergent topographic":
                    projection_operator = convergent_topo_transform
                elif projection_pattern == "divergent topographic":
                    projection_operator = divergent_topo_transform
                else:
                    raise RuntimeError(
                        f"Unknown projection pattern {projection_pattern}"
                    )

                connections = projection_operator(
                    self.rng_connections,
                    Npre,
                    Npost,
                    coords_pre,
                    coords_post,
                    prob,
                    sigma_scale,
                    exclude_self=True,
                )

                for post_id, (pre_ids, dists) in connections.items():
                    post_gid = post_id + offset_post
                    for dist, pre_id in zip(dists, pre_ids):
                        pre_gid = pre_id + offset_pre
                        w = 1.0
                        netcon_params_dict = defaultdict(lambda: dict())
                        mech_params_dict = defaultdict(lambda: dict())
                        for syn_mech_type, syn_params in synapse_mechs.items():
                            syn_mech_name = self.synapse_mechanisms[syn_mech_type]
                            for syn_param_name, syn_param_value in syn_params.items():
                                if isinstance(syn_param_value, dict):
                                    syn_param_value = get_param_val_from_distribution(
                                        syn_param_value, self.rng_connections
                                    )
                                if (
                                    syn_param_name
                                    in self.synapse_parameter_rules[syn_mech_name][
                                        "netcon"
                                    ]
                                ):
                                    netcon_param_index = self.synapse_parameter_rules[
                                        syn_mech_name
                                    ]["netcon"][syn_param_name]
                                    netcon_params_dict[syn_mech_name][
                                        netcon_param_index
                                    ] = syn_param_value
                                elif (
                                    syn_param_name
                                    in self.synapse_parameter_rules[syn_mech_name][
                                        "mechanism"
                                    ]
                                ):
                                    mech_params_dict[syn_mech_name][
                                        syn_param_name
                                    ] = syn_param_value
                                else:
                                    raise RuntimeError(
                                        f"Unknown parameter {syn_param_name} in synapse mechanism {syn_mech_name}"
                                    )
                        syn_id = pnm.make_connection(
                            pre_gid,
                            post_gid,
                            synapse_sections,
                            delay=dist * 10.0,
                            synapse_mechanisms=self.synapse_mechanisms,
                            mech_params=dict(mech_params_dict),
                            netcon_params=dict(netcon_params_dict),
                        )

    def prepare_run(self):
        self.pnm.set_maxstep(10)
        self.pnm.want_all_spikes()

    def run(self, tstop):
        self.prepare_run()
        self.pnm.run(tstop)

    def done(self):
        self.pnm.pc.barrier()
        self.pnm.done()

    def gather_spikes(self, populations=None):
        cell_spikes = []
        if populations is None:
            populations = self.populations
        for population in populations:
            cell_spikes.append(self.pnm.get_cell_spikes(population))

        all_spikes = gather_spikes(self.pnm.pc, *cell_spikes)
        return all_spikes

    def save_spikes(self, output_name, populations=None):
        cell_spikes = []
        if populations is None:
            populations = self.populations
        for population in populations:
            cell_spikes.append(self.pnm.get_cell_spikes(p))

        save_spikes(self.pnm.pc, output_name, *cell_spikes)

    def gid2cell(self, gid):
        return self.pnm.pc.gid2cell(gid)

    def gid_exists(self, gid):
        return self.pnm.gid_exists(gid)
