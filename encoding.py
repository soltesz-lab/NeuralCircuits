import functools
import numpy as np
from numpy import ndarray
from typing import Tuple, Optional, Iterable, Iterator, Union
from scipy.interpolate import Akima1DInterpolator


@functools.cache
def transfer_linear_rf(input_range, output_freq_range, **kwargs):
    ip = Akima1DInterpolator(input_range, output_freq_range)
    return ip


@functools.cache
def transfer_gaussian_rf(
    input_range, output_freq_range, unit_no, module_index, n_fields
):
    input_span = input_range[1] - input_range[0]
    output_freq_span = output_freq_range[1] - output_freq_range[0]
    field_width = input_span / n_fields
    center = module_index * field_width + 0.5 * field_width
    input_x = np.arange(input_range[0], input_range[1], input_span / 100.0)
    output_freq = output_freq_range[0] + output_freq_span * np.exp(
        -4 * np.log(2) * (input_x - center) ** 2 / field_width**2
    )
    ip = Akima1DInterpolator(input_x, output_freq)
    return ip


def poisson_rate_generator(
    signal: Union[ndarray, Iterable[ndarray]],
    t_start: float = 0.0,
    time_window: int = 100,
    dt: float = 0.02,
    **kwargs,
) -> Iterator[ndarray]:
    """
    Lazily invokes ``PoissonEncoder`` to iteratively encode a sequence of
    data.

    :param data: NDarray of shape ``[n_samples, n_1, ..., n_k]``.
    :param time_window: Length of Poisson spike train per input variable.
    :param dt: Spike generator time step.
    :return: NDarray of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    t_start_ = t_start
    encoder = PoissonRateEncoder(time_window=time_window, dt=dt, **kwargs)
    for chunk_index in range(signal.shape[0]):
        chunk = signal[chunk_index, :].reshape((-1, 1))
        output, t_next = encoder.encode(chunk, t_start=t_start_)
        yield output
        t_start_ = t_next


class PoissonRateEncoder:
    def __init__(
        self,
        time_window: float = 0.01,  # seconds
        input_range: Tuple[int, int] = (0, 1),
        output_freq_range: Tuple[int, int] = (0, 100),
        generator: Optional[np.random.RandomState] = None,
        transfer_function=None,
        transfer_kwargs={},
        dt: float = 0.001,
    ) -> None:
        assert input_range[1] - input_range[0] > 0
        assert output_freq_range[1] - output_freq_range[0] > 0
        assert time_window > 0
        self.min_input, self.max_input = input_range[0], input_range[1]
        self.min_output, self.max_output = (
            output_freq_range[0],
            output_freq_range[1],
        )
        self.time_window = time_window
        if transfer_function is None:
            transfer_function = transfer_linear_rf
        self.transfer_function = transfer_function
        self.transfer_kwargs = transfer_kwargs
        if generator is None:
            generator = np.random
        self.generator = generator
        self.ndim = 1
        self.dt = dt

    def encode(
        self,
        signal: ndarray,
        return_spike_array: bool = False,
        t_start: Optional[float] = None,
    ) -> ndarray:
        assert (
            len(signal.shape) == 2
        ), "encode requires input signal of shape number_samples x input_dimensions"

        nsamples = signal.shape[0]
        ndim = signal.shape[1]

        assert (
            ndim == self.ndim
        ), f"input signal has dimension {ndim} but encoder has input dimension {self.ndim}"

        transfer_ip = self.transfer_function(
            (self.min_input, self.max_input),
            (self.min_output, self.max_output),
            **self.transfer_kwargs,
        )
        spike_train = []
        freq = transfer_ip(signal)

        time_window_len = int(round(self.time_window / self.dt))

        spike_array = self.generator.uniform(0, 1, nsamples * time_window_len).reshape(
            (nsamples, time_window_len)
        )
        dt = self.dt  # seconds
        for i in range(nsamples):
            if freq[i] > 0:
                spike_array[
                    i,
                    np.where(
                        np.logical_and(
                            spike_array[i] > 0.0, spike_array[i] < freq[i] * dt
                        )
                    ),
                ] = 1.0
            spike_array[i, np.where(spike_array[i] != 1.0)] = 0

        t_next = None
        if t_start is not None:
            t_next = t_start + self.time_window * nsamples

        if return_spike_array:
            return np.copy(self.spike_array), t_next
        else:
            if t_start is None:
                t_start = 0.0
            spike_times = []
            for i in range(nsamples):
                this_spike_inds = np.argwhere(spike_array[i, :] == 1).reshape((-1,))
                this_spike_times = []
                if len(this_spike_inds) > 0:
                    this_spike_times = (
                        t_start
                        + np.asarray(this_spike_inds, dtype=np.float32) * self.dt
                    )
                spike_times.append(this_spike_times)
            return np.concatenate(spike_times, dtype=np.float32), t_next
