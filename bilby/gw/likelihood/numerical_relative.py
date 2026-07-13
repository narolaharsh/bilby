from copy import deepcopy
import types
import numpy as np
from .base import GravitationalWaveTransient
from ...core.fisher import FisherMatrixPosteriorEstimator
from ...core.utils import logger
from scipy.optimize import differential_evolution


class NumericalRelativeBinningGravitationalWaveTransient(GravitationalWaveTransient):
    """
    A gravitational-wave transient likelihood object implementing numerical
    relative binning, in contrast to the analytic scheme in relative.py.
    Designed for parameter estimation on long-duration signals. This likelihood does not yet work with earth rotation.
    # FIXME: Include earth rotation feature
    Parameters
    ==========
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances.
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        Must use a relative-binning source model (e.g.
        `lal_binary_black_hole_relative_binning`) so that `frequency_bin_edges`
        in `waveform_arguments` is respected.
    fiducial_parameters: dict
        Parameter set used to construct the sparse frequency grid.
    minimum_bins: int, optional
        The bin construction algorithm iterates until at least this many bins
        are produced.
    delta: float, optional
        Total allowed likelihood error budget shared across all bins.
    """

    def __init__(
            self, interferometers, waveform_generator,
            fiducial_parameters=None,
            time_marginalization=False,
            distance_marginalization=False, phase_marginalization=False, priors=None,
            distance_marginalization_lookup_table=None,
            jitter_time=True, reference_frame="sky",
            time_reference="geocenter", earth_rotation=False,
            minimum_bins=100, delta=0.1
    ):

        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            priors=priors,
            distance_marginalization=distance_marginalization,
            time_marginalization=time_marginalization,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=jitter_time,
            reference_frame=reference_frame,
            time_reference=time_reference,
            phase_marginalization=phase_marginalization,
            earth_rotation=earth_rotation,
        )
        self.delta = delta
        self.minimum_bins = minimum_bins
        self.uniform_frequency_array_resolution = (
            self.waveform_generator.frequency_array[1]
            - self.waveform_generator.frequency_array[0]
        )

        self.fiducial_parameters = fiducial_parameters
        # self.parameters_to_be_updated = [key for key in priors if not isinstance(
        #        priors[key], (DeltaFunction, Constraint, float, int))]
        self.parameters_to_be_updated = ['chirp_mass', 'mass_ratio']

        logger.info(f"Parameter keys to be used: {self.parameters_to_be_updated}")

        self.parameter_bounds = self.get_bounds_from_priors(priors)

        self.per_detector_fiducial_waveforms = self.compute_uniform_grid_detector_response(
            fiducial_parameters)
        self.maximum_frequency = self.compute_last_nonzero_frequency(
            self.per_detector_fiducial_waveforms)
        self.minimum_frequency = self.waveform_generator.waveform_arguments["minimum_frequency"]
        logger.info(f"Last nonzero frequency: {self.maximum_frequency}")
        self.inverse_psd = self.compute_inverse_psd()

        self.bin_centers = self.waveform_generator.frequency_array
        # self.perturbed_parameters = self.find_maximum_likelihood_parameters(self.parameter_bounds,
        #                                                                    )
        self.sigma_bounds = self.fisher_matrix_based_perturbation(parameter_keys=self.parameters_to_be_updated)
        self.perturbed_parameters = self.perturb_parameters_from_sigma_bounds()

        self.perturbed_detector_response = self.compute_uniform_grid_detector_response(
            self.perturbed_parameters)

        self.bin_inds = self.setup_bins(
            self.minimum_bins,
            self.per_detector_fiducial_waveforms,
            self.perturbed_detector_response,
        )
        logger.info(f"Finished setting up {len(self.bin_inds) - 1} bins.")
        self.bin_freqs = self.waveform_generator.frequency_array[self.bin_inds]

        # Tell the waveform model to evaluate at the sparse bin-edge frequencies.
        # Do NOT reassign waveform_generator.frequency_array — that would corrupt
        # duration and sampling_frequency (series.py derives them from the array).
        self.waveform_generator.waveform_arguments["frequency_bin_edges"] = self.bin_freqs
        self.waveform_generator.waveform_arguments["fiducial"] = 0
        self.waveform_generator._cache["parameters"] = None

        self.bin_centers = 0.5 * (self.bin_freqs[1:] + self.bin_freqs[:-1])
        self.bin_widths = np.diff(self.bin_freqs)

        # Size of each bin in number of uniform-grid samples
        self.bin_sizes = np.diff(self.bin_inds)
        self.bin_sizes[-1] += 1

        self.per_detector_fiducial_waveform_points = (
            self.compute_per_detector_fiducial_waveform_points()
        )
        self.summary_data = self.compute_summary_data(self.bin_inds)

    def compute_inverse_psd(self):
        """
        Returns
        =======
        inverse_psd: dict
            Dictionary keyed by interferometer name of inverse PSD arrays on
            the full uniform frequency grid.
        """
        inverse_psd = {}
        for interferometer in self.interferometers:
            name = interferometer.name
            _temp_psd = np.power(interferometer.power_spectral_density_array, -1)
            _temp_psd[np.isnan(_temp_psd)] = 0.0
            _temp_psd[np.isinf(_temp_psd)] = 0.0
            inverse_psd[name] = _temp_psd
        return inverse_psd

    def get_bounds_from_priors(self, priors):
        bounds = []
        for key in self.parameters_to_be_updated:
            bounds.append([priors[key].minimum, priors[key].maximum])
        return bounds

    def compute_per_detector_fiducial_waveform_points(self):
        """
        Returns
        =======
        per_detector_fiducial_waveform_points: dict
            Fiducial waveform evaluated at the bin edges for each detector.
            Zero-valued entries are replaced with ``np.inf`` so they produce
            zero when used as a denominator in the waveform ratio.
        """
        per_detector_fiducial_waveform_points = {}
        for interferometer in self.interferometers:
            name = interferometer.name
            points = self.per_detector_fiducial_waveforms[name][self.bin_inds].copy()
            points[points == 0] = np.inf
            per_detector_fiducial_waveform_points[name] = points
        return per_detector_fiducial_waveform_points

    def compute_uniform_grid_detector_response(self, parameters):
        """
        Evaluate the detector response on the full uniform frequency grid.

        Parameters
        ==========
        parameters: dict
            Waveform and extrinsic parameters.

        Returns
        =======
        detector_response: dict
            Dictionary keyed by interferometer name of complex strain arrays.
        """
        self._set_fiducial()
        polarizations = self.waveform_generator.frequency_domain_strain(parameters)
        self._unset_fiducial()
        detector_response = {}
        for interferometer in self.interferometers:
            name = interferometer.name
            detector_response[name] = interferometer.get_detector_response(
                polarizations, parameters, earth_rotation=self.earth_rotation)
        return detector_response

    def compute_last_nonzero_frequency(self, detector_response):
        """
        Find the highest frequency at which the fiducial waveform is non-zero
        across all detectors.

        Parameters
        ==========
        detector_response: dict
            Dictionary keyed by interferometer name of complex strain arrays.

        Returns
        =======
        last_nonzero_frequency: float
            Frequency in Hz of the last non-zero sample.
        """
        last_nonzero_indices = []
        for _, value in detector_response.items():
            if value[-1] != 0j:
                last_nonzero_indices.append(-1)
                continue
            nonzero_indices = np.where(value[::-1] != 0j)[0]
            last_nonzero_index = len(value) - 1 - nonzero_indices[0]
            last_nonzero_indices.append(last_nonzero_index)
        return self.waveform_generator.frequency_array[min(last_nonzero_indices)]

    def setup_bins(self, n_target_bins, fiducial_detector_response, perturbed_detector_response):
        """
        Construct a sparse frequency grid by bisecting the uniform grid until
        the per-bin likelihood error falls below ``delta / n_target_bins``.

        Parameters
        ==========
        n_target_bins: int
            Target number of bins.
        fiducial_detector_response: dict
            Fiducial waveform on the uniform grid, keyed by interferometer name.
        perturbed_detector_response: dict
            Perturbed waveform on the uniform grid, keyed by interferometer name.

        Returns
        =======
        bin_inds: list of int
            Sorted unique frequency-array indices of the bin edges, including
            the left edge of the first bin.
        """
        catch_errors = []
        proposed_bin_inds = self.setup_bins_using_bisection_method(
            n_target_bins,
            fiducial_detector_response,
            perturbed_detector_response,
            self.minimum_frequency,
            self.maximum_frequency,
            catch_errors,
        )

        if n_target_bins != len(proposed_bin_inds):
            logger.info(
                f"Rerunning the binning algorithm with {len(proposed_bin_inds)} proposed bins "
                f"and {n_target_bins} target bins"
            )
            return self.setup_bins(
                len(proposed_bin_inds), fiducial_detector_response, perturbed_detector_response)

        elif len(proposed_bin_inds) < self.minimum_bins:
            perturbed_parameters = self.perturb_parameters_from_sigma_bounds()
            perturbed_detector_response = self.compute_uniform_grid_detector_response(
                perturbed_parameters)
            logger.info(
                f"Too few bins produced. Updated perturbed chirp mass to "
                f"{perturbed_parameters['chirp_mass']}, "
                f"mass ratio to {perturbed_parameters['mass_ratio']}."
            )
            self.perturbed_parameters = perturbed_parameters
            self.perturbed_detector_response = perturbed_detector_response
            return self.setup_bins(
                self.minimum_bins, fiducial_detector_response, perturbed_detector_response)

        else:
            logger.info(
                f"Bisection complete. "
                f"Total error: {np.sum(np.array(catch_errors)):.4f} "
                f"(budget: {self.delta}). "
                f"Bins produced: {len(proposed_bin_inds)}"
            )
            global_minimum_index = self.find_nearest_index(
                self.minimum_frequency, self.waveform_generator.frequency_array)
            bin_inds = sorted(np.unique(np.insert(proposed_bin_inds, 0, global_minimum_index)))
            return bin_inds

    def exact_log_likelihood_ratio(self, parameters):
        """
        Compute the exact (non-relative-binned) log likelihood ratio on the
        full frequency grid.

        Relative binning requires bin edges (``self.bin_freqs`` and friends),
        which don't exist at the time we are using scipy maximizer to
        find the maximum likelihood point. So we use this function to compute
        likelihood without any apprximation.

        # FIXME: There must be an easier way to do this but this works for now.
        """
        original_calculate_snrs = self.calculate_snrs  # backup of the RB method
        original_compute_full_waveform = self._compute_full_waveform  # backup of the RB method
        self.calculate_snrs = types.MethodType(GravitationalWaveTransient.calculate_snrs, self)
        self._compute_full_waveform = types.MethodType(
            GravitationalWaveTransient._compute_full_waveform, self)
        self._set_fiducial()
        try:
            return self.log_likelihood_ratio(parameters)
        finally:
            self._unset_fiducial()
            self.calculate_snrs = original_calculate_snrs
            self._compute_full_waveform = original_compute_full_waveform

    def lnlike_scipy_maximize(self, parameter_list, parameters):
        parameters = parameters.copy()
        parameters.update(self.get_parameter_dictionary_from_list(parameter_list))
        return -self.exact_log_likelihood_ratio(parameters)

    def get_parameter_dictionary_from_list(self, parameter_list):
        parameter_dictionary = dict(zip(self.parameters_to_be_updated, parameter_list))
        excluded_parameter_keyss = set(self.fiducial_parameters) - set(self.parameters_to_be_updated)
        for key in excluded_parameter_keyss:
            parameter_dictionary[key] = self.fiducial_parameters[key]
        return parameter_dictionary

    def find_maximum_likelihood_parameters(self, parameter_bounds,
                                           iterations=5, maximization_kwargs=None):
        """
        Function to compute maximum likelihood paraemters
        """
        if maximization_kwargs is None:
            maximization_kwargs = dict()
        parameters = deepcopy(self.fiducial_parameters)
        updated_parameters_list = self.get_parameter_list_from_dictionary(self.fiducial_parameters)
        old_fiducial_ln_likelihood = self.exact_log_likelihood_ratio(self.fiducial_parameters)
        logger.info(f"Fiducial ln likelihood ratio: {old_fiducial_ln_likelihood:.2f}")
        updated_parameters = parameters
        for it in range(iterations):
            logger.info(f"Optimizing fiducial parameters. Iteration : {it + 1}")
            output = differential_evolution(
                self.lnlike_scipy_maximize,
                bounds=parameter_bounds,
                args=(parameters,),
                x0=updated_parameters_list,
                **maximization_kwargs,
            )
            updated_parameters_list = output['x']
            updated_parameters = deepcopy(self.fiducial_parameters)
            updated_parameters.update(self.get_parameter_dictionary_from_list(updated_parameters_list))
            new_fiducial_ln_likelihood = self.exact_log_likelihood_ratio(updated_parameters)
            logger.info(f"Fiducial ln likelihood ratio: {new_fiducial_ln_likelihood:.2f}")
            if new_fiducial_ln_likelihood - old_fiducial_ln_likelihood < 0.1:
                break
            old_fiducial_ln_likelihood = new_fiducial_ln_likelihood

        logger.info("Fiducial waveforms updated")
        return updated_parameters

    def setup_bins_using_bisection_method(self, n_bins, fiducial_detector_response,
                                          perturbed_detector_response,
                                          minimum_frequency, maximum_frequency,
                                          catch_errors):
        """
        Recursively bisect the frequency range ``[minimum_frequency,
        maximum_frequency]`` until the likelihood error is below the per-bin
        budget or the uniform-grid resolution is reached.

        Parameters
        ==========
        n_bins: int
            Total number of bins (used to compute the per-bin error budget).
        fiducial_detector_response: dict
            Fiducial waveform on the uniform grid.
        perturbed_detector_response: dict
            Perturbed waveform on the uniform grid.
        minimum_frequency: float
            Lower edge of the current interval in Hz.
        maximum_frequency: float
            Upper edge of the current interval in Hz.
        catch_errors: list
            Accumulates the observed error for each accepted bin.

        Returns
        =======
        bin_inds: np.ndarray of int
            Index of the right edge of each accepted bin within this interval.
        """
        frequency_array = self.waveform_generator.frequency_array
        idx_low = self.find_nearest_index(minimum_frequency, frequency_array)
        idx_high = self.find_nearest_index(maximum_frequency, frequency_array)

        allowed_error = self.delta / n_bins
        observed_error = self.compute_likelihood_error_per_bin(idx_low, idx_high)

        if (idx_high - idx_low) <= 1:
            logger.debug(
                f"Reached uniform grid limit at [{frequency_array[idx_low]:.2f}, "
                f"{frequency_array[idx_high]:.2f}] Hz."
            )
            return np.array([idx_high])

        if observed_error < allowed_error:
            logger.debug(
                f"Bin accepted: [{frequency_array[idx_low]:.2f}, "
                f"{frequency_array[idx_high]:.2f}] Hz, "
                f"merged {idx_high - idx_low} samples."
            )
            catch_errors += [observed_error]
            return np.array([idx_high])

        idx_mid = (idx_low + idx_high) // 2
        logger.debug(
            f"Error too large — splitting at {frequency_array[idx_mid]:.2f} Hz."
        )
        left_inds = self.setup_bins_using_bisection_method(
            n_bins, fiducial_detector_response, perturbed_detector_response,
            frequency_array[idx_low], frequency_array[idx_mid], catch_errors)
        right_inds = self.setup_bins_using_bisection_method(
            n_bins, fiducial_detector_response, perturbed_detector_response,
            frequency_array[idx_mid], frequency_array[idx_high], catch_errors)
        return np.append(left_inds, right_inds)

    def compute_waveform_ratio_per_interferometer(self, waveform_polarizations, interferometer, parameters=None):
        """
        Compute the waveform ratio r0, r1 at the bin centers for one detector.

        Parameters
        ==========
        waveform_polarizations: dict
            Waveform polarizations evaluated at ``bin_freqs``.
        interferometer: bilby.gw.detector.Interferometer
        parameters: dict, optional

        Returns
        =======
        [r0, r1]: list of np.ndarray
            Zeroth- and first-order waveform ratio coefficients, one per bin.
        """

        name = interferometer.name
        strain = interferometer.get_detector_response(
            waveform_polarizations=waveform_polarizations,
            parameters=parameters,
            frequencies=self.bin_freqs,
        )
        waveform_ratio = strain / self.per_detector_fiducial_waveform_points[name]
        r0 = (waveform_ratio[1:] + waveform_ratio[:-1]) / 2
        r1 = (waveform_ratio[1:] - waveform_ratio[:-1]) / self.bin_widths
        return [r0, r1]

    def _compute_full_waveform(self, signal_polarizations, interferometer, parameters=None):
        fiducial_waveform = self.per_detector_fiducial_waveforms[interferometer.name]
        r0, r1 = self.compute_waveform_ratio_per_interferometer(
            waveform_polarizations=signal_polarizations,
            interferometer=interferometer,
            parameters=parameters,
        )

        idxs = slice(self.bin_inds[0], self.bin_inds[-1] + 1)
        duplicated_r0 = np.repeat(r0, self.bin_sizes)
        duplicated_r1 = np.repeat(r1, self.bin_sizes)
        duplicated_fm = np.repeat(self.bin_centers, self.bin_sizes)

        f = interferometer.frequency_array
        full_waveform_ratio = np.zeros(f.shape[0], dtype=complex)
        full_waveform_ratio[idxs] = duplicated_r0 + duplicated_r1 * (f[idxs] - duplicated_fm)
        return fiducial_waveform * full_waveform_ratio

    def calculate_snrs(self, waveform_polarizations, interferometer, return_array=True, parameters=None):
        """
        Compute SNR quantities for one interferometer using the relative
        binning approximation.

        Parameters
        ==========
        waveform_polarizations: dict
            Waveform polarizations evaluated at ``bin_freqs``.
        interferometer: bilby.gw.detector.Interferometer
        return_array: bool, optional
            If True and time marginalisation is active, compute and return
            ``d_inner_h_array``.
        parameters: dict, optional

        Returns
        =======
        calculated_snrs: _CalculatedSNRs
        """
        r0, r1 = self.compute_waveform_ratio_per_interferometer(
            waveform_polarizations=waveform_polarizations,
            interferometer=interferometer,
            parameters=parameters,
        )
        a0, a1, b0, b1 = self.summary_data[interferometer.name]

        d_inner_h = np.sum(a0 * np.conjugate(r0) + a1 * np.conjugate(r1))
        h_inner_h = np.sum(b0 * np.abs(r0) ** 2 + 2 * b1 * np.real(r0 * np.conjugate(r1)))
        optimal_snr_squared = h_inner_h
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared ** 0.5)

        if return_array and self.time_marginalization:
            full_waveform = self._compute_full_waveform(
                signal_polarizations=waveform_polarizations,
                interferometer=interferometer,
                parameters=parameters,
            )
            d_inner_h_array = 4 / self.waveform_generator.duration * np.fft.fft(
                full_waveform[0:-1]
                * interferometer.frequency_domain_strain.conjugate()[0:-1]
                / interferometer.power_spectral_density_array[0:-1])
        else:
            d_inner_h_array = None

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared.real,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=d_inner_h_array,
        )

    def generate_perturbed_parameters(self):
        """
        Generate a slightly perturbed copy of the fiducial parameters by
        applying small random fractional changes to chirp mass and mass ratio.
        # FIXME: Replace this feature by a maximum likelihood estimator
        # This function will be deprecated
        Returns
        =======
        perturbed_parameters: dict
        """
        perturbed_parameters = self.fiducial_parameters.copy()

        # FIXME: use a Fisher matrix to choose perturbation size
        chirp_mass_perturbation = 1 + 1e-2 * np.random.uniform(-1, 1)
        mass_ratio_perturbation = 1 + 1e-2 * np.random.uniform(-1, 1)

        perturbed_parameters['chirp_mass'] *= chirp_mass_perturbation
        perturbed_parameters['mass_ratio'] *= mass_ratio_perturbation

        logger.info(
            f"Perturbed chirp mass: {perturbed_parameters['chirp_mass']}, "
            f"mass ratio: {perturbed_parameters['mass_ratio']}"
        )
        return perturbed_parameters

    def compute_summary_data(self, bin_inds):
        """
        Compute the per-bin summary data integrals a0, a1, b0, b1 over the
        full uniform frequency grid.

        Parameters
        ==========
        bin_inds: array-like of int
            Indices into ``waveform_generator.frequency_array`` of the bin edges.

        Returns
        =======
        summary_data: dict
            Keyed by interferometer name; each value is a tuple ``(a0, a1, b0, b1)``
            of complex arrays with length ``len(bin_inds) - 1``.
        """
        bin_inds = np.asarray(bin_inds)
        n_bins = len(bin_inds) - 1
        frequency_array = self.waveform_generator.frequency_array
        bin_frequencies = frequency_array[bin_inds]
        bin_centers = 0.5 * (bin_frequencies[1:] + bin_frequencies[:-1])
        normalization = 4.0 / self.waveform_generator.duration

        summary_data = {}
        for interferometer in self.interferometers:
            name = interferometer.name
            a0 = np.zeros(n_bins, dtype=complex)
            a1 = np.zeros(n_bins, dtype=complex)
            b0 = np.zeros(n_bins, dtype=complex)
            b1 = np.zeros(n_bins, dtype=complex)

            fiducial = self.per_detector_fiducial_waveforms[name]
            strain = interferometer.frequency_domain_strain
            ipsd = self.inverse_psd[name]

            for i in range(n_bins):
                sl = slice(bin_inds[i], bin_inds[i + 1])
                freq = frequency_array[sl]
                df = freq - bin_centers[i]
                kernel = normalization * np.conj(fiducial[sl]) * ipsd[sl]
                a0[i] = np.sum(strain[sl] * kernel)
                a1[i] = np.sum(strain[sl] * kernel * df)
                hh_kernel = normalization * fiducial[sl] * np.conj(fiducial[sl]) * ipsd[sl]
                b0[i] = np.sum(hh_kernel)
                b1[i] = np.sum(hh_kernel * df)

            summary_data[name] = (a0, a1, b0, b1)

        return summary_data

    def find_nearest_index(self, frequency_value, frequency_array):
        """
        Return the index in ``frequency_array`` closest to ``frequency_value``.

        Parameters
        ==========
        frequency_value: float
            Target frequency in Hz.
        frequency_array: np.ndarray

        Returns
        =======
        index: int
        """
        return np.argmin(np.abs(frequency_array - frequency_value))

    def compute_likelihood_error_per_bin(self, minimum_frequency_index, maximum_frequency_index):
        """
        Compute the absolute difference between the exact likelihood and the
        relative binning approximation over one frequency bin.

        Parameters
        ==========
        minimum_frequency_index: int
            Index of the lower bin edge in ``waveform_generator.frequency_array``.
        maximum_frequency_index: int
            Index of the upper bin edge in ``waveform_generator.frequency_array``.

        Returns
        =======
        likelihood_error: float
        """
        bin_edges = np.array([minimum_frequency_index, maximum_frequency_index])
        summary_data = self.compute_summary_data(bin_edges)
        frequency_array = self.waveform_generator.frequency_array
        bandwidth = frequency_array[maximum_frequency_index] - frequency_array[minimum_frequency_index]
        normalization = 4.0 / self.waveform_generator.duration

        logger.debug(
            f"Bin under consideration: "
            f"[{frequency_array[minimum_frequency_index]:.2f}, "
            f"{frequency_array[maximum_frequency_index]:.2f}] Hz"
        )

        # Relative binning likelihood
        d_inner_h = 0.0
        h_inner_h = 0.0
        for interferometer in self.interferometers:
            name = interferometer.name
            a0_ifo, a1_ifo, b0_ifo, b1_ifo = summary_data[name]
            perturbed = self.perturbed_detector_response[name][bin_edges]
            fiducial = self.per_detector_fiducial_waveforms[name][bin_edges].copy()
            fiducial[fiducial == 0] = np.inf
            ratios = perturbed / fiducial
            r0 = (ratios[1:] + ratios[:-1]) / 2
            r1 = (ratios[1:] - ratios[:-1]) / bandwidth
            d_inner_h += np.sum(np.real(a0_ifo * np.conj(r0) + a1_ifo * np.conj(r1)))
            h_inner_h += np.sum(
                np.real(b0_ifo * r0 * np.conj(r0)
                        + b1_ifo * (r0 * np.conj(r1) + np.conj(r0) * r1))
            )
        rb_likelihood = np.real(d_inner_h) - 0.5 * h_inner_h

        # Exact likelihood
        exact_d_inner_h = 0.0
        exact_h_inner_h = 0.0
        sl = slice(minimum_frequency_index, maximum_frequency_index)
        for interferometer in self.interferometers:
            name = interferometer.name
            d = interferometer.frequency_domain_strain[sl]
            h = self.perturbed_detector_response[name][sl]
            ipsd = self.inverse_psd[name][sl]
            exact_d_inner_h += np.sum(d * np.conj(h) * ipsd)
            exact_h_inner_h += np.sum(h * np.conj(h) * ipsd)
        exact_likelihood = normalization * (np.real(exact_d_inner_h) - 0.5 * np.real(exact_h_inner_h))

        logger.debug(f"Exact likelihood: {exact_likelihood:.4f}")
        logger.debug(f"Relative binning likelihood: {rb_likelihood:.4f}")

        return np.abs(exact_likelihood - rb_likelihood)

    def _set_fiducial(self):
        """
        Set the internal waveform generator to generate fiducial waveforms,
        e.g., for setting up the reference point.

        This is handled automatically within the likelihood for most applications
        and users shouldn't need to use this method.
        """
        self.waveform_generator.waveform_arguments["fiducial"] = 1
        self.waveform_generator._cache["parameters"] = None

    def _unset_fiducial(self):
        """
        Set the internal waveform generator to generate waveforms at the binning
        frequencies.

        This is handled automatically within the likelihood for most applications
        and users shouldn't need to use this method.
        """
        self.waveform_generator.waveform_arguments["fiducial"] = 0
        self.waveform_generator._cache["parameters"] = None

    def get_parameter_list_from_dictionary(self, parameter_dict):
        return [parameter_dict[k] for k in self.parameters_to_be_updated]

    def fisher_matrix_based_perturbation(self, parameter_keys, fd_eps=1e-6):
        """
        Estimate the 1-sigma measurement accuracy of each parameter in
        ``parameter_keys`` from the Fisher information matrix, evaluated at
        ``self.fiducial_parameters``.

        The Fisher matrix requires the exact (non-relative-binned)
        likelihood

        Parameters
        ==========
        parameter_keys: list of str
            Names of the parameters whose Fisher-matrix uncertainties are
            estimated.
        fd_eps: float
            Finite-difference step size (as a fraction of the prior width)
            used to estimate the Fisher matrix, see
            ``bilby.core.fisher.FisherMatrixPosteriorEstimator``.

        Returns
        =======
        sigma_bounds: dict
            Keyed by parameter name, the 1-sigma measurement accuracy taken
            from the diagonal of the Fisher matrix.
        """
        original_calculate_snrs = self.calculate_snrs
        original_compute_full_waveform = self._compute_full_waveform
        self.calculate_snrs = types.MethodType(GravitationalWaveTransient.calculate_snrs, self)
        self._compute_full_waveform = types.MethodType(
            GravitationalWaveTransient._compute_full_waveform, self)
        self._set_fiducial()
        try:
            fisher_estimator = FisherMatrixPosteriorEstimator(
                likelihood=self, priors=self.priors, parameters=parameter_keys, fd_eps=fd_eps,
            )
            iFIM = fisher_estimator.calculate_iFIM(self.fiducial_parameters)
        finally:
            self._unset_fiducial()
            self.calculate_snrs = original_calculate_snrs
            self._compute_full_waveform = original_compute_full_waveform

        sigma_bounds = {}
        for i, key in enumerate(parameter_keys):
            sigma_bounds[key] = np.sqrt(iFIM[i, i])
            logger.info(f"Fisher-matrix estimated {key} accuracy (1-sigma): {sigma_bounds[key]:.3e}")
        return sigma_bounds

    def perturb_parameters_from_sigma_bounds(self):
        """
        Perturb ``self.fiducial_parameters`` by a random draw scaled to the
        per-parameter 1-sigma bounds from ``fisher_matrix_based_perturbation``.

        Parameters
        ==========
        sigma_bounds: dict
            Keyed by parameter name, 1-sigma measurement accuracy, as
            returned by ``fisher_matrix_based_perturbation``.

        Returns
        =======
        perturbed_parameters: dict
        """
        perturbed_parameters = self.fiducial_parameters.copy()
        for key, parameter_sigma in self.sigma_bounds.items():
            perturbed_parameters[key] = (
                self.fiducial_parameters[key] + parameter_sigma * np.random.uniform(-2, 2)
            )
            logger.info(f"Perturbed {key}: {perturbed_parameters[key]}")
        return perturbed_parameters
