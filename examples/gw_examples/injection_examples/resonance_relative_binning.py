import bilby
import matplotlib.pyplot as plt
from bilby.core.utils import random


def check_likelihood_curve(likelihood, priors, key, injection_parameters):
    parameter_values = priors[key].sample(100)
    collect_likelihood = []

    for x in parameter_values:
        injection_parameters[key] = x
        templ = likelihood.log_likelihood_ratio(parameters=injection_parameters)
        collect_likelihood.append(templ)

    return collect_likelihood, parameter_values


random.seed(2)


duration = 512
sampling_frequency = 2048.0
minimum_frequency = 20


# Specify the output directory and the name of the simulation.
outdir = "deleteme_resonance"
label = "relative"
bilby.core.utils.setup_logger(outdir=outdir, label=label)


injection_parameters = dict(
    mass_1=1.5,
    mass_2=1.3,
    chi_1=0.02,
    chi_2=0.02,
    luminosity_distance=50.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
    lambda_1=545,
    lambda_2=1346,
)


injection_parameters["chirp_mass"] = bilby.gw.conversion.component_masses_to_chirp_mass(
    injection_parameters["mass_1"], injection_parameters["mass_2"]
)
injection_parameters["mass_ratio"] = (
    injection_parameters["mass_2"] / injection_parameters["mass_1"]
)
injection_parameters.pop("mass_1")
injection_parameters.pop("mass_2")

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant="IMRPhenomD_NRTidalv2",
    reference_frequency=100.0,
    minimum_frequency=minimum_frequency,
)

# Create the waveform_generator
injection_waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments,
)

# The likelihood requires a relative-binning-aware source model, which accepts
# the `fiducial` waveform argument used to switch between the dense fiducial
# waveform and the sparse bin-edge evaluation.
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star_relative_binning,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(["ET"])
for ifo in ifos:
    ifo.minimum_frequency = minimum_frequency

ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)

ifos.inject_signal(
    waveform_generator=injection_waveform_generator,
    parameters=injection_parameters,
    earth_rotation=False,
)


# For this analysis, we implement the standard BNS priors defined.
priors = bilby.gw.prior.BNSPriorDict()
priors["chirp_mass"] = bilby.core.prior.TruncatedGaussian(
    injection_parameters["chirp_mass"],
    1e-5,
    minimum=injection_parameters["chirp_mass"] - 1e-2,
    maximum=injection_parameters["chirp_mass"] + 1e-2,
)
priors["geocent_time"] = bilby.core.prior.Uniform(
    injection_parameters["geocent_time"] - 0.1,
    injection_parameters["geocent_time"] + 0.1,
)
priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
    minimum=10.0, maximum=500.0, name="luminosity_distance"
)

fiducial_parameters = priors.sample(1)
fiducial_parameters = (
    injection_parameters  # {key:value[0] for key, value in fiducial_parameters.items()}
)


likelihood = bilby.gw.likelihood.NumericalRelativeBinningGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    priors=priors,
    earth_rotation=False,
    fiducial_parameters=fiducial_parameters,
    delta=0.01,
)

l_of_theta, theta_value = check_likelihood_curve(
    likelihood, priors, "chirp_mass", injection_parameters
)

fig, ax = plt.subplots(1, 1)
ax.scatter(theta_value, l_of_theta)
fig.savefig("theta.pdf")

exit()
