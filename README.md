# Simulation corrections with Normalizing Flows

Repository dedicated to correcting simulation inaccuracies with neural spline auto-regressive flows.

We utilize the monotonically increasing transformation characteristic of normalizing flows, ensuring that the quantiles in the target and latent spaces are preserved. This enables us to morph the quantiles of simulated and data distributions, thus performing the necessary corrections.

The main problem is that the diferences in simulation and data leads to diferent mva ID curves, where teh data has a more background like shape due to our incapability to include all possible detector effects into the simulation, which is used to train the mva ID. To correct that we need to include correction factors that may come with big uncertanties.

This project aims to use normalizing flows to correct the simulation distirbutions based on data distributions and with that achieve a better closure betwenn corrected simulation and data and decrese one of the major source of uncertanties in this analysis.

The images below shows preliminary results where the blue histogram is the nominal simulated distributions, the red are the corrected samples using normalizng flows and black dots are data.

HoverE is a example of distirbution not well modeled and that needs to be corrected.

![plot](./plot/probe_hoe.png)[width=0.5\textwidth]

After correcting all variables used as input to the mvaID BDT, we can reevaluate it, and check that the flow corrected values achieve a better closure than the nominal simulation.

![plot](./plot/mvaID_barrel.png)[width=0.5\textwidth]

## Usage

To run the tests one need to have the set the configurations of the flow in the flow_configuration.yaml and after run the main.py script.

The code is not fully finished, so one may need to set one or two paths.

