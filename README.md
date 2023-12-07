# Simulation corrections with Normalizing Flows

Repository dedicated to correcting simulation inaccuracies with neural spline auto-regressive flows.

We utilize the monotonically increasing transformation characteristic of normalizing flows, ensuring that the quantiles in the target and latent spaces are preserved. This enables us to morph the quantiles of simulated and data distributions, thus performing the necessary corrections.

![plot](./plots/results/hoe.png)

