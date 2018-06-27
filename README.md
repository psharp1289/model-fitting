# Hierarchical Model Fitting

**demo.m**:   demonstrates fitting two models to simulated data

**simulate_data.m**:   simulate data from a reinforcement learning agent

**lik_rl.m**:          reinforcement learning likelihood function

**lik_bayes.m**:       Bayesian inference likelihood function

**mfUtil.m**:          various functions including the below
 - *.randomP*            Sample parameter values from prior distributions
 - *.computeEstimates*   Resample all parameters based on their posterior probability
 - *.computeEstimate*    Resample one parameter based on its posterior probability
 - *.fit_prior*          Update the hyperparameters of the prior distribution to reflect the posterior
 - *.logsumexp*          Compute log(sum(exp(x),dim)) avoiding numerical underflow
 - *.randmultinomial*    Generate multinomial random numbers
