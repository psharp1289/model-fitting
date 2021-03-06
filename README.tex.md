# Hierarchical Model Fitting

MATLAB code for hierarchical model fitting using iterative importance sampling. 

The fitting procedure is demonstrated by fitting RL and Bayesian models to simulated data.

### Fitting algorithm

Define model parameters' prior distributions $ \text{p}(\theta) $, and repeat until mean likelihood stops increasing:
 1. Sample parameter values from  $ \text{p}(\theta) $
 2. Compute likelihood $ \text{p}(\text{data}|\theta) $ for each sample 
 3. Derive $ \text{p}(\theta|\text{data}) \propto \text{p}(\theta)\text{p}(\text{data}|\theta) $, by resampling the parameter values with likelihoods as weights
 4. Fit prior distributions to the resampled values: $ \text{p}(\theta) \leftarrow \text{p}(\theta|\text{data}) $

### Reinforcement learning model
Parameters: $ \eta $ - learning rate
$$ \begin{matrix} \forall i & Q_0(i) = 0 & Q_0(i) = 0 \end{matrix} $$
$$ Q_{t+1}(c_t) = Q_t(c_t) + \eta(\text{outcome}_t - Q_t(c_t)) $$

### Bayesian inference model
Parameters: $ \phi $ - shape of prior distribution
$$ \begin{matrix} \forall i & A_0(i) = \phi & B_0(i) = \phi \end{matrix} $$
$$ \begin{matrix} \text{if } \text{outcome}_t = 1 & A _{t+1}(c_t) = A_t(c_t) + 1 \\\ \text{if } \text{outcome}_t = 0 & B _{t+1}(c_t) = B_t(c_t) + 1 \end{matrix} $$ 
$$ Q_t(c_t) = \frac{A_t(c_t)}{A_t(c_t)+B_t(c_t)} $$

### Choice probability (for both models)
Parameters: $ \beta $ - inverse temperature
$$ \text{p}(c_t) = \frac{e^{\beta Q_t(c_t)}}{\sum_i{e^{\beta Q_t(i)}}} $$

Legend: $ Q $ - expected value, $ c $ - choice, $ t $ - trial
## Scripts
 - **demo.m**:   demonstrates fitting the two models to simulated data, then using one of the models to simulate new data and examine it against the original data
 - **simulate_data.m**:   simulate original data from a reinforcement learning agent
 - **lik_rl.m**:          reinforcement learning likelihood function (also simulates choices)
 - **lik_bayes.m**:       Bayesian inference likelihood function
 - **mfUtil.m**:          various functions including the below
    - *.randomP*           - Sample parameter values from prior distributions
    - *.computeEstimates*  - Resample all parameters based on their posterior probability
    - *.computeEstimate*   - Resample one parameter based on its posterior probability
    - *.fit_prior*         - Update the hyperparameters of the prior distribution to reflect the posterior
    - *.logsumexp*         - Compute log(sum(exp(x),dim)) avoiding numerical underflow
    - *.randmultinomial*   - Generate multinomial random numbers
    - *.fit2P*   		   - organize the results of model fitting for one subject such that they can be fed back into the likelihiid function for simulation 

