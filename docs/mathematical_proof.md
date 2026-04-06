# Mathematical Demonstration of the Flaws

This document provides a rigorous mathematical proof demonstrating why the current implementations in [src/deliverables.py](file:///d:/Projects/MMM-Figshare-eCommerce/src/deliverables.py) and [src/insights.py](file:///d:/Projects/MMM-Figshare-eCommerce/src/insights.py) produce mathematically invalid outputs for channel contributions and budget optimization.

---

## 1. The True Data Generating Process (DGP)

The PyMC model defined in [src/models/hierarchical_bayesian.py](file:///d:/Projects/MMM-Figshare-eCommerce/src/models/hierarchical_bayesian.py) applies a log transformation to the target variable (`Revenue`) during preprocessing $\tilde{Y} = \ln(Y)$. 

The model specification for the expected value of $\tilde{Y}$ at time $t$ is additive:
$$ \mu_t = \text{Base}_t + \sum_{c=1}^{C} \beta_c \cdot \text{Sat}_c(\text{Adstock}_c(X_{c,t})) $$

Where:
- $\mu_t = E[\ln(Y_t)]$
- $X_{c,t}$ is the spend for channel $c$ at time $t$.
- $\text{Sat}_c(\dots)$ is the Hill saturation function.

The observation likelihood is robust Student-T:
$$ \ln(Y_t) \sim \text{StudentT}(\mu_t, \sigma, \nu) $$

## 2. Inferred Expected Value in Linear Space

To analyze contributions in actual dollars (linear space), we must exponentiate. The relationship between log-space and linear-space expectations for $Y_t$ relies on the properties of the distribution.

For a normally distributed error in log-space $\ln(Y_t) \sim \mathcal{N}(\mu_t, \sigma^2)$, the linear expectation is given by the log-normal mean:
$$ E[Y_t] = \exp\left(\mu_t + \frac{\sigma^2}{2}\right) $$

For a Student-T distribution, the exact expectation in linear space is generally undefined without truncation, but it is typically approximated or empirically computed from posterior predictive samples:
$$ E[Y_t] \approx \frac{1}{S} \sum_{s=1}^{S} \exp(\mu_t^{(s)} + \epsilon_t^{(s)}) $$

Critically, **the additive model in log-space becomes multiplicative in linear-space**:
$$ \exp(\mu_t) = \exp(\text{Base}_t) \cdot \prod_{c=1}^{C} \exp\left( \beta_c \cdot \text{Sat}_c(...) \right) $$

## 3. The Flawed Log-to-Linear Scaling Implementation

In [src/deliverables.py](file:///d:/Projects/MMM-Figshare-eCommerce/src/deliverables.py), the code ignores the non-linear relationship and calculates a global linear scaling factor:
$$ C_{scale} = \frac{\sum Y_t^{true}}{\sum \mu_t^{pred}} $$

It then computes the dollar contribution for channel $c$ by simply multiplying the log-space attribution by this scalar:
$$ \text{Contribution}_c^{linear} = C_{scale} \cdot \left( \sum_{t} \beta_c \cdot \text{Sat}_c(\dots) \right) $$

### Why is this mathematically invalid?
This formulation assumes:
$$ E[\exp(X)] = C_{scale} \cdot E[X] $$

This severely violates **Jensen's Inequality**, which dictates that for a convex function like $\exp(\dots)$:
$$ E[\exp(X)] \geq \exp(E[X]) $$

By treating an additive component of a log-normal distribution as a simple fraction of a linear sum, the code entirely ignores the compounding, multiplicative effect of multiple channels operating simultaneously. 

### What is the Correct Calculation?
The true marginal contribution of a channel $c$ in linear space dollars is the difference in expected revenue *with* and *without* that channel:
$$ \text{Contribution}_c = E[Y_t | X_{all}] - E[Y_t | X_{all \setminus c}=0] $$

$$ \text{Contribution}_c = E\left[\exp\left(\text{Base}_t + \dots + \beta_c \text{Sat}_c \right)\right] - E\left[\exp\left(\text{Base}_t + \dots + 0 \right)\right] $$

This represents the actual dollar drop if the channel was turned off, accounting for the correct synergistic scales of the overall baseline.

---

## 4. Compounding Error in Marginal ROAS and Budget Optimization

Because the baseline $\text{Contribution}_c^{linear}$ is mathematically corrupted by the scalar factor $C_{scale}$, everything downstream is uniquely impacted.

In [src/insights.py](file:///d:/Projects/MMM-Figshare-eCommerce/src/insights.py), the budget optimizer creates an objective function based on a reverse-engineered `scale` parameter:
$$ \text{scale}_{opt} = \frac{\text{Contribution}_c^{linear}}{ \text{obs} \cdot \text{Sat}_c(\dots) } $$

The optimizer's objective function (to be maximized) then becomes:
$$ \text{Maximize } \sum_{c=1}^{C} \left( \text{scale}_{opt} \cdot \text{Sat}_c(x) \right) $$

### Why is this mathematically invalid?
The objective function above defines a **strictly additive model in linear space**. It treats total revenue simply as $R(x_{1 \dots n}) = \sum k_c \cdot \text{Sat}_c(x_c)$.
However, as demonstrated in section 2, the true PyMC posterior defines a model that is **strictly multiplicative in linear space**:
$$ R(x_{1 \dots n}) = \exp(\text{Base}) \cdot \prod \exp(...) $$

Because the solver is optimizing an additive proxy of a multiplicative posterior, the resulting marginal derivatives (Marginal ROAS) and the final optimized budget allocations will drift heavily away from the true optimal state discovered by the NUTS MCMC sampler.

**Conclusion:** To fix these issues, the deliverables and optimization engine must compute expected values directly over the exponentiated posterior predictive tensor matrices, rather than proxying via `scale_factor`.
