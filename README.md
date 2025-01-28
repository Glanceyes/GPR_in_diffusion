**# [KAIST AI503 Project] Gaussian Process Regression in Diffusion Models for Time Series Data**

In this project, we explore how to incorporate ***\*Gaussian Process Regression (GPR)\**** into a ***\*Diffusion Model\**** for one-dimensional time series data. The general idea is as follows:

<br/>

1. ***\*Diffusion Model Setup\****:  

   We start with a diffusion model that progressively adds noise to the data through a forward noising process. Formally, if we denote the original data as $x_0$, at each diffusion step $t$ we obtain:
$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim N(0,I),
$$


   where $\bar{\alpha}_t$ is a function of the noise schedule. The model then learns to reverse this process, ultimately attempting to recover a clean estimate of $x_0$ from a noisy $x_t$.

<br/>

2. ***\*Gaussian Process Regression (GPR) as an Energy Function\****:  

   GPR will be employed to guide the diffusion reverse process. The GPR model provides a predictive mean and variance, which can be interpreted as an energy landscape:
$$
E(x) = \frac{(x - \mu_{\text{GPR}})^2}{2 \sigma_{\text{GPR}}^2},$$
$$


   where $\mu_{\text{GPR}}$ and $\sigma_{\text{GPR}}$ are the GPR predictive mean and standard deviation for the target time step. By treating the GPR output as an energy function, we can modify the reverse diffusion step to push our sample $x_t$ towards regions favored by the GPR prediction, potentially achieving more accurate estimates of the underlying signal.

<br/>

3. ***\*Time Series Application\****:  

   We focus on a real-world time series dataset - PM2.5 measurement. After applying the diffusion model to denoise the data, we use GPR to guide the denoising trajectory. We will also compare scenarios with and without GPR guidance, and even introduce Stein Variational Gradient Descent (SVGD) steps to refine the final samples.