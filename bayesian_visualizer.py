import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma, poisson, bernoulli

st.set_page_config(page_title="Bayesian Conjugate Priors", layout="centered")
st.title("📊 Bayesian Conjugate Prior Visualizer")

# --- Simulate data ---
n_trials = st.sidebar.slider("Number of Observations", 10, 100, 50)
true_p = st.sidebar.slider("True Probability (p)", 0.1, 0.9, 0.7)
true_lambda = st.sidebar.slider("True Rate (λ)", 1, 10, 4)

binary_data = bernoulli.rvs(true_p, size=n_trials)
poisson_data = poisson.rvs(true_lambda, size=n_trials)

# --- Beta-Binomial Posterior ---
st.header("🔵 Beta-Binomial Model")

alpha_prior = st.slider("Beta Prior: α", 1, 10, 2)
beta_prior = st.slider("Beta Prior: β", 1, 10, 2)

successes = np.sum(binary_data)
failures = n_trials - successes
alpha_post = alpha_prior + successes
beta_post = beta_prior + failures

x = np.linspace(0, 1, 100)
fig1, ax1 = plt.subplots()
ax1.plot(x, beta.pdf(x, alpha_prior, beta_prior), '--', label='Prior')
ax1.plot(x, beta.pdf(x, alpha_post, beta_post), label='Posterior')
ax1.set_title("Beta-Binomial Posterior")
ax1.set_xlabel("Probability of Success (p)")
ax1.set_ylabel("Density")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# --- Gamma-Poisson Posterior ---
st.header("🟣 Gamma-Poisson Model")

shape_prior = st.slider("Gamma Prior: Shape", 1, 10, 2)
rate_prior = st.slider("Gamma Prior: Rate", 1, 10, 1)

sum_counts = np.sum(poisson_data)
shape_post = shape_prior + sum_counts
rate_post = rate_prior + n_trials

x = np.linspace(0, 10, 100)
fig2, ax2 = plt.subplots()
ax2.plot(x, gamma.pdf(x, a=shape_prior, scale=1/rate_prior), '--', label='Prior')
ax2.plot(x, gamma.pdf(x, a=shape_post, scale=1/rate_post), label='Posterior')
ax2.set_title("Gamma-Poisson Posterior")
ax2.set_xlabel("Rate Parameter (λ)")
ax2.set_ylabel("Density")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# --- Summary ---
st.header("📌 Posterior Parameters")
st.markdown(f"""
- **Beta-Binomial Final Posterior**: α = `{alpha_post}`, β = `{beta_post}`
- **Gamma-Poisson Final Posterior**: Shape = `{shape_post}`, Rate = `{rate_post}`
""")
