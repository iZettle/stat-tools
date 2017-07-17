import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt


def get_axis():
    ax = plt.subplot(1, 1, 1)
    ax.set_facecolor('#FAFAFA')
    ax.figure.set_facecolor('#FAFAFA')
    return ax


def bernoulli_test(control_data, treatment_data, prior_kw={}, sample_kw={}):
    _prior_kw = {'alpha': 1, 'beta': 1}
    _prior_kw.update(prior_kw)

    with pm.Model():
        # Priors
        prob_of_success_control = pm.Beta('prob_of_success_control', **_prior_kw)
        prob_of_success_treatment = pm.Beta('prob_of_success_treatment', **_prior_kw)

        # Likelihood
        pm.Bernoulli('control', p=prob_of_success_control, observed=control_data)
        pm.Bernoulli('treatment', p=prob_of_success_treatment, observed=treatment_data)

        # Metrics
        pm.Deterministic('difference_of_probability', prob_of_success_treatment - prob_of_success_control)

        return pm.sample(**sample_kw)


def bernoulli_probs_plot(trace, ax=None):
    ax = ax if ax else get_axis()

    control = trace.get_values('prob_of_success_control')
    treatment = trace.get_values('prob_of_success_treatment')

    sns.distplot(control, kde_kws={'label': 'control'}, color='#DB6E58', ax=ax)
    sns.distplot(treatment, kde_kws={'label': 'treatment'}, color='#54BF79', ax=ax)
    ax.axvline(control.mean(), linestyle='--', color='#DB6E58', label='mean_control')
    ax.axvline(treatment.mean(), linestyle='--', color='#54BF79', label='mean_treatment')
    plt.legend()


def bernoulli_diff_plot(trace, ax=None):
    ax = ax if ax else get_axis()

    sns.distplot(trace.get_values('difference_of_probability'), kde_kws={'label': 'treatment - control'},
                 color="#1E2649", ax=ax)
    a, b = pm.hpd(trace.get_values('difference_of_probability'))
    ax.axvline(a, linestyle='--', color='#1E2649', label='95% HPD')
    ax.axvline(b, linestyle='--', color='#1E2649')
    plt.legend()


def exponential_test(control_data, treatment_data, prior=pm.Exponential, prior_kw={'lam': 1}, sample_kw={}):
    _prior_kw = {}
    _prior_kw.update(prior_kw)

    with pm.Model():
        # Priors
        lam_control = prior('lam_control', **_prior_kw)
        lam_treatment = prior('lam_treatment', **_prior_kw)

        # Likelihood
        pm.Exponential('control', lam=lam_control, observed=control_data)
        pm.Exponential('treatment', lam=lam_treatment, observed=treatment_data)

        # Metrics
        mean_control = pm.Deterministic('mean_control', 1/lam_control)
        mean_treatment = pm.Deterministic('mean_treatment', 1/lam_treatment)
        pm.Deterministic('difference_of_mean', mean_treatment - mean_control)

        return pm.sample(**sample_kw)


def exponential_mean_plot(trace, ax=None):
    ax = ax if ax else get_axis()

    control = trace.get_values('mean_control')
    treatment = trace.get_values('mean_treatment')

    sns.distplot(control, kde_kws={'label': 'control'}, color='#DB6E58', ax=ax)
    sns.distplot(treatment, kde_kws={'label': 'treatment'}, color='#54BF79', ax=ax)
    ax.axvline(control.mean(), linestyle='--', color='#DB6E58', label='mean_control')
    ax.axvline(treatment.mean(), linestyle='--', color='#54BF79', label='mean_treatment')
    plt.legend()

def exponential_diff_plot(trace, ax=None):
    ax = ax if ax else get_axis()

    sns.distplot(trace.get_values('difference_of_mean'), kde_kws={'label': 'treatment - control'},
                 color="#1E2649", ax=ax)
    a, b = pm.hpd(trace.get_values('difference_of_mean'))
    ax.axvline(a, linestyle='--', color='#1E2649', label='95% HPD')
    ax.axvline(b, linestyle='--', color='#1E2649')
    plt.legend()
