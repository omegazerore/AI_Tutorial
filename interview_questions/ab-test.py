"""
Say you’re testing if adding reactions increases engagement rate:

Baseline reaction rate: 10%

You want to detect at least a 1% lift (i.e., 11% in treatment)

α = 0.05, Power = 0.8

Using power analysis, you calculate that you'll need ~15,000 users per group.
"""


from statsmodels.stats.power import NormalIndPower

effect_size = 0.2  # Cohen's h (for proportion difference)
alpha = 0.05
power = 0.8

analysis = NormalIndPower()
sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided')

print(sample_size)