__author__ = 'antonio franco'

'''
Copyright (C) 2019  Antonio Franco (antonio_franco@live.it)
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
from mpmath import *
mp.dps = 25
mp.pretty = True


def gen_erl_sum_MC(rates, shapez, n_samples):
    """
    Generates n_samples samples from the sum of len(rates) Erlang distributions with parameters rates(i) and shapez(i).
    :param rates (list of floats): list of rates
    :param shapez (list of int): list of shapes. must be have the same length as len(rates)
    :param n_samples (int): number of samples to be generated
    :return (list of floats): samples.
    """
    assert(isinstance(rates, list))
    assert(isinstance(shapez, list))
    assert(isinstance(n_samples, int))
    assert(all(isinstance(x, int) for x in shapez))
    assert(len(rates) == len(shapez))

    S = np.zeros((n_samples,))
    for l, n in zip(rates, shapez):
        for i in range(0, n):
            betas = 1.0/l * np.ones((n_samples,))
            S += np.random.exponential(betas)

    return S


def erl_sum_CDF(rates, shapez, y):
    """
    Evaluates the CDF of the sum of len(rates) Erlang distributions with parameters rates(i) and shapez(i) in y,
    according to: Imran Shafique Ansari and Ferkan Yilmaz and Mohamed-Slim Alouini and Oguz Kucur:
    "New Results on the Sum of Gamma Random Variates With Application to the Performance of Wireless Communication Systems
    over Nakagami-m Fading Channels", https://arxiv.org/abs/1202.2576
    :param rates (list of floats): list of rates
    :param shapez (list of int): list of shapes. must be have the same length as len(rates)
    :param y (float): point where to evaluate the CDF
    :return (float): CDF in y.
    """
    assert(isinstance(rates, list))
    assert(isinstance(shapez, list))
    assert(all(isinstance(x, int) for x in shapez))
    assert(len(rates) == len(shapez))

    K = 1
    psi1 = []
    psi2 = []

    z = exp(-y)

    for l, s in zip(rates, shapez):
        psi1.extend((1.0 + l)*ones(1, s))
        psi2.extend(l*ones(1, s))
        K *= l**s

    psi1.append(1.0)
    psi2.append(0.0)

    F = K * meijerg([[], psi1], [psi2, []], z)

    return float(F)


def erl_sum_PDF(rates, shapez, y):
    """
    Evaluates the PDF of the sum of len(rates) Erlang distributions with parameters rates(i) and shapez(i) in y,
    according to: Imran Shafique Ansari and Ferkan Yilmaz and Mohamed-Slim Alouini and Oguz Kucur:
    "New Results on the Sum of Gamma Random Variates With Application to the Performance of Wireless Communication Systems
    over Nakagami-m Fading Channels", https://arxiv.org/abs/1202.2576
    :param rates (list of floats): list of rates
    :param shapez (list of int): list of shapes. must be have the same length as len(rates)
    :param y (float): point where to evaluate the PDF
    :return (float): CDF in y.
    """
    assert(isinstance(rates, list))
    assert(isinstance(shapez, list))
    assert(all(isinstance(x, int) for x in shapez))
    assert(len(rates) == len(shapez))

    K = 1
    psi1 = []
    psi2 = []

    z = exp(-y)

    for l, s in zip(rates, shapez):
        psi1.extend((1.0 + l)*ones(1, s))
        psi2.extend(l*ones(1, s))
        K *= l**s

    f = K * meijerg([[], psi1], [psi2, []], z)

    return float(f)


import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    Tests the algorithm against a Montecarlo simulation
    """
    np.random.seed(19680801) # For reproducibility

    # Main parameters
    n_samples = int(1e4)
    n_erls = 5
    n_exps = 5
    n_bins = 50

    lambdas = np.random.rand(n_erls).tolist()
    shapes = np.random.randint(low=1, high=n_exps+1, size=n_erls, dtype='int').tolist()

    S = gen_erl_sum_MC(lambdas, shapes, n_samples)

    plt.figure()
    n, bins, patches = plt.hist(S, n_bins, density=True, facecolor='b', alpha=0.75, label='Montecarlo')
    Y = np.linspace(bins.min(), bins.max(), n_bins*2)
    F = [erl_sum_PDF(lambdas, shapes, y) for y in Y]
    plt.plot(Y, F, label='Analytical')
    plt.xlabel('Y')
    plt.ylabel('PDF')
    plt.title('Sum of ' + str(n_erls) + ' Erlang random variables')
    plt.legend()

    plt.figure()
    n, bins, patches = plt.hist(S, n_bins, density=True, cumulative=True, facecolor='b', alpha=0.75, label='Montecarlo')
    Y = np.linspace(bins.min(), bins.max(), n_bins*2)
    F = [erl_sum_CDF(lambdas, shapes, y) for y in Y]
    plt.plot(Y, F, label='Analytical')
    plt.xlabel('Y')
    plt.ylabel('CDF')
    plt.title('Sum of ' + str(n_erls) + ' Erlang random variables')
    plt.legend()

    plt.show()
