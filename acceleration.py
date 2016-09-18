"""Training algorithms."""
import logging
import itertools
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from six.moves import reduce

from picklable_itertools.extras import equizip

import theano
from six import add_metaclass
from theano import tensor

from blocks.graph import ComputationGraph
from blocks.utils import dict_subset, named_copy, pack, shared_floatx
from blocks.theano_expressions import l2_norm

logger = logging.getLogger(__name__)

from blocks.algorithms import StepRule

# TODO: generalize for more higher order terms...
class Acceleration(StepRule):
    """
        Acceleration comes down to using this matrix and no nonlinearities in an RNN:
        [A']   [a 0 1] * [A]
        [V'] = [a b 1]   [V]
                         [G]
        and then the update is just V'
    """
    def __init__(self, learning_rate, dynamics_decay_params):
        self.learning_rate = learning_rate
        # in decreasing order
        self.dynamics_decay_params = [shared_floatx(item) for item in dynamics_decay_params]

    # previous_step = gradient
    def compute_step(self, param, previous_step):
        dynamics = [shared_floatx(param.get_value() * 0.) for item in self.dynamics_decay_params]
        dynamics_updates = []
        for n in range(len(dynamics)):
            dynamics_updates.append( (dynamics[n], 
                previous_step + sum([dynamics[k] * self.dynamics_decay_params[k] for k in range(n+1)]) ) )
        step = self.learning_rate * dynamics_updates[-1][1]
        return step, dynamics_updates


from numpy.testing import assert_allclose, assert_raises

def test_acceleration():
    a = shared_floatx([3, 4])
    cost = (a ** 2).sum()
    learning_rate = .1
    dynamics_decay_params = [.5]
    steps, updates = Acceleration(learning_rate,  dynamics_decay_params).compute_steps(
        OrderedDict([(a, tensor.grad(cost, a))]))
    f = theano.function([], [steps[a]], updates=updates)
    assert_allclose(f()[0], [0.6, 0.8])
    assert_allclose(f()[0], [0.9, 1.2])
    assert_allclose(f()[0], [1.05, 1.4])

test_acceleration()

lr = .1
acc, mom = [.5, .5]

import numpy
np = numpy
def compute_updates(lr, acc, mom, A_V_G):
    A_V = np.dot(np.array([ [acc, 0, 1], [acc, mom, 1] ]),  A_V_G)
    print A_V
    return A_V

def gg(x):
    return 2*x

pp = 5
a_v_g = [0,0,0]
stps = []
for i in range(3):
    a_v_g[-1] = gg(pp)
    upd = compute_updates(lr, acc, mom, a_v_g)
    a_v_g[0] = upd[0]
    a_v_g[1] = upd[1]
    stps.append(lr * upd[1])

if 1:
    # N.B. NOT ACTUALLY APPLYING UPDATES (just accumulating them!)
    def test_acceleration():
        a = shared_floatx([pp])
        cost = (a ** 2).sum()
        dynamics_decay_params = [acc, mom]
        accel = Acceleration(lr,  dynamics_decay_params)
        steps, updates = accel.compute_steps(OrderedDict([(a, tensor.grad(cost, a))]))
        upd_fn = theano.function([], [ uu[1] for uu in updates])
        f = theano.function([], [steps[a]], updates=updates)
        r1, r2, r3 = stps#[0], stps[1], stps[2]
        assert_allclose(f()[0], r1)
        assert_allclose(f()[0], r2)
        assert_allclose(f()[0], r3)

    test_acceleration()

