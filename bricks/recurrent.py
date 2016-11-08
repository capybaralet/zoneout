import copy
import numpy as np
numpy = np

from theano import tensor

from blocks.bricks.parallel import Fork
from blocks.bricks import (application, Initializable, Tanh,
                           Logistic, Rectifier, lazy, application)
from blocks.bricks.recurrent import BaseRecurrent, recurrent, LSTM
from blocks.roles import VariableRole, add_role, WEIGHT, INITIAL_STATE

from blocks.utils import shared_floatx_nans, shared_floatx_zeros
from picklable_itertools.extras import equizip


##############################################################
#
#   FOR NORM STABILIZER VARIABLE FILTER
#
##############################################################
class MemoryCellRole(VariableRole):
    pass

MEMORY_CELL = MemoryCellRole()
##############################################################


 
##############################################################
#
#   David GRU
#
##############################################################

class DropGRU(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None, **kwargs):
        self.dim = dim

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        children = [self.activation, self.gate_activation]
        kwargs.setdefault('children', []).extend(children)
        super(DropGRU, self).__init__(**kwargs)

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 3
        if name in ['states', 'drops_states', 'drops_cells', 'drops_igates']:
            return self.dim
        if name == 'mask':
            return 0
        return super(DropGRU, self).get_dim(name)

    def _allocate(self):
        self.W_rz = shared_floatx_nans((self.dim, 2 * self.dim),
                                          name='W_state')
        self.W_htilde = shared_floatx_nans((self.dim, self.dim),
                                          name='W_state')
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = shared_floatx_zeros((self.dim,),
                                                  name="initial_state")
        add_role(self.W_rz, WEIGHT)
        add_role(self.W_htilde, WEIGHT)
        add_role(self.initial_state_, INITIAL_STATE)

        #self.parameters = [self.W_state, self.initial_state_, self.initial_cells]
        self.parameters = [self.W_rz, self.W_htilde, self.initial_state_]

    def _initialize(self):
        for weights in self.parameters[:2]:
            self.weights_init.initialize(weights, self.rng)

    # NTS: scan may complain about unused input?
    @recurrent(sequences=['inputs', 'drops_states', 'drops_cells', 'drops_igates', 'mask'],
               states=['states'],
               contexts=[], outputs=['states'])
    # naming (r, z, htilde) comes from Wojciech's "Empirical Evaluation..."
    def apply(self, inputs, drops_states, drops_cells, drops_igates, states, mask=None):
        def slice_last(x, no):
            return x[:, no * self.dim: (no + 1) * self.dim]

        rz = self.gate_activation.apply(tensor.dot(states, self.W_rz) + inputs[:, self.dim:])
        r = slice_last(rz, 0)
        z = slice_last(rz, 1)
        htilde = self.activation.apply(tensor.dot(r * states, self.W_htilde) + inputs[:, :self.dim])
        next_states = z * states + (1 - z) * htilde * drops_igates
        next_states = next_states * drops_states + (1 - drops_states) * states

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.initial_state_[None, :], batch_size, 0)]
#END David GRU#####################################################
 
 
##############################################################
#
#   LSTM
#
##############################################################

class DropLSTM(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None,
                 model_type=6, ogates_zoneout=False, **kwargs):
        self.dim = dim
        self.model_type = model_type
        self.ogates_zoneout = ogates_zoneout

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        children = [self.activation, self.gate_activation]
        kwargs.setdefault('children', []).extend(children)
        super(DropLSTM, self).__init__(**kwargs)

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells', 'drops_states', 'drops_cells', 'drops_igates']:
            return self.dim
        if name == 'mask':
            return 0
        return super(DropLSTM, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4 * self.dim),
                                          name='W_state')
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        # self.initial_state_ = shared_floatx_zeros((self.dim,),
        #                                          name="initial_state")
        # self.initial_cells = shared_floatx_zeros((self.dim,),
        #                                         name="initial_cells")
        add_role(self.W_state, WEIGHT)

        self.parameters = [
            self.W_state]  # , self.initial_state_, self.initial_cells]

    def _initialize(self):
        for weights in self.parameters[:1]:
            self.weights_init.initialize(weights, self.rng)

    @recurrent(sequences=['inputs', 'drops_states', 'drops_cells', 'drops_igates', 'mask'],
               states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, drops_states, drops_cells, drops_igates, states, cells, mask=None):
        def slice_last(x, no):
            return x[:, no * self.dim: (no + 1) * self.dim]

        activation = tensor.dot(states, self.W_state) + inputs
        in_gate = self.gate_activation.apply(slice_last(activation, 0)) * drops_igates #elephant
        forget_gate_input = slice_last(activation, 1)
        forget_gate = self.gate_activation.apply(
            forget_gate_input + tensor.ones_like(forget_gate_input))
        next_cells = (
            forget_gate * cells +
            in_gate * self.activation.apply(slice_last(activation, 2)))
        out_gate = self.gate_activation.apply(slice_last(activation, 3))
        next_states = out_gate * self.activation.apply(next_cells)

        # Apply zoneout
        next_states = next_states * drops_states + (1 - drops_states) * states
        next_cells = next_cells * drops_cells + (1 - drops_cells) * cells
            
        if self.ogates_zoneout:
            next_states = drops_igates * next_states + (1 - drops_igates) * forget_gate * states

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells

#    @application(outputs=apply.states)
#    def initial_states(self, batch_size, *args, **kwargs):
#        import ipdb
#        ipdb.set_trace()
#        return [tensor.repeat(self.initial_state_[None, :], batch_size, 0),
#                tensor.repeat(self.initial_cells[None, :], batch_size, 0)]
    
#END LSTM#####################################################




##############################################################
#
#   Simple RNN
#
##############################################################

class DropSimpleRecurrent(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, **kwargs):
        self.dim = dim
        children = [activation] + kwargs.get('children', [])
        super(DropSimpleRecurrent, self).__init__(children=children, **kwargs)

    @property
    def W(self):
        return self.parameters[0]

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim
        if name in ['states', 'drops_states', 'drops_cells', 'drops_igates']:
            return self.dim
        if name == 'mask':
            return 0
        return super(DropSimpleRecurrent, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                                                  name="W"))
        add_role(self.parameters[0], WEIGHT)
        self.parameters.append(shared_floatx_zeros((self.dim,),
                                                   name="initial_state"))
        add_role(self.parameters[1], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @recurrent(sequences=['inputs', 'drops_states', 'drops_cells', 'drops_igates', 'mask'], states=['states'],
               outputs=['states'], contexts=[])
    def apply(self, inputs, drops_states, drops_cells, drops_igates, states, mask=None):
        next_states = inputs + tensor.dot(states, self.W)
        next_states = self.children[0].apply(next_states)

        # apply zoneout
        next_states = drops_states * next_states + (1 - drops_states) * states

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return tensor.repeat(self.parameters[1][None, :], batch_size, 0)

#END SRNN#####################################################

