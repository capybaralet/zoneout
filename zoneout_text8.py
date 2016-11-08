# standard libraries
import argparse
import logging
import os
import sys
import time

# numpy / theano
import numpy
np = numpy
import theano
import theano.tensor as T

# blocks / fuel
from blocks.algorithms import (GradientDescent, StepClipping, CompositeRule,
                               Momentum, Adam, RMSProp)
from blocks.bricks import Tanh, Softmax, Linear, Rectifier
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks.recurrent import LSTM, GatedRecurrent, SimpleRecurrent
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.extensions.saveload import Load
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.monitoring import aggregation
from blocks.model import Model
from blocks.roles import WEIGHT, OUTPUT
from fuel.datasets import IndexableDataset

# this repo
from utils_DATA import get_text8_stream
from bricks.recurrent import DropBidirectionalGraves, DropLSTM, DropGRU, DropSimpleRecurrent, MEMORY_CELL
from extensions import SaveLog, SaveParams
from initialization import NormalizedInitialization, OrthogonalInitialization


floatX = theano.config.floatX
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)


def learning_algorithm(args):
    name = args.algorithm
    learning_rate = float(args.learning_rate)
    momentum = args.momentum
    clipping_threshold = args.clipping
    if name == 'adam':
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        adam = Adam(learning_rate=learning_rate)
        step_rule = CompositeRule([adam, clipping])
    elif name == 'rms_prop':
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        rms_prop = RMSProp(learning_rate=learning_rate)
        step_rule = CompositeRule([clipping, rms_prop])
    else:
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        sgd_momentum = Momentum(learning_rate=learning_rate, momentum=momentum)
        step_rule = CompositeRule([clipping, sgd_momentum])
    return step_rule


def parse_args():
    parser = argparse.ArgumentParser(description='Text8experiment')
    parser.add_argument('--experiment_path', type=str,
                        default='./text8',
                        help='Location for writing results')
    parser.add_argument('--state_dim', type=int,
                        default=2000,
                        help='States dimensions')
    parser.add_argument('--epochs', type=int,
                        default=2000,
                        help='Number of epochs to train for')
    parser.add_argument('--seed', type=int,
                        default=123,
                        help='Random generator seed')
    parser.add_argument('--load_path',
                        default=argparse.SUPPRESS,
                        help='File with parameter to be loaded)')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_noise', type=float, default=0,
                        help='stdv of weight noise')
    parser.add_argument('--momentum',
                        default=0.99,
                        type=float,
                        help='Momentum for SGD')
    parser.add_argument('--clipping',
                        default=1,
                        type=float,
                        help='Gradient clipping norm')
    parser.add_argument('--test_cost', action='store_true',
                        default=False,
                        help='Report test set cost')
    parser.add_argument('--algorithm', choices=['rms_prop', 'adam',
                                                'adam'],
                        default='adam',
                        help='Learning algorithm to use')
    parser.add_argument('--subsample', action='store_true',
                        default=False,
                        help='Subsample input')
    parser.add_argument('--features',
                        default='fbank',
                        help='Type of features to use',
                        choices=['log_power_spectrum', 'fbank'])
    parser.add_argument('--pool_size', type=int,
                        default=200,
                        help='Pool size for dataset')
    parser.add_argument('--maximum_frames', type=int,
                        default=10000,
                        help='Pool size for dataset')
    parser.add_argument('--window_features', type=int,
                        default=1,
                        help='Use neighbour frames (window_features / 2'
                             'before and window_features / 2 after).'
                             'Should be odd.')
    parser.add_argument('--initialization', choices=['glorot', 'uniform', 'identity', 'ortho'],
                        default='ortho')
    parser.add_argument('--drop_prob',
                        type=str,
                        default='1',
                        help='(for rnn with shared mask and for SRNN/GRU) Despite the name, is the update probability. Sorry.')
    parser.add_argument('--drop_prob_states',
                        type=float,
                        default=1.0,
                        help='(for rnn with non-shared mask) Despite the name, is the update probability. Sorry.')
    parser.add_argument('--drop_prob_cells',
                        type=float,
                        default=1.0,
                        help='(for rnn with non-shared mask) Despite the name, is the update probability. Sorry.')
    parser.add_argument('--drop_prob_igates',
                        type=float,
                        default=1.0,
                        help='(for rnn with drop on input gate ala Recurrent dropout without memory loss paper.')
    parser.add_argument('--ogates_zoneout', action='store_true',
                        default=False,
                        help='Zone out the output gates reusing the masks from the input gate dropout')
    parser.add_argument('--stoch_depth', action='store_true',
                        default=False,
                        help='Use stochastic depth, i.e. dropout on full cell/state layers')
    parser.add_argument('--share_mask', action='store_true',
                        default=False,
                        help='Use the same mask for cells and states')
    parser.add_argument('--rnn_type',
                        default='LSTM',
                        help='options: LSTM, GRU, SRNN')
    parser.add_argument('--norm_cost_coeff',
                        type=float,
                        default=0,
                        help='options: LSTM, GRU, SRNN')
    parser.add_argument('--input_drop',
                        type=float,
                        default=0,
                        help='options: 0.x (e.g. 0.5) for probability per timestep')
    parser.add_argument('--penalty', type=str, default='hids',
                        choices=['hids', 'cells'])
    parser.add_argument('--seq_len', type=int, default='180', help='for chopping up data')
    parser.add_argument('--batch_size', default=128, type=int)
    return parser.parse_args()


def train(step_rule, state_dim, epochs, seed, 
          test_cost, experiment_path, 
          features, pool_size, maximum_frames, initialization, weight_noise,
          drop_prob, drop_prob_states, drop_prob_cells, drop_prob_igates, ogates_zoneout, batch_size,
          stoch_depth, share_mask, rnn_type, norm_cost_coeff, penalty, seq_len, input_drop,
          **kwargs):

    print '.. PTB experiment'
    print '.. arguments:', ' '.join(sys.argv)
    t0 = time.time()

    ###########################################
    #
    # MAKE STREAMS
    #
    ###########################################
    rng = np.random.RandomState(seed)

    if share_mask:
        drop_prob_cells = drop_prob
        # when the mask is shared, we use the drop_prob_cells mask for both cells and states
        drop_prob_states = None

    print '.. initializing iterators'
    
    train_stream = get_text8_stream(
        'train', batch_size, seq_len, drop_prob_states, drop_prob_cells, drop_prob_igates, state_dim, False)
    train_stream_evaluation = get_text8_stream(
        'train', batch_size, seq_len, drop_prob_states, drop_prob_cells, drop_prob_igates, state_dim, True)
    dev_stream = get_text8_stream(
        'valid', batch_size, seq_len, drop_prob_states, drop_prob_cells, drop_prob_igates, state_dim, True)

    data = train_stream.get_epoch_iterator(as_dict=True).next()
    ####################


    ###########################################
    #
    # BUILD MODEL
    #
    ###########################################

    print '.. building model'
    
    # the input data, axes are: (timestep, example index in minibatch, input dimension)  (aka T, B, N)
    x = T.tensor3('features', dtype=floatX)
    x, y = x[:-1], x[1:]
    drops_states = T.tensor3('drops_states')
    drops_cells = T.tensor3('drops_cells')
    drops_igates = T.tensor3('drops_igates')

    x.tag.test_value = data['features']
    drops_states.tag.test_value = data['drops_states']
    drops_cells.tag.test_value = data['drops_cells']
    drops_igates.tag.test_value = data['drops_igates']


    if initialization == 'glorot':
        weights_init = NormalizedInitialization()
    elif initialization == 'uniform':
        weights_init = Uniform(width=.2)
    elif initialization == 'ortho':
        weights_init = OrthogonalInitialization()
    else:
        raise ValueError('No such initialization')
    
    
    if rnn_type.lower()=='lstm':
        in_to_hid = Linear(27, state_dim*4, name='in_to_hid',
                       weights_init=weights_init, biases_init=Constant(0.0))
        recurrent_layer = DropLSTM(dim=state_dim, weights_init=weights_init, activation=Tanh(), name='rnn', ogates_zoneout=ogates_zoneout)
    elif rnn_type.lower()=='gru':
        in_to_hid = Linear(27, state_dim*3, name='in_to_hid',
                       weights_init=weights_init, biases_init=Constant(0.0))
        recurrent_layer = DropGRU(dim=state_dim, weights_init=weights_init, activation=Tanh(), name='rnn')
    elif rnn_type.lower()=='srnn':
        in_to_hid = Linear(27, state_dim, name='in_to_hid',
                       weights_init=weights_init, biases_init=Constant(0.0))
        recurrent_layer = DropSimpleRecurrent(dim=state_dim, weights_init=weights_init, activation=Rectifier(), name='rnn')
    else:
        raise NotImplementedError
    
    hid_to_out = Linear(state_dim, 27, name='hid_to_out',
                        weights_init=weights_init, biases_init=Constant(0.0))
    
    in_to_hid.initialize()
    recurrent_layer.initialize()
    hid_to_out.initialize()
    
    h = in_to_hid.apply(x)
    
    if rnn_type.lower() == 'lstm':
        yh = recurrent_layer.apply(h, drops_states, drops_cells, drops_igates)[0]
    else:
        yh = recurrent_layer.apply(h, drops_states, drops_cells, drops_igates)
    
    y_hat_pre_softmax = hid_to_out.apply(yh)
    shape_ = y_hat_pre_softmax.shape

    ####################


    ###########################################
    #
    # SET UP COSTS AND MONITORS
    #
    ###########################################

    def crossentropy_lastaxes(yhat, y):
        # for sequence of distributions/targets
        return -(y * T.log(yhat)).sum(axis=yhat.ndim - 1)

    def softmax_lastaxis(x):
        # for sequence of distributions
        return T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape)

    yhat = softmax_lastaxis(y_hat_pre_softmax)
    cross_entropies = crossentropy_lastaxes(yhat, y)
    cross_entropy = cross_entropies.mean().copy(name="cross_entropy")
    cost = cross_entropy.copy(name="cost")


    batch_cost = cost.copy(name='batch_cost')
    nll_cost = cost.copy(name='nll_cost')
    bpc = (nll_cost/np.log(2.0)).copy(name='bpr')

    
    cost_monitor = aggregation.mean(
        batch_cost, batch_size).copy(name='sequence_cost_monitor')
    cost_per_character = aggregation.mean(
        batch_cost, (seq_len - 1) * batch_size).copy(name='character_cost')
    cost_train = cost.copy(name='train_batch_cost')
    cost_train_monitor = cost_monitor.copy('train_batch_cost_monitor')
    cg_train = ComputationGraph([cost_train, cost_train_monitor])

    ##################### DK ADD COST ########################
    ##################### DK ADD COST ########################
    norm_cost = 0.

    def _magnitude(x, axis=-1):
        return T.sqrt(T.maximum(T.sqr(x).sum(axis=axis), numpy.finfo(x.dtype).tiny))

    if penalty == 'cells':
        assert VariableFilter(roles=[MEMORY_CELL])(cg_train.variables)
        for cell in VariableFilter(roles=[MEMORY_CELL])(cg_train.variables):
            norms = _magnitude(cell)
            norm_cost += T.mean(T.sum((norms[1:] - norms[:-1])**2, axis=0) / (seq_len - 1))
    elif penalty == 'hids':
        assert 'rnn_apply_states' in [o.name for o in VariableFilter(roles=[OUTPUT])(cg_train.variables)]
        for output in VariableFilter(roles=[OUTPUT])(cg_train.variables):
            if output.name == 'rnn_apply_states':
                norms = _magnitude(output)
                norm_cost += T.mean(T.sum((norms[1:] - norms[:-1])**2, axis=0) / (seq_len - 1))

    norm_cost.name = 'norm_cost'
    #cost_valid = cost_train
    cost_train += norm_cost_coeff * norm_cost
    cost_train = cost_train.copy('cost_train') #should this be cost_train.outputs[0]?

    cg_train = ComputationGraph([cost_train, cost_train_monitor])#, norm_cost])

    ##################### DK ADD COST ########################
    ##################### DK ADD COST ########################


    if weight_noise > 0:
        weights = VariableFilter(roles=[WEIGHT])(cg_train.variables)
        cg_train = apply_noise(cg_train, weights, weight_noise)
        cost_train = cg_train.outputs[0].copy(name='cost_train')
        cost_train_monitor = cg_train.outputs[1].copy(
            'train_batch_cost_monitor')

    model = Model(cost_train)
    train_cost_per_character = aggregation.mean(
        cost_train_monitor,
        (seq_len - 1) * batch_size).copy(name='train_character_cost')

    algorithm = GradientDescent(step_rule=step_rule, cost=cost_train,
                                parameters=cg_train.parameters)

    observed_vars = [cost_train,
                     cost_train_monitor, train_cost_per_character,
                     aggregation.mean(algorithm.total_gradient_norm)]
    train_monitor = TrainingDataMonitoring(
        variables=observed_vars,
        prefix="train", after_epoch=True)

    dev_monitor = DataStreamMonitoring(
        variables=[nll_cost, bpc],
        data_stream=dev_stream, prefix="dev"
    )
    extensions = []
    if 'load_path' in kwargs:
        with open(kwargs['load_path']) as f:
            loaded = np.load(f)
            model = Model(cost_train)
            params_dicts = model.get_parameter_dict()
            params_names = params_dicts.keys()
            for param_name in params_names:
                param = params_dicts[param_name]
                # '/f_6_.W' --> 'f_6_.W'
                slash_index = param_name.find('/')
                param_name = param_name[slash_index + 1:]
                if param.get_value().shape == loaded[param_name].shape:
                    print 'Found: ' + param_name
                    param.set_value(loaded[param_name])
                else:
                    print 'Not found: ' + param_name

    extensions.extend([FinishAfter(after_n_epochs=epochs),
                       train_monitor,
                       dev_monitor])

    if test_cost:
        test_monitor = DataStreamMonitoring(
            variables=[cost_monitor, cost_per_character],
            data_stream=test_stream,
            prefix="test"
        )
        extensions.append(test_monitor)

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    log_path = os.path.join(experiment_path, 'log.txt')
    fh = logging.FileHandler(filename=log_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    extensions.append(SaveParams('dev_nll_cost', model, experiment_path,
                                 every_n_epochs=1))
    extensions.append(SaveLog(every_n_epochs=1))
    extensions.append(ProgressBar())
    extensions.append(Printing())

    main_loop = MainLoop(model=model, data_stream=train_stream,
                         algorithm=algorithm, extensions=extensions)
    t1 = time.time()
    print "Building time: %f" % (t1 - t0)
    main_loop.run()
    print "Execution time: %f" % (time.time() - t1)


if __name__ == '__main__':
    args = parse_args()
    argsdict = args.__dict__
    flags = [flag.lstrip('--') for flag in sys.argv[1:]]
    experiment_path = '_'.join(flags)
    i = 0
    while os.path.exists(experiment_path + "." + str(i)):
        i += 1
    experiment_path = experiment_path + "." + str(i)
    os.mkdir(experiment_path)
    argsdict['experiment_path'] = experiment_path
    with open (os.path.join(experiment_path,'exp_params.txt'), 'w') as f:
        for key in sorted(argsdict):
            f.write(key+'\t'+str(argsdict[key])+'\n')
            
    step_rule = learning_algorithm(args)
    train(step_rule, **args.__dict__)
