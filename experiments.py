# Copyright 2022 The Balsa Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Balsa experiment configs.

See README.md for the main configurations to run.
"""
import balsa
from balsa import hyperparams

# 19 most slow-running queries sorted by Postgres latency.
SLOW_TEST_QUERIES = [
    '6d.sql', '6f.sql',
    '8c.sql',
    '9d.sql',
    '10c.sql',
    '16b.sql',
    '17a.sql', '17b.sql', '17c.sql', '17d.sql', '17e.sql', '17f.sql',
    '18c.sql',
    '19c.sql', '19d.sql',
    '20a.sql',
    '25c.sql',
    '26c.sql',
    '30a.sql'
]

# A random split using seed 52.  Test latency is chosen to be close to the
# bootstrapped mean.
RAND_52_TEST_QUERIES = [
    '8a.sql', '16a.sql', '2a.sql', '30c.sql', '17e.sql', '20a.sql', '26b.sql',
    '12b.sql', '15b.sql', '15d.sql', '10b.sql', '15a.sql', '4c.sql', '4b.sql',
    '22b.sql', '17c.sql', '24b.sql', '10a.sql', '22c.sql'
]


# ===================================================================

RANDOM_SPLIT_1__TEST_QUERIES = [
    '1c.sql', '2c.sql', '4b.sql', '4c.sql', '5c.sql', '6a.sql', '6c.sql', '6e.sql', '8b.sql', '8c.sql', '9c.sql',
    '11d.sql', '15a.sql', '17b.sql', '17e.sql', '18b.sql',
    '20a.sql', '21a.sql', '25c.sql', '28b.sql',
    '32b.sql', '33a.sql'
]
    
RANDOM_SPLIT_2__TEST_QUERIES = [
    '1a.sql', '4c.sql', '5c.sql', '6c.sql', '6d.sql', '7b.sql', '8c.sql',
    '10a.sql', '11a.sql', '11d.sql', '13c.sql', '13d.sql', '15d.sql', '16a.sql', '17b.sql', '19a.sql',
    '20a.sql', '22b.sql', '25b.sql', '29b.sql',
    '31a.sql', '32b.sql'
]
    
RANDOM_SPLIT_3__TEST_QUERIES = [
    '2a.sql', '3b.sql', '6d.sql', '9b.sql',
    '10b.sql', '11b.sql', '11c.sql', '13c.sql', '13d.sql', '16b.sql', '18c.sql', '19c.sql',
    '21c.sql', '22a.sql', '22d.sql', '26a.sql', '26b.sql', '27c.sql', '28a.sql', '28c.sql',
    '30a.sql', '33c.sql'
]


BASE_QUERY_SPLIT_1__TEST_QUERIES = [
    '2a.sql', '2b.sql', '2c.sql', '2d.sql',
    '7a.sql', '7b.sql', '7c.sql',
    '15a.sql', '15b.sql', '15c.sql', '15d.sql',
    '24a.sql', '24b.sql', 
    '25a.sql', '25b.sql', '25c.sql',
    '31a.sql', '31b.sql', '31c.sql',
]

BASE_QUERY_SPLIT_2__TEST_QUERIES = [
    '13a.sql', '13b.sql', '13c.sql', '13d.sql',
    '15a.sql', '15b.sql', '15c.sql', '15d.sql',
    '20a.sql', '20b.sql', '20c.sql',
    '26a.sql', '26b.sql', '26c.sql',
    '29a.sql', '29b.sql', '29c.sql',
    '30a.sql', '30b.sql', '30c.sql',
    '33a.sql', '33b.sql', '33c.sql'
]

BASE_QUERY_SPLIT_3__TEST_QUERIES = [
    '1a.sql', '1b.sql', '1c.sql', '1d.sql',
    '5a.sql', '5b.sql', '5c.sql',
    '12a.sql', '12b.sql', '12c.sql',
    '17a.sql', '17b.sql', '17c.sql', '17d.sql', '17e.sql', '17f.sql',
    '22a.sql', '22b.sql', '22c.sql', '22d.sql',
    '27a.sql', '27b.sql', '27c.sql',
    '28a.sql', '28b.sql', '28c.sql'
]


LEAVE_ONE_OUT_SPLIT_1__TEST_QUERIES = [
    '1c.sql', '2a.sql', '3b.sql', '4a.sql', '5a.sql', '6b.sql', '7c.sql', '8c.sql', '9c.sql',
    '10b.sql', '11b.sql', '12c.sql', '13b.sql', '14a.sql', '15b.sql', '16c.sql', '17c.sql', '18b.sql', '19a.sql',
    '20c.sql', '21c.sql', '22b.sql', '23b.sql', '24a.sql', '25a.sql', '26c.sql', '27c.sql', '28a.sql', '29b.sql',
    '30a.sql', '31b.sql', '32b.sql', '33c.sql'
]

LEAVE_ONE_OUT_SPLIT_2__TEST_QUERIES = [
    '1d.sql', '2d.sql', '3a.sql', '4b.sql', '5c.sql', '6d.sql', '7a.sql', '8c.sql', '9c.sql',
    '10a.sql', '11a.sql', '12a.sql', '13d.sql', '14b.sql', '15b.sql', '16a.sql', '17f.sql', '18a.sql', '19d.sql',
    '20a.sql', '21b.sql', '22c.sql', '23b.sql', '24b.sql', '25a.sql', '26a.sql', '27b.sql', '28c.sql', '29a.sql',
    '30b.sql', '31a.sql', '32b.sql', '33b.sql'
]

LEAVE_ONE_OUT_SPLIT_3__TEST_QUERIES = [
    '1c.sql', '2d.sql', '3b.sql', '4a.sql', '5c.sql', '6d.sql', '7b.sql', '8a.sql', '9a.sql',
    '10c.sql', '11d.sql', '12a.sql', '13a.sql', '14b.sql', '15a.sql', '16d.sql', '17b.sql', '18b.sql', '19d.sql',
    '20b.sql', '21a.sql', '22a.sql', '23b.sql', '24a.sql', '25b.sql', '26a.sql', '27a.sql', '28b.sql', '29c.sql',
    '30a.sql', '31a.sql', '32a.sql', '33c.sql'
]

# ===================================================================


COMP_CAST_TEST_QUERIES = [
    '20a.sql', '20b.sql', '20c.sql',
    '26a.sql', '26b.sql', '26c.sql',
    '27a.sql', '27b.sql', '27c.sql',
    '28a.sql', '28b.sql', '28c.sql',
    '29a.sql', '29b.sql', '29c.sql',
    '30a.sql', '30b.sql', '30c.sql'
]

RAND_BASE_QUERIES = [
    '2a.sql', '2b.sql', '2c.sql', '2d.sql',
    '7a.sql', '7b.sql', '7c.sql',
    '15a.sql', '15b.sql', '15c.sql', '15d.sql',
    '24a.sql', '24b.sql', 
    '25a.sql', '25b.sql', '25c.sql',
    '31a.sql', '31b.sql', '31c.sql',
]


RAND_JOB_LIGHT_SPLIT = [
    '10c.sql', '11b.sql', '12a.sql', '18b.sql', '19a.sql',
    '1c.sql', '20a.sql', '20b.sql', '21a.sql', '22b.sql',
    '23a.sql', '26b.sql', '4a.sql', '8c.sql'
]

# ========================= STACK ===================================

STACK__RANDOM_SPLIT_1__TEST_QUERIES = [
'q1__q1-031.sql', 'q1__q1-035.sql', 'q4__q4-042.sql', 'q4__q4-064.sql',
'q4__q4-089.sql', 'q5__q5-032.sql', 'q6__q6-060.sql', 'q6__q6-064.sql', 'q7__q7-099.sql',
'q11__6c5cba419c5b7b02d431aeb5e766d775d812967a.sql', 'q11__c1ae2a992cde4ea2c4922d852df22043254b4f84.sql',
'q12__55de941e8497cfeeb93d3f8f2d7a18489e0e6c32.sql', 'q14__63c0776f1727638316b966fe748df7cc585a335b.sql',
'q14__74fd1af68d23f0690e3d0fc80bd9b42fa90a7e94.sql', 'q14__97e68ad5c2ced4c182366b3118a1f5f69b423fa6.sql',
'q14__b49361f85785200ed6ec1f2eec357b7598c9e564.sql', 'q15__3e37e62655ceaebc14e79edad518e5710752f51d.sql',
'q15__543ab3f730e494a69e3d15e59675f491544cb15d.sql', 'q15__b8ddf65b0c0c7867a9b560e571d457fec410715c.sql',
'q15__d5546c01928a687eb1f54e9f8eb4e1aff68fc381.sql', 'q16__1e863562a79ca1f7754c759ebab6a2addda0bde8.sql',
'q16__ea9efde510227beb8d624b8c4a6941b9d5e6e637.sql', 'q16__ed2ffeaefcf5ad8bbadc713ccc766541e12080aa.sql',
]

STACK__LEAVE_ONE_OUT_SPLIT_1__TEST_QUERIES = [
'q1__q1-035.sql', 'q2__q2-032.sql', 'q3__q3-043.sql', 'q4__q4-041.sql',
'q5__q5-041.sql', 'q6__q6-060.sql', 'q7__q7-047.sql', 'q8__q8-046.sql',
'q11__6c5cba419c5b7b02d431aeb5e766d775d812967a.sql', 'q12__547c6bf1994c9b2ba82a7ae32f4b051beabf46fd.sql',
'q13__935e2051bf80eeafe91aeb6eb719b6b64b9592c2.sql', 'q14__5e4835cd72aaa2d7be15b2a5ffa2e66156b3656f.sql',
'q15__543ab3f730e494a69e3d15e59675f491544cb15d.sql', 'q16__b1a96cd48ba297dd93bce73c27b491069ad7449f.sql',
]

STACK__BASE_QUERY_SPLIT_1__TEST_QUERIES = [
'q2__q2-001.sql', 'q2__q2-012.sql', 'q2__q2-032.sql', 'q2__q2-035.sql',
'q2__q2-050.sql', 'q2__q2-081.sql', 'q2__q2-094.sql', 'q2__q2-098.sql',
'q7__q7-034.sql', 'q7__q7-036.sql', 'q7__q7-047.sql', 'q7__q7-077.sql',
'q7__q7-082.sql', 'q7__q7-085.sql', 'q7__q7-095.sql', 'q7__q7-099.sql',
'q13__13ad1b8c6bea4fda1892b9fa82cc1ceb9ceb85fc.sql', 'q13__1ddcc8650e17b292bc7344902baffc90c5ae5761.sql',
'q13__935e2051bf80eeafe91aeb6eb719b6b64b9592c2.sql', 'q13__a091adce62743b65c04532e98e8ff3d7e546ea77.sql',
'q13__a3d03772d880754fc4e150d82908757477ae2186.sql', 'q13__add0df9dccb2790c14508e19c9e0deb79fad6ea2.sql',
'q13__d383cd5b4aee7d3f73508e2a1fe5f6d0f7dd42a2.sql', 'q13__d4707be2adfdbc842f42acb1fc16e3a43faf7474.sql',
]

# ===================================================================


LR_SCHEDULES = {
    'C': {
        'lr_piecewise': [
            (0, 0.001),
            (50, 0.0005),
            (100, 0.00025),
            (150, 0.000125),
            (200, 0.0001),
        ]
    },
    # Delay C's decay by 10 iters.
    'C10': {
        'lr_piecewise': [
            (0, 0.001),
            (50 + 10, 0.0005),
            (100 + 10, 0.00025),
            (150 + 10, 0.000125),
            (200 + 10, 0.0001),
        ]
    },
}


class BalsaParams(object):
    """Params for run.BalsaAgent."""

    @classmethod
    def Params(cls):
        p = hyperparams.InstantiableParams(cls)
        p.Define('db', 'imdbload', 'Name of the Postgres database.')
        p.Define('query_dir', 'queries/join-order-benchmark',
                 'Directory of the .sql queries.')
        p.Define(
            'query_glob', '*.sql',
            'If supplied, glob for this pattern. Otherwise, use all queries. Example: 29*.sql.'
        )
        p.Define(
            'test_query_glob', None,
            'Similar usage as query_glob. If None, treat all queries as training nodes.'
        )
        p.Define('engine', 'postgres',
                 'The execution engine.  Options: postgres.')
        p.Define('engine_dialect_query_dir', None,
                 'Directory of the .sql queries in target engine\'s dialect.')
        p.Define('run_baseline', False,
                 'If true, just load the queries and run them.')
        p.Define(
            'drop_cache', True,
            'If true, drop the buffer cache at the end of each value iteration.'
        )
        p.Define(
            'plan_physical', True,
            'If true, plans physical scan/join operators.  '\
            'Otherwise, just join ordering.'
        )
        p.Define('cost_model', 'postgrescost',
                 'A choice of postgrescost, mincardcost.')
        p.Define('bushy', True, 'Plans bushy query execution plans.')
        p.Define('search_space_join_ops',
                 ['Hash Join', 'Merge Join', 'Nested Loop'],
                 'Action space: join operators to learn and use.')
        p.Define('search_space_scan_ops',
                 ['Index Scan', 'Index Only Scan', 'Seq Scan', 'Bitmap Heap Scan', 'Tid Scan'],
                 'Action space: scan operators to learn and use.')

        # LR.
        p.Define('lr', 1e-3, 'Learning rate.')
        p.Define('lr_decay_rate', None, 'If supplied, use ExponentialDecay.')
        p.Define('lr_decay_iters', None, 'If supplied, use ExponentialDecay.')
        p.Define('lr_piecewise', None, 'If supplied, use Piecewise.  Example:'\
                 '[(0, 1e-3), (200, 1e-4)].')
        p.Define('use_adaptive_lr', None, 'Experimental.')
        p.Define('use_adaptive_lr_decay_to_zero', None, 'Experimental.')
        p.Define('final_decay_rate', None, 'Experimental.')
        p.Define('linear_decay_to_zero', False,
                 'Linearly decay from lr to 0 in val_iters.')
        p.Define('reduce_lr_within_val_iter', False,
                 'Reduce LR within each val iter?')

        # Training.
        p.Define('inherit_optimizer_state', False, 'Experimental.  For Adam.')
        p.Define('epochs', 100, 'Num epochs to train.')
        p.Define('bs', 1024, 'Batch size.')
        p.Define('val_iters', 500, '# of value iterations.')
        p.Define('increment_iter_despite_timeouts', False,
                 'Increment the iteration counter even if timeouts occurred?')
        p.Define('loss_type', None, 'Options: None (MSE), mean_qerror.')
        p.Define('cross_entropy', False, 'Use cross entropy loss formulation?')
        p.Define('l2_lambda', 0, 'L2 regularization lambda.')
        p.Define('adamw', None,
                 'If not None, the weight_decay param for AdamW.')
        p.Define('label_transforms', ['log1p', 'standardize'],
                 'Transforms for labels.')
        p.Define('label_transform_running_stats', False,
                 'Use running mean and std to standardize labels?'\
                 '  May affect on-policy.')
        p.Define('update_label_stats_every_iter', True,
                 'Update mean/std stats of labels every value iteration?  This'\
                 'means the scaling of the prediction targers will shift.')
        p.Define('gradient_clip_val', 0, 'Clip the gradient norm computed over'\
                 ' all model parameters together. 0 means no clipping.')
        p.Define('early_stop_on_skip_fraction', None,
                 'If seen plans for x% of train queries produced, early stop.')
        # Validation.
        p.Define('validate_fraction', 0.1,
                 'Sample this fraction of the dataset as the validation set.  '\
                 '0 to disable validation.')
        p.Define('validate_every_n_epochs', 5,
                 'Run validation every this many training epochs.')
        p.Define(
            'validate_early_stop_patience', 3,
            'Number of validations with no improvements before early stopping.'\
            '  Thus, the maximum # of wasted train epochs = '\
            'this * validate_every_n_epochs).'
        )
        # Testing.
        p.Define('test_every_n_iters', 1,
                 'Run test set every this many value iterations.')
        p.Define('test_after_n_iters', 0,
                 'Start running test set after this many value iterations.')
        p.Define('test_using_retrained_model', False,
                 'Whether to retrain a model from scratch just for testing.')
        p.Define('track_model_moving_averages', False,
                 'Track EMA/SWA of the agent?')
        p.Define('ema_decay', 0.95, 'Use an EMA model to evaluate on test.')

        # Pre-training.
        p.Define('sim', True, 'Initialize from a pre-trained SIM model?')
        p.Define('finetune_out_mlp_only', False, 'Freeze all but out_mlp?')
        p.Define(
            'sim_checkpoint', None,
            'Path to a pretrained SIM checkpoint.  Load it instead '
            'of retraining.')
        p.Define(
            'param_noise', 0.0,
            'If non-zero, add Normal(0, std=param_noise) to Linear weights '\
            'of the pre-trained net.')
        p.Define(
            'param_tau', 1.0,
            'If non-zero, real_model_t = tau * real_model_tm1 + (1-tau) * SIM.')
        p.Define(
            'use_ema_source', False,
            'Use an exponential moving average of source networks?  If so, tau'\
            ' is used as model_t := source_t :='\
            ' tau * source_(t-1) + (1-tau) * model_(t-1).'
        )
        p.Define(
            'skip_sim_init_iter_1p', False,
            'Starting from the 2nd iteration, skip initializing from '\
            'simulation model?'
        )
        p.Define(
            'generic_ops_only_for_min_card_cost', False,
            'This affects sim model training and only if MinCardCost is used. '\
            'See sim.py for documentation.')
        p.Define(
            'sim_data_collection_intermediate_goals', True,
            'This affects sim model training.  See sim.py for documentation.')

        # Training data / replay buffer.
        p.Define(
            'init_experience', 'data/initial_policy_data.pkl',
            'Initial data set of query plans to learn from. By default, this'\
            ' is the expert optimizer experience collected when baseline'\
            ' performance is evaluated.'
        )
        p.Define('skip_training_on_expert', True,
                 'Whether to skip training on expert plan-latency pairs.')
        p.Define(
            'dedup_training_data', True,
            'Whether to deduplicate training data by keeping the best cost per'\
            ' subplan per template.'
        )
        p.Define('on_policy', False,
                 'Whether to train on only data from the latest iteration.')
        p.Define(
            'use_last_n_iters', -1,
            'Train on data from this many latest iterations.  If on_policy,'\
            ' this flag is ignored and treated as 1 (latest iter).  -1 means'\
            ' train on all previous iters.')
        p.Define('skip_training_on_timeouts', False,
                 'Skip training on executions that were timeout events?')
        p.Define(
            'use_new_data_only', False,
            'Experimental; has effects if on_policy or use_last_n_iters > 0.'\
            '  Currently only implemented in the dedup_training_data branch.')
        p.Define(
            'per_transition_sgd_steps', -1, '-1 to disable.  Takes effect only'\
            ' for when p.use_last_n_iters>0 and p.epochs=1.  This controls the'\
            ' average number of SGD updates taken on each transition.')
        p.Define('physical_execution_hindsight', False,
                 'Apply hindsight labeling to physical execution data?')
        p.Define(
            'replay_buffer_reset_at_iter', None,
            'If specified, clear all agent replay data at this iteration.')
        # Offline replay.
        p.Define(
            'prev_replay_buffers_glob', None,
            'If specified, load previous replay buffers and merge them as training purpose.'
        )
        p.Define(
            'prev_replay_buffers_glob_val', None,
            'If specified, load previous replay buffers and merge them as validation purpose.'
        )
        p.Define(
            'agent_checkpoint', None,
            'Path to a pretrained agent checkpoint.  Load it instead '
            'of retraining.')
        p.Define('prev_replay_keep_last_fraction', 1,
                 'Keep the last fraction of the previous replay buffers.')
        # Modeling: tree convolution (suggested).
        p.Define('tree_conv', True,
                 'If true, use tree convolutional neural net.')
        p.Define('tree_conv_version', None, 'Options: None.')
        p.Define('sim_query_featurizer', True,
                 'If true, use SimQueryFeaturizer to produce query features.')
        # Featurization.
        p.Define('perturb_query_features', None,
                 'If not None, randomly perturb query features on each forward'\
                 ' pass, and this flag specifies '\
                 '(perturb_prob_per_table, [scale_min, scale_max]).  '\
                 'A multiplicative scale is drawn from '\
                 'Unif[scale_min, scale_max].  Only performed when training '\
                 'and using a query featurizer with perturbation implemented.')

        # Modeling: Transformer (deprecated).  Enabled when tree_conv is False.
        p.Define('v2', True, 'If true, use TransformerV2.')
        p.Define('pos_embs', True, 'Use positional embeddings?')
        p.Define('dropout', 0.0, 'Dropout prob for transformer stack.')

        # Inference.
        p.Define('check_hint', True, 'Check hints are respected?')
        p.Define('beam', 20, 'Beam size.')
        p.Define(
            'search_method', 'beam_bk',
            'Algorithm used to search for execution plans with cost model.')
        p.Define(
            'search_until_n_complete_plans', 10,
            'Keep doing plan search for each query until this many complete'\
            ' plans have been found.  Returns the predicted cheapest one out'\
            ' of them.  Recommended: 10.')
        p.Define('planner_config', None, 'See optimizer.py#PlannerConfig.')
        p.Define(
            'avoid_eq_filters', False,
            'Avoid certain equality filters during planning (required for Ext-JOB).'
        )

        p.Define('sim_use_plan_restrictions', True, 'Experimental.')
        p.Define('real_use_plan_restrictions', True, 'Experimental.')

        # Exploration during inference.
        p.Define(
            'epsilon_greedy', 0,
            'Epsilon-greedy policy: with epsilon probability, execute a'\
            ' randomly picked plan out of all complete plans found, rather'\
            ' than the predicted-cheapest one out of them.')
        p.Define('epsilon_greedy_random_transform', False,
                 'Apply eps-greedy to randomly transform the best found plan?')
        p.Define('epsilon_greedy_random_plan', False,
                 'Apply eps-greedy to randomly pick a plan?')
        p.Define('epsilon_greedy_within_beam_search', False,
                 'Apply eps-greedy to within beam search?')
        p.Define('explore_soft_v', False,
                 'Sample an action from the soft V-distribution?')
        p.Define('explore_visit_counts', False, 'Explores using a visit count?')
        p.Define('explore_visit_counts_sort', False,
                 'Explores by executing the plan with the smallest '\
                 '(visit count, predicted latency) out of k-best plans?')
        p.Define('explore_visit_counts_latency_sort', False,
                 'Explores using explore_visit_counts_sort if there exists '\
                 'a plan that has a 0 visit count. Else sorts by predicted latency.')

        # Safe execution.
        p.Define('use_timeout', True, 'Use a timeout safeguard?')
        p.Define('initial_timeout_ms', None, 'Timeout for iter 0 if not None.')
        p.Define('special_timeout_label', True,
                 'Use a constant timeout label (4096 sec)?')
        p.Define('timeout_slack', 2,
                 'A multiplier: timeout := timeout_slack * max_query_latency.')
        p.Define('relax_timeout_factor', None,
                 'If not None, a positive factor to multiply with the current'\
                 ' timeout when relaxation conditions are met.')
        p.Define('relax_timeout_on_n_timeout_iters', None,
                 'If there are this many timeout iterations up to now, relax'\
                 ' the current timeout by relax_timeout_factor.')

        # Execution.
        p.Define('use_local_execution', False,
                 'For query executions, connect to local engine or the remote'\
                 ' cluster?  Non-execution EXPLAINs are always issued to'\
                 ' local.')
        p.Define('use_cache', True, 'Skip executing seen plans?')
        p.Define('workload_order_file', None, 'If there, the workload order is specified.  Otherwise, the order is random.')
        p.Define('target_checkpoint_dir', None, 'If specified, save checkpoints to this directory.')
        p.Define('test_query_dir', None,
                 'If specified, use this directory for test queries.  '\
                 'Otherwise, use query_dir.')
        return p


########################## Baselines ##########################


@balsa.params_registry.Register
class Baseline(BalsaParams):

    def Params(self):
        p = super().Params()
        p.run_baseline = True
        return p


@balsa.params_registry.Register
class BaselineExtJOB(Baseline):

    def Params(self):
        p = super().Params()
        p.query_glob = ['*.sql']
        p.query_dir = 'queries/join-order-benchmark-extended'
        p.test_query_glob = ['e*.sql']
        return p

@balsa.params_registry.Register
class BaselineTPCH(Baseline):

    def Params(self):
        p = super().Params()
        p.query_glob = ['*.sql']
        p.query_dir = 'queries/tpch'
        p.test_query_glob = ['e*.sql']
        return p

@balsa.params_registry.Register
class BaselineTPCDS(Baseline):

    def Params(self):
        p = super().Params()
        p.query_glob = ['*.sql']
        p.query_dir = 'queries/tpcds'
        p.test_query_glob = ['e*.sql']
        return p

@balsa.params_registry.Register
class BaselineJOBLight(Baseline):
    def params(self):
        p = super().Params()
        p.query_glob = ['*.sql']
        p.query_dir = 'queries/join-order-benchmark-light'
        p.test_query_glob = RAND_JOB_LIGHT_SPLIT

        return p

########################## Main Balsa agents ##########################


@balsa.params_registry.Register
class MinCardCost(BalsaParams):

    def Params(self):
        p = super().Params()
        p.cost_model = 'mincardcost'
        p.sim_checkpoint = None
        # Exploration schemes.
        p.explore_visit_counts = True
        return p


@balsa.params_registry.Register
class MinCardCostSortCnts(MinCardCost):

    def Params(self):
        return super().Params().Set(
            explore_visit_counts=False,
            explore_visit_counts_sort=True,
        )


@balsa.params_registry.Register
class MinCardCostOnPol(MinCardCostSortCnts):

    def Params(self):
        p = super().Params()

        from_p = BalsaParams().Params()
        from_p.cost_model = 'mincardcost'
        from_p.query_glob = ['*.sql']
        from_p.test_query_glob = 'TODO: Subclasses should fill this.'
        from_p.sim_checkpoint = None
        # Exploration schemes.
        from_p.explore_visit_counts = False
        from_p.explore_visit_counts_sort = True

        p = hyperparams.CopyFieldsTo(from_p, p)
        return p.Set(on_policy=True)


@balsa.params_registry.Register
class Rand52MinCardCostOnPol(MinCardCostOnPol):

    def Params(self):
        p = super().Params()
        p.test_query_glob = RAND_52_TEST_QUERIES
        p.sim_checkpoint = 'checkpoints/sim-MinCardCost-rand52split-680secs.ckpt'
        return p


@balsa.params_registry.Register
class Rand52MinCardCostOnPolLrC(Rand52MinCardCostOnPol):

    def Params(self):
        return super().Params().Set(**LR_SCHEDULES['C'])


@balsa.params_registry.Register  # keep
class Balsa_JOBRandSplit(Rand52MinCardCostOnPolLrC):

    def Params(self):
        p = super().Params()
        p.increment_iter_despite_timeouts = True
        p = p.Set(**LR_SCHEDULES['C10'])
        return p

@balsa.params_registry.Register
class Balsa_TPCH(BalsaParams):

    def Params(self):
        p = super().Params()
        p.query_glob = ['*.sql']
        p.query_dir = 'queries/tpch'
        p.test_query_glob = ['e*.sql']
        return p

@balsa.params_registry.Register
class Balsa_TPCDS(BalsaParams):

    def Params(self):
        p = super().Params()
        p.query_glob = ['*.sql']
        p.query_dir = 'queries/tpcds'
        p.test_query_glob = ['e*.sql']
        return p


@balsa.params_registry.Register
class Balsa_JOBRandSplitReplay(Balsa_JOBRandSplit):  # keep

    def Params(self):
        p = super().Params()

        p.validate_fraction = 0.1
        p.epochs = 100
        p.lr_piecewise = [(0, 5 * 1e-4), (1, 1e-4)]

        p.val_iters = 100
        p.validate_every_n_epochs = 1
        p.validate_early_stop_patience = 5

        # Change path to point to the desired buffers:
        p.prev_replay_buffers_glob = './data/replay-Balsa_JOBRandSplit-*'
        # Choose one buffer as a hold-out validation set if desired:
        p.prev_replay_buffers_glob_val = None
        p.skip_training_on_timeouts = False
        return p


@balsa.params_registry.Register
class SlowMinCardCost(MinCardCostOnPol):

    def Params(self):
        p = super().Params()
        p.use_timeout = True
        p.test_query_glob = SLOW_TEST_QUERIES
        p.sim_checkpoint = 'checkpoints/sim-MinCardCost-slowsplit-610secs.ckpt'
        # ExponentialDecay(2e-3, 0.1, 100).
        p = p.Set(lr=2e-3, lr_decay_rate=0.1, lr_decay_iters=100)
        return p


@balsa.params_registry.Register
class SlowMinCardCostNoDecay(SlowMinCardCost):

    def Params(self):
        # No decay.
        return super().Params().Set(lr=1e-3,
                                    lr_decay_rate=None,
                                    lr_decay_iters=None)


@balsa.params_registry.Register
class SlowMinCardCostLrC(SlowMinCardCostNoDecay):

    def Params(self):
        return super().Params().Set(**LR_SCHEDULES['C'])


@balsa.params_registry.Register  # keep
class Balsa_JOBSlowSplit(SlowMinCardCostLrC):

    def Params(self):
        p = super().Params()
        p.increment_iter_despite_timeouts = True
        p = p.Set(**LR_SCHEDULES['C10'])
        return p


# ===========================================================================
# Base Experiment Class

@balsa.params_registry.Register
class Balsa_JOB_EvaluationBase(Balsa_JOBSlowSplit):
    def Params(self):
        p = super().Params()
        p.sim_checkpoint = None
        p.val_iters = 200
        return p

# ===========================================================================
# RANDOM SPLIT 1-3

@balsa.params_registry.Register
class Balsa_JOBRandomSplit1(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = RANDOM_SPLIT_1__TEST_QUERIES
        return p
    
@balsa.params_registry.Register
class Balsa_JOBRandomSplit2(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = RANDOM_SPLIT_2__TEST_QUERIES
        return p
    
@balsa.params_registry.Register
class Balsa_JOBRandomSplit3(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = RANDOM_SPLIT_3__TEST_QUERIES
        return p
# ===========================================================================
# BASE QUERY SPLIT 1-3

# Identical to Balsa_JOBLeakageTest2
@balsa.params_registry.Register
class Balsa_JOBBaseQuerySplit1(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = BASE_QUERY_SPLIT_1__TEST_QUERIES
        return p
    
@balsa.params_registry.Register
class Balsa_JOBBaseQuerySplit2(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = BASE_QUERY_SPLIT_2__TEST_QUERIES
        return p
    
@balsa.params_registry.Register
class Balsa_JOBBaseQuerySplit3(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = BASE_QUERY_SPLIT_3__TEST_QUERIES
        return p
# ===========================================================================
# LEAVE ONE OUT SPLIT 1-3

@balsa.params_registry.Register
class Balsa_JOBLeaveOneOutSplit1(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = LEAVE_ONE_OUT_SPLIT_1__TEST_QUERIES
        return p
    
@balsa.params_registry.Register
class Balsa_JOBLeaveOneOutSplit2(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = LEAVE_ONE_OUT_SPLIT_2__TEST_QUERIES
        return p
    
@balsa.params_registry.Register
class Balsa_JOBLeaveOneOutSplit3(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = LEAVE_ONE_OUT_SPLIT_3__TEST_QUERIES
        return p


# lehl@2022-11-09:
# - Only give full base queries to train/test, do not test
#   on the partial sub-structures
# - Separate complete_cast / comp_cast_type structure by giving
#   Q20, Q26-30 into test set and only keeping Q23 in train set
#   (all other queries do not use these tables)
#
# The aim is to show the existence (or lack of) leakage between queries
# and running with a newly trained sim to further reduce problems
#
@balsa.params_registry.Register  # keep
class Balsa_JOBLeakageTest(Balsa_JOBSlowSplit):
    def Params(self):
        p = super().Params()
        p.test_query_glob = COMP_CAST_TEST_QUERIES
        p.sim_checkpoint = None
        return p

# Identical to Balsa_JOBBaseQuerySplit1
@balsa.params_registry.Register
class Balsa_JOBLeakageTest2(Balsa_JOBSlowSplit):
    def Params(self):
        p = super().Params()
        p.test_query_glob = BASE_QUERY_SPLIT_1__TEST_QUERIES
        p.sim_checkpoint = None
        return p
    

@balsa.params_registry.Register
class Balsa_JOBLightRandom(Balsa_JOBSlowSplit):
    def Params(self):
        p = super().Params()
        p.sim_checkpoint = None
        p.query_dir = 'queries/join-order-benchmark-light'
        p.test_query_glob = RAND_JOB_LIGHT_SPLIT
        return p

@balsa.params_registry.Register
class Balsa_JOBSlowSplitReplay(Balsa_JOBSlowSplit):  # keep

    def Params(self):
        p = super().Params()

        p.validate_fraction = 0.1
        p.epochs = 100
        p.lr_piecewise = [(0, 5 * 1e-4), (1, 1e-4)]

        p.val_iters = 100
        p.validate_every_n_epochs = 1
        p.validate_early_stop_patience = 5

        # Change path to point to the desired buffers:
        p.prev_replay_buffers_glob = './data/replay-Balsa_JOBSlowSplit-*'
        # Choose one buffer as a hold-out validation set if desired:
        p.prev_replay_buffers_glob_val = None
        p.skip_training_on_timeouts = False
        return p


########################## Generalizing to STACK ##########################
# STACK Base Experiment Class

@balsa.params_registry.Register
class Balsa_STACK_EvaluationBase(Balsa_JOBSlowSplit):
    def Params(self):
        p = super().Params()
        p.db = 'so'
        p.query_dir = 'queries/stack'
        p.init_experience = 'data/initial_policy_data__stack.pkl'
        p.query_glob = ['*.sql']
        p.sim_checkpoint = None
        p.initial_timeout_ms = 3 * 60 * 1000
        p.val_iters = 100
        return p

# ===========================================================================
# STACK RANDOM SPLIT

@balsa.params_registry.Register
class Balsa_STACKRandomSplit1(Balsa_STACK_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = STACK__RANDOM_SPLIT_1__TEST_QUERIES
        return p
    
# ===========================================================================
# STACK BASE QUERY SPLIT

@balsa.params_registry.Register
class Balsa_STACKBaseQuerySplit1(Balsa_STACK_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = STACK__BASE_QUERY_SPLIT_1__TEST_QUERIES
        return p
    
# ===========================================================================
# STACK LEAVE ONE OUT SPLIT

@balsa.params_registry.Register
class Balsa_STACKLeaveOneOutSplit1(Balsa_STACK_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = STACK__LEAVE_ONE_OUT_SPLIT_1__TEST_QUERIES
        return p

########################## Generalizing to Ext-JOB ##########################


@balsa.params_registry.Register
class ExtJOBMinCardCostOnPol(Rand52MinCardCostOnPol):

    def Params(self):
        p = super().Params()
        p.query_dir = 'queries/join-order-benchmark-extended'
        p.query_glob = ['*.sql']
        p.test_query_glob = ['e*.sql']

        p.sim_checkpoint = ('checkpoints/' +
                            'sim-MinCardCost-train113JOB-784s-13epochs.ckpt')

        # Save some time.
        p.test_after_n_iters = 80
        p.test_every_n_iters = 1

        # This required so plan hints do not fail for certain queries
        # in Ext-JOB that contain equality filters.
        p.avoid_eq_filters = True
        return p


@balsa.params_registry.Register  # keep
class Balsa_TrainJOB_TestExtJOB(ExtJOBMinCardCostOnPol):

    def Params(self):
        return super().Params().Set(inherit_optimizer_state=True)


@balsa.params_registry.Register  # keep
class Balsa1x_TrainJOB_TestExtJOB(Balsa_TrainJOB_TestExtJOB):

    def Params(self):
        p = super().Params()
        p.validate_every_n_epochs = 1
        p.validate_early_stop_patience = 5
        p.lr_piecewise = [(0, 1e-3), (1, 1e-4)]
        # Change path to point to the desired buffers:
        p.prev_replay_buffers_glob = './replays/EXTJOB/train/replay-Balsa_TrainJOB_TestExtJOB-<fill_me_in>.pkl'
        p.test_after_n_iters = 0
        p.test_every_n_iters = 1
        # Disable timeout for further exploration
        p.use_timeout = False
        return p


@balsa.params_registry.Register  # keep
class Balsa8x_TrainJOB_TestExtJOB(Balsa1x_TrainJOB_TestExtJOB):

    def Params(self):
        p = super().Params()
        # Change paths to point to the desired buffers:
        p.prev_replay_buffers_glob = './replays/EXTJOB/train/*pkl'
        # Choose one buffer as a hold-out validation set if desired:
        p.prev_replay_buffers_glob_val = './replays/EXTJOB/val/*pkl'
        return p


########################## Neo-impl experiments ##########################


@balsa.params_registry.Register
class NeoImplRand52(BalsaParams):

    def Params(self):
        p = super().Params()
        p.query_glob = ['*.sql']
        p.test_query_glob = RAND_52_TEST_QUERIES

        p.test_every_n_iters = 1

        # Algorithmic choices below.

        # No simulator.
        p.sim = False
        p.sim_checkpoint = None

        # Off-policy, retrain always.
        p.on_policy = False
        p.param_tau = 1

        # Use demonstrations from expert.
        p.skip_training_on_expert = False

        # No timeouts.
        p.use_timeout = False
        # NOTE: this flag is necessary because there could be some 'treating as
        # timeouts' feedback even when use_timeout is set to False.  In those
        # cases, skip training on those labels.
        p.skip_training_on_timeouts = True
        p.use_cache = False  # Do not cache plans.

        return p


@balsa.params_registry.Register
class NeoImplRand52Reset(NeoImplRand52):

    def Params(self):
        return super().Params().Set(param_tau=0.0)


@balsa.params_registry.Register  # keep
class NeoImpl_JOBRandSplit(NeoImplRand52Reset):

    def Params(self):
        return super().Params().Set(dedup_training_data=False)

# ===========================================================================
# Base Experiment Class

@balsa.params_registry.Register
class Neo_JOB_EvaluationBase(NeoImplRand52):
    def Params(self):
        p = super().Params()
        p.epochs = 50
        p.val_iters = 50
        return p

# ===========================================================================
# RANDOM SPLIT 1-3

@balsa.params_registry.Register
class Neo_JOBRandomSplit1(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = RANDOM_SPLIT_1__TEST_QUERIES
        return p
    
@balsa.params_registry.Register
class Neo_JOBRandomSplit2(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = RANDOM_SPLIT_2__TEST_QUERIES
        return p
    
@balsa.params_registry.Register
class Neo_JOBRandomSplit3(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = RANDOM_SPLIT_3__TEST_QUERIES
        return p
# ===========================================================================
# BASE QUERY SPLIT 1-3

# Identical to Balsa_JOBLeakageTest2
@balsa.params_registry.Register
class Neo_JOBBaseQuerySplit1(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = BASE_QUERY_SPLIT_1__TEST_QUERIES
        return p
    
@balsa.params_registry.Register
class Neo_JOBBaseQuerySplit2(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = BASE_QUERY_SPLIT_2__TEST_QUERIES
        return p
    
@balsa.params_registry.Register
class Neo_JOBBaseQuerySplit3(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = BASE_QUERY_SPLIT_3__TEST_QUERIES
        return p
# ===========================================================================
# LEAVE ONE OUT SPLIT 1-3

# Identical to Balsa_JOBLeakageTest2
@balsa.params_registry.Register
class Neo_JOBLeaveOneOutSplit1(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = LEAVE_ONE_OUT_SPLIT_1__TEST_QUERIES
        return p
    
@balsa.params_registry.Register
class Neo_JOBLeaveOneOutSplit2(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = LEAVE_ONE_OUT_SPLIT_2__TEST_QUERIES
        return p
    
@balsa.params_registry.Register
class Neo_JOBLeaveOneOutSplit3(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = LEAVE_ONE_OUT_SPLIT_3__TEST_QUERIES
        return p




########################## Neo - STACK ##########################
# STACK Base Experiment Class

@balsa.params_registry.Register
class Neo_STACK_EvaluationBase(NeoImplRand52):
    def Params(self):
        p = super().Params()
        p.db = 'so'
        p.query_dir = 'queries/stack'
        p.query_glob = ['*.sql']
        p.sim_checkpoint = None
        p.init_experience = 'data/initial_policy_data__stack.pkl'

        # Need to set this to true, since Neo will not progress its training
        # if there is no increment with any timeouts
        p.increment_iter_despite_timeouts = True

        p.val_iters = 50
        return p

# ===========================================================================
# RANDOM SPLIT

@balsa.params_registry.Register
class Neo_STACK_RandomSplit1(Neo_STACK_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = STACK__RANDOM_SPLIT_1__TEST_QUERIES
        return p
    
# ===========================================================================
# BASE QUERY SPLIT

@balsa.params_registry.Register
class Neo_STACK_BaseQuerySplit1(Neo_STACK_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = STACK__BASE_QUERY_SPLIT_1__TEST_QUERIES
        return p
    
# ===========================================================================
# LEAVE ONE OUT SPLIT

@balsa.params_registry.Register
class Neo_STACK_LeaveOneOutSplit1(Neo_STACK_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.test_query_glob = STACK__LEAVE_ONE_OUT_SPLIT_1__TEST_QUERIES
        return p


########################## Ablation: sim ##########################


@balsa.params_registry.Register  # keep
class JOBRandSplit_NoSim(Balsa_JOBRandSplit):

    def Params(self):
        # TODO: there's a minor bug in this code path:
        #   iter 0: rand-init weights W0
        #           plan & execute
        #
        #   iter 1: rand-init weights W1
        #           train W1 on iter 0's data
        #           plan & execute
        #
        #   iter 2: use last iter's updated weights (self.model)
        #           train, plan, execute
        #
        # The minor bug is that iter 1 should train W0 rather than
        # re-initializes the weights.  The code is currently not set up to do
        # that.
        #
        # This is very minor and we should spend time on other things.
        p = super().Params()
        # No simulator.
        p.sim = False
        p.sim_checkpoint = None
        return p


@balsa.params_registry.Register  # keep
class JOBRandSplit_PostgresSim(Balsa_JOBRandSplit):

    def Params(self):
        p = super().Params()
        # Use PostgresCost as the simulator.
        p.cost_model = 'postgrescost'
        p.sim_checkpoint = 'checkpoints/sim-PostgresCost-rand52split-26epochs.ckpt'
        return p


########################## Ablation: timeouts ##########################


@balsa.params_registry.Register  # keep
class JOBRandSplit_NoTimeout(Balsa_JOBRandSplit):

    def Params(self):
        p = super().Params()
        # No timeouts.
        p.use_timeout = False
        # NOTE: this flag is necessary because there could be some 'treating as
        # timeouts' feedback even when use_timeout is set to False.  In those
        # cases, skip training on those labels.
        p.skip_training_on_timeouts = True
        return p


########################## Ablation: Training scheme ##########################


@balsa.params_registry.Register  # keep
class JOBRandSplit_RetrainScheme(Balsa_JOBRandSplit):

    def Params(self):
        p = super().Params()
        # Off pol, tau=0, no inherit, no LR decay.
        p.on_policy = False
        p.param_tau = 0
        p.lr_piecewise = None
        return p


######################### Ablation: exploration #########################


@balsa.params_registry.Register  # keep
class JOBRandSplit_EpsGreedy(Balsa_JOBRandSplit):

    def Params(self):
        p = super().Params()
        p.explore_visit_counts_sort = False
        p.epsilon_greedy_within_beam_search = 0.0025
        return p


@balsa.params_registry.Register  # keep
class JOBRandSplit_NoExplore(Balsa_JOBRandSplit):

    def Params(self):
        p = super().Params()
        p.explore_visit_counts_sort = False
        return p
