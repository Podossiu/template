from utils.Config import FLAGS

def get_experiment_setting():
    experiment_setting = 'ema_decay_{ema_decay}/fp_pretrained_{fp_pretrained}/bit_list_{bit_list}'.format(ema_decay=getattr(FLAGS, 'ema_decay', None), fp_pretrained=getattr(FLAGS, 'fp_pretrained_file', None) is not None,  bit_list='_'.join([str(i) for i in getattr(FLAGS, 'bits_list', None)]))
    if getattr(FLAGS, 'act_bits_list', False):
        experiment_setting = os.path.join(experiment_setting, 'act_bits_list_{}'.format('_'.join([str(i) for i in FLAGS.act_bits_list])))
    if getattr(FLAGS, 'double_side', False):
        experiment_setting = os.path.join(experiment_setting, 'double_side_True')
    if not getattr(FLAGS, 'rescale', False):
        experiment_setting = os.path.join(experiment_setting, 'rescale_False')
    if not getattr(FLAGS, 'calib_pact', False):
        experiment_setting = os.path.join(experiment_setting, 'calib_pact_False')
    experiment_setting = os.path.join(experiment_setting, 'kappa_{kappa}'.format(kappa=getattr(FLAGS, 'kappa', 1.0)))
    if getattr(FLAGS, 'target_bitops', False):
        experiment_setting = os.path.join(experiment_setting, 'target_bitops_{}'.format(getattr(FLAGS, 'target_bitops', False)))
    if getattr(FLAGS, 'target_size', False):
        experiment_setting = os.path.join(experiment_setting, 'target_size_{}'.format(getattr(FLAGS, 'target_size', False)))
    if getattr(FLAGS, 'init_bit', False):
        experiment_setting = os.path.join(experiment_setting, 'init_bit_{}'.format(getattr(FLAGS, 'init_bit', False)))
    if getattr(FLAGS, 'unbiased', False):
        experiment_setting = os.path.join(experiment_setting, f'unbiased_True')
    print('Experiment settings: {}'.format(experiment_setting))
    return experiment_setting

def profiling(model, use_cuda):
    """profiling on either gpu or cpu"""
    print('Start model profiling, use_cuda:{}.'.format(use_cuda))
    flops, params, bitops, bitops_max, bytesize, energy, latency = model_profiling(model, FLAGS.image_size, FLAGS.image_size, verbose = getattr(FLAGS, 'model_profiling_verbose', False))
    return bitops, bytesize


