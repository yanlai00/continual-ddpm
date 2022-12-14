import argparse

args = argparse.Namespace()
args.width_mult = 1.0
args.num_tasks = 10
args.sparsity = 0.5
args.mode = "fan_in"
args.nonlinearity = 'relu'


# args.save = True
# args.model = GEMResNet18
# output_size: 5
# er_sparsity: True

# optimizer: adam
# epochs: 100
# lr: 0.001
# batch_size: 128
# test_batch_size: 128

# adaptor: gt
# adapt: True
# hard_alphas: True
# adapt_lrs: [200]
# eval_ckpts: []
