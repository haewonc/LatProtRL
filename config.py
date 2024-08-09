import argparse

def config_rep(device, protein, level, reduce_dim=None):
    assert protein in ['GFP', 'AAV']
    args = argparse.Namespace()
    args.name = protein
    args.device = device 
    args.level = level 
    args.embed_dim = 1280
    args.num_layers = 33
    args.hidden_dim = 256
    if protein == 'GFP':
        args.length = 237
        args.num_tokens = 237 + 2
        args.reduce_dim = 32
        args.num_trainable_layers = 4
    elif protein == 'AAV':
        args.length = 28
        args.num_tokens = 28 + 2
        args.reduce_dim = 16
        args.num_trainable_layers = 4
    else:
        raise NotImplementedError()
    if reduce_dim != None:
        args.reduce_dim = reduce_dim
    return args

def get_fitness_info(protein):
    if protein == 'GFP':
        length = 237
        min_fitness = 1.283419251
        max_fitness = 4.123108864
    elif protein == 'AAV':
        length = 28
        min_fitness = 0.0
        max_fitness = 19.53645667061
    else:
        raise NotImplementedError()
    return length, min_fitness, max_fitness

def create_base(args):
    assert args.protein in ['GFP', 'AAV']
    opt = argparse.Namespace()
    opt.name = args.protein
    opt.device = args.device 
    opt.level = args.level
    opt.length, opt.min_fitness, opt.max_fitness = get_fitness_info(args.protein)
    opt.seq_pretrained =f'saved/{args.protein}_{args.level}_LM.pt'
    opt.rew_pretrained = f'ckpt/{opt.name}/oracle.ckpt' 
    opt.reduce_dim = None
    return opt

def create_opt(args):
    assert args.protein in ['GFP', 'AAV']
    opt = argparse.Namespace()
    opt.name = args.protein
    opt.device = args.device 
    opt.level = args.level
    opt.not_sparse = args.not_sparse
    opt.length, opt.min_fitness, opt.max_fitness = get_fitness_info(args.protein)
    opt.step_mut = args.step_mut
    if args.protein == 'GFP':
        opt.action_size = 0.3
        opt.topk = 18
        opt.done_cond = argparse.Namespace(max_steps=5, max_mutation=15, step_mut=opt.step_mut)
    elif args.protein == 'AAV':
        opt.action_size = 0.1
        opt.topk = 8
        opt.done_cond = argparse.Namespace(max_steps=3, max_mutation=15, step_mut=opt.step_mut)
    opt.seq_pretrained =f'saved/{args.protein}_{args.level}_LM.pt'
    opt.rew_pretrained = f'ckpt/{opt.name}/oracle.ckpt' if args.use_oracle else f'ckpt/{opt.name}/{args.level}.ckpt'
    opt.reduce_dim = None
    return opt

def create_rep_from_opt(opt):
    return config_rep(opt.device, opt.name, opt.level, opt.reduce_dim)
