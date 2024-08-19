import argparse

def argparse_common():
    #Common Parser
    parser = argparse.ArgumentParser()
    
    #Seed
    parser.add_argument('--seed', type=int, default=2024, 
                        help='random seed')
    #BRP Param
    parser.add_argument('-s','--max_stacks' ,type=int ,default=5,
                        help='number of stacks')
    
    parser.add_argument('-t','--max_tiers',type=int,default=7, help="number of tiers")
    #Model Param
    parser.add_argument('-em', '--embed_dim', type=int, default=128, help='embedding size')
    parser.add_argument('-nh', '--n_heads',  type=int, default=8, help='number of heads in MHA')
    parser.add_argument('-ne', '--n_encode_layers', type=int, default=3, help='number of MHA encoder layers')
    parser.add_argument('-ff', '--ff_hidden_dim', type=int, default=512, help='ff_hidden dimension')
    return parser

def argparse_train_IL():
    parser = argparse_common()
    parser.add_argument('-b', '--batch', type=int, default = 256, help='batch size')
    parser.add_argument('-bv', '--batch_verbose', type=int, default = 100, help='batch verbose size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-E', '--epoch', type=int, default=50, help = "epoch num")

    return parser.parse_args()

def argparse_train_SSIL():
    parser = argparse_common()
    #Model Directory
    parser.add_argument('-mp', '--model_path', type=str, default = "uBRP_IL.pt", help="Model path, note that it should be in train/pre_trained/~.")
    #Training Parameter
    parser.add_argument('-b', '--batch', type=int, default = 256, help='batch size')
    parser.add_argument('-bv', '--batch_verbose', type=int, default = 100, help='batch verbose size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('-E', '--epoch', type=int, default=200, help = "epoch num")

    #Sampling Parameter
    parser.add_argument('-pn', '--problem_num', type=int, default=1024, help = "number of problems")
    parser.add_argument('-sm', '--sampling_num', type=int, default=512, help = "number of sampling number (i.e. How many time solving same problem)")
    #SSIL baseline Parameter
    parser.add_argument('-ca', '--commit_alpha', type=float, default=0.05, help = "significance for commit (alpha_a in paper)")
    parser.add_argument('-ra', '--rollback_alpha', type=float, default=0.1, help = "significance for rollback (alpha_b in paper)")
    return parser.parse_args()

def argparse_test():
    parser = argparse_common()
    #Test ALL
    parser.add_argument("--test_all", action=argparse.BooleanOptionalAction)#--no-test_all if you want to test each
    parser.add_argument('-mp', '--model_path', type=str, default = "uBRP_IL_SSIL.pt", help="Model path, note that it should be in train/pre_trained/~.")
    parser.add_argument('-dt','--decode_type', type=str, default = 'greedy', choices=['greedy', 'sampling', 'ESS'], help='Greedy or Sampling type')
    parser.add_argument('-T', '--temp', type=float, default=1, help='Temperature for Softmax, can be applied on sampling and ESS')
    parser.add_argument('-b', '--batch', type=int, default = 2560, help='Sampling batch-size')
    parser.add_argument('-N', '--sampling_num', type=int, default=2560, help='Total sampling number. Should be multiple of batch')
    return parser.parse_args()


