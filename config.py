import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(
        description="FedHUA: Federated Heterogeneity and Uncertainty Guided Alignment via Query Generation"
    )

    # General federated-learning settings.
    parser.add_argument('--gpu', type=str, default='0', help='CUDA device identifier(s).')
    parser.add_argument('--model', type=str, default='simplecnn', help='Neural network used in training.')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset: cifar10, cifar100, or fashionmnist.')
    parser.add_argument('--partition', type=str, default='noniid', help='Partition strategy: iid, noniid, or noniid-skew-<k>.')
    parser.add_argument('--num_local_iterations', type=int, default=400, help='Number of local SGD iterations.')
    parser.add_argument('--batch_size', type=int, default=64, help='Training and generated-query mini-batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Local SGD learning rate.')
    parser.add_argument('--epochs', type=int, default=10, help='Reserved for epoch-based variants.')
    parser.add_argument('--n_parties', type=int, default=10, help='Number of clients.')
    parser.add_argument('--comm_round', type=int, default=55, help='Total number of communication rounds.')
    parser.add_argument('--init_seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--datadir', type=str, default='./data/', help='Data directory.')
    parser.add_argument('--beta', type=float, default=0.5, help='Dirichlet concentration for noniid partitioning.')
    parser.add_argument('--skew_class', type=int, default=2, help='Classes per client for noniid-skew partitioning.')
    parser.add_argument('--reg', type=float, default=1e-5, help='L2 weight decay.')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'amsgrad'], help='Local optimizer.')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='Fraction of selected clients per round.')
    parser.add_argument('--load_path', type=str, default=None, help='Optional global-model checkpoint path.')
    parser.add_argument('--save_model', action='store_true', help='Save the final global model.')

    # FedHUA settings.  The aliases retain compatibility with the earlier package.
    parser.add_argument('--warmup_rounds', '--start_round', dest='warmup_rounds', default=50, type=int,
                        help='FedAvg warm-up rounds before FedHUA query generation.')
    parser.add_argument('--base_budget', '--synthesis_batch_size', dest='base_budget', default=256, type=int,
                        help='Base generation budget N_0 per client and round.')
    parser.add_argument('--generation_steps', '--g_steps', dest='generation_steps', default=500, type=int,
                        help='Input-optimization steps G for each query-generation attempt.')
    parser.add_argument('--lr_g', default=1e-2, type=float, help='Input-optimization step size eta_x.')
    parser.add_argument('--alpha', '--ua_budget_alpha', dest='alpha', default=1.0, type=float,
                        help='Drift-to-budget sensitivity alpha.')
    parser.add_argument('--budget_min_scale', '--ua_budget_min_scale', dest='budget_min_scale', default=0.5, type=float,
                        help='Lower clipping bound s_min for the budget scale.')
    parser.add_argument('--budget_max_scale', '--ua_budget_max_scale', dest='budget_max_scale', default=2.0, type=float,
                        help='Upper clipping bound s_max for the budget scale.')
    parser.add_argument('--gamma', '--ua_uncertainty_beta', dest='gamma', default=1.0, type=float,
                        help='Class-allocation trade-off gamma for local uncertainty.')
    parser.add_argument('--entropy_batches', '--ua_entropy_batches', dest='entropy_batches', default=2, type=int,
                        help='Local real-data batches used to estimate class-wise entropy.')
    parser.add_argument('--lambda_u', '--ua_entropy_weight', dest='lambda_u', default=0.3, type=float,
                        help='Weight lambda_u for local predictive entropy in generation.')
    parser.add_argument('--lambda_b', '--ua_boundary_lambda', dest='lambda_b', default=0.1, type=float,
                        help='Weight lambda_b for local boundary-margin guidance.')
    parser.add_argument('--tau_g', default=0.70, type=float,
                        help='Teacher-confidence threshold for retaining a synthetic query.')
    parser.add_argument('--tau_b', default=0.30, type=float,
                        help='Local top-1/top-2 margin threshold for retaining a synthetic query.')
    parser.add_argument('--max_query_attempts', default=10, type=int,
                        help='Maximum reinitialization attempts for each pending synthetic query batch.')
    parser.add_argument('--temperature', '--T', dest='temperature', default=1.0, type=float,
                        help='Knowledge-distillation temperature.')
    parser.add_argument('--fashion_lambda_kd', default=0.01, type=float,
                        help='Fixed KD coefficient for Fashion-MNIST.')

    # Baseline-method options retained by the comparison scripts.
    parser.add_argument('--server_momentum', type=float, default=0.0)
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--lambda_1', type=float, default=0.01)

    args = parser.parse_args()

    cfg = {}
    if args.dataset == 'cifar10':
        cfg['classes_size'] = 10
        cfg['channel'] = 3
        cfg['image_size'] = 32
    elif args.dataset == 'cifar100':
        cfg['classes_size'] = 100
        cfg['channel'] = 3
        cfg['image_size'] = 32
    elif args.dataset == 'fashionmnist':
        cfg['classes_size'] = 10
        cfg['channel'] = 1
        cfg['image_size'] = 28
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

    args.channel = cfg['channel']
    args.image_size = cfg['image_size']
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args, cfg
