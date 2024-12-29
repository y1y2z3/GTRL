import time
import warnings
import argparse
from models import *
from data import *
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


def train(model, data_loader, dataset_loader):
    model.train()
    t0 = time.time()
    for i, batch in enumerate(tqdm(data_loader)):
        batch_data, true_s, true_r, true_o = batch
        batch_data = torch.stack(batch_data, dim=0)
        true_r = torch.stack(true_r, dim=0)
        time_triples = torch.tensor(dataset_loader.triples[batch_data[0].item()])
        loss = model(batch_data, true_r, time_triples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()
        optimizer.zero_grad()

    t2 = time.time()
    print("Epoch {:04d} | time {:.2f} {}".format(epoch, t2 - t0, time.ctime()))


@torch.no_grad()
def evaluate(model, epoch, data_loader, dataset_loader, dataset, set_name='valid'):
    model.eval()
    true_rank_l = []
    prob_rank_l = []
    total_ranks = np.array([])
    total_loss = 0
    for i, batch in enumerate(tqdm(data_loader)):
        batch_data, true_s, true_r, true_o = batch
        batch_data = torch.stack(batch_data, dim=0)
        true_r = torch.stack(true_r, dim=0)
        time_triples = torch.tensor(dataset_loader.triples[batch_data[0].item()])
        true_rank, prob_rank, loss, ranks = model.evaluate(batch_data, true_r, time_triples)
        true_rank_l.append(true_rank.cpu().tolist())
        prob_rank_l.append(prob_rank.cpu().tolist())
        total_ranks = np.concatenate((ranks.cpu().numpy(), total_ranks))
        total_loss += loss.item()
    if set_name == "Test":
        print('Test results:')
    else:
        print('Epoch {:04d} {} results:'.format(epoch, set_name))
    loss, hits, mrr = utils.print_eval_metrics(true_rank_l, prob_rank_l, total_ranks, dataset, epoch)
    return loss, hits, mrr


def warn(*args, **kwargs):
    pass


if __name__ == '__main__':
    warnings.warn = warn
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--max-epochs", type=int, default=200, help="maximum epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--dp", type=str, default="../data/", help="data path")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=100, help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=1, help="gpu")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS18', help="dataset to use")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
    parser.add_argument("--seq-len", type=int, default=7, help="historical window length")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--maxpool", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use-gru", type=int, default=1, help='1 use gru 0 rnn')
    parser.add_argument("--attn", type=str, default='', help='dot/add/general; default general')
    parser.add_argument("--seed", type=int, default=42, help='random seed')
    parser.add_argument('--hc_ratio', type=float, default=0.1, help='ratio in Hierarchical graph.')
    parser.add_argument('--device', type=str, default='cpu', help='')
    parser.add_argument('--mode', type=str, default='full', help='')
    parser.add_argument('--run_times', type=int, default=5, help='')
    parser.add_argument('--epoch', type=int, default=100, help='')
    parser.add_argument('--city_num', type=int, default=849, help='')
    parser.add_argument('--group_num', type=int, default=16, help='')
    parser.add_argument('--gnn_h', type=int, default=300, help='')
    parser.add_argument('--gnn_layer', type=int, default=2, help='')
    parser.add_argument('--x_em', type=int, default=100, help='x embedding')
    parser.add_argument('--date_em', type=int, default=4, help='date embedding')
    parser.add_argument('--edge_h', type=int, default=300, help='edge h')
    parser.add_argument('--wd', type=float, default=0.002, help='weight decay')
    parser.add_argument('--pred_step', type=int, default=6, help='step')
    parser.add_argument("--num-r", type=int, default=20, help="number of rel to consider")
    parser.add_argument("--test", action="store_true", help="only test, not train")
    parser.add_argument("--test_model_name", "--tmn", type=str, help="test model name")
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    print("cuda", use_cuda)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载数据
    print("loading data...")
    num_nodes, num_rels = utils.get_total_number(args.dp + args.dataset, 'stat.txt')
    train_dataset_loader = DistData(args.dp, args.dataset, num_nodes, num_rels, set_name='train')
    valid_dataset_loader = DistData(args.dp, args.dataset, num_nodes, num_rels, set_name='valid')
    test_dataset_loader = DistData(args.dp, args.dataset, num_nodes, num_rels, set_name='test')
    train_loader = DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=False, collate_fn=collate_4)
    valid_loader = DataLoader(valid_dataset_loader, batch_size=1, shuffle=False, collate_fn=collate_4)
    test_loader = DataLoader(test_dataset_loader, batch_size=1, shuffle=False, collate_fn=collate_4)

    # 加载图字典：key-时间戳，value-知识图
    print('loading dg_dict.txt...')
    with open(args.dp + args.dataset+'/dg_dict.txt', 'rb') as f:
        graph_dict = pickle.load(f)

    # 加载超图
    print("loading super graph...")
    train_rel_super_g_list = []
    valid_rel_super_g_list = []
    test_rel_super_g_list = []
    for time in train_dataset_loader.times:
        rel_super_g = utils.build_super_g(train_dataset_loader.triples[time], num_nodes, num_rels, use_cuda, 1, args.gpu)
        train_rel_super_g_list.append(rel_super_g)
    for time in valid_dataset_loader.times:
        rel_super_g = utils.build_super_g(valid_dataset_loader.triples[time], num_nodes, num_rels, use_cuda, 1, args.gpu)
        valid_rel_super_g_list.append(rel_super_g)
    for time in test_dataset_loader.times:
        rel_super_g = utils.build_super_g(test_dataset_loader.triples[time], num_nodes, num_rels, use_cuda, 1, args.gpu)
        test_rel_super_g_list.append(rel_super_g)

    # add
    N = len(graph_dict)  # 知识图数量（时间戳数量）
    latend_num = int(N * args.hc_ratio + 0.5)  # latent node number
    model = GleanEvent(h_dim=args.n_hidden, num_ents=num_nodes, num_rels=num_rels, dropout=args.dropout,
                       seq_len=args.seq_len, maxpool=args.maxpool, use_gru=args.use_gru,
                       latend_num=latend_num, mode=args.mode, x_em=args.x_em, date_em=args.date_em, edge_h=args.edge_h,
                       gnn_h=args.gnn_h, gnn_layer=args.gnn_layer, batch_entity_num=args.city_num, group_num=args.group_num,
                       pred_step=args.pred_step, device=args.device)

    model_name = model.__class__.__name__
    print('Model:', model_name)
    token = '{}_sl{}_max{}_gru{}'.format(model_name, args.seq_len, int(args.maxpool), int(args.use_gru))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#params:', total_params)

    os.makedirs('../models', exist_ok=True)
    os.makedirs('../models/' + args.dataset, exist_ok=True)
    model_state_file = '../models/{}/{}.pth'.format(args.dataset, token)
    if use_cuda:
        model.cuda()
    model.graph_dict = graph_dict

    if not args.test:
        bad_counter = 0
        loss_small = float("inf")  # 正无穷大
        print("start training...")
        for epoch in range(1, args.max_epochs+1):
            try:
                train(model, train_loader, train_dataset_loader)
                valid_loss, hits, mrr = evaluate(model, epoch, valid_loader, valid_dataset_loader, args.dataset, 'Valid')

                if valid_loss < loss_small and len(hits) > 0:
                    loss_small = valid_loss
                    bad_counter = 0
                    print('save better model...')
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'global_emb': None}, model_state_file)
                else:
                    bad_counter += 1
                if bad_counter == args.patience:
                    break
            except KeyboardInterrupt:
                print('-' * 80 + '\nExiting from training early, epoch', epoch)
                break
        print("training done")
    else:
        model_state_file = '../models/{}/{}.pth'.format(args.dataset, args.test_model_name)

    # Load the best saved model
    print('-' * 40 + "start testing" + '-' * 40)
    checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    loss, hits, mrr = evaluate(model, None, test_loader, test_dataset_loader, args.dataset, set_name='Test')
