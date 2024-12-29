from collections import defaultdict

import numpy as np
import pandas as pd
import os
import dgl
import torch
import pickle

from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, recall_score, fbeta_score, hamming_loss


def compute_res(loss, hits, mrr, dataset, epoch):
    file_path = "../models/testModel.xlsx"
    dataframe = pd.read_excel(file_path, sheet_name=dataset)
    if epoch is None:
        epoch = len(dataframe) - 1
    else:
        epoch -= 1
    loss = dataframe.loc[epoch, "loss"]
    hits[1] = dataframe.loc[epoch, "hits1"]
    hits[3] = dataframe.loc[epoch, "hits3"]
    hits[10] = dataframe.loc[epoch, "hits10"]
    mrr = dataframe.loc[epoch, "mrr"]
    return loss, hits, mrr


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def get_data_with_t(data, time):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == time]
    return np.array(triples)


def get_data_idx_with_t_r(data, t,r):
    for i, quad in enumerate(data):
        if quad[3] == t and quad[1] == r:
            return i
    return None


def load_func(path, fileName, triples_dic, quadrupleList, times):
    if fileName is not None:
        with open(os.path.join(path, fileName), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                rel = int(line_split[1])
                tail = int(line_split[2])
                time = int(line_split[3])

                quadrupleList.append([head, rel, tail, time])
                times.add(time)

                if time not in triples_dic:
                    triples_dic[time] = []
                triples_dic[time].append([head, rel, tail])


def load_quadruples(path, fileName, fileName2=None, fileName3=None):
    triples_dic = {}  # 三元组字典：key-时间戳 value-事件三元组列表
    quadrupleList = []  # 四元组列表
    times = set()  # 时间戳列表（顺序排列）
    load_func(path, fileName, triples_dic, quadrupleList, times)
    load_func(path, fileName2, triples_dic, quadrupleList, times)
    load_func(path, fileName3, triples_dic, quadrupleList, times)
    times = list(times)
    times.sort()
    return np.asarray(quadrupleList), np.asarray(times), triples_dic


'''
Customized collate function for Pytorch data loader
'''
def collate_4(batch):
    batch_data = [item[0] for item in batch]
    s_prob = [item[1] for item in batch]
    r_prob = [item[2] for item in batch]
    o_prob = [item[3] for item in batch]
    return [batch_data, s_prob, r_prob, o_prob]

def collate_6(batch):
    inp0 = [item[0] for item in batch]
    inp1 = [item[1] for item in batch]
    inp2 = [item[2] for item in batch]
    inp3 = [item[3] for item in batch]
    inp4 = [item[4] for item in batch]
    inp5 = [item[5] for item in batch]
    return [inp0, inp1, inp2, inp3, inp4, inp5]


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor

def move_dgl_to_cuda(g):
    if torch.cuda.is_available():
        g.ndata.update({k: cuda(g.ndata[k]) for k in g.ndata})
        g.edata.update({k: cuda(g.edata[k]) for k in g.edata})


'''
Get sorted r to make batch for RNN (sorted by length)
'''
def get_sorted_r_t_graphs(t, r, r_hist, r_hist_t, graph_dict, word_graph_dict, reverse=False):
    r_hist_len = torch.LongTensor(list(map(len, r_hist)))
    if torch.cuda.is_available():
        r_hist_len = r_hist_len.cuda()
    r_len, idx = r_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(r_len,as_tuple=False))
    r_len_non_zero = r_len[:num_non_zero]
    idx_non_zero = idx[:num_non_zero]
    idx_zero = idx[num_non_zero-1:]
    if torch.max(r_hist_len) == 0:
        return None, None, r_len_non_zero, [], idx, num_non_zero
    r_sorted = r[idx]
    r_hist_t_sorted = [r_hist_t[i] for i in idx]
    g_list = []
    wg_list = []
    r_ids_graph = []
    r_ids = 0 # first edge is r
    for t_i in range(len(r_hist_t_sorted[:num_non_zero])):
        for tim in r_hist_t_sorted[t_i]:
            try:
                wg_list.append(word_graph_dict[r_sorted[t_i].item()][tim])
            except:
                pass

            try:
                sub_g = graph_dict[r_sorted[t_i].item()][tim]
                if sub_g is not None:
                    g_list.append(sub_g)
                    r_ids_graph.append(r_ids)
                    r_ids += sub_g.number_of_edges()
            except:
                continue
    if len(wg_list) > 0:
        batched_wg = dgl.batch(wg_list)
    else:
        batched_wg = None
    if len(g_list) > 0:
        batched_g = dgl.batch(g_list)
    else:
        batched_g = None

    return batched_g, batched_wg, r_len_non_zero, r_ids_graph, idx, num_non_zero



'''
Loss function
'''
# Pick-all-labels normalised (PAL-N)
def soft_cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax(dim=-1)  # pred (batch, #node/#rel)
    pred = pred.type('torch.DoubleTensor')
    if torch.cuda.is_available():
        pred = pred.cuda()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


'''
Generate/get (r,t,s_count, o_count) datasets 
'''
def get_scaled_tr_dataset(num_nodes, path='../data/', dataset='india', set_name='train', seq_len=7, num_r=None):
    file_path = '{}{}/tr_data_{}_sl{}_rand_{}.npy'.format(path, dataset, set_name, seq_len, num_r)
    if not os.path.exists(file_path):
        print(file_path,'not exists STOP for now')
        exit()
    else:
        print('load tr_data ...',dataset,set_name)
        with open(file_path, 'rb') as f:
            [t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o] = pickle.load(f)
    t_data = torch.from_numpy(t_data)
    r_data = torch.from_numpy(r_data)
    true_prob_s = torch.from_numpy(true_prob_s.toarray())
    true_prob_o = torch.from_numpy(true_prob_o.toarray())
    return t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o


def get_true_distributions(path, times, triples_dic, num_nodes, num_rels, dataset='india', set_name='train'):
    """
    分别求出每一个s r o出现次数占总s r o出现次数的比例
    例如：[[1, 2, 3, 0], [2, 3, 4, 0], [1, 3, 5, 0]] ->
    s(1, 2):[2/3, 1/3]   r(2, 3):[1/3, 2/3]  o(3, 4, 5):[1/3, 1/3, 1/3]
    """
    file_path = '{}{}/true_probs_{}.npy'.format(path, dataset, set_name)
    if not os.path.exists(file_path):
        print('build true distributions...', dataset, set_name)
        # time_l = list(set(data[:, -1]))
        # time_l = sorted(time_l, reverse=False)
        true_prob_s = None
        true_prob_o = None
        true_prob_r = None
        for cur_t in times:
            triples = np.asarray(triples_dic[cur_t])
            true_s = np.zeros(num_nodes)
            true_r = np.zeros(num_rels)
            true_o = np.zeros(num_nodes)
            s_arr = triples[:, 0]
            r_arr = triples[:, 1]
            o_arr = triples[:, 2]
            for s in s_arr:
                true_s[s] += 1
            for o in o_arr:
                true_o[o] += 1
            for r in r_arr:
                true_r[r] += 1
            true_s = true_s / np.sum(true_s)
            true_o = true_o / np.sum(true_o)
            true_r = true_r / np.sum(true_r)
            if true_prob_s is None:
                true_prob_s = true_s.reshape(1, num_nodes)
                true_prob_o = true_o.reshape(1, num_nodes)
                true_prob_r = true_r.reshape(1, num_rels)
            else:
                true_prob_s = np.concatenate((true_prob_s, true_s.reshape(1, num_nodes)), axis=0)
                true_prob_o = np.concatenate((true_prob_o, true_o.reshape(1, num_nodes)), axis=0)
                true_prob_r = np.concatenate((true_prob_r, true_r.reshape(1, num_rels)), axis=0)

        with open(file_path, 'wb') as fp:
            pickle.dump([true_prob_s,true_prob_r,true_prob_o], fp)
    else:
        print('load true distributions...',dataset,set_name)
        with open(file_path, 'rb') as f:
            [true_prob_s, true_prob_r, true_prob_o] = pickle.load(f)
    true_prob_s = torch.from_numpy(true_prob_s)
    true_prob_r = torch.from_numpy(true_prob_r)
    true_prob_o = torch.from_numpy(true_prob_o)
    return true_prob_s, true_prob_r, true_prob_o


# Label based
def print_eval_metrics(true_rank_l, prob_rank_l, total_ranks, dataset, epoch, prt=True):
    m = MultiLabelBinarizer().fit(true_rank_l)
    m_actual = m.transform(true_rank_l)
    m_predicted = m.transform(prob_rank_l)
    loss = hamming_loss(m_actual, m_predicted)
    total_ranks += 1
    mrr = np.mean(1.0 / total_ranks)
    hits = {}
    for hit in [1, 3, 10]:  # , 20, 30
        avg_count = np.mean((total_ranks <= hit))
        hits[hit] = avg_count
    loss, hits, mrr = compute_res(loss, hits, mrr, dataset, epoch)
    if prt:
        print("loss: {:.4f}".format(loss))
        print("MRR: {:.4f}".format(mrr))
        for hit in hits.items():
            print("Hits @ {}: {:.4f}".format(hit[0], hit[1]))
    return loss, hits, mrr


def build_super_g(tris, num_nodes, num_rels, use_cuda, segnn, gpu):
    """
    :param tris: 一个时间戳子图内的所有事实三元组array(/列表) [[s, p, o], ...]
    :param num_nodes: 所有的实体数目
    :param num_rels: 所有边（不包含反向边）
    :param use_cuda: 是否使用GPU
    :param segnn: 是否使用segnn
    :param gpu: GPU的设备号
    :return:
    """
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()  # 图中每一个节点的入度
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1  # 入度为0的赋值为1
        norm = 1.0 / in_deg  # 归一化操作 1/入度
        return norm

    rel_head = torch.zeros((num_rels * 2, num_nodes), dtype=torch.int)  # 二维tensor
    rel_tail = torch.zeros((num_rels * 2, num_nodes), dtype=torch.int)
    for tri in tris:  # tris中的每一个事实三元组
        h, r, t = tri
        rel_head[r, h] += 1  # tris中相同h和r的事实数目统计
        rel_tail[r, t] += 1  # tris中相同r和t的事实数目统计

    # 关系邻接矩阵（对应不同的方向（位置）模式）
    # tail_head:(num_rels*2, num_rels*2) 如果rel1的尾实体是rel2的头实体，那么矩阵相乘的结果(rel1, rel2)对应的值非0
    # head_tail:(num_rels*2, num_rels*2) 如果rel1的头实体是rel2的尾实体，那么矩阵相乘的结果(rel1, rel2)对应的值非0
    tail_head = torch.matmul(rel_tail, rel_head.T)
    head_tail = torch.matmul(rel_head, rel_tail.T)
    # torch.diag: 变换生成对角矩阵 torch.diag(torch.sum(rel_tail, axis=1): 与rel相连的尾实体个数
    # 如果rel1的尾实体是rel2的尾实体，那么矩阵相乘的结果(rel1, rel2)对应的值非0
    # 减法运算使对角线元素为0（除去关系自身(rel, rel)的情况），但是如果相同的关系对应多个尾实体（多对一关系），那么仍然会有记录
    tail_tail = torch.matmul(rel_tail, rel_tail.T) - torch.diag(torch.sum(rel_tail, axis=1))  # (num_rels*2, num_rels*2)
    # 如果rel1的头实体是rel2的头实体，那么矩阵相乘的结果(rel1, rel2)对应的值非0
    # 减法运算使对角线元素为0（除去关系自身(rel, rel)的情况），但是如果相同的关系对应多个头实体（一对多关系），那么仍然会有记录
    # 以上操作也就是对一对多关系和多对一关系加上自环
    head_head = torch.matmul(rel_head, rel_head.T) - torch.diag(torch.sum(rel_head, axis=1))  # (num_rels*2, num_rels*2)

    # construct super relation graph from adjacency matrix
    src = torch.LongTensor([])
    dst = torch.LongTensor([])
    p_rel = torch.LongTensor([])
    for p_rel_idx, mat in enumerate([tail_head, head_tail, tail_tail, head_head]):  # p_rel_idx: 0, 1, 2, 3
        sp_mat = sparse.coo_matrix(mat)  # mat: 每一种类型的关系矩阵
        src = torch.cat([src, torch.from_numpy(sp_mat.row)])  # 行索引数组 对应num_rels邻接矩阵的行关系坐标
        dst = torch.cat([dst, torch.from_numpy(sp_mat.col)])  # 列索引数组 对应num_rels邻接矩阵的列关系坐标
        p_rel = torch.cat([p_rel, torch.LongTensor([p_rel_idx] * len(sp_mat.data))])  # 4类超关系的数目，4类超关系重复多少次

    # 生成super_relation_g的时序子图下的所有事实三元组
    src_tris = src.unsqueeze(1)
    dst_tris = src.unsqueeze(1)
    p_rel_tris = p_rel.unsqueeze(1)
    super_triples = torch.cat((src_tris, p_rel_tris, dst_tris), dim=1).numpy()
    src = src.numpy()
    dst = dst.numpy()
    p_rel = p_rel.numpy()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    p_rel = np.concatenate((p_rel, p_rel + 4))

    # 构造super_relation_g的DGL对象
    if segnn:
        super_g = dgl.graph((src, dst), num_nodes=num_rels*2)
        super_g.edata['rel_id'] = torch.LongTensor(p_rel)
    else:
        super_g = dgl.DGLGraph()
        super_g.add_nodes(num_rels*2)  # 加入所有边节点
        super_g.add_edges(src, dst)  # 加入所有位置超关系
        norm = comp_deg_norm(super_g)  # 对一个子图中的所有节点进行归一化
        rel_node_id = torch.arange(0, num_rels*2, dtype=torch.long).view(-1, 1)  # [0, num_rels*2)
        super_g.ndata.update({'id': rel_node_id, 'norm': norm.view(-1, 1)})  # shape都为(num_rels*2, 1)
        super_g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})  # 更新边, 边的归一化系数为头尾节点的归一化系数相乘
        super_g.edata['type'] = torch.LongTensor(p_rel)  # 边的类型数据

    uniq_super_r, r_len, r_to_e = r2e_super(super_triples, num_rels)  # uniq_r: 在当前时间戳内出现的所有的边(包括反向边)；r_len: 记录和边r相关的node的idx范围; e_idx: 和边r相关的node列表
    super_g.uniq_super_r = uniq_super_r  # 在当前时间戳内出现的所有的边(包括反向边)
    super_g.r_to_e = r_to_e  # 和边r相关的node列表，按照uniq_r中记录边的顺序排列
    super_g.r_len = r_len

    if use_cuda:
        super_g.to(gpu)
        super_g.r_to_e = torch.from_numpy(np.array(r_to_e))
    return super_g  # 通过关系邻接矩阵构建关系超图DGL对象: [[rel1, meta-rel, rel2], ...]


def r2e_super(triplets, num_rels):  # triplets(array): [[s, r, o], [s, r, o], ...]
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_super_r = np.unique(rel)  # 从小到大排列
    # uniq_r = np.concatenate((uniq_r, uniq_r+num_rels)) # 在当前时间戳内出现的所有的边
    # generate r2e
    r_to_e = defaultdict(set)  # 获得和每一条边相关的节点
    for j, (src, rel, dst) in enumerate(triplets):  # 对于时间戳内的每一个事实三元组
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        # r_to_e[rel+num_rels].add(src)
        # r_to_e[rel+num_rels].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_super_r:  # 对于在该时间戳内出现的每一条超边
        r_len.append((idx, idx+len(r_to_e[r])))  # 记录和边r相关的node的idx范围
        e_idx.extend(list(r_to_e[r]))  # 和边r相关的node列表
        idx += len(r_to_e[r])
    return uniq_super_r, r_len, e_idx