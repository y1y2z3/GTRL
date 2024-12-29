import sys

sys.path.append('..')
import numpy as np
import os
import argparse
import pickle
import dgl
import torch
from src.utils import load_quadruples

print(os.getcwd())

# get direct graph

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def get_data_with_t(data, time):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == time]
    return np.array(triples)


def get_indices_with_t(data, time: int):
    """
    获取时间戳为time的所有四元组在四元组总序列中的index集合
    """
    idx = [i for i in range(len(data)) if data[i][3] == time]
    return np.array(idx)


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0, as_tuple=False).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm


def check_exist(outf):
    return os.path.isfile(outf)


def get_all_graph_dict(args):
    """构建某数据集的时态知识图"""
    file = os.path.join(args.dp+args.dn, 'dg_dict.txt')
    if not check_exist(file):
        graph_dict = {}  # 图字典：key-时间戳，value-知识图
        total_data, total_times, triples_dic = load_quadruples(args.dp+args.dn, 'train.txt', 'valid.txt', 'test.txt')
        print(total_data.shape, total_times.shape)

        for time in total_times:
            if time % 100 == 0:
                print(str(time)+'\tof '+str(max(total_times)))
            data = np.array(triples_dic[time])
            # edge_indices: 边缘索引
            edge_indices = get_indices_with_t(total_data, time)  # search from total_data (unsplitted)

            g = get_big_graph_w_idx(data, edge_indices)
            graph_dict[time] = g
        
        with open(file, 'wb') as fp:
            pickle.dump(graph_dict, fp)
        print('dg_dict.txt saved! ')
    else:
        print('dg_dict.txt exists! ')


def get_big_graph_w_idx(data, edge_indices):
    """构建某个时间戳的知识图"""
    src, rel, dst = data.transpose()  # 转置最后两个维度
    uniq_v, edges = np.unique((src, dst), return_inverse=True)  
    src, dst = np.reshape(edges, (2, -1))
    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    g.add_edges(src, dst, {'eid': torch.from_numpy(edge_indices)}) # array list
    norm = comp_deg_norm(g)
    g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)})
    g.edata['type'] = torch.LongTensor(rel)
    g.ids = {}
    idx = 0
    for id in uniq_v:
        g.ids[id] = idx
        idx += 1
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", default="../data/", help="dataset path")
    ap.add_argument("--dn", default="AFG", help="dataset name")

    args = ap.parse_args()
    print(args)

    if not os.path.exists(os.path.join(args.dp, args.dn)):
        print('Check if the dataset exists')
        exit()

    get_all_graph_dict(args)