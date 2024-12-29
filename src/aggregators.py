from propagations import *
from modules import *
# add
import torch.nn.functional as F
from hierarchical_graph_conv import SCConv
# Sequential是一个容器，用于按顺序组合多个神经网络层（如线性层、卷积层、激活函数等）和操作
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.nn import TransformerEncoderLayer
from torch.nn.parameter import Parameter


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        if n_layers < 2:
            self.layers.append(GCNLayer(in_feats, n_classes, activation, dropout))
        else:
            self.layers.append(GCNLayer(in_feats, n_hidden, activation, dropout))
            for i in range(n_layers - 2):
                self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
            self.layers.append(GCNLayer(n_hidden, n_classes, activation, dropout))  # activation or None

    def forward(self, g, features=None):  # no reverse
        if features is None:
            h = g.ndata['h']
        else:
            h = features
        for layer in self.layers:
            h = layer(g, h)
        return h


# aggregator for event forecasting
class AggregatorEvent(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, seq_len=10, maxpool=1, latend_num=0,
                 mode=0, x_em=0, date_em=0, edge_h=0, gnn_h=0, gnn_layer=0, batch_entity_num=0, group_num=0,
                 pred_step=0, device=0, encoder="lstm", w=None, w_init="rand"):
        super().__init__()
        # old
        self.h_dim = h_dim  # feature
        self.latend_num = latend_num
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.se_aggr = GCN(100, h_dim, h_dim, 2, F.relu, dropout)

        # new
        self.mode = mode
        self.x_em = x_em  # 实体embedding特征维度 default = 100
        self.date_em = date_em  # date_embedding特征维度 default = 4
        self.edge_h = edge_h  # 关系embedding特征维度 default = 300
        self.gnn_h = gnn_h
        self.gnn_layer = gnn_layer
        self.batch_entity_num = batch_entity_num
        self.group_num = group_num
        self.pred_step = pred_step
        self.device = device
        self.w_init = w_init
        self.encoder = encoder
        self.new_x = w

        if self.encoder == 'self':
            self.encoder_layer = TransformerEncoderLayer(100, nhead=2, dim_feedforward=100)
            self.x_embed = Lin(100, x_em)
        elif self.encoder == 'lstm':
            self.input_LSTM = nn.LSTM(100, x_em, num_layers=1, batch_first=True)
        if self.w_init == 'rand':
            self.w = Parameter(torch.randn(batch_entity_num, group_num).to(device, non_blocking=True), requires_grad=True)
        elif self.w_init == 'group':
            self.w = Parameter(self.new_w, requires_grad=True)

        # nn.Embedding(num_embeddings, embedding_dim):将num_embeddings个单词用embedding_dim维的特征向量来表示
        self.u_embed1 = nn.Embedding(12, date_em)  # month
        self.u_embed2 = nn.Embedding(7, date_em)  # week
        self.u_embed3 = nn.Embedding(24, date_em)  # hour
        self.edge_inf = Seq(Lin(x_em * 2, edge_h), ReLU(inplace=True))
        self.group_gnn = nn.ModuleList([NodeModel(x_em, edge_h, gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.group_gnn.append(NodeModel(gnn_h, edge_h, gnn_h))
        self.global_gnn = nn.ModuleList([NodeModel(x_em + gnn_h, 100, gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.global_gnn.append(NodeModel(gnn_h, 1, gnn_h))
        if self.mode == 'ag':
            self.decoder = DecoderModule(x_em, edge_h, gnn_h, gnn_layer, batch_entity_num, group_num, device)
            self.predMLP = Seq(Lin(gnn_h, 16), ReLU(inplace=True), Lin(16, 1), ReLU(inplace=True))
        if self.mode == 'full':
            self.decoder = DecoderModule(x_em, edge_h, gnn_h, gnn_layer, batch_entity_num, group_num, device)
            self.predMLP = Seq(Lin(gnn_h, 16), ReLU(inplace=True), Lin(16, self.pred_step), ReLU(inplace=True))

        # SCConv
        hid_dim = 32
        dropout = 0.5
        alpha = 0.2
        self.SCConv = SCConv(in_features=hid_dim + 50, out_features=hid_dim, dropout=dropout, alpha=alpha, latend_num=latend_num, gcn_hop=1)

        if maxpool == 1:
            self.dgl_global_edge_f = dgl.max_edges
            self.dgl_global_node_f = dgl.max_nodes
        else:
            self.dgl_global_edge_f = dgl.mean_edges
            self.dgl_global_node_f = dgl.mean_nodes

        out_feat = int(h_dim // 2)
        self.re_aggr1 = CompGCN_dg(h_dim, out_feat, h_dim, out_feat, True, F.relu, self_loop=True, dropout=dropout)
        self.re_aggr2 = CompGCN_dg(out_feat, h_dim, out_feat, h_dim, True, F.relu, self_loop=True, dropout=dropout)

    def batchInput(self, x, edge_w, edge_index):
        sta_num = x.shape[1]
        x = x.reshape(-1, x.shape[-1])
        edge_w = edge_w.reshape(-1, edge_w.shape[-1])
        for i in range(edge_index.size(0)):
            edge_index[i, :] = torch.add(edge_index[i, :], i * sta_num)
        edge_index = edge_index.transpose(0, 1)
        edge_index = edge_index.reshape(2, -1)
        return x, edge_w, edge_index

    def forward(self, t_list, ent_embeds, rel_embeds, graph_dict):
        """
        t_list:当前batch的时间戳序号，逆序的
        ent_embeds:实体表示
        rel_embeds:关系表示
        graph_dict:图字典
        """
        # t_list, ent_embeds, rel_embeds: torch.Size([16]) torch.Size([208, 100]) torch.Size([164, 100])
        all_time_list = list(graph_dict.keys())
        all_time_list.sort(reverse=False)  # 0 to future
        pre_time_list = []
        pre_time_len_list = []
        nonzero_idx = torch.nonzero(t_list, as_tuple=False).view(-1)
        t_list = t_list[nonzero_idx]  # usually no duplicates

        for tim in t_list:
            length = all_time_list.index(tim)
            if self.seq_len <= length:
                # 拿当前time下的前seq_len个time
                pre_time_list.append(torch.LongTensor(all_time_list[length - self.seq_len:length]))
                pre_time_len_list.append(self.seq_len)
            else:
                pre_time_list.append(torch.LongTensor(all_time_list[:length]))
                pre_time_len_list.append(length)

        # 初始化一个batch内所有时间戳的图
        unique_t = torch.unique(torch.cat(pre_time_list))
        t_idx = list(range(len(unique_t)))
        # time2id mapping  {0: 0, 24: 1, 48: 2, 72: 3, 96: 4, 120: 5, ... , 312: 13, 336: 14}
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        # entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        batched_g = dgl.batch(g_list)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batched_g = batched_g.to(device)  # torch.device('cuda:0')
        batched_g.ndata['h'] = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1])
        if torch.cuda.is_available():
            type_data = batched_g.edata['type'].cuda()
        else:
            type_data = batched_g.edata['type']
        batched_g.edata['e_h'] = rel_embeds.index_select(0, type_data)
        x = batched_g.ndata["h"]  # 整体图中的所有实体特征表示 torch.Size([1, 791, 100])
        edge_w = batched_g.edata["e_h"]  # 整体图中的所有关系特征表示

        # change
        edges_one = batched_g.edges()[0].unsqueeze(1)
        edges_two = batched_g.edges()[1].unsqueeze(1)
        edge_index_a = torch.cat((edges_one, edges_two), 1)
        edge_index = edge_index_a.transpose(0, 1).unsqueeze(0)
        x = x.unsqueeze(0).unsqueeze(2)  # torch.Size([1, 791, 1, 100])

        # start
        x = x.reshape(-1, x.shape[2], x.shape[3])  # torch.Size([791, 1, 100])
        self.batch_entity_num = x.shape[0]  # 一个batch的整体知识图中实体总数
        # w是论文中提到的M矩阵，表示所有实体分别属于每个实体组的概率（整体知识图共享）
        if self.w_init == 'rand':
            self.w = Parameter(torch.randn(self.batch_entity_num, self.group_num).to(device, non_blocking=True), requires_grad=True)  # torch.Size([791, 16])
        elif self.w_init == 'group':
            self.w = Parameter(self.new_x, requires_grad=True)

        if self.mode == 'ag':
            self.decoder = DecoderModule(self.x_em, self.edge_h, self.gnn_h, self.gnn_layer, self.batch_entity_num, self.group_num, device)
            self.predMLP = Seq(Lin(self.gnn_h, 16), ReLU(inplace=True), Lin(16, 1), ReLU(inplace=True))
        if self.mode == 'full':
            self.decoder = DecoderModule(self.x_em, self.edge_h, self.gnn_h, self.gnn_layer, self.batch_entity_num, self.group_num, device)
            self.predMLP = Seq(Lin(self.gnn_h, 16), ReLU(inplace=True), Lin(16, self.pred_step), ReLU(inplace=True))

        if self.encoder == 'self':
            x = self.encoder_layer(x)
            x = self.x_embed(x)
            # x = x.reshape(-1, self.city_num, TIME_WINDOW, x.shape[-1])
            x = x.reshape(-1, self.batch_entity_num, 1, x.shape[-1])
            x = torch.max(x, dim=-2).values
        elif self.encoder == 'lstm':
            _, (x, _) = self.input_LSTM(x)
            x = x.reshape(-1, self.batch_entity_num, x.shape[-1])  # torch.Size([1, 791, 100])

        # graph pooling
        w = F.softmax(self.w)  # torch.Size([791, 16])
        w1 = w.transpose(0, 1)  # torch.Size([16, 791])
        w1 = w1.unsqueeze(dim=0)  # torch.Size([1, 16, 791])
        w1 = w1.repeat_interleave(x.size(0), dim=0)  # torch.Size([1, 16, 791]) 将w1的第dim每个元素重复 x.size(0) = 1 次
        g_x = torch.bmm(w1, x)  # 初始化实体组的特征表示，通过聚合属于该组的实体特征表示来实现  对应公式2 torch.Size([1, 16, 100])

        # 实体组间关系特征表示训练
        # group gnn
        for i in range(self.group_num):
            for j in range(self.group_num):
                if i == j:
                    continue
                # g_edge_input: 实体组i和j间的初始关系特征表示，通过拼接两者特征表示来实现
                g_edge_input = torch.cat([g_x[:, i], g_x[:, j]], dim=-1)  # torch.Size([1, 200])
                tmp_g_edge_w = self.edge_inf(g_edge_input)  # torch.Size([1, 300])
                tmp_g_edge_w = tmp_g_edge_w.unsqueeze(dim=0)  # torch.Size([1, 1, 300])  对应公式3
                tmp_g_edge_index = torch.tensor([i, j]).unsqueeze(dim=0).to(self.device, non_blocking=True)  # torch.Size([1, 2])
                # g_edge_w: 实体组间关系表示  g_edge_index: 实体组间关系表示对应的索引
                if i == 0 and j == 1:
                    g_edge_w = tmp_g_edge_w
                    g_edge_index = tmp_g_edge_index
                else:
                    g_edge_w = torch.cat([g_edge_w, tmp_g_edge_w], dim=0)
                    g_edge_index = torch.cat([g_edge_index, tmp_g_edge_index], dim=0)
        # g_edge_w: torch.Size([240, 1, 300])  g_edge_index: torch.Size([240, 2])   240代表16个组间的240种关系（同一实体组没有）
        g_edge_w = g_edge_w.transpose(0, 1)  # torch.Size([1, 240, 300])
        g_edge_index = g_edge_index.unsqueeze(dim=0)  # torch.Size([1, 240, 2])
        # g_edge_index = g_edge_index.repeat_interleave(u_em.shape[0], dim=0)
        g_edge_index = g_edge_index.transpose(1, 2)  # torch.Size([1, 2, 240])
        g_x, g_edge_w, g_edge_index = self.batchInput(g_x, g_edge_w, g_edge_index)
        # g_x: torch.Size([16, 100]) g_edge_w: torch.Size([240, 300]) g_edge_index: torch.Size([2, 240])
        for i in range(self.gnn_layer):
            g_x = self.group_gnn[i](g_x, g_edge_index, g_edge_w)  # 对应公式4、5、6

        g_x = g_x.reshape(-1, self.group_num, g_x.shape[-1])
        w2 = w.unsqueeze(dim=0)  # torch.Size([1, 791, 16])
        w2 = w2.repeat_interleave(g_x.size(0), dim=0)
        new_x = torch.bmm(w2, g_x)  # 对应公式7
        # x = x.expand(new_x.shape[0], x.shape[1], x.shape[2])
        new_x = new_x[0: 1, :, :]
        new_x = torch.cat([x, new_x], dim=-1)

        # edge_w = edgte_w.unsqueeze(dim=-1)

        new_x, edge_w, edge_index = self.batchInput(new_x, edge_w, edge_index)
        zeros = torch.zeros(new_x.shape[0], 400 - new_x.shape[1])
        new_x = torch.cat((new_x, zeros), dim=1)
        for i in range(self.gnn_layer):
            ref = nn.Linear(400, 100)
            edge_w_new = ref(new_x)
            batched_g.edata["e_h"] = edge_w
            batched_g.ndata["h"] = edge_w_new
            self.re_aggr1(batched_g, False)
            self.re_aggr2(batched_g, False)  # 对应公式8

        # e_h
        e_h_data = batched_g.edata["e_h"]
        e_h_data_trans = torch.transpose(e_h_data, 0, 1)
        re_size = rel_embeds.size()[0]
        e_h_linear = nn.Linear(e_h_data_trans.size()[1], re_size)
        e_h_data = e_h_linear(e_h_data_trans)
        e_h_data = torch.transpose(e_h_data, 0, 1)

        e_h_seq_tensor = torch.zeros(len(pre_time_len_list), self.seq_len, self.h_dim)
        if torch.cuda.is_available():
            e_h_seq_tensor = e_h_data.cuda()
        for i, all_time_list in enumerate(pre_time_list):
            for j, t in enumerate(all_time_list):
                e_h_seq_tensor[i, j, :] = e_h_data[time_to_idx[t.item()]]

        # h
        h_data = batched_g.ndata["h"]
        h_data_trans = torch.transpose(h_data, 0, 1)
        h_re_size = ent_embeds.size()[0]
        h_linear = nn.Linear(h_data_trans.size()[1], h_re_size)
        h_data = h_linear(h_data_trans)
        h_data = torch.transpose(h_data, 0, 1)

        h_seq_tensor = torch.zeros(len(pre_time_len_list), self.seq_len, self.h_dim)
        if torch.cuda.is_available():
            h_seq_tensor = e_h_data.cuda()
        for i, all_time_list in enumerate(pre_time_list):
            for j, t in enumerate(all_time_list):
                h_seq_tensor[i, j, :] = h_data[time_to_idx[t.item()]]

        return e_h_seq_tensor, pre_time_len_list, h_seq_tensor, e_h_data, h_data


class DecoderModule(nn.Module):
    def __init__(self, x_em, edge_h, gnn_h, gnn_layer, city_num, group_num, device):
        super(DecoderModule, self).__init__()
        self.device = device
        self.city_num = city_num
        self.group_num = group_num
        self.gnn_layer = gnn_layer
        self.x_embed = Lin(gnn_h, x_em)
        self.group_gnn = nn.ModuleList([NodeModel(x_em, edge_h, gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.group_gnn.append(NodeModel(gnn_h, edge_h, gnn_h))
        self.global_gnn = nn.ModuleList([NodeModel(x_em + gnn_h, 100, gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.global_gnn.append(NodeModel(gnn_h, 100, gnn_h))

    def forward(self, x, trans_w, g_edge_index, g_edge_w, edge_index, edge_w):
        x = self.x_embed(x)
        x = x.reshape(-1, self.city_num, x.shape[-1])
        w = Parameter(trans_w, requires_grad=False).to(self.device, non_blocking=True)
        w1 = w.transpose(0, 1)
        w1 = w1.unsqueeze(dim=0)
        w1 = w1.repeat_interleave(x.size(0), dim=0)
        g_x = torch.bmm(w1, x)
        g_x = g_x.reshape(-1, g_x.shape[-1])
        for i in range(self.gnn_layer):
            g_x = self.group_gnn[i](g_x, g_edge_index, g_edge_w)
        g_x = g_x.reshape(-1, self.group_num, g_x.shape[-1])
        w2 = w.unsqueeze(dim=0)
        w2 = w2.repeat_interleave(g_x.size(0), dim=0)
        new_x = torch.bmm(w2, g_x)
        new_x = torch.cat([x, new_x], dim=-1)
        new_x = new_x.reshape(-1, new_x.shape[-1])
        # print(new_x.shape,edge_w.shape,edge_index.shape)
        for i in range(self.gnn_layer):
            new_x = self.global_gnn[i](new_x, edge_index, edge_w)

        return new_x


class NodeModel(torch.nn.Module):

    def node2edge(self, x):
        # receivers = torch.matmul(rel_rec, x)
        # senders = torch.matmul(rel_send, x)
        # edges = torch.cat([senders, receivers], dim=2)
        return x

    def edge2node(self, node_num, x, rel_type):
        mask = rel_type.squeeze()
        x = x + x * (mask.unsqueeze(0))
        # rel = rel_rec.t() + rel_send.t()
        rel = torch.tensor(np.ones(shape=(node_num, x.size()[0])))
        incoming = torch.matmul(rel.to(torch.float32), x)
        return incoming / incoming.size(1)

    def __init__(self, node_h, edge_h, gnn_h, channel_dim=120, time_reduce_size=1):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(node_h + edge_h, gnn_h), ReLU(inplace=True))
        self.node_mlp_2 = Seq(Lin(node_h + gnn_h, gnn_h), ReLU(inplace=True))

        self.conv3 = nn.Conv1d(edge_h, channel_dim * time_reduce_size * 2, kernel_size=1,
                               stride=1)
        self.bn3 = nn.BatchNorm1d(channel_dim * time_reduce_size * 2)

        self.conv4 = nn.Conv1d(channel_dim * time_reduce_size * 2, 1, kernel_size=1, stride=1)

        self.conv5 = nn.Conv1d(channel_dim * time_reduce_size * 2, channel_dim * time_reduce_size * 2, kernel_size=1,
                               stride=1)

    def forward(self, x, edge_index, edge_attr):
        """
        x:实体组的特征表示
        edge_index:实体组间关系特征表示对应的索引
        edge_attr:实体组间关系特征表示
        """
        # x: torch.Size([16, 100]) edge_attr: torch.Size([240, 300]) edge_index: torch.Size([2, 240])
        edge = edge_attr  # torch.Size([240, 300])
        node_num = x.size()[0]
        edge = edge.unsqueeze(1)
        edge = edge.permute(0, 2, 1)
        edge = F.relu(self.conv3(edge))  # relu:返回一个与输入形状相同的张量，所有负值被替换为零，正值保持不变
        x = self.conv4(edge)
        rel_type = F.sigmoid(x)
        s_input_2 = self.edge2node(node_num, edge, rel_type)
        return s_input_2
