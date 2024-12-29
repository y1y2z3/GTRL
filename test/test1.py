import numpy as np
import pandas as pd


class Test1:
    def __init__(self, name, age):
        self.name = name
        self.age = age


if __name__ == '__main__':
    print("---------------test1----------------")
    # stu = Test1("yyz", 23)
    # print(stu.age)
    # list = []
    # list.append(1)
    # list.append("jjd")
    # print(list)
    # time_l = list(set(data[:, -1]))
    # list = [[1, 2, 3, 0], [2, 3, 5, 1]]
    # npList = np.asarray(list)
    # print(npList[-1, -1])
    # value = 10290
    # print('-' * 80 + '\nExiting from training early, epoch', value)
    # hits = {}
    # hits[1] = 0.1
    # hits[3] = 0.4
    # hits[10] = 1
    # for hit in hits.items():
    #     print(hit[0], hit[1])
    # def func1(hits, loss):
    #     hits[1] = 78
    #     loss = 89
    # loss = 90
    # func1(hits, loss)
    # print(hits[1])
    # print(loss)

    def compute_res(loss, hits, mrr, dataset, epoch):
        file_path = "../models/testModel.xlsx"
        dataframe = pd.read_excel(file_path, sheet_name=dataset)
        if epoch is None:
            epoch = len(dataframe) - 1
        loss = dataframe.loc[epoch, "loss"]
        hits[1] = dataframe.loc[epoch, "hits1"]
        hits[3] = dataframe.loc[epoch, "hits3"]
        hits[10] = dataframe.loc[epoch, "hits10"]
        mrr = dataframe.loc[epoch, "mrr"]
        return loss, hits, mrr

    loss, hits, mrr = compute_res(1, {}, 0, "example", 34)
    print("loss: {:.4f}".format(loss))
    print("MRR: {:.4f}".format(mrr))
    for hit in hits.items():
        print("Hits @ {}: {:.4f}".format(hit[0], hit[1]))
