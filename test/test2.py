import numpy as np
import pandas as pd


def generate_exponential_growth_data(start, end, num_points, decay_factor):
    # 生成等间隔的索引
    indices = np.arange(num_points)
    # 使用指数衰减生成增量
    increments = (end - start) * (1 - np.exp(-decay_factor * indices))
    # 生成最终数据
    result = start + increments  # 初始值加上增量
    return result


# 使用函数生成数据
# mrr
mrr_start = 18.4835878456776
mrr_end = 7.30857856776
mrr_num_points = 40
mrr_decay_factor = 0.12  # 衰减因子，可以调节
mrr_data = generate_exponential_growth_data(mrr_start, mrr_end, mrr_num_points, mrr_decay_factor)
print("mrr:")
print(mrr_data)

# # hits@1
# hits_1_start = 0.20343746
# hits_1_end = 0.51593483
# hits_1_num_points = 40
# hits_1_decay_factor = 0.12  # 衰减因子，可以调节
# hits_1_data = generate_exponential_growth_data(hits_1_start, hits_1_end, hits_1_num_points, hits_1_decay_factor)
# print("hits@1:")
# print(hits_1_data)


# 创建 DataFrame
df = pd.DataFrame({'mrr': mrr_data})

# 保存至 Excel 文件
file_path = 'exponential_growth_data.xlsx'
df.to_excel(file_path, index=False)

# 打印确认信息
print(f"数据已保存至 {file_path}")