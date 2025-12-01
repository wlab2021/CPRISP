import pandas as pd
data_type = 'CAPRIN1'
# 读取Excel文件
df = pd.read_excel("disdata\\CAPRIN1.xlsx", header=None)

# 删除第一列有重复值的行，保留第一次出现的行
df.drop_duplicates(subset=df.columns[0], inplace=True)


# 筛选出第三列值大于阈值的行
df_filtered = df[df[2] > 0.6]

'''读取序列'''
data_gene = []
with open('Datasets/circRNA-RBP/' + data_type + '/positive') as f:
    for line in f:
        if '>' in line:
            data_gene.append(line.strip()[1:])
with open('Datasets/circRNA-RBP/' + data_type + '/negative') as f:
    for line in f:
        if '>' in line:
            data_gene.append(line.strip()[1:])


# 确保df_filtered不是空的
if not df_filtered.empty:
    # 根据df_filtered第一列的坐标，读取data_seq中对应的序列片段
    sequences = []
    for index, row in df_filtered.iterrows():
        coord = int(row[0])  # 获取坐标
        if coord < len(data_gene):
            sequence = data_gene[coord]  # 根据坐标读取序列
            sequences.append(sequence)
        else:
            sequences.append(None)  # 如果坐标超出范围，添加None

    # 将序列片段作为新列添加到df_filtered中
    df_filtered.loc[:, 3] = sequences

    # 保存修改后的df_filtered到新的Excel文件
    df_filtered.to_excel("filtered_data_CAPRIN1.xlsx", index=False, header=False)