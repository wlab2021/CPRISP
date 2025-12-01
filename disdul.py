import pandas as pd
data_type = 'CAPRIN1'
df = pd.read_excel("disdata\\CAPRIN1.xlsx", header=None)

df.drop_duplicates(subset=df.columns[0], inplace=True)


df_filtered = df[df[2] > 0.6]

data_gene = []
with open('Datasets/circRNA-RBP/' + data_type + '/positive') as f:
    for line in f:
        if '>' in line:
            data_gene.append(line.strip()[1:])
with open('Datasets/circRNA-RBP/' + data_type + '/negative') as f:
    for line in f:
        if '>' in line:
            data_gene.append(line.strip()[1:])

if not df_filtered.empty:
    sequences = []
    for index, row in df_filtered.iterrows():
        coord = int(row[0])
        if coord < len(data_gene):
            sequence = data_gene[coord]
            sequences.append(sequence)
        else:
            sequences.append(None)

    df_filtered.loc[:, 3] = sequences

    df_filtered.to_excel("filtered_data_CAPRIN1.xlsx", index=False, header=False)