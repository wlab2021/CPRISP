import csv

csvFile = open("global_representation.csv", 'w', newline='', encoding='utf-8')
writer = csv.writer(csvFile)
csvRow = []

f = open("F:\project\RBP\\resnet\global_representation.txt", 'r', encoding='utf8')
for line in f:
    csvRow = line.split()
    temp_label = csvRow.pop()  # 得到最后一个元素
    csvRow = ["".join(csvRow), temp_label]  # join合并元素
    print(csvRow)
    writer.writerow(csvRow)
f.close()
csvFile.close()

# # 转成tsv文件
with open('global_representation.csv', encoding='utf-8') as f:
    data = f.read().replace(',', '\t')
    # data = f.read()
with open('global_representation.tsv', 'w', encoding='utf-8') as f:
    f.write(data)
f.close()