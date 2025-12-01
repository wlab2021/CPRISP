import csv

csvFile = open("global_representation.csv", 'w', newline='', encoding='utf-8')
writer = csv.writer(csvFile)
csvRow = []

f = open("D:\project\RBP\\resnet\global_representation.txt", 'r', encoding='utf8')
for line in f:
    csvRow = line.split()
    writer.writerow(csvRow)
f.close()
csvFile.close()

# 由于在txt中也包含了空格，所以得到的csv文件是根据空格划分的。
# 我的数据集的格式分成了两部分: （一个中文短句：string, 一个标签：int）,但是短句中包括了空格，所以这样变换会分成三个或者更多个部分。

# 所以改进代码，将list中的前[:-1]个合并成一个部分。
# 改进后代码如下：

