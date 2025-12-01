
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("multienrichjam")

BiocManager::install("multienrichjam")
install.packages("C:/Users/L/Downloads/org.Hs.eg.db_3.20.0.tar.gz", repos = NULL, type = "source")

# 安装 remotes 
install.packages("remotes")

# 使用 remotes 安装 multienrichjam
remotes::install_github("jmw86069/multienrichjam", dependencies=TRUE)

library(DOSE)
library(org.Hs.eg.db)
library(topGO)
library(clusterProfiler)
library(pathview)
library(ggplot2)
library(GOplot)
library(stringr)
library(dplyr)
library(ggpubr)

library(ggsci)



keytypes(org.Hs.eg.db) 
# 基因
x<-read.table("./id_tras/CAPRIN1_common_gene.txt",header=F,na.strings = c("NA"))

x1 <- as.character(x[, 1])


# 基因 ID 转换
gene.df = bitr(x1,  # 数据集
               fromType="SYMBOL",  # 输入为 symbol 格式
               toType="ENTREZID",  # 转为 ENTREZID 格式
               OrgDb="org.Hs.eg.db")  # 人类 数据集
gene <- gene.df$ENTREZID


# GO
ego_ALL <- enrichGO(gene = gene,
                   OrgDb=org.Hs.eg.db,
                   keyType = "ENTREZID",
                   ont = "ALL",
                   pAdjustMethod = "BH",
                   pvalueCutoff = 0.05,
                   qvalueCutoff = 0.05,
                   readable = TRUE)


ek_ALL_df <- ego_ALL@result


ego_result_all <- as.data.frame(ego_ALL)

# 展示画图
barplot(ego_ALL, drop = TRUE, showCategory =10,split="ONTOLOGY",label_format=100) + facet_grid(ONTOLOGY~., scale='free')

p + theme(axis.text.y = element_text(family = "Times New Roman", face = "bold",size = 15))



# KEGG
data(geneList, package = "DOSE")
g_list <- names(geneList)[1:100]
head(g_list)

ek <- enrichKEGG(gene = gene,  # 需要分析的基因的EntrezID
                 organism = "hsa",  # 人类
                 pvalueCutoff =0.05,  # 设置 pvalue 界值
                 qvalueCutoff =0.05)  # 设置 qvalue 界值 (FDR校正后的p值)

eKEGG <- as.data.frame(ek)

barplot(ek, x = "Count", showCategory = 10)  # 绘制图形




# 将 id 转化为基因名
geneIds_list <- strsplit(eKEGG$geneID[c(5)], "/")

flattened_geneIds <- unlist(geneIds_list)


geneNames <- mapIds(org.Hs.eg.db, keys = flattened_geneIds, column = "SYMBOL", keytype = "ENTREZID")
print(geneNames)
writeLines(geneNames, con = "./id_tras/dia_tdp43.txt")













