BiocManager::install("readxl")
#模型扫描后去重

library(biomaRt)
library(readxl)
library(dplyr)


# 读取数据


# tsv_data <- read.table("D:\\data\\datasets\\clip_data\\TARDBP_HEK293.tsv", header = TRUE, sep = "\t")
# df1 <- read_excel("filtered_data_METTL14_Hela.xlsx")
# df2 <- read_excel("filtered_data_HNRNPC_HepG2.xlsx")

df1 <- read_excel("filtered_data_CAPRIN1.xlsx")
df2 <- read_excel("filtered_data_CAPRIN1.xlsx")


# 首先，确保两个数据框的列名相同
df1 <- df1 %>% select(geneName)
df2 <- df2 %>% select(geneName)




gene1 <- df1$geneName
gene2 <- df2$geneName


mart <- useMart("ensembl","hsapiens_gene_ensembl")  ##人类选择 hsapiens_gene_ensembl
gene_name1<-getBM(attributes=c("ensembl_transcript_id","external_gene_name","ensembl_gene_id"),filters = "ensembl_transcript_id",values = gene1, mart = mart)
gene_name2<-getBM(attributes=c("ensembl_transcript_id","external_gene_name","ensembl_gene_id"),filters = "ensembl_transcript_id",values = gene2, mart = mart)


gene_name_unique1 <- unique(gene_name1[ , "external_gene_name"])
gene_name_unique2 <- unique(gene_name2[ , "external_gene_name"])


# 使用 set operations 来找到两个向量中的共同元素
# common_data <- intersect(gene_name_unique1, gene_name_unique2)

# common_data <- intersect(intersect(gene_name_unique1, gene_name_unique2), gene_name_unique3)



# writeLines(common_data, con = "id_tras/HNRNPC_common_gene.txt")  # 多个细胞系

writeLines(gene_name_unique1, con = "./id_tras/CAPRIN1_common_gene.txt")  # 仅一个细胞系
