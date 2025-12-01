library(readxl)
library(ggplot2)
# library(showtext)

# font_add("TNR", regular = "C:/Windows/WinSxS/amd64_microsoft-windows-f..etype-timesnewroman_31bf3856ad364e35_10.0.26100.1_none_da632f5801e3ddfb/times.ttf")
# showtext_auto()

windowsFonts(TNR = windowsFont("Times New Roman"))

# 读取数据
data <- read_excel("results/DM1-male.xlsx")

# 设置因子顺序
data$Name = factor(data$Name,
                   levels = c("CSCRSites", "CRIP", "iCircRBP-DHN", "CRBPDL",
                              "HCRNet", "CircSSNN", "CRSP"))

# 设置颜色
Color = c("#9F79EE", "#FFCC66", "#1ABCC2", "#CA6FAC", "#6F94CD",
          "#8FBC8F", "#FF2400", "purple", "red")

# 创建棒棒糖图
p <- ggplot(data, aes(x=Value, y=Name)) +
  geom_segment(aes(x=0, xend=Value, yend=Name, color=Name), linewidth=1) +
  geom_point(aes(color=Name), size=8) +
  geom_point(aes(color=Name), size=10, shape=21, fill=NA) +
  geom_text(aes(label=Value), hjust=0.5, vjust=2.7, fontface = "bold") +
  scale_color_manual(values=Color) +
  coord_cartesian(xlim = c(0, 1)) +  # 限制可视范围为0~1，避免截断图元
  labs(x="AUC", y="") +
  ggtitle("DM1-male") +
  theme_classic() +
  theme(
    text = element_text(family = "TNR"),
    plot.title = element_text(face = "bold", hjust = 0.4, size = 16),
    legend.position = "none",
    panel.border = element_rect(color="black", fill=NA, linewidth=1),
    axis.line = element_line(color="black"),
    panel.grid.major = element_line(color="grey", linewidth=0.2),
    panel.grid.minor = element_line(color="grey", linewidth=0.1),
    axis.title = element_text(face="bold", size=14),
    axis.title.x = element_text(margin = margin(t = 15)),  # x轴标题下移
    axis.title.y = element_text(margin = margin(r = 20)),  # y轴标题左移
    axis.text = element_text(face="bold", size=12)
  )

# 显示图
print(p)

# 保存为SVG文件
ggsave("res/DM1-male.svg", plot = p, width=8, height=5.4)
