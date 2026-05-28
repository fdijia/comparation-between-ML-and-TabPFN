library(lavaan)
library(readr)
library(dplyr)
library(glue)

# # 读取数据
# data <- read_csv("特征整理与数据构建/output/D1_data.csv")
# opposite_cols <- c("第1题", "第2题", "第3题", "第4题", "第5题", "第6题", "第8题", "第21题", "第32题", "第36题")
# data <- data %>%
#   mutate(across(all_of(opposite_cols), ~ 6 - .x))
# rename_dict <- c(
#   setNames(
#     paste0("第", c(1:10, 12:28, 30:36), "题"),
#     paste0("x", c(1:10, 12:28, 30:36))
#   ),
#   "anxiety" = "问题-焦虑",
#   "depression" = "问题-抑郁",
#   "risk" = "危机-自己",
#   "gender" = "性别"
# )
# data <- data %>%
#   rename(!!!rename_dict)

data <- read_csv("可解释性分析/sem/data.csv")
opposite_cols <- c("x1", "x2", "x3", "x4", "x5", "x6", "x8", "x21", "x32", "x36")
data <- data %>%
  mutate(across(all_of(opposite_cols), ~ 6 - .x))
# =========================
# SEM模型
# =========================
desc_anxiety = '
internal =~ x13 + x34 + x31
internet =~ x26 + x35 + x17
relation =~ x7 + x20 + x28

relation ~ ri*internet
internal ~ ir*relation
anxiety ~ ai*internal

med_ii := ri*ir
med_i := med_ii*ai
med_r := ir*ai
'
desc_depression = '
internal =~ x34 + x2 + x31 + x14
relation =~ x10 + x7 + x19 + x28 + x20

internal ~ ir*relation
depression ~ di*internal + dr*relation

med_i := ir*di
total_i := dr + med_i
'
desc_selfharm = '
internal =~ x13 + x34 + x31 + x10
family =~ x4 + x36

internal ~ fi*family
risk ~ ri*internal + rf*family

med := ri*fi
total := med + rf
'
# =========================
# 拟合模型 + Bootstrap
# =========================
pre_name <- 'selfharm'
fit <- sem(
  model = desc_selfharm,
  data = data,
  se = "bootstrap",
  bootstrap = 10000
)

print("sem finished")
# =========================
# 参数估计
# =========================
estimates <- parameterEstimates(
  fit,
  standardized = TRUE,
  ci = TRUE
)

# =========================
# 模型拟合指标
# =========================
fit_stats <- fitMeasures(
  fit,
  c(
    "chisq",
    "df",
    "pvalue",
    "cfi",
    "tli",
    "rmsea",
    "srmr",
    "aic",
    "bic"
  )
)

print(fit_stats)

# =========================
# 所有路径
# op == "~"
# =========================
estimates <- parameterEstimates(fit, standardized = TRUE)
paths <- estimates %>%
  filter(op == "~") %>%
  select(
    lhs,
    rhs,
    est,
    est.std= std.all,
    se,
    z,
    pvalue,
    ci.lower,
    ci.upper
  )


# =========================
# 显著路径
#
# 判断标准:
# p < 0.05
# =========================
sig_paths <- paths %>%
  filter(pvalue < 0.05)

print(sig_paths)


# =========================
# 中介效应
# op == ":="
# =========================
mediation_results <- estimates %>%
  filter(op == ":=") %>%
  select(
    lhs,
    est,
    est.std= std.all,
    se,
    z,
    pvalue,
    ci.lower,
    ci.upper
  )

print(mediation_results)



# =========================
# 显著中介效应
#
# 判断标准:
# bootstrap CI 不包含0
#
# 即:
# ci.lower * ci.upper > 0
# =========================
sig_mediation <- mediation_results %>%
  filter(ci.lower * ci.upper > 0)

print(sig_mediation)


# =========================
# 保存全部结果
# =========================
write.csv(
  estimates,
  glue("可解释性分析/sem/{pre_name}_estimates_all.csv"),
  row.names = FALSE
)

write.csv(
  paths,
  glue("可解释性分析/sem/{pre_name}_paths.csv"),
  row.names = FALSE
)

write.csv(
  sig_paths,
  glue("可解释性分析/sem/{pre_name}_sig_paths.csv"),
  row.names = FALSE
)

write.csv(
  mediation_results,
  glue("可解释性分析/sem/{pre_name}_mediation.csv"),
  row.names = FALSE
)

write.csv(
  sig_mediation,
  glue("可解释性分析/sem/{pre_name}_sig_mediation.csv"),
  row.names = FALSE
)

fit_stats_df <- data.frame(as.list(fit_stats))

write.csv(
  fit_stats_df,
  glue("可解释性分析/sem/{pre_name}_fit_stats.csv"),
  row.names = FALSE
)

cat("\n=========================\n")
cat("显著路径数量:", nrow(sig_paths), "\n")
cat("显著中介效应数量:", nrow(sig_mediation), "\n")
cat("=========================\n")

# 10000 显著路径10, 其中4+3+3, 按理说应该4+3+6, 我需要焦虑也在里面