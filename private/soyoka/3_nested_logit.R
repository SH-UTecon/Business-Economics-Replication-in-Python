library(tidyverse)
library(fixest)

setwd("Business-Economics-Replication-in-Python")
dir("data")
df <- read_csv("data/data_2.csv")
df <- read_csv("data/data_2_NIPYO.csv") 

# add type information
df_type <- read_csv("data/CleanData_20180222.csv", locale = locale(encoding = "shift-jis")) |> 
select(maker = Maker, type = Type, name = Name, year = Year, price)
df <- df |> left_join(df_type, by = c("year", "price"))

# Nested logit ----
# まず、マーケット・企業レベルにおける、各製品属性の和と自乗和を計算する。
# ここでacross関数は、最初に文字列ベクトルで指定した変数について、後ろにリスト内で定義した操作を適用している。
df <-
  df %>%
  dplyr::group_by(year, maker, type) %>%
  dplyr::mutate(
    dplyr::across( c("hppw", "FuelEfficiency", "size"),
                   list(sum_own = ~ sum(.x, na.rm = TRUE) )   ),
    dplyr::across( c("hppw", "FuelEfficiency", "size"),
                   list(sqr_sum_own = ~ sum(.x^2, na.rm = TRUE) ) ),
    group_n = n()
  ) %>%
  dplyr::ungroup()

# 次に、マーケットレベルでの、各製品属性の和を計算する。
df <- 
  df %>% 
  dplyr::group_by(year, type) %>%
  dplyr::mutate( 
    dplyr::across( c("hppw", "FuelEfficiency", "size"),
                   list( sum_mkt = ~ sum(.x, na.rm = TRUE) )  ),
    dplyr::across( c("hppw", "FuelEfficiency", "size"),
                   list( sqr_sum_mkt = ~ sum(.x^2, na.rm = TRUE) )    ),
    mkt_n = n()
  ) %>%
  dplyr::ungroup() 

# 以上で定義した変数を利用して、まずBLP操作変数を構築する。
df <- 
  df %>% 
  dplyr::mutate(
    iv_BLP_own_hppw_nest = hppw_sum_own - hppw,
    iv_BLP_own_FuelEfficiency_nest = FuelEfficiency_sum_own - FuelEfficiency,
    iv_BLP_own_size_nest = size_sum_own - size,
    iv_BLP_other_hppw_nest = hppw_sum_mkt - hppw_sum_own,
    iv_BLP_other_FuelEfficiency_nest = FuelEfficiency_sum_mkt - FuelEfficiency_sum_own,
    iv_BLP_other_size_nest = size_sum_mkt - size_sum_own, 
    iv_BLP_own_num_nest = group_n - 1, 
    iv_BLP_other_num_nest = mkt_n - group_n) 

# 続いて、Differentiation IVを構築する。
df <- 
  df %>% 
  mutate(
    iv_GH_own_hppw_nest = (group_n - 1) * hppw^2 + (hppw_sqr_sum_own - hppw^2) - 2 * hppw * (hppw_sum_own - hppw),
    iv_GH_own_FuelEfficiency_nest = (group_n - 1) * FuelEfficiency^2 + (FuelEfficiency_sqr_sum_own - FuelEfficiency^2) - 2 * FuelEfficiency * (FuelEfficiency_sum_own - FuelEfficiency),
    iv_GH_own_size_nest = (group_n - 1) * size^2 + (size_sqr_sum_own - size^2) - 2 * size * (size_sum_own - size),
    iv_GH_other_hppw_nest = (mkt_n - group_n) * hppw^2 + (hppw_sqr_sum_mkt - hppw_sqr_sum_own) - 2 * hppw * (hppw_sum_mkt - hppw_sum_own),
    iv_GH_other_FuelEfficiency_nest = (mkt_n - group_n) * FuelEfficiency^2 + (FuelEfficiency_sqr_sum_mkt - FuelEfficiency_sqr_sum_own) - 2 * FuelEfficiency * (FuelEfficiency_sum_mkt - FuelEfficiency_sum_own),
    iv_GH_other_size_nest = (mkt_n - group_n) * size^2 + (size_sqr_sum_mkt - size_sqr_sum_own) - 2 * size * (size_sum_mkt - size_sum_own),
  ) %>%
  dplyr::select(
    -starts_with("sum_own"),
    -starts_with("sum_mkt"),
    -starts_with("sqr_sum_own"),
    -starts_with("sqr_sum_mkt"),
    -mkt_n,
    -group_n
  )

df <-
  df %>%
  dplyr::mutate(logit_share = log(share) - log(share0))


# インサイドシェアを定義する。

df <- 
  df %>% 
  dplyr::group_by(year, type) %>% 
  dplyr::mutate(sum_year_body = sum(Sales)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(inside_share = Sales / sum_year_body, 
    log_inside_share = log(Sales / sum_year_body))

# estimation
# OLS
est_ols_nest <- feols(logit_share ~ price + log_inside_share + hppw + FuelEfficiency + size | 0, data = df)
# BLP (price and insideshare)
est_BLP_nest <- feols(
    logit_share ~ hppw + FuelEfficiency + size | 0 |
      price + log_inside_share ~ iv_BLP_own_hppw_nest + iv_BLP_own_FuelEfficiency_nest + iv_BLP_own_size_nest +
      iv_BLP_other_hppw_nest + iv_BLP_other_FuelEfficiency_nest + iv_BLP_other_size_nest + iv_BLP_own_num_nest +  iv_BLP_other_num_nest,
    data = df
  )
# results
etable(list(est_ols_nest, est_BLP_nest),  
                se = "hetero",
                fitstat = c("r2", "n", "ivf" ) , 
                signifCode = NA, 
                dict = c(price = "自動車価格",
                         hppw = "馬力／重量",
                         FuelEfficiency = "燃費(キロメートル／ 1 リットル)",
                         size = "サイズ",
                         `(Intercept)` = "定数項"),
                digits = 2,
                digits.stats = 2,
                depvar = FALSE)

# Mixed logit ----
# まずデータをソートする。マーケット順かつモデル順
df  |> arrange(year, NameID)
info <- df |> select(year, NameID) 
n <- length(info$year)

data %>% 
    arrange(year, NameID) -> data

# マーケットとモデルの情報をGET
data %>% 
    select(year, NameID) -> Info

N <- length(Info$year)
T <- length(unique(Info$year))

# 平均効用に入ってくる部分。内生性がある価格を含んでいる。
X1 <-
  data %>%
  mutate(cons = 1) %>%
  select(cons, price, FuelEfficiency, hppw, size) %>%
  as.matrix()

# ランダム係数とInteractする部分。
X2 <-
  data %>%
  mutate(cons = 1) %>%
  select(price, cons, size) %>%
  as.matrix()


# 操作変数行列。ここは外生変数と追加的な操作変数を含む
Z <-
  data %>%
  mutate(cons = 1) %>%
  select(
    cons, FuelEfficiency, hppw, size,
    starts_with("iv_GH"),
    -ends_with("nest")
  ) %>%
  as.matrix()

# 市場シェア
ShareVec <-
  data %>%
  select(share) %>%
  as.matrix()

datalist = list()
datalist$X1 <- X1
datalist$X2 <- X2
datalist$Z <- Z
datalist$ShareVec <- ShareVec
datalist$marketindex <- data$year
datalist$logitshare <- data$logit_share

set.seed(111)

Nsim = 500

draw_vec = rnorm(Nsim*ncol(X2))

datalist$draw_vec <- draw_vec

parameter = list()
parameter$Nsim = Nsim
parameter$T = T 
parameter$N = N
parameter$theta2 = c(0.001, 0.001)

marketindex <- datalist$marketindex
uniquemarketindex = sort(unique(marketindex))
temp1 = matrix( rep(uniquemarketindex, N), T, N) %>% t()
temp2 = matrix( rep(marketindex, T), N, T)
tempmat = (temp1 == temp2)*1
datalist$tempmat <- tempmat
