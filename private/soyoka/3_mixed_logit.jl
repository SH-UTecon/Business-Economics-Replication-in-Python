using Parameters, CSV, DataFrames, Random, LinearAlgebra, Statistics, Optim

pwd()
cd("Business-Economics-Replication-in-Python")
readdir()
# CSVファイルを読み込む
data = CSV.read("data/data_2.csv", DataFrame)
data = sort(data, [:year, :NameID])
# 読み込んだデータを表示
println(data)
# cons 列を追加する
data.cons .= 1
names(data)

Random.seed!(111)
# 必要な行列を定義する
# 平均効用
data_x1 = select(data, [:cons, :price, :FuelEfficiency, :hppw, :size])
matrix_x1 = Matrix(data_x1)
# ランダム係数
data_x2 = select(data, [:cons, :price, :size])
matrix_x2 = Matrix(data_x2)
# 操作変数
data_iv = select(data, [:cons, 
                        :FuelEfficiency,
                        :size, 
                        :hppw, 
                        :iv_GH_own_hppw, 
                        :iv_GH_own_FuelEfficiency, 
                        :iv_GH_own_size, 
                        :iv_GH_other_hppw, 
                        :iv_GH_other_FuelEfficiency, 
                        :iv_GH_other_size])
matrix_iv = Matrix(data_iv)
# market index
# create temporary matrix: market indexを作りたいのでは？
vec_market_index = Vector(data.year)
tmp1 = reshape(repeat(vec_market_index, T), N, T)
vec_unique_market_index = unique(data.year)
tmp2 = transpose(reshape(repeat(vec_unique_market_index, N), T, N))
matrix_tmp = (tmp1 .== tmp2).*1
# share
vec_share = Vector(data.share)
# parameters
N = size(data, 1)
T = size(unique(data.year), 1)
@with_kw struct Parameter
    Nsim::Int = 500
    T::Int = T
    N::Int = N
    theta2::Vector{Float64}
end

# 市場シャアを計算する関数
"""
  f_mktshare
Calculate market market share using mean utility
# Arguments
- parameter::Parameter: parameters of x2
- delta::Vector{Float64}: vector of x1
# Return
- s_jt_i::Vector{Float64}: predicted market share
"""
function f_mktshare(parameter::Parameter, delta::Vector{Float64})
  @unpack Nsim, T, N, theta2 = parameter

  N = size(data, 1)
  T = size(unique(data.year), 1)

  vec_draw = randn(Nsim*size(matrix_x2, 2))
  # market_index = vec_draw[1:Nsim*length(theta2)] # 使ってない？
  # matrix_x2かける必要なくない？
  mu = matrix_x2 * diagm(theta2) * reshape(vec_draw, length(theta2), Nsim)
  # mu = matrix_x2 * reshape(vec_draw, length(theta2), Nsim) # なぜこれではだめなのか
  denom_outside = exp.(zeros(Int, T, Nsim))
  
  # delta = zeros(Int, N, 1) # ここで定義しなおしてはダメなのでは？
  delta_mu = delta * ones(Int, 1, Nsim) + mu
  exp_delta_mu = exp.(delta_mu)
  
  denom_tmp = transpose(transpose(exp_delta_mu) * matrix_tmp) + denom_outside# simulationごとに各delta_muを足し合わせて denom_outsideを足している? > marketってyear?
  denom = matrix_tmp * denom_tmp
  
  s_jt_i = exp_delta_mu ./ denom
  s_jt_i = mean(s_jt_i, dims = 2)

  return vec(s_jt_i)
end

# 平均効用のinversionの関数
"""
  f_contraction
# Argument
- parameter::Parameter: parameters including N of simulation, N of obs, N of time, theta2
- delta_ini::Vector{Float64}: vector of initial guess of mean utility
# Return
- delta_old::Vector{Float64}: predicted mean utility which is fitted good to the data (share)
"""
function f_contraction(parameter::Parameter, delta_ini::Vector{Float64})
  tol = 1e-11
  share_obs = vec_share
  norm = 1e+10
  max_itr = 1000
  # 初期値を一番初めのoldにセット
  delta_old = delta_ini
  exp_delta_old = exp.(delta_old)

  itr = 0 
  while (norm > tol && itr < max_itr)
    pred_mkt_share = f_mktshare(parameter, delta_old)
    # updateの式がテキストと異なるが、expをとっているため
    exp_delta = exp_delta_old .* share_obs ./ pred_mkt_share
    # tolを超えているかチェック>textでは抜けてる？
    norm = maximum(abs.(exp_delta - exp_delta_old))
    # Update
    # name
    exp_delta_old = exp_delta
    delta_old = log.(exp_delta)
    # N of itr
    itr += 1
  end 

  return delta_old
end 

# 二段階目の推定を行う関数
""" 
  f_GMMobj
# Argument
theta2::Vector{Float64}: vector of parameters of x2
option::{1, 0}: specify the output of optimization problem
# Return
output::{Dict}or{Float64}: value of obj func
"""
function f_GMMobj(theta2::Vector{Float64}, option)
  parameter = Parameter(theta2 = theta2)
  @unpack Nsim, T, N, theta2 = parameter
  # updateする平均効用logitを推定していないので0から始める
  delta_ini_global = zeros(Float64, N)
  delta_ini = delta_ini_global
  delta = f_contraction(parameter, delta_ini)
  delta_ini_global = delta
  X1 = matrix_x1
  Z = matrix_iv
  
  wht_mat = "2SLS"
  if wht_mat == "2SLS"
    W = inv(transpose(Z) * Z)
  elseif wht_mat == "Ident"
    W = Matrix{Float16}(I, size(Z, 2), size(Z, 2))
  else 
    error("Invarid option for 'wht_mat'")
  end
  
  beta_hat = inv(transpose(X1)*Z*W*transpose(Z)*X1)*transpose(X1)*Z*W*transpose(Z)*delta
  
  Xi = delta - X1 * beta_hat
  
  val_obj = transpose(Xi)*Z*W*transpose(Z)*Xi

  if option == 1
    output = Dict("val_obj" => val_obj, "beta_hat" => beta_hat, "delta" => delta)
  elseif option == 0
    output = val_obj
  else 
    error("Invalid option for 'option'")
  end
return output
end

# optimization
result = optimize(theta2 -> f_GMMobj(theta2, 0), 
                  [0.3, 0.18, 0.01], 
                  LBFGS(), 
                  Optim.Options(iterations=1000))

result
