# LayerProbe 截止日前实验汇总

- 状态：`selected_results_semantics_checked`
- 配置：`ica3pp_review_response_8c32g` / `paper`
- Python：`python.exe`
- 物理核 / 正式主 worker / SMT 吞吐 worker：8 / 8 / 16
- 总任务数：60

## 方法阶梯

| 规模 | 方法 | worker | 重复 | 中位秒 | 95% bootstrap CI | 中位峰值 RSS MB |
|---|---|---:|---:|---:|---:|---:|
| 24624k_18p | factorized | 1 | 10 | 64.204 | [63.338, 64.832] | 102.4 |
| 24624k_18p | factorized | 8 | 10 | 10.466 | [10.352, 10.657] | 290.9 |
| 24624k_18p | flat | 1 | 10 | 169.954 | [168.737, 171.908] | 94.1 |
| 24624k_18p | flat_parallel | 8 | 10 | 27.669 | [27.430, 28.093] | 289.1 |
| 24624k_18p | kernel_memo | 1 | 10 | 65.659 | [64.896, 66.236] | 93.5 |
| 24624k_18p | kernel_memo_parallel | 8 | 10 | 10.772 | [10.636, 10.978] | 289.9 |

## 复用效应

| 规模 | flat→kernel_memo | kernel_memo→factorized(1) | 同调度并行 memo→factorized | flat(1)→factorized(P) | 同调度并行 flat→factorized | 语义步调用减少 |
|---|---:|---:|---:|---:|---:|---:|
| 24624k_18p | 2.603× | 1.024× | 1.031× | 16.209× | 2.654× | 32.3% |

## 单机强扩展

| worker | 重复 | 中位秒 | 相对 1 worker 加速 | 并行效率 | worker-slot 秒 |
|---:|---:|---:|---:|---:|---:|

## 呈现族规模

| 呈现数 | 配对数 | kernel_memo→factorized 中位加速 | 范围 | 语义步调用减少 |
|---:|---:|---:|---:|---:|

## 使用边界

这些数据只支持当前离散刹车任务、4 个声明代理和 18 种呈现。
它们不支持真实学习效果、真实受众诊断、多领域泛化、GPU 或多节点扩展性主张。
