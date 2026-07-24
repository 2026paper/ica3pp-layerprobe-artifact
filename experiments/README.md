# LayerProbe 实验入口

所有命令均从仓库根目录执行，路径保持相对：

```powershell
$env:PYTHONPATH = "$PWD\src"
$pythonExe = (Get-Command python).Source
```

不要覆盖已经完成的证据目录。复现时使用新的 `results/_repro_*` 目录。

## 状态总览

| 项目 | 状态 | 证据或计划路径 |
|---|---|---|
| 冻结 distancefix 主结果 | 已完成 | `results/deadline_paper_distancefix_20260723_xeon/` |
| 独立轨迹 oracle | 已完成 | `results/independent_trace_oracle_full_24624_distancefix_20260723_xeon/` |
| 完整键组成项消融 | 已完成 | `results/cache_key_ablation_full_24624_distancefix_20260723_xeon/` |
| 二维 grid 语义迁移域 | 已完成 | `results/grid_transfer_audit_full_20260724/` |
| 固定种子 60-mutant 审计 | 已完成 | `results/randomized_mutation_audit_seed20260724_128/` |
| `n=10` method ladder（含 Flat-P8） | 尚待 clean 隔离完成 | `results/review_method_ladder_n10_clean_20260724/` |
| `n=10` 1--16 worker scaling | 尚待 clean 隔离完成 | `results/parallel_scaling_n10_clean_20260724/` |
| `n=10` deliberative cost-regime | 尚待 clean 隔离完成 | `results/deliberative_policy_n10_clean_20260724/` |
| `n=10` scheduler sensitivity | 尚待 clean 隔离完成 | `results/scheduler_sensitivity_n10_clean_20260724/` |
| 单进程 coarse cProfile | 尚待 clean 完成 | `results/cost_profile_full_24624_n10_clean_20260724/` |

最近一次已经完成的测试结果是 **39 passed**；全部修改结束后仍需最终重跑。smoke 只验证实现与门禁，不提供论文性能数字。

## 已完成的 distancefix 冻结证据

修复 distance sentinel 后，论文和 artifact 应读取以下真实目录：

- `results/deadline_paper_distancefix_20260723_xeon/`
- `results/communication_full_24624_distancefix_provenance_v2_20260723_xeon/`
- `results/independent_trace_oracle_full_24624_distancefix_20260723_xeon/`
- `results/cache_key_ablation_full_24624_distancefix_20260723_xeon/`
- `results/agent_sensitivity_full_24624_distancefix_provenance_v2_20260723_xeon/`
- `results/range_extension_stress_distancefix_provenance_v2_20260723_xeon/`

这些结果对应 `frozen_source_snapshots/distancefix_pre_flatp8/`，而不是当前已增加 Flat-P8 和审稿增强实验的源代码。冻结核验入口是：

```powershell
.\scripts\02_verify_frozen_outputs.ps1
```

已有 `results/verify_enhanced_frozen_source_20260724/` 给出冻结源下的验证报告。缺少 `distancefix` 的旧目录以及当前源下的 diagnostic mismatch 均不得替代上述证据。

## 二维 grid 语义迁移域

`grid_transfer_audit.py` 使用与刹车域不同的二维有限状态结构：`x`、`y`、朝向、逻辑步，四类动作、障碍布局和 18 个 observation-only presentations。完整说明见 `GRID_TRANSFER_AUDIT_README.md`。

已完成结果覆盖 1,296 个声明机制，其中 1,270 个在有限 horizon 内可达；完整键 LayerProbe 与 Flat、独立 oracle 在 canonical/reverse 顺序下均为零轨迹差异，删除 state、memory 或 observation 的弱键在两种顺序下均产生端到端见证。该实验是第二个有限状态结构迁移案例，不是现实系统或跨平台性能验证。

复现时使用新目录：

```powershell
& $pythonExe experiments\grid_transfer_audit.py `
  --output results\_repro_grid_transfer_full `
  --workers 8 `
  --chunk-size 32
```

## 固定种子随机 mutation 审计

`randomized_mutation_audit.py` 使用固定种子 `20260724`，从冻结刹车域中分层抽取 128 个有效机制，并生成三个 family、每类 20 个、共 60 个 mutants：

- agent policy；
- boundary offset；
- cache-key projection。

已完成审计的完整轨迹检测为 56/60，下游六比特签名检测为 49/60。其余 4 个 mutant 在该固定样本上 `affected_kernels=0`，因此只能表述为未激活或行为等价，不能表述为 checker 漏检，也不能外推生产缺陷概率。

```powershell
& $pythonExe experiments\randomized_mutation_audit.py `
  --output results\_repro_randomized_mutation_seed20260724_128 `
  --seed 20260724 `
  --per-family 20 `
  --sample-kernels 128 `
  --workers 8 `
  --no-resume
```

## 正式单机计时的强制隔离规则

所有可引用 wall-clock 运行必须通过 `scripts/run_isolated_formal.ps1`。完整协议见 `scripts/FORMAL_ISOLATION.md`。wrapper 要求启动前连续 30 秒没有 Python-family 进程，并在运行期间监控 runner 进程树；发现外部 Python 后只终止自己记录的 PID。

只有包含有效 `FORMAL_PROVENANCE.json`、匹配结果树 hash，且其隔离记录为退出码 0 的 `CLEAN_RUN.json` 的目录才是正式计时证据。`CONTAMINATED.json`、`REFUSED.json`、`PROGRAM_FAILED.json`、`DO_NOT_USE_*.json`、smoke 和部分目录都不能引用。

正式计时必须串行执行；期间不要启动测试、绘图、其他实验或任何 Python 子任务。

### `n=10` method ladder

配置 `deadline_profile_review_8c32g.json` 在 24,624 个机制和 18 个 presentations 上执行 10 次配对重复。梯子包含：

- Flat-1 与 Flat-P8；
- KernelMemo-1 与 KernelMemo-P8；
- LayerProbe-1 与 LayerProbe-P8。

Flat-P8 使用与其他 P8 方法匹配的 8-worker 调度，用于拆分“复用”和“并行”的贡献。计划 clean 路径当前尚未完成；受干扰的 `results/review_method_ladder_n10_20260724/` 必须排除。

```powershell
.\scripts\run_isolated_formal.ps1 `
  -PythonExe $pythonExe `
  -ScriptPath experiments\deadline_runner.py `
  -ArgumentList @(
    "--config", "experiments\deadline_profile_review_8c32g.json",
    "--mode", "paper",
    "--only", "method_ladder",
    "--output", "results\review_method_ladder_n10_clean_20260724",
    "--ignore-freeze"
  ) `
  -OutputLabel method_ladder_n10 `
  -ExpectedOutputDir results\review_method_ladder_n10_clean_20260724 `
  -InputFiles experiments\deadline_profile_review_8c32g.json
```

### `n=10` deliberative cost-regime

`deliberative_policy_benchmark.py` 比较匹配调度的 KernelMemo-P8 与 LayerProbe-P8。默认固定设计从 24,624 个机制中等距抽取 512 个，使用全部 18 个 presentations、8 workers、深度 `{0,2,4,6}`、10 次配对重复和 10,000 次 bootstrap。正深度执行有限枚举式 lookahead；它不是 `sleep` 或 busy loop，也不是新现实域。

每个配对必须同时通过完整 candidate digest 和四代理 observation/action/status trace digest 门禁。

```powershell
.\scripts\run_isolated_formal.ps1 `
  -PythonExe $pythonExe `
  -ScriptPath experiments\deliberative_policy_benchmark.py `
  -ArgumentList @(
    "--output", "results\deliberative_policy_n10_clean_20260724",
    "--workers", "8",
    "--repeats", "10",
    "--depths", "0", "2", "4", "6",
    "--population-kernels", "24624",
    "--sample-size", "512",
    "--presentations", "18",
    "--bootstrap-samples", "10000"
  ) `
  -OutputLabel deliberative_policy_n10 `
  -ExpectedOutputDir results\deliberative_policy_n10_clean_20260724
```

### `n=10` scheduler sensitivity

`scheduler_sensitivity.py` 固定语义工作量、8 workers、cache scope 和按 mechanism ID 的确定性归约，只改变本机任务分配：

- `current_chunksize`：生产中心任务队列及 `max(1, kernels // (workers * 4))`；
- `fine_chunksize_1`：中心队列，每个提交 chunk 含一个 mechanism group；
- `static_contiguous`：8 个独立 spawned 进程各执行一个不可分割的固定连续分块。

这不是 work stealing。代码没有设置 CPU affinity，因此也不得写成 core-pinned。每次重复在落盘前检查三种调度的完整 candidate-signature digest。

```powershell
.\scripts\run_isolated_formal.ps1 `
  -PythonExe $pythonExe `
  -ScriptPath experiments\scheduler_sensitivity.py `
  -ArgumentList @(
    "--output", "results\scheduler_sensitivity_n10_clean_20260724",
    "--workers", "8",
    "--repeats", "10",
    "--kernels", "24624",
    "--presentations", "18"
  ) `
  -OutputLabel scheduler_sensitivity_n10 `
  -ExpectedOutputDir results\scheduler_sensitivity_n10_clean_20260724
```

### 单进程 coarse cProfile

`cost_profile.py` 对生产 `run_factorized(..., workers=1)` 路径做一次函数级 cProfile，默认覆盖 24,624 个机制和 18 个 presentations。它报告 additive self time 和重叠的 cumulative time；`evaluator/cache-loop residual` 同时包含容器操作、生成的 hash/equality、解释器工作和 profiler overhead。

因此该实验只能支持 coarse attribution。绝对 profile 时间不是 speedup，residual 也不是 lookup、hash、serialization 或 allocation 的精确分解。

```powershell
.\scripts\run_isolated_formal.ps1 `
  -PythonExe $pythonExe `
  -ScriptPath experiments\cost_profile.py `
  -ArgumentList @(
    "--output", "results\cost_profile_full_24624_clean_20260724",
    "--kernels", "24624",
    "--presentations", "18"
  ) `
  -OutputLabel cost_profile_full_24624 `
  -ExpectedOutputDir results\cost_profile_full_24624_clean_20260724
```

## 测试与非计时任务

```powershell
& $pythonExe -m pytest -p no:cacheprovider
```

最近一次已完成运行是 39 passed；最终 artifact 封包前必须再执行一次。correctness 任务可以为了吞吐并行，但其 `elapsed` 字段不得进入性能表；正式计时任务则必须独占 Python 工作负载。

## 论文主张边界

- 本轮接受这台电脑能够完成的工作，不要求多机或跨体系结构实验；
- 二维 grid 支持有限状态结构迁移，不支持真实系统普适性；
- distancefix 的区间外压力测试仍属于刹车任务族，不是第三个域；
- 独立 oracle 和 mutation audit 是差分证据，不是形式化证明；
- scheduler 实验只比较单机中心队列与固定连续分块，不含 work stealing 或 core pinning；
- cProfile 是诊断性 coarse attribution，不是独立 wall-clock 性能结论。
