# FWI 反演程序说明

本次新增 `fwi_optimization.py`，并把多炮目标函数和梯度求和放入
`fwi_gradient.py` 的 `fwi_objective_gradient`。主程序为 `main_fwi.ipynb`。
新增/修正入口均用 `FIX` 或 `FIX/NEW` 注释标识。

## 运行方法

在项目目录用包含 NumPy、Numba、Matplotlib、SciPy、Jupyter 的 Python 环境，
重启 notebook 内核并从第一单元顺序运行到最后一个单元。
不要复用旧版本的 `d_obs`、`sigma` 或已经导入的旧函数。

当前主程序保留 101×101 网格、9 炮、10 Hz、1 s 数据长度；
最后一单元执行 5 次梯度下降更新。可通过 `max_iterations` 增加迭代次数。
为使正演、梯度和步长搜索对应同一个目标函数，程序会重新生成观测数据。

输出保存在 `fwi_results/`：

- `objective_history.png`：真实目标函数及归一化下降曲线。
- `velocity_models.png`：真实、初始、反演速度模型，共用色标。
- `inversion_result.npz`：速度模型、各次接受的模型、目标函数、步长和停止原因。

每次重跑示例会更新该目录内同名的结果文件。

## 目标函数与梯度

目标函数定义为 `J(v) = dt/2 * sum_shots,time,receivers((d_syn-d_obs)**2)`。
它使用所有炮、所有时间样本和所有接收道，不按炮数取平均。
梯度返回数组是该离散目标函数关于每个网格速度的普通偏导数。

`fwi_objective_gradient(v, d_obs, setup)` 返回：

```python
objective, gradient_m, gradient_v, d_syn
```

其中 `m=1/v**2`，因此 `gradient_m = -v**3/2 * gradient_v`。
优化直接使用 `gradient_v`。传入 `compute_gradient=False` 时仅正演和计算目标函数，
两个梯度返回 `None`，无需分配全时刻波场。

`AcousticSetup` 统一保存采集坐标、网格、时间步、差分系数和固定的吸收系数。
每炮在函数内部依次正演、反传并累加梯度，内存只需保存一炮的敏感度历史。
同一套配置贯穿数据生成、初始梯度、每次迭代和全部试探步。

### 为什么增加离散伴随

原 `fwi_gradient(wavefield_tt, lambda_wavefield, velocity, dt)` 保留兼容性，
用于物理域波场互相关的连续伴随近似；完整迭代现在使用 `_discrete_adjoint_gradient`。
原来的 `backward_acoustic_solver.py` 同样保留，可用于反传演示，但不承担新优化器的梯度计算。

新的实现对 `forward_acoustic_solver` 的实际更新式逐步反向求导，包括：

1. 可变速度空间算子的转置。`diag(v²)*L` 的转置为 `L.T*diag(v²)`，不能简单原样反跑。
2. 非零吸收系数对应的更新分母和前一时刻系数。
3. 震源注入中的 `v²` 导数。
4. 第一个更新步和最后一个记录样本。
5. `np.pad(mode="edge")` 的链式求导，把吸收层速度偏导累加回物理域边缘。

`sigma` 由参考速度 3500 m/s 一次生成，之后视为固定的数值边界参数。
若每次根据试探模型的最大速度重新计算 sigma，则目标函数还会包含 sigma 对速度的导数；
那是另一个参数化问题，不能与本实现的梯度混用。

为支持严格伴随，正演函数增加三个可选参数，原调用参数和默认返回行为兼容：

- `store_wavefield_tt=False`：不保存物理域二阶时间导数。
- `store_velocity_sensitivity=True`：第三项返回扩展域每一步的局部速度导数。
- `verbose=False`：关闭逐时间步打印。

默认参数仍返回原先的 `(snapshots, shotgather, wavefield_tt)`。

## 优化与抛物线法

每轮先得到速度梯度，并在有速度上下界和可选 `update_mask` 时计算可行方向：

```python
p = -projected_gradient / max(abs(projected_gradient))
velocity_next = velocity + alpha * p
```

归一化仅改变步长标度，保持最速下降方向。`alpha` 的单位为 m/s，
也就是本轮单网格速度变化的最大绝对值；原梯度对应的步长为
`alpha / max(abs(projected_gradient))`，保存于 `raw_step_history`。

`parabolic_line_search` 对 `phi(alpha)=J(v+alpha*p)` 执行：

1. 从 `initial_step` 开始正演试探；无下降时减半，有下降时扩大，寻找三点区间。
2. 对三点 `(a,fa),(b,fb),(c,fc)` 拟合抛物线。设
   `s_ab=(fb-fa)/(b-a)`、`s_bc=(fc-fb)/(c-b)`、
   `q=(s_bc-s_ab)/(c-a)`，顶点为 `(a+b)/2-s_ab/(2*q)`。
3. 对顶点对应的真实模型再次正演，更新区间。非正曲率、越界顶点或重复点触发区间收缩。
4. 仅接受真实评价过、严格下降且满足 Armijo 条件的候选点。

步长限制由 `max_step`、速度上下界和 CFL 稳定性共同决定。
若最优下降试探点达到边界，可接受该边界点；不对未评价的抛物线预测值直接更新模型。
搜索失败时保留原模型，状态为 `line_search_failed`。

`objective_history` 从第 0 次初始模型开始，只记录接受的更新；
`line_searches` 则包含全部试探步及其真实目标函数，可用于复查搜索过程。
零梯度、相对下降量达阈值和达到迭代上限分别返回明确状态。

## 验证

运行 `python -m unittest -v test_fwi`。
测试覆盖含非均匀速度、非方形网格和吸收层的方向导数，边缘及震源导数，
Taylor 余项二阶收敛，多炮求和，零残差，短时间窗端点，重复接收器，
抛物线解析解、搜索失败、步长上限、速度界限、更新掩码和多次更新的真实下降。

本次首次严格梯度检验得到约 `10^-9` 的相对误差，Taylor 余项减半比约为 4。
这比此前单次约 0.76% 的近似检验提供了更强的一致性证据。
7 项自动化测试全部通过，包括原正演接口兼容性检查。

主 notebook 按当前 101×101、9 炮、1001 个时间样本配置实际运行 5 次更新：

| 迭代 | 真实目标函数 J | 接受步长 (m/s) |
| --- | ---: | ---: |
| 0 | 0.603601352 | — |
| 1 | 0.100916828 | 31.1981 |
| 2 | 0.031263178 | 14.1815 |
| 3 | 0.016163406 | 6.1480 |
| 4 | 0.009902374 | 5.7590 |
| 5 | 0.006958964 | 2.8840 |

目标函数累计下降 **98.8471%**，每轮均采用已评价的抛物线候选步。
模型 RMSE 从 **19.8020 m/s** 降到 **14.4005 m/s**。
最终速度范围约 2482.74–2553.39 m/s，仍未完全恢复 2600 m/s 的异常体峰值。
停止状态为 `max_iterations`，表示到达示例的 5 次迭代上限，并非宣称已收敛。
精确数组保存在 `fwi_results/inversion_result.npz`，图像及已执行单元输出一并保留。
另起 Python 进程重新生成 9 炮观测数据并对保存的最终模型独立正演，
得到 `J = 0.006958964459390798`，与保存的曲线末值完全一致。

有限观测和少量迭代不能保证恢复唯一的真实速度模型或达到全局最优解；
下降曲线检验的是指定数据目标函数。示例同时输出模型 RMSE，便于分别观察数据拟合和模型恢复。
