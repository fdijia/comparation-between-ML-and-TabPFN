# 青少年心理风险的早期预警模型研究: 机器学习与 TabPFN的对比分析

------

这是论文`青少年心理风险的早期预警模型研究: 机器学习与 TabPFN的对比分析`的代码部分

------

## Start
对于该代码, 建议使用conda环境
```bash
conda create -n py13 python=3.13
```
下载tabpfn相关库, 注意tabpfn正在持续更新, 因此代码方面可能发生变化. 
```bash
pip install tabpfn
pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"
```

## utils
该文件夹定义了一些常数, 以及一些通用函数. 

## Run
- 1st Step: 进入`特征整理与数据构建`文件夹
  - 通过`P1_data_transform.py`获取已有数据并读取X和y, 由于原数据有部分敏感信息, 这里仅展示处理后的数据, 输出在`output`文件夹下. 
  - 通过`P2_data_split.py`对所有数据集进行5CV, 得到每个数据集的folds, 也输出在`output`文件夹下. 
  - `P3_correlation.py`是论文中`第三章 心理健康及其相关因素`的代码, 主要是各种检验以及画图函数. 
- 2nd Step: 
  - 机器学习模型: 进入`机器学习模型`文件夹, 并先运行`params_search.py`, 通过对`D1`数据集进行超参数网格搜索, 得到各模型最佳参数, 保存在`comparation`文件夹下. 再运行`classify.py`进行训练.
  - TabPFN: 进入`TabPFN`文件夹, 先在[官网](https://huggingface.co/Prior-Labs)下载模型权重: v2.5-default, v2.5-default-2, v2.6-default并分别重命名为default-1.ckpt, default-2.ckpt, default-3.ckpt, 保存在`TabPFN/model`文件夹下. 再运行`classify.py`进行训练. **官方刚发布了v3权重**
- 3rd Step: 进入`模型性能评估与比较`文件夹
  - 通过运行`compare.py`综合第二步的运行结果, 并对`D1`的结果进行boostrap(其他数据集可自行设置n_boostrap的值), 对比模型结果并保存. 此外还会对各个数据集的结果进行latex格式化输出, 得到latex的表格. 
  - 对于`D4`数据集, 需运行`balanced.py`, 其中会先进行超参数搜索, 再训练模型得到输出. 
- 4th Step: 进入`可解释性分析`文件夹
  - 运行`shapley.py`, 会对`D1`的模型结果进行shap分析, 运行结果较长, 对于其他数据集可以改变对应参数, 但是建议不要对KNN进行shap分析, 可能运行内存太大而卡死, 最后得到的`D1`数据集的shap结果保存在`output`文件夹下, 可以直接使用, 其中画的图在`figure`文件夹中. 
  - 运行`residual_erase.py`对`D1`数据集操作, 将其中的所有自变量(性别除外)通过残差化消除性别的影响, 并保存新的`sem/data.csv`; 再通过`EFA.py`设置合理参数, 对shap选择的变量进行EFA
  - 在`sem.R`中, 需要先下载对应库, 然后运行SEM, 注意模型描述基于EFA的输出. 结果保存在`sem`文件夹下
