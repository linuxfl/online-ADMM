## LBFGS算法和ADMM算法解决Logistic Regression的分布式计算问题
###下载代码
git clone git@github.com:linxfl/ADMM.git
###编译
cd src;make
###配置与运行
####数据划分
需要将数据划分到每一个节点，有统一的前缀，admm.conf中需要在train_file中添加这个前缀文件名，划分到每一个节点的数据文件以这个前缀加‘_’再加进程号命名，<br>
例如：train_file=./data/data,那么划分到每一个节点的数据文件以data_0,data_1,data_2,...,data_n命名
####数据格式
数据以稀疏格式存储
####配置文件
#####admm.conf中有下面配置选项：
  * 数据集相关的：<br>
train_file：训练数据文件前缀<br>
valid_file：测试集<br>
model_file：保存的模型文件<br>
out_file：预测结果<br>
num_fea：特征数<br>
ins_len：一条样本的列数<br>

  * LBFGS算法和LR模型相关的：<br>
epsilon：lbfgs收敛条件阈值<br>
hasL1reg：有没有L1正则<br>
l1reg：L1正则系数<br>
l2reg：2正则系数，没有L2正则直接写0<br>
lbfgs_maxIter：lbfgs最大的迭代数<br>
m：lbfgs保存的前m次曲率信息<br>

  * admm算法相关参数：<br>
rho：admm惩罚参数<br>
admm_maxIter：admm最大迭代数<br>
####脚本文件（script）
run.sh:运行，需要自已去配置进程数.<br>
sync_model.py:修改完配置文件需要同步到所有机器上去.<br>
cal_metric.py:计算结果精度，有AUC、aopc、MAE、MSE<br>
####具体执行过程
./scritp/run.sh 运行程序并生成模型文件<br>
./bin/predict ./conf/admm.conf 对测试集进行预测<br>
python ./script/cal_metirc.py result.data 计算结果精度<br>
