# README.md
 ![](https://img.shields.io/badge/python-MachineLearning-brightgreen.svg)

## 目录

#### 1 [线性回归模型](#1)

##### 1.1 [公式推导](#1_1)

##### 1.2 [代码实现](https://github.com/LuohaoXu/ML/blob/main/src/linear_model/LinearRegression.py)




## <span id="1">1 线性回归模型</span>

#### <span id="1_1">1.1 公式推导</span>
数据

$$
\begin{matrix} 
x_{11},\ x_{12},\ ...,\ x_{1n},\ y_1\\
x_{21},\ x_{22},\ ...,\ x_{2n},\ y_2\\
...\\
x_{m1},\ x_{m2},\ ...,\ x_{mn},\ y_m\\
\end{matrix}
$$

推导

$$
令 \pmb X= \begin{bmatrix} 
x_{11},\ x_{12},\ ...,\ x_{1n},\ 1\\
x_{21},\ x_{22},\ ...,\ x_{2n},\ 1\\
...\\
x_{m1},\ x_{m2},\ ...,\ x_{mn},\ 1\\
\end{bmatrix}\ , \   
\pmb w=\begin{bmatrix}
w_{1}\\
w_{2}\\
...\\
w_{n}\\
b
\end{bmatrix}
$$

$假设 f(x)=\pmb X\pmb w$

$$
则E(\pmb w)=(\pmb X \pmb w-\pmb y)^T(\pmb X \pmb w-\pmb y)=\pmb w^T\pmb X^T\pmb X\pmb w-\pmb w^T\pmb X^T\pmb y-\pmb y^T\pmb X\pmb w+\pmb y^T\pmb y
$$

$根据最小二乘，需求使得E(\pmb w)最小的\pmb w^*$

$$
\frac{\partial E(\pmb w)}{\partial\pmb w}=\frac{\partial\pmb w^T\pmb X^T\pmb X\pmb w}{\partial \pmb w}-\frac{\partial\pmb w^T\pmb X^T\pmb y}{\partial\pmb w}-\frac{\partial\pmb y^T\pmb X\pmb w}{\partial \pmb w}+\frac{\partial\pmb y^T\pmb y}{\partial \pmb w}
$$

$$
=(\pmb X^T\pmb X+\pmb X^T\pmb X)\pmb w-\pmb X^T\pmb y-\pmb X^T\pmb y+0\\
$$

$$
=2(\pmb X^T\pmb X\pmb w -\pmb X^T\pmb y)\\
$$

$$
令\frac{\partial E(\pmb w)}{\partial\pmb w}=0，得\pmb w^*=(\pmb X^T\pmb X)^{-1}\pmb X^T\pmb y
$$



#### 1.2 代码实现

[代码地址](https://github.com/LuohaoXu/ML/blob/main/src/linear_model/LinearRegression.py)

