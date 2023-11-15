# stAdv: Spatially Transformed Adversarial Attack (PyTorch)

本仓库由[as791/atAadv-Pytorch](https://github.com/as791/stAdv-PyTorch/tree/main) fork 得到。并做了一点修改：

1. 将原本的一些numpy运算改为tensor计算，可以更好利用GPU
2. 将优化器由L-BFGS改为SGD，效果更好
