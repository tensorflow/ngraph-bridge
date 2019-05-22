#!/bin/bash



echo "**************************************************************************"
echo "Run MNIST ADAM OPTIMIZER on NGRAPH"
echo "**************************************************************************"
python mnist_deep_simplified.py --train_loop_count=20

echo "**************************************************************************"
echo "Run MNIST ADAM OPTIMIZER on TF"
echo "**************************************************************************"
NGRAPH_TF_DISABLE=1 python mnist_deep_simplified.py --train_loop_count=20

echo "**************************************************************************"
echo "Run MNIST GRADIENT DESCENT on NGRAPH"
echo "**************************************************************************"

python mnist_deep_simplified_gradient_descent.py --train_loop_count=20

echo "**************************************************************************"
echo "Run MNIST GRADIENT DESCENT on TF"
echo "**************************************************************************"

NGRAPH_TF_DISABLE=1 python mnist_deep_simplified_gradient_descent.py --train_loop_count=20
