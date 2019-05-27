# Adversarial Regression with Multiple Learners

### Overview
Currently, you can use this code to replicate the experimental results for **redwine** dataset (Figure 1, Figure 2, etc). The  experimental results for other datasets can be similarly generated. 


### Installation
**(Necessary: Python3.7 and conda)**

1. First, clone the project folder to your computer.
2. Then, create an environment and activate it:
  ```
  conda create -n multiple-learner python=3.7
  conda activate multiple-learner
  ```
3. After the environment is activated, install the following required packages:
   ```
   conda install numpy scipy pandas scikit-learn seaborn matplotlib
   pip install cvxpy
   pip install cvxopt
   ```
  

### Run 
1. Inside the project folder, create a folder to store experimental outputs:
```
mkdir result/
```

2.  Enter into **src/** folder, run the following command to generate experimental outputs:
```
./run_exp.sh
```

3. Insider **src/** folder, run the following command to generate Figure 1 (complete information), Figure 2 (incomplete information + over-estimatd z), and Figure 3 (incomplete information + under-estimated z):
```
python plot.py redwine
```

4. I generated the figures on MacOS Mojave. If you see some error like the following, [Try this solution: ](https://github.com/palantir/python-language-server/issues/217)
```
libc++abi.dylib: terminating with uncaught exception of type NSException
```

### Reference
```
@inproceedings{tong2018adversarial,
  title={Adversarial Regression with Multiple Learners},
  author={Tong, Liang and Yu, Sixie and Alfeld, Scott and others},
  booktitle={International Conference on Machine Learning},
  pages={4953--4961},
  year={2018}
}
```
