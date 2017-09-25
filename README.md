# 10707-A1-xiyuanl1
Preparation:  
In matlab, add root folder and all sub folders to path

To run the network:  
run `src\main.m` to run the network  
Sections are seprated to run independently if wanted.

To change network setup:
change `intialized network` section of `src\main.m`.
Network structure is defined by vector `layers`, other parameters can be changed accordingly. Activation functions are passed as function parameters: `activ_fun`, its corresponding gradient function is also passed as `dactiv_fun`.

Loss and accuracy:
In this code I refer `loss` to cross-entropy loss and `accuracy` to classification accuracy. The classification error in the write-up can be aquired by `1-accuracy`.

Plots:
Running `src\main.m` should plot cross-entropy loss over epochs and classification error over epochs. Weights can be plot running `src\plot_weight.m`. Help script `src\plot_script.m` has some plots function I used to generate plots for integrated plots.w
