%% 6_d
load('/Users/Cyrus/Documents/MATLAB/deep-learning/HW1/results/hu_20.mat')
hold on; plot(0:epoch,train_loss); hold on; plot(0:epoch,valid_loss);
load('/Users/Cyrus/Documents/MATLAB/deep-learning/HW1/results/hu_100.mat')
hold on; plot(0:epoch,train_loss); hold on; plot(0:epoch,valid_loss);
load('/Users/Cyrus/Documents/MATLAB/deep-learning/HW1/results/hu_200.mat')
hold on; plot(0:epoch,train_loss); hold on; plot(0:epoch,valid_loss);
load('/Users/Cyrus/Documents/MATLAB/deep-learning/HW1/results/hu_500.mat')
hold on; plot(0:epoch,train_loss); hold on; plot(0:epoch,valid_loss);

legend('train error, 20 hidden units', 'valid error, 20 hidden units', ... 
'train error, 100 hidden units', 'valid error, 100 hidden units', ...
'train error, 200 hidden units', 'valid error, 200 hidden units', ...
'train error, 500 hidden units', 'valid error, 500 hidden units')
title('cross-entropy error')

%% 6_h
load('/Users/Cyrus/Documents/MATLAB/deep-learning/HW1/results/batch_norm.mat')
hold on; plot(0:epoch,train_loss); hold on; plot(0:epoch,valid_loss);
load('/Users/Cyrus/Documents/MATLAB/deep-learning/HW1/results/no_batch_norm.mat')
hold on; plot(0:epoch,train_loss); hold on; plot(0:epoch,valid_loss);

legend('train error, batch norm', 'valid error, batch norm', ... 
'train error, no norm', 'valid error, no norm')
title('cross-entropy error')

%% 6_i
load('/Users/Cyrus/Documents/MATLAB/deep-learning/HW1/results/sigmoid.mat')
hold on; plot(0:epoch,train_loss); hold on; plot(0:epoch,valid_loss);
load('/Users/Cyrus/Documents/MATLAB/deep-learning/HW1/results/tanh.mat')
hold on; plot(0:epoch,train_loss); hold on; plot(0:epoch,valid_loss);
load('/Users/Cyrus/Documents/MATLAB/deep-learning/HW1/results/relu.mat')
hold on; plot(0:epoch,train_loss); hold on; plot(0:epoch,valid_loss);
legend('train error, sigmoid', 'valid error, sigmoid', ... 
'train error, tanh', 'valid error, tanh', ...
'train error, ReLU', 'valid error, ReLU')
title('cross-entropy error')