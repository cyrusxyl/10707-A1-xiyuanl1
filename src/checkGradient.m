epsilon=1e-3;
num_check = 10;

X = test_data(1,:);
Y = test_labels(1,:);

[output, act_h, act_a] = Forward(W, b, X);
[grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a);

sum_error = 0;

for l = 1:length(W)
  [out,in]=size(W{l});
  for i = 1:num_check
      m = randperm(out,1);
      n = randperm(in,1);
      W1 = W;
      W2 = W;
      W1{l}(m,n) = W1{l}(m,n)+epsilon;
      W2{l}(m,n) = W2{l}(m,n)-epsilon;

      [~, loss1] = ComputeAccuracyAndLoss(W1, b, X, Y);
      [~, loss2] = ComputeAccuracyAndLoss(W2, b, X, Y);

      g = (loss2-loss1)/(2*epsilon);
      error = g-grad_W{l}(m,n);

      fprintf('g=%f, g_est=%f, error=%f\n',g,grad_W{l}(m,n),error);
      sum_error = sum_error + error;
  end
end

average_error = sum_error/(l*i)

  