% function [W, b] = UpdateParameters(W, b, grad_W, grad_b, learning_rate)
% % [W, b] = UpdateParameters(W, b, grad_W, grad_b, learning_rate) computes and returns the
% % new network parameters 'W' and 'b' with respect to the old parameters, the
% % gradient updates 'grad_W' and 'grad_b', and the learning rate.
% 
% update = @(x,y) x+learning_rate*y;
% W = cellfun(update, W, grad_W, 'UniformOutput', false);
% b = cellfun(update, b, grad_b, 'UniformOutput', false);
% 
% end

function [W, b, delta_W, delta_b] = UpdateParameters(W, b, grad_W, grad_b, learning_rate, delta_W, delta_b, momentum)
% [W, b] = UpdateParameters(W, b, grad_W, grad_b, learning_rate) computes and returns the
% new network parameters 'W' and 'b' with respect to the old parameters, the
% gradient updates 'grad_W' and 'grad_b', and the learning rate.
for i = 1:length(W)
    delta_W{1,i} = -learning_rate.*grad_W{1,i} + momentum.*delta_W{1,i} ...
        -learning_rate*(1/3000)*W{1,i};
    W{1,i} = W{1,i} + delta_W{1,i};
    
    delta_b{1,i} = -learning_rate.*grad_b{1,i} + momentum.*delta_b{1,i};
    b{1,i} = b{1,i} + delta_b{1,i};
end
end