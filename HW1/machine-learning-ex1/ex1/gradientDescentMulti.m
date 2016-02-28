function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
t_len = length(theta); % number of thetas to update
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    %fprintf('J = %i\n', J_history(iter));
    tmpTheta = theta;
    diff = X*theta-y;%compute difference b/w hypothesis and observed
    %loop over all theta values 
    for i = 1:t_len
        %multiply the difference by the correspond column in matrix X
        preSum = diff.*X(:,i); 
        s = sum(preSum(:));%sum it
        tmpTheta(i) = theta(i) - ((alpha/m)*s);
    end
    theta = tmpTheta; 
end

end
