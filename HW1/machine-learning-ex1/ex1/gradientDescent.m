function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
t_len = length(theta); % number of thetas to update
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    %fprintf('J = %f\n',J_history(iter));        
        
    tmpTheta = theta;
    diff = X*theta-y;%compute difference b/w hypothesis and observed
    %loop over all theta values 
    for i = 1:t_len
        h = X*theta; 
        %multiply the difference between the hypothesis and the observed by the correspond column in matrix X
        preSum = diff.*X(:,i); 
        s = sum(preSum(:));%sum it
        tmpTheta(i) = theta(i) - ((alpha/m)*s);
    end
    theta = tmpTheta; 
    
    %fprintf('Updated Theta = %i\n',theta);
    
end

end
