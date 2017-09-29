function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%

%PART 1
%replace numbers's with 1's and 0's in y
yMod = zeros(m, num_labels);
for i = 1:m
    yMod(i, y(i)) = 1;
endfor
%add ones for X
a1 = [ones(m,1) X];

%for theta1
z2 = a1*Theta1';
a2 = sigmoid(z2);

%for Theta2
a2 = [ones(size(a2,1),1) a2];

%final step
z3 = a2*Theta2';
a3 = sigmoid(z3);

%vectorized J matrix of size m, n
Jvect = -1*yMod .* log(a3)-(1-yMod) .* log(1-a3);

%Cost function without regularization
J = sum(sum(Jvect))/m ; 

%PART 1 - regularization
%do not regularize for Thetas (:,0)
th1 = Theta1(:, 2:end) .^2;
th2 = Theta2(:, 2:end) .^2;

%Sum the squared matrices
regTerm1 = sum(sum(th1));
regTerm2 = sum(sum(th2));

%Compute total regularization term
regTerm = (regTerm1 + regTerm2) * lambda / (2*m);

%Regularized Cost Function
J = J + regTerm;

%PART 2 - BACK PROPOGATION


for t = 1:m
  %Select t-ht row and make it a vector num_labels x 1
  a1 = [1; X(t,:)'];
  
  %Compute z2, a2 and add bias term before a2. z2 is a vector
  z2 = Theta1 * a1;
  a2 = sigmoid(z2);
  a2 = [1; a2];
  
  %Compute z3, a3. a3 needs to have dimension num_labels x 1
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  
  %Select tht row from the logical [0 0 0 1 0 0 0] y matrix. Convert to vector
  yy = yMod(t,:)';
  
  %dr is just difference between last layer and true y values
  d3 = a3- yy;

  %compute d2 with including 1 as a bias term for a2   
  d2 = (Theta2' * d3) .* [1; sigmoidGradient(z2)];
  
  %remove bias term
  d2 = d2(2:end);
  
  %Update gradients without regularization
  Theta1_grad = Theta1_grad + d2*a1';
  Theta2_grad = Theta2_grad + d3*a2';

endfor

%PART 3 -- regularized backpropogation

% set 0 for first terms as Theta0 is not regularized
Theta1 = Theta1(:, 2:end);
Theta2 = Theta2(:, 2:end);

Theta1 = [zeros(size(Theta1, 1),1) Theta1];
Theta2 = [zeros(size(Theta2, 1),1) Theta2];

%compute regularization terms
regThet1 = (lambda/m) * Theta1;
regThet2 = (lambda/m) * Theta2;

%update gradients with regularization
Theta1_grad = (1/m) * Theta1_grad + regThet1;
Theta2_grad = (1/m) * Theta2_grad + regThet2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
