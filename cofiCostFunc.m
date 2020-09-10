function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)

X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

J=(1/2)*sum((((X*Theta'-Y).^2).*R),"all")+(lambda/2)*(sum(Theta.^2,"all")+sum(X.^2,"all"));

for i=1:num_movies
    idx=find(R(i, :)==1);
    Thetatemp = Theta(idx, :);
    Ytemp = Y(i, idx);
    X_grad(i, :) = (X(i, :)*Thetatemp'-Ytemp)*Thetatemp+lambda*X(i,:);
end

for j=1:num_users
    idx=find(R(:, j)==1);
    Xtemp = X(idx, :);
    Ytemp = Y(idx, j);
    Theta_grad(j, :) = ((Theta(j,:)*Xtemp')'-Ytemp)'*Xtemp+lambda*Theta(j,:);
end


grad = [X_grad(:); Theta_grad(:)];

end
