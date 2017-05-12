% ***************************************
% 16 April 2017
% Carlo P. Las Marias | carlol@gmail.com
% ***************************************


function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% disp('size X: '), disp(size(X));      % 300 x 2
% disp('size idx: '), disp(size(idx));  % 300 x 1
% disp('K: '), disp(K);                 % 3
% disp('size of centroid: '), disp(size(centroids));  %
% disp('centroid: '), disp(centroids);  %
% disp(X);

for i = 1:size(X,1)
  min_dist = inf;

  for j = 1:K
    diff = X(i,:)' - centroids(j,:)';
    dist = diff' * diff;
    if (dist < min_dist)
      min_dist = dist;
      idx(i) = j;
    endif
  end

end

% =============================================================

end

