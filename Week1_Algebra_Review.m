% The ; denotes we are going back to a new row.
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12];

% Initialize a vector 
v = [1;2;3] ;

% Get the dimension of the matrix A where m = rows and n = columns
[m,n] = size(A);

% You could also store it this way
dim_A = size(A);

% Get the dimension of the vector v 
dim_v = size(v);

% Now let's index into the 2nd row 3rd column of matrix A
A_23 = A(2,3);

% Matrix addition
S = [1,3,2;
      4,0,1];
F = [1,3;0,1;5,2];


%Inverse Matrix
invF = pinv(F);

A = [3,4;2,16];

aTransp = A';


%Basic use

v = 1:0.1:4;
v = 1:6;

C = ones(5);
C = 2*ones(6);

rand(3,2);

w = -5 + sqrt(20)*(randn(1,10000));
%hist(w,50);

%moving data around

size(A);


A = magic(3)
[r c] = find( A >=5)



%subplot(1,2,1)






