function [ Y ] = Kernel(U, V, p1)
%KERNEL 此处显示有关此类的摘要
% 核函数
%   此处显示详细说明
[ m, ~ ] = size(V);
[ n, ~ ] = size(U);
Y = exp(-(repmat(sum(U.*U,2)',m,1)+repmat(sum(V.*V,2),1,n) - 2*V*U')/(2*p1^2));
Y = Y';
end