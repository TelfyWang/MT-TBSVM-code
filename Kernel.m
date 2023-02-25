function [ Y ] = Kernel(U, V, p1)
%KERNEL �˴���ʾ�йش����ժҪ
% �˺���
%   �˴���ʾ��ϸ˵��
[ m, ~ ] = size(V);
[ n, ~ ] = size(U);
Y = exp(-(repmat(sum(U.*U,2)',m,1)+repmat(sum(V.*V,2),1,n) - 2*V*U')/(2*p1^2));
Y = Y';
end