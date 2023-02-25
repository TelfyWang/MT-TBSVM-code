function [ H ] = Cond(H)
%COND 此处显示有关此函数的摘要
%   此处显示详细说明

    H = H + 1e-5*speye(size(H));
end

