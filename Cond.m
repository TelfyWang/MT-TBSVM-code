function [ H ] = Cond(H)
%COND �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

    H = H + 1e-5*speye(size(H));
end

