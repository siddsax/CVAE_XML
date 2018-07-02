addpath('/homeappl/home/saxenasi/fastxml/manik/Tools/matlab/')
addpath('/homeappl/home/saxenasi/fastxml/manik/tools/')
addpath('/homeappl/home/saxenasi/fastxml/manik/Tools/metrics/')
addpath('/homeappl/home/saxenasi/fastxml/manik/FastXML/')

A = .55;
B = 1.5;

load score_matrix.mat
[I, J, S] = find(score_matrix);
[sorted_I, idx] = sort(I);
J = J(idx);
S = S(idx);
score_matrix = sparse(J, sorted_I, S);

load ty.mat
[I, J, S] = find(ty);
[sorted_I, idx] = sort(I);
J = J(idx);
S = S(idx);
ty = sparse(J, sorted_I, S);
ip = inv_propensity(ty,A,B);
[metrics] = get_all_metrics(score_matrix , ty, ip)
