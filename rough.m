addpath('/scratch/work/saxenas2/fastxml/manik/Tools/matlab/')
addpath('/scratch/work/saxenas2/fastxml/manik/tools/')
addpath('/scratch/work/saxenas2/fastxml/manik/Tools/metrics/')
addpath('/scratch/work/saxenas2/fastxml/manik/FastXML/')

A = .55;
B = 1.5;

[tx, ty] = read_data('/scratch/work/saxenas2/fastxml/manik/FastXML/orig_data/eurlex_test.txt');
params.num_thread = 10;
param.start_tree =0;
param.num_tree = 50;
param.bias = 1.0;
param.log_loss_coeff = 1.0;
param.lbl_per_leaf = 100;
param.max_leaf = 10;

[score_matrix] = fastXML_test(tx, params, '/scratch/work/saxenas2/fastxml/manik/train_datasets/original_model');