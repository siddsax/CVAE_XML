from header import *
from precision_k import precision_k
import time
import pdb

# - ------ legacy sampling ----------------------------------------------
# for i in range(np.shape(y_tr)[0]):
#     labels = ksort[:np.sum(k<3)]
#     fin_labels = np.random.choice(labels, x, replace=False)
#     new_y[i, fin_labels] = 1
#     # print(pcaed)
# ---------------------------------------------------------------------





def gen(x_tr, y_tr, params):
    new_y = np.load('../new_y.npy')
    # # ---------- Cluster Sampling ---------------------------------------
    # label_counts = np.sum(y_tr, axis=0)
    # x = np.sum(y_tr, axis=1)
    # new_y = np.zeros(np.shape(y_tr))#[0], np.shape(y_tr)[1])
    # ## clusters = np.load('cluster_assignments_1.npy')[0,:]
    # # num_clusters = int(np.max(clusters))

    # lives = label_counts - 40
    # lives[np.argwhere(lives>0)] = 0
    # clusters[np.argwhere(lives==0)] = num_clusters + 1 
    # data_pts_num = []
    # data_pts = []
    # for i in range(int(num_clusters)):
    #     data_pts.append(np.argwhere(clusters==i))           
    #     data_pts_num.append(len(data_pts[i]))


    # data = 0
    # priority_list = []
    # stuck_count = 0
    # while(np.sum(lives) < 0 and data < y_tr.shape[0]):
    #     if(len(priority_list)):
    #         clst_num = priority_list[0]
    #         priority_list.remove(clst_num)
    #     else:    
    #         clst_num = np.random.randint(0, high=num_clusters)
    #     num_labels = np.random.choice(x)
    #     if(num_labels>data_pts_num[clst_num]):
    #         if(stuck_count>10):
    #             stuck_count = 0
    #         else:
    #             stuck_count+=1
    #             priority_list.append(clst_num)
    #             print(" ---- stuck ---- at {1} for {0} ----".format(num_labels, clst_num))
    #             continue
    #     else:
    #         x = np.delete(x, np.argwhere(x==num_labels)[0])
    #         fin_labels = np.random.choice(data_pts[clst_num][:,0], int(num_labels), replace=False)
    #         lives[fin_labels] += 1
    #         clusters[np.argwhere(lives==0)] = num_clusters + 1
    #         data_pts_num = []
    #         data_pts = []
    #         for i in range(num_clusters):
    #             data_pts.append(np.argwhere(clusters==i))           
    #             data_pts_num.append(len(data_pts[i]))

    #         new_y[data, fin_labels] = 1
    #         data+=1
    #         print(data)
    # np.save('../new_y.npy', new_y)
    # ----------------  -----------------------------------------------------
    print(new_y.shape)
    c = Variable(torch.from_numpy(new_y.astype('float32'))).type(params.dtype)
    X_dim = x_tr.shape[1]
    model = fnn_model_class(params)
    if(torch.cuda.is_available()):
        model.cuda()
        print("--------------- Using GPU! ---------")
    else:
        print("=============== Using CPU =========")
    model = load_model(model,params.load_model + '/model_best_test_regen')
    model.eval()
    eps = Variable(torch.randn(np.shape(y_tr)[0], params.Z_dim)).type(params.dtype)
    X_sample = model.decoder(eps, c)

    new_x = params.scaler.inverse_transform(X_sample.data)

    name = params.load_model.split('/')[-1]
    np.save('new_x_' + name, np.around(new_x, decimals=4))
    np.save('new_y_' + name, new_y)
