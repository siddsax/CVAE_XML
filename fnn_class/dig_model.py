from header import *
def dig(x_tr, y_tr, x_te, y_te, params):
    model = fnn_model_class(params)
    model.load_state_dict(torch.load(params.load_model + "/model_best"))

    all_params = []
    for i,param in enumerate(model.parameters()):
        print(i)
        print(param)
        all_params.append(param.cpu().detach().numpy().tolist())
        print(torch.mean(param))
        print("-------------------")

    fig, ax = plt.subplots()
    # print(all_params)
    ax.hist(all_params, 10)#, weights=num_sold)
    plt.show()