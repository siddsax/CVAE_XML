    # Building model
    # ==================================================
    print('-'*50)
    print("Building model..."); start_time = timeit.default_timer();
    model = CNN_model(args)
    model.model_file = os.path.join('./CNN_runtime_models', gen_model_file(args))
    if not os.path.isdir(model.model_file):
        os.makedirs(model.model_file)
    else:
        print('Warning: model file already exist!\n %s' % (model.model_file))

    model.add_data(X_trn, Y_trn)
    model.add_pretrain(vocabulary, vocabulary_inv)
    model.build_train()
    print('Process time %.3f (secs)\n' % (timeit.default_timer() - start_time))

    # Training model
    # ==================================================
    print('-'*50)
    print("Training model..."); start_time = timeit.default_timer();
    store_params_time = 0.0;
    for epoch_idx in xrange(args.num_epochs + 1):
        loss = model.train()
        print 'Iter:', epoch_idx, 'Trn loss ', loss
        if epoch_idx % 5 == 0:
            print 'saving model...'; tmp_time = timeit.default_timer();
            model.store_params(epoch_idx)
            store_params_time += timeit.default_timer() - tmp_time
    total_time = timeit.default_timer() - start_time
    print('Total time %.4f (secs), training time %.4f (secs), IO time %.4f (secs)' \
          % (total_time, total_time - store_params_time, store_params_time))

