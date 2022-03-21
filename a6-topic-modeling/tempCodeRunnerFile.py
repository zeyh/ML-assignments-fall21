    lda = LDATopicModel(K=3, dataset='20_newsgroups_subset')
    lda.initialize_counts()

    test2 = lda.phi_i_v(0, 'space')
    print(math.fabs(test2 - 0.009111) < 0.001, test2)
    test2 = lda.phi_i_v(2, 'sell')
    print(math.fabs(test2 - 0.001948) < 0.001, test2)
    
    print("--------------------")
    test = lda.theta_d_i(42, 1)
    print(math.fabs(test - 0.323809) < 0.001, test)
    