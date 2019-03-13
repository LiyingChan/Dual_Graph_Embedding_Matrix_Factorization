class ConfigX(object):
    """
    docstring for ConfigX

    configurate the global parameters and hyper parameters

    """

    def __init__(self):
        super(ConfigX, self).__init__()
        self.dataset_name = "yelp"

        # self.rating_path = '../data/las_vegas_reviews_filtered_index.txt'
        self.rating_train_path = "../data/yelp_train.csv"
        self.rating_test_path = "../data/yelp_test.csv"
        self.rating_valid_path = "../data/yelp_valid.csv"
        # self.trust_path = '../data/ft_trust.txt'
        self.sep = ','
        self.random_state = 0
        self.size = 0.8  # 0.8 0.7 0.6 0.5 0.4
        self.min_val = 1.0  # 0.5 1.0
        self.max_val = 5.0  # 4.0 5.0
        self.isEarlyStopping = True

        # HyperParameter
        self.coldUserRating = 5  # 用户评分数少于5条定为cold start users 5,10
        self.hotUserRating = 30  # 397个，50 40个， 30
        self.factor = 40  # 隐含因子个数户 或者5
        self.threshold = 1e-4  # 收敛的阈值
        self.lr = 0.01  # 学习率
        self.maxIter = 100
        self.lambdaP = 0.001  # 0.02
        self.lambdaQ = 0.001  # 0.02
