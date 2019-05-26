import argparse
from model import *
import numpy.linalg as lin
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('numExp', type=int)
    parser.add_argument('Lambda', type=float)
    parser.add_argument('numLearner', type=int)
    parser.add_argument('data_path', type=str)
    parser.add_argument('defender_Beta', type=float)
    parser.add_argument('defender_Lambda', type=float)
    parser.add_argument('complete_information', type=int)
    parser.add_argument('overEstimate', type=int)

    args = parser.parse_args()
    numLearner, numExp, data_path, complete_information, defender_Beta, defender_Lambda, Lambda, overEstimate = \
        args.numLearner, args.numExp, args.data_path, args.complete_information, args.defender_Beta, args.defender_Lambda, args.Lambda, args.overEstimate


    # different parameters of the attacker
    Beta_range = np.linspace(0, 1.0, 11)
    preprocessor = StandardScaler

    # instantiation of the Environment
    Env = Environment(data_path, numLearner, preprocessor)
    Env.generate_train_test_data()


    # imaginary target used in MLSG
    shift = 5
    if complete_information:
        train_target = Env.train_y + shift
    else:
        if overEstimate:
            train_target = Env.train_y + np.random.uniform(low=shift, high=10)
        else:
            train_target = Env.train_y + np.random.uniform(low=0, high=shift)
    train_target[train_target.squeeze() > 10, :] = 10
    train_target[train_target.squeeze() < 0, :] = 0

    # instantiations of different learners, the instantiation
    # of MLSG will be done later
    OLS = OLS_learner(Env)
    Ridge = Ridge_learner(Env)
    Lasso = Lasso_learner(Env)

    ret = pd.DataFrame()
    for Beta in Beta_range:  
        RMSE_MLSG = []
        RMSE_OLS = []
        RMSE_Lasso = []
        RMSE_Ridge = []  

        numAdv = np.int(np.ceil(len(Env.test_y)*Beta))

        for exp in range(numExp):
            # generate different train/test data
            Env.generate_train_test_data()
            to_be_attacked = Env.generate_to_be_attacked_data(numAdv)

            # attacker's real target
            test_target = Env.test_y[:numAdv] + shift
            attacker = Attacker(test_target, Lambda, numLearner)

            # best case: defender knows all the parameters of the attacker
            if complete_information:
                MLSG = MLSG_learner(Env, train_target, Beta, Lambda)
            else:
                MLSG = MLSG_learner(Env, train_target, defender_Beta, defender_Lambda)

            # learn models
            OLS.learn()
            Lasso.learn()
            Ridge.learn()
            MLSG.learn()

            # attack OLS
            attacked_X = attacker.generate_adversarial_data(OLS.get_theta(), to_be_attacked)
            RMSE_OLS.append(OLS.get_RMSE(attacked_X, numAdv))

            # attack Lasso
            attacked_X = attacker.generate_adversarial_data(Lasso.get_theta(), to_be_attacked)
            RMSE_Lasso.append(Lasso.get_RMSE(attacked_X, numAdv))

            # attack Ridge
            attacked_X = attacker.generate_adversarial_data(Ridge.get_theta(), to_be_attacked)
            RMSE_Ridge.append(Ridge.get_RMSE(attacked_X, numAdv))

            # attack MLSG
            attacked_X = attacker.generate_adversarial_data(MLSG.get_theta(), to_be_attacked)
            RMSE_MLSG.append(MLSG.get_RMSE(attacked_X, numAdv))

        RMSE_MLSG = pd.DataFrame(RMSE_MLSG)
        RMSE_OLS = pd.DataFrame(RMSE_OLS)
        RMSE_Lasso = pd.DataFrame(RMSE_Lasso)
        RMSE_Ridge = pd.DataFrame(RMSE_Ridge)

        ret = pd.concat([ret, pd.concat([RMSE_MLSG, RMSE_OLS, RMSE_Lasso, RMSE_Ridge])], axis=1)

    # save result to disk
    ret['type'] = pd.Series(['MLSG'] * numExp + ['OLS'] * numExp + ['Lasso'] * numExp +['Ridge'] * numExp, index=ret.index)
    ret.columns = ['%.1f' % i for i in np.linspace(0, 1.0, 11)] + ['type']


    if complete_information:
        ret.to_csv('../result/redwine_Lambda=%.2f.csv' % Lambda)
    else:
        if overEstimate:
            ret.to_csv('../result/redwine_defenderLambda=%.2f_defenderBeta=%.2f_overEstimateTarget_attackerLambda=%.2f.csv' % (defender_Lambda, defender_Beta, Lambda)) 
        else:
            ret.to_csv('../result/redwine_defenderLambda=%.2f_defenderBeta=%.2f_underEstimateTarget_attackerLambda=%.2f.csv' % (defender_Lambda, defender_Beta, Lambda)) 










