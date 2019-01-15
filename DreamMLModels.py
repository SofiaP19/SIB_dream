import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy import stats


class DreamMLModels:

    def __init__(self, data, n_div=(1000, 100), feature_selection=False):
        self.data = clean(data)
        self.feature_selection = feature_selection
        self.x, self.y, self.train_in, self.train_out, self.test_in, self.test_out, self.train_inFinal, self.train_outFinal, self.test_inFinal, self.test_outFinal = self.set_train_and_test_data(self.data, n_div)
        self.LR_model = None
        self.NNT_model = None
        self.SVM_model = None
        self.RF_model = None

    def set_train_and_test_data(self, data, n):
        """
        :n: number of lines of test data
        :return: in values (x) and response values/variables (y) and divided sets of train and test values
        """
        y = data.standard_value.copy()
        x = data.drop('standard_value', axis=1)
        x = x.values
        y = y.values
        np.around(x, 5)
        np.around(y, 5)
        if self.feature_selection:
            sel = VarianceThreshold(threshold=0.2)
            x = sel.fit_transform(x)
        indices = np.random.permutation(len(x))
        leave_out = n[0]+n[1]
        train_in = x[indices[:-leave_out]]
        train_out = y[indices[:-leave_out]]
        test_in = x[indices[-n[0]:-n[1]]]
        test_out = y[indices[-n[0]:-n[1]]]
        train_inFinal = x[indices[:-n[1]]]
        train_outFinal =y[indices[:-n[1]]]
        test_inFinal = x[indices[-n[1]:]]
        test_outFinal = y[indices[-n[1]:]]

        return x, y, train_in, train_out, test_in, test_out, train_inFinal, train_outFinal, test_inFinal, test_outFinal

    def create_test_eval(self, model_name, final=False):
        if model_name == 'ALL':
            models_list = ['LR', 'NNT', 'SVM', 'RF']
            res = []
            for mod in models_list:
                print('\n----------------------' + mod + '----------------------------\n')
                m = self.create_model(mod, final)
                pred_values = self.predict(mod, final)
                res.append((mod, m, pred_values))
                self.error_measure(pred_values, final)
                print('\n-------------------------------------------------------------\n')
            return res

        else:
            print('\n----------------------' + model_name + '----------------------------\n')
            m = self.create_model(model_name, final)
            pred_values = self.predict(model_name,final)
            self.error_measure(pred_values, final)
            print('\n-------------------------------------------------------------\n')

            return m, pred_values

    def create_model(self, model_name, final=False, param_opt = True):
        """
        :param model: which model to train
        :return: nothing, just set the desired model to a variable
        model options:
        'LR' - Linear Regression
        'NNT' - Neural Networks
        'SVM' - Suport Vector Machines
        'RF' - Random Forests
        """
        #X_train, y_train = None, None
        if final:
            X_train, y_train = self.train_in, self.train_out
        else:
            X_train, y_train = self.train_inFinal, self.train_outFinal


        if model_name == 'LR':
            model = LinearRegression()
            if param_opt:
                param_lr = {"copy_X": [True, False],
                            "fit_intercept": [True, False],
                            "normalize": [True, False]}
                clf_lr = GridSearchCV(model, param_grid=param_lr, cv=5)
                clf_lr.fit(X_train, y_train)
                self.LR_model = clf_lr.best_estimator_
            else:
                self.LR_model = model.fit(X_train, y_train)

            return self.LR_model

        elif model_name == 'NNT':
            model = MLPRegressor()
            if param_opt:
                param_nn = {"activation": ['identity', 'logistic', 'tanh', 'relu'],
                            "solver": ["lbfgs", "sgd", "adam"],
                            "learning_rate": ["constant", "invscaling", "adaptive"],
                            "max_iter": stats.randint(1, 500)}
                clf_nn = RandomizedSearchCV(model, param_distributions=param_nn, cv=5)

                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #não está a funcionar com isto... a função clean era suposto resolver o erro que está a dar...
                clf_nn.fit(X_train, y_train)

                #clf_nn.fit(self.x, self.y)

                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



                self.NNT_model = clf_nn.best_estimator_
            else:
                self.NNT_model = model.fit(X_train, y_train)

            return self.NNT_model
        elif model_name == 'SVM':
            model = SVR()
            if param_opt:
                #parameters_svm = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
                parameters_svm = {"C": stats.uniform(2, 10), "gamma": stats.uniform(0.1, 1)}
                clf_svm = RandomizedSearchCV(model, param_distributions=parameters_svm, n_iter=20, cv=5)
                clf_svm.fit(X_train, y_train)
                self.SVM_model = clf_svm.best_estimator_
            else:
                self.SVM_model = model.fit(X_train, y_train)

            return self.SVM_model
        else:
            model = RandomForestRegressor()
            if param_opt:
                param_rf = {"max_depth": [3, 5],
                              "max_features": stats.randint(1, 11),
                              "min_samples_split": stats.randint(2, 11),
                              "min_samples_leaf": stats.randint(1, 11),
                              "bootstrap": [True, False]
                              }
                clf_rf = RandomizedSearchCV(model, param_distributions=param_rf, n_iter=20, cv=5)
                clf_rf.fit(X_train, y_train)
                self.RF_model = clf_rf.best_estimator_
            else:
                self.RF_model = model.fit(X_train, y_train)

            return self.RF_model


    def predict(self, model, final=False):
        """
        :param to_test: values to test in the model, by default use the test_in values
        :param model: indicates which model use to test predict the values
        :return: the predicted values
        """
        if final:
            to_test = self.test_inFinal
        else:
            to_test = self.test_in
        if model == 'LR':
            return self.LR_model.predict(to_test)
        elif model == 'NNT':
            return self.NNT_model.predict(to_test)
        elif model == 'SVM':
            return self.SVM_model.predict(to_test)
        else:
            return self.RF_model.predict(to_test)

    def RF_feature_selection(self, n=50, see_only_profet_features=True):
        """
        Selection of top important features based on RandomForestersRegressor model
        :param n: number of top features to select
        :param see_only_profet_features: indicates if it is to select only ProFet features
        :return: a list (len = n) of top features
        """
        if self.RF_model == None:
            self.create_model('RF')
        importance_list = self.RF_model.feature_importances_



        features_list = self.data.columns[1:]
        sorted_list = [x for _, x in sorted(zip(importance_list, features_list))]
        sorted_list = sorted_list[::-1]
        if see_only_profet_features:
            # get list with name of profet features
            with open('data/profFet_keys_list.pkl', 'rb') as f:
                pf_features = pickle.load(f)
            sorted_list_only_profet = []
            for feat in sorted_list:
                if feat in pf_features:
                    sorted_list_only_profet.append(feat)
            sorted_list = sorted_list_only_profet
        return sorted_list[:n]


    def error_measure(self, y_pred, final=False): #, metric):
        """
        Calcules and print all the error metrics
        :param y_true: real values
        :param y_pred: predicted values
        :return:
        """
        if final:
            y_true = self.test_outFinal
        else:
            y_true = self.test_out
        #explained_variance_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred), mean_squared_log_error(y_true, y_pred), median_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)
        try:
            print('explained_variance_score: ', explained_variance_score(y_true, y_pred))
        except:
            print('Not possible to perform explained_variance_score.')
        try:
            print('mean_absolute_error: ', mean_absolute_error(y_true, y_pred))
        except:
            print('Not possible to perform mean_absolute_error.')
        try:
            print('mean_squared_error: ', mean_squared_error(y_true, y_pred))
        except:
            print('Not possible to perform mean_squared_error.')
        try:
            print('mean_squared_log_error: ', mean_squared_log_error(y_true, y_pred))
        except:
            print('Not possible to perform mean_squared_log_error.')
        try:
            print('median_absolute_error: ', median_absolute_error(y_true, y_pred))
        except:
            print('Not possible to perform median_absolute_error.')
        try:
            print('r2_score: ', r2_score(y_true, y_pred))
        except:
            print('Not possible to perform r2_score.')


    '''def do_fit(self, model, final):
        """
        :param model: model to fit the data
        :param final: indicates if is the final model (use all the data) or not (use only train data)
        :return:
        """
        if not final:
            model = model.fit(self.train_in, self.train_out)
        else:
            model = model.fit(self.train_inFinal, self.train_outFinal)
        return model'''


def clean(data):
    """
    :param data: pandas data frame
    :return: data frame ready to be used in ML models
    """

    extra_cols = ['compound_id','standard_inchi_key','compound_name',
                  'synonym',
                  'target_id',
                  'target_pref_name',
                  'gene_names',
                  'wildtype_or_mutant',
                  'mutation_info',
                  'pubmed_id',
                  'standard_type',
                  'standard_relation',
                  'standard_units',
                  'ep_action_mode',
                  'assay_format',
                  'assaytype',
                  'assay_subtype',
                  'inhibitor_type',
                  'detection_tech',
                  'assay_cell_line',
                  'compound_concentration_value',
                  'compound_concentration_value_unit',
                  'substrate_type',
                  'smiles',
                  'substrate_relation',
                  'substrate_value',
                  'substrate_units',
                  'assay_description',
                  'title',
                  'journal',
                  'doc_type',
                  'annotation_comments',
                  'standard_inchi',
                  'standard_inchi_key_chembl',
                  'sequence']



    data = data[[col for col in data.columns if col not in extra_cols]]
    assert isinstance(data, pd.DataFrame), "df needs to be a pd.DataFrame"
    data.dropna(axis=1, how='all', inplace=True)
    data.dropna(axis=0, inplace=True)
    indices_to_keep = ~data.isin([np.nan, np.inf, -np.inf, np.string_]).any(1)
    data = data[indices_to_keep].astype(np.float64)
    return data




