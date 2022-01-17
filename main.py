import sklearn
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, Normalizer, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
import numpy as np


def cut_quantiles(df):
    Q1 = df.quantile(q=0.25)
    Q3 = df.quantile(q=0.75)
    IRQ = Q3 - Q1
    left_cond = Q1 - 1.5 * IRQ
    right_cond = Q3 + 1.5 * IRQ
    cond_train = np.logical_and(df > left_cond, df < right_cond)

    return df[cond_train]


class FluidPredictor:
    def __init__(self):
        pd.set_option('display.expand_frame_repr', False)
        self.db = pd.read_csv('2021.12.21_project_data.csv', delimiter="\t")
        self.interleukina = ['Interleukina – 11B',
                                  'Interleukina – 11P',
                                  'Interleukina – 16B',
                                  'Interleukina – 16P',
                                  'Interleukina – 24B',
                                  'Interleukina – 24P',
                                  'Interleukina – 31B',
                                  'Interleukina – 31P',
                                  'Interleukina – 36B',
                                  'Interleukina – 36P',
                                  'Interleukina – 44B',
                                  'Interleukina – 44P']

        self.predict_columns = ['16-B',
                                '16-P',
                                '11-B',
                                '11-P',
                                '24-B',
                                '24-P',
                                '36-B',
                                '36-P',
                                '31-B',
                                '31-P',
                                '44-B',
                                '44-P']
        self.zab_11 = ['PI - 11', 'GI - 11', 'PPD - 11', 'PPD - 11 B', 'PPD - 11 P', 'TWI - 11 suma', 'API', 'SBI',
                       'wiek', 'plec_1', 'plec_2']
        self.zab_16 = ['PI - 16', 'GI - 16', 'PPD - 16', 'PPD - 16 B', 'PPD - 16 P', 'TWI - 16 suma', 'API', 'SBI',
                       'wiek', 'plec_1', 'plec_2']
        self.zab_24 = ['PI - 24', 'GI - 24', 'PPD - 24', 'PPD - 24 B', 'PPD - 24 P', 'TWI - 24 suma', 'API', 'SBI',
                       'wiek', 'plec_1', 'plec_2']
        self.zab_36 = ['PI - 36', 'GI - 36', 'PPD - 36', 'PPD - 36 B', 'PPD - 36 P', 'TWI - 36 suma', 'API', 'SBI',
                       'wiek', 'plec_1', 'plec_2']
        self.zab_31 = ['PI - 31', 'GI - 31', 'PPD - 31', 'PPD - 31 B', 'PPD - 31 P', 'TWI - 31 suma', 'API', 'SBI',
                       'wiek', 'plec_1', 'plec_2']
        self.zab_44 = ['PI - 44', 'GI - 44', 'PPD - 44', 'PPD - 44 B', 'PPD - 44 P', 'TWI - 44 suma', 'API', 'SBI',
                       'wiek', 'plec_1', 'plec_2']
        self.zeby = [self.zab_16, self.zab_11, self.zab_24, self.zab_36, self.zab_31, self.zab_44]
        self.questions = ['zgrzytanie', 'zaciskanie', 'sztywnosc', 'ograniczone otwieranie',
                          'bol miesni', 'przygryzanie', 'cwiczenia', 'szyna', 'starcie-przednie',
                          'starcie-boczne', 'ubytki klinowe', 'pekniecia szkliwa',
                          'impresje jezyka', 'linea alba', 'przerost zwaczy', 'tkliwosc miesni']
        self.PI = ['PI - 11', 'PI - 16', 'PI - 24', 'PI - 36', 'PI - 31', 'PI - 44']
        self.GI = ['GI - 11', 'GI - 16', 'GI - 24', 'GI - 36', 'GI - 31', 'GI - 44']
        self.TWI = ['TWI - 11 suma', 'TWI - 16 suma', 'TWI - 24 suma', 'TWI - 36 suma', 'TWI - 31 suma',
                    'TWI - 44 suma']
        self.PPD = ['PPD - 11', 'PPD - 11 B', 'PPD - 11 P',
                    'PPD - 16', 'PPD - 16 B', 'PPD - 16 P',
                    'PPD - 24', 'PPD - 24 B', 'PPD - 24 P',
                    'PPD - 36', 'PPD - 36 B', 'PPD - 36 P',
                    'PPD - 31', 'PPD - 31 B', 'PPD - 31 P',
                    'PPD - 44', 'PPD - 44 B', 'PPD - 44 P']

    def preprocessing(self):
        self.db.drop(columns=['Unnamed: 0'], inplace=True)
        missing_columns = list(self.db.isnull().sum()[self.db.isnull().sum() > 0].index)
        db_preproc = self.db
        db_preproc[['SBI', 'API']] = db_preproc[['SBI', 'API']].applymap(lambda x: x.replace('%', '')).astype(
            float)
        # boolean to int questions
        db_preproc[self.questions] = db_preproc[self.questions].astype(int)

        # add feature that tell us if the aperature is healthy or not
        for ppd in self.PPD:
            name = ppd + ' - healthy'
            db_preproc[name] = 0

        for ppd in self.PPD:
            name = ppd + ' - healthy'
            indexes = db_preproc.loc[db_preproc[ppd] < 2].index
            db_preproc[name].iloc[indexes] = 1

        # clean interleukina
        interleukina_outliered = []
        for i in range(len(missing_columns)):
            temp = db_preproc[self.interleukina[i]].dropna()
            new_name = self.interleukina[i] + "_outliered"
            interleukina_outliered.append(new_name)
            interleukina_fixed = cut_quantiles(temp)
            db_preproc[new_name] = interleukina_fixed

        db_preproc.drop(self.interleukina, axis=1, inplace=True)
        interleukina_mean_fill = db_preproc[interleukina_outliered].mean().to_dict()
        db_preproc.fillna(value=interleukina_mean_fill, inplace=True)
        self.interleukina = interleukina_outliered

        return db_preproc

    def encoding_data(self, df):
        df[['API', 'SBI', 'wiek'] + self.interleukina] = MinMaxScaler().fit_transform(df[['API', 'SBI', 'wiek'] + self.interleukina])
        # oht the plec column
        cat_variables = ['plec', *self.questions]
        for cat in cat_variables:
            encoded_feat = OneHotEncoder().fit_transform(df[cat].values.reshape(-1, 1)).toarray()
            n = df[cat].nunique()
            cols = ['{}_{}'.format(cat, n) for n in range(1, n + 1)]
            encoded_df = pd.DataFrame(encoded_feat, columns=cols)
            encoded_df.index = df.index
            # delete actual plec kolumn and add encoded
            df.drop(cat, axis=1, inplace=True)
            df = pd.concat([df, encoded_df], axis=1)

        return df

    def predicting(self):
        df = self.preprocessing()
        df = self.encoding_data(df)

        sum_result = 0
        for i, aperture in enumerate(self.predict_columns):
            X = df.drop(self.predict_columns, axis=1)
            y = df[aperture]

            # stratify continous data
            min_y = np.amin(y)
            max_y = np.amax(y)
            bins = np.linspace(start=min_y, stop=max_y, num=5)
            y_binned = np.digitize(y, bins, right=True)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,
                                                                stratify=y_binned)

            model = SVR(C=10, kernel='linear')

            print(self.zeby[int(i / 2)], aperture)

            model.fit(X_train, y_train)
            y_predicted = model.predict(X_test)

            result = mean_absolute_error(y_test, y_predicted)
            perc = (result / max_y * 100)
            print(f"MAE dla {aperture}: ", result, f" Min: {min_y}, Max: {max_y}, perc: {perc}")
            sum_result += result

        print('Final result: ', sum_result / len(self.predict_columns))


p = FluidPredictor()
p.predicting()
