from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, df, target_col='Churn'):
        self.df = df
        self.target_col = target_col

    def transform(self):
        self.df['Churn'] = self.df['Churn'].astype(int)
        self.df['International plan'] = self.df['International plan'].replace({'No': 0, 'Yes': 1})
        self.df['Voice mail plan'] = self.df['Voice mail plan'].replace({'No': 0, 'Yes': 1})
        self.df['State'] = LabelEncoder().fit_transform(self.df['State'])
        return self.df

    def split_and_scale(self):
        X = self.df.drop(columns=self.target_col)
        y = self.df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

        return X_train, X_test, y_train, y_test
