import pandas as pd

class DataLoader:
    def __init__(self, url1, url2):
        self.url1 = url1
        self.url2 = url2

    def load_data(self):
        df1 = pd.read_csv(self.url1)
        df2 = pd.read_csv(self.url2)
        df = pd.concat([df1, df2], ignore_index=True)
        return df.sample(frac=1).reset_index(drop=True)
