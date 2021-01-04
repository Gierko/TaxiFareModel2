import pandas as pd
from TaxiFareModel2.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"


def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
    return df


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

def set_pipe():

    dist_pipe = Pipeline([("distance", DistanceTransformer()),
                       ("scaler", StandardScaler())])

    time_features_pipe = Pipeline([("time", TimeFeaturesEncoder("pickup_datetime")),
                             ("encoding", OneHotEncoder(handle_unknown="ignore"))])

    preprocessing_pipe = ColumnTransformer([
                                ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
                               ("time", time_features_pipe, ["pickup_datetime"])])

    final_pipe = Pipeline([("preprocess", preprocessing_pipe),
                        ("algo", RandomForestRegressor())])

    return final_pipe




if __name__ == '__main__':
    df = set_pipe()
    print(df)
