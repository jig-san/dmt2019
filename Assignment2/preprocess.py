import numpy as np
import pandas as pd
import random


def preprocess(df, train):
    todrop = []

    if train:
        # add target
        df = create_target(df)
        # df = df.drop(["click_bool", "booking_bool", "gross_bookings_usd", "position"], axis=1)

    # MISSING VALUES #

    # 2nd place,
    # missing values in hotel description - worst case scenario
    df = replace_missing_worst(df)

    # Alternative from the paper
    # replace with first quartile
    # df = replace_missing_first_quartile(df)

    # 2nd place,
    # historical data - 95% missing, so use them to create highlighting features
    df["usd_diff"] = np.abs(df["price_usd"] - df["visitor_hist_adr_usd"])
    df["usd_diff"].fillna(value=0)
    df["starrating_diff"] = np.abs(df["visitor_hist_starrating"] - df["prop_starrating"])
    df["starrating_diff"].fillna(value=0)
    todrop.extend(["visitor_hist_adr_usd", "visitor_hist_starrating"])

    # FEATURE ENGINEERING #

    # 4th place, Opera Solutions
    df = process_numerical(df)

    # date_time into year, month, day, hour
    df = split_datetime(df)
    todrop.append("date_time")

    # remove price outliers from training set
    if train:
        df = df[df["price_usd"] < 10000]

    # from the paper
    df["count_window"] = df["srch_room_count"] * max(df["srch_booking_window"]) + df["srch_booking_window"]

    # dropping irrelevant features and features with too many missing values
    print("Dropping useless columns ...")
    for label in todrop:
        df = df.drop(label, axis=1)
    for n in range(1, 9):
        df = df.drop(["comp%d_inv" % n, "comp%d_rate" % n, "comp%d_rate_percent_diff" % n], axis=1)

    return df


def normalize(df, column_name):
    print("Normalizing " + column_name + " ...")
    df[column_name] = (df[column_name] - df[column_name].mean()) / df[column_name].std()


def replace_missing_worst(df):
    print("Process missing hotel disctriptions ...")
    df["prop_review_score"].fillna(value=-1, inplace=True)
    df["prop_location_score2"].fillna(value=-1, inplace=True)
    df["srch_query_affinity_score"].fillna(value=-1000, inplace=True)  # log of probability, the less the less the probability
    df["orig_destination_distance"].fillna(value=40075, inplace=True)

    return df


def replace_missing_first_quartile(df):
    print("Process missing hotel disctriptions ...")
    location_quartile = df.groupby("prop_country_id")["prop_location_score2"].quantile(q=0.25)
    df["prop_location_score2_quartile"] = location_quartile[df.prop_country_id].values
    df["prop_location_score2"].fillna(df["prop_location_score2_quartile"], inplace=True)
    del df["prop_location_score2_quartile"]

    review_quartile = df.groupby("prop_country_id")["prop_review_score"].quantile(q=0.25)
    df["prop_review_score_quartile"] = review_quartile[df.prop_country_id].values
    df["prop_review_score"].fillna(df["prop_review_score_quartile"], inplace=True)
    del df["prop_review_score_quartile"]

    affinity_score_quartile = df.groupby("prop_country_id")["srch_query_affinity_score"].quantile(q=0.25)
    df["srch_query_affinity_score_quartile"] = affinity_score_quartile[df.prop_country_id].values
    df["srch_query_affinity_score"].fillna(df["srch_query_affinity_score_quartile"], inplace=True)
    del df["srch_query_affinity_score_quartile"]

    destination_distance_quartile = df.groupby("prop_country_id")["orig_destination_distance"].quantile(q=0.25)
    df["orig_destination_distance_quartile"] = destination_distance_quartile[df.prop_country_id].values
    df["orig_destination_distance"].fillna(df["orig_destination_distance_quartile"], inplace=True)
    del df["orig_destination_distance_quartile"]

    return df


def split_datetime(df):
    print("Processing date time ...")
    df['year'] = pd.DatetimeIndex(df['date_time']).year
    df['month'] = pd.DatetimeIndex(df['date_time']).month
    df['day'] = pd.DatetimeIndex(df['date_time']).day
    df['hour'] = pd.DatetimeIndex(df['date_time']).hour

    return df


def process_numerical(df):
    print("Processing numeric features ...")

    # adding mean, stddev, median features for each numerical hotel feature for each prop_id
    numeric_features = ["prop_starrating", "prop_review_score", "prop_location_score1", "prop_location_score2"]

    for label in numeric_features:
        mean = df.groupby("prop_id")[label].mean().fillna(value=-1)
        median = df.groupby("prop_id")[label].median().fillna(value=-1)
        stddev = df.groupby("prop_id")[label].std().fillna(value=-1)

        df[label + "_mean"] = mean[df.prop_id].values
        del mean
        df[label + "_median"] = median[df.prop_id].values
        del median
        df[label + "_stddev"] = stddev[df.prop_id].values
        del stddev

    return df


def create_target(df):
    print("Creating target ...")
    df["target"] = 0
    df.loc[df["click_bool"] == 1, "target"] = 1
    df.loc[df["booking_bool"] == 1, "target"] = 5

    return df


def get_random_sample(file):
    # returns dataset of s random rows
    # https://stackoverflow.com/questions/22258491/read-a-small-random-sample-from-a-big-csv-file-into-a-python-data-frame
    n = sum(1 for line in open(file)) - 1  # number of records in file (excludes header)
    s = 800000  # desired sample size
    skip = sorted(random.sample(range(1, n + 1), n - s))  # the 0-indexed header will not be included in the skip list

    return pd.read_csv(file, header=0, skiprows=skip)


def get_first_rows_sample(file):
    # returns dataset of nrows first rows

    return pd.read_csv(file, header=0, nrows=int(300000))


def process_original_data(trainfile, testfile):
    print("Reading .csv's ...")
    # traindata = get_random_sample(trainfile)
    # testdata = get_random_sample(testfile)
    traindata = get_first_rows_sample(trainfile)
    print("Loaded train file: " + str(traindata.shape[0]) + " rows, " + str(traindata.shape[1]) + " columns")
    testdata = get_first_rows_sample(testfile)
    print("Loaded test file: " + str(testdata.shape[0]) + " rows, " + str(testdata.shape[1]) + " columns")

    print("Writing .csv's...")
    traindata.to_csv("data/train_first_rows_" + str(traindata.shape[0]) + ".csv", index=False)
    testdata.to_csv("data/test_first_rows_" + str(testdata.shape[0]) + ".csv", index=False)
    print("Finished writing")

    return [traindata, testdata]


train_original = "data/training_set_VU_DM.csv"
test_original = "data/test_set_VU_DM.csv"
# process_original_data(train_original, test_original)

# train_filename = "data/train_first_rows_300000.csv"
# test_filename = "data/test_first_rows_300000.csv"

# print("Reading .csv's ...")
# train_df = pd.read_csv(train_filename, header=0)
# print("Loaded train: " + str(train_df.shape[0]) + " rows, " + str(train_df.shape[1]) + " columns")
# test_df = pd.read_csv(test_filename, header=0)
# print("Loaded test: " + str(test_df.shape[0]) + " rows, " + str(test_df.shape[1]) + " columns")

chunksize = 1000000
i = 0
for chunk_train in pd.read_csv(train_original, chunksize=chunksize):
    print("... Preprocessing train " + str(i) + "...")
    train = preprocess(chunk_train, True)
    del chunk_train
    train = train.fillna(value=0)
    print("writing result...")
    train.to_csv("data/train/train_" + str(i) + ".csv", index=False)
    print("Finished writing to data/train_" + str(i) + ".csv")
    del train
    i += 1

i = 0
for chunk_test in pd.read_csv(test_original, chunksize=chunksize):
    print("... Preprocessing test " + str(i) + "...")
    test = preprocess(chunk_test, False)
    del chunk_test
    test = test.fillna(value=0)
    print("writing result...")
    test.to_csv("data/test/test_" + str(i) + ".csv", index=False)
    print("Finished writing to data/test_" + str(i) + ".csv")
    del test
    i += 1
