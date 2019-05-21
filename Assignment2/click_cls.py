import pandas as pd
from sklearn.linear_model import SGDClassifier

print('reading...')
data = pd.read_csv("data/training_set_VU_DM.csv", nrows=1000000)
print('read...')

columns = ['site_id', 'visitor_location_country_id',
           'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
           'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
           'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price',
           'position', 'price_usd', 'promotion_flag', 'srch_destination_id',
           'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
           'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
           'srch_query_affinity_score', 'orig_destination_distance', 'random_bool']

data.fillna(0, inplace=True)

train = data.loc[:len(data) // 2]
test = data.loc[len(data) // 2:]

train_X = train.loc[:, columns].to_numpy()
train_Y = train.loc[:, ['click_bool']].to_numpy().T[0]

test_X = test.loc[:, columns].to_numpy()
test_Y = test.loc[:, ['click_bool']].to_numpy().T[0]

clf = SGDClassifier(verbose=1)
print('running fit...', len(train_Y))
clf.fit(train_X, train_Y)
print('running score...', len(test_Y))
print(clf.score(test_X, test_Y))
