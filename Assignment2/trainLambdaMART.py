import pickle
import pandas as pd
from pyltr.models.lambdamart import LambdaMART
from pyltr.metrics import NDCG

print("Reading train_0.csv ...")
train = pd.read_csv("data/train/train_0.csv").sort_values("srch_id")
print("Finished reading")
qids = train["srch_id"].copy()
train_Y = train["target"]
train_X = train.drop(["target", "srch_id", "click_bool", "booking_bool", "gross_bookings_usd", "position"], axis=1)

model = LambdaMART(
    metric=NDCG(len(train)),
    n_estimators=200,
    min_samples_leaf=6,
    max_depth=10,
    max_leaf_nodes=7,
    verbose=1
)
del train
print("Fitting lambdaMART ...")
model.fit(train_X, train_Y, qids)

outfile = "data/lambdamart_EST10_SL20_D10_LN7"
print("Dumping model ...")
pickle.dump(model, open(outfile, 'wb'))
print("Model dumped")

print("Calculating feature importances ...")
importances = zip(train_X.columns.values, list(model.feature_importances_))
features = [feature for feature in importances]
features.sort(key=lambda x: x[1], reverse=True)
for feature in features:
    print(feature)

print("Finished")
