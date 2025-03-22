import pandas as pd

from lenskit.als import BiasedMFScorer
from lenskit.batch import recommend
from lenskit.data import ItemListCollection, UserIDKey, load_movielens
from lenskit.knn import ItemKNNScorer
from lenskit.metrics import NDCG, RBP, RecipRank, RunAnalysis
from lenskit.pipeline import topn_pipeline
from lenskit.splitting import SampleFrac, crossfold_users
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pyprojroot.here import here


ml100k = load_movielens(Path("C:/SCHOOL/Spring2024/HPML/Project/100k/1monoine/data/ml-100k.zip"))
print(ml100k.interaction_table(format="pandas", original_ids=True).head())

model_ii = ItemKNNScorer(k=20)
model_als = BiasedMFScorer(features=50)

pipe_ii = topn_pipeline(model_ii)
pipe_als = topn_pipeline(model_als)



# test data is organized by user
all_test = ItemListCollection(UserIDKey)
# recommendations will be organized by model and user ID
all_recs = ItemListCollection(["model", "user_id"])

for split in crossfold_users(ml100k, 5, SampleFrac(0.2)):
    # collect the test data
    all_test.add_from(split.test)

    # train the pipeline, cloning first so a fresh pipeline for each split
    fit_als = pipe_als.clone()
    fit_als.train(split.train)
    # generate recs
    als_recs = recommend(fit_als, split.test.keys(), 100)
    all_recs.add_from(als_recs, model="ALS")

    # do the same for item-item
    fit_ii = pipe_ii.clone()
    fit_ii.train(split.train)
    ii_recs = recommend(fit_ii, split.test.keys(), 100)
    all_recs.add_from(ii_recs, model="II")



