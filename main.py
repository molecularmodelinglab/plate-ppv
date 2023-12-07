import numpy as np
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder as LE

from rdkit import Chem
from rdkit.Chem import AllChem

from tqdm import tqdm

from sklearn.utils import column_or_1d
from imblearn.under_sampling import RandomUnderSampler
from copy import deepcopy

import pickle


def bedroc_score(y_true, y_pred, decreasing=True, alpha=20.0):
    assert len(y_true) == len(y_pred)  # The number of scores must be equal to the number of labels
    big_n = len(y_true)
    n = sum(y_true == 1)
    if decreasing:
        order = np.argsort(-y_pred)
    else:
        order = np.argsort(y_pred)
    m_rank = (y_true[order] == 1).nonzero()[0]
    s = np.sum(np.exp(-alpha * m_rank / big_n))
    r_a = n / big_n
    rand_sum = r_a * (1 - np.exp(-alpha))/(np.exp(alpha/big_n) - 1)
    fac = r_a * np.sinh(alpha / 2) / (np.cosh(alpha / 2) -
                                      np.cosh(alpha/2 - alpha * r_a))
    cte = 1 / (1 - np.exp(alpha * (1 - r_a)))
    return s * fac / rand_sum + cte


class LabelEncoder(LE):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self


def get_fps(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return list(AllChem.GetHashedMorganFingerprint(mol, nBits=2048, radius=4, useChirality=True))


rus = RandomUnderSampler(random_state=42)


overall_res = {}


for dataset in os.listdir("./data"):
    df = pd.read_csv(os.path.join("./data", dataset))
    df = df[df['SMILES'].apply(lambda x: '.' not in x)]  # need to remove the mixtures
    df_name = "_".join(dataset.split("_")[:2])
    le = LabelEncoder()
    le.fit(["Inactive", "Active"])
    df["Label"] = le.transform(df["Label"])
    X = np.array([get_fps(smi) for smi in tqdm(df["SMILES"].tolist(), desc=f'{dataset} featurize')])
    y = df["Label"].to_numpy()

    sp = ShuffleSplit(n_splits=10, test_size=0.1)
    overall_res[df_name] = {}

    for i, (train_idx, test_idx) in tqdm(enumerate(sp.split(X, y)), total=10, desc=f'{dataset} CV',):
        overall_res[df_name][i] = {}
        test_X = X[test_idx, :]
        test_y = y[test_idx]
        train_X = X[train_idx, :]
        train_y = y[train_idx]

        clf_unbalanced = XGBClassifier(n_estimators=300)
        clf_balanced = XGBClassifier(n_estimators=300)

        us_train_X, us_train_y = rus.fit_resample(train_X, train_y)
        us_test_X, us_test_y = rus.fit_resample(test_X, test_y)

        clf_balanced.fit(us_train_X, us_train_y)
        clf_unbalanced.fit(train_X, train_y)

        # naming convention is <training set balanced/unbalanced>_<test set balanced/unbalanced>
        # the paper only looks at unbalanced test sets, so only III (balanced train) and IV (unbalanced train)
        y_pred_bal_unbal = clf_balanced.predict(test_X)  # III
        y_pred_unbal_unbal = clf_unbalanced.predict(test_X)  # IV
        y_pred_unbal_bal = clf_unbalanced.predict(us_test_X)  # II
        y_pred_bal_bal = clf_balanced.predict(us_test_X)  # I

        y_pred_prob_bal_unbal = clf_balanced.predict_proba(test_X)[:, 1]
        y_pred_prob_unbal_unbal = clf_unbalanced.predict_proba(test_X)[:, 1]
        y_pred_prob_unbal_bal = clf_unbalanced.predict_proba(us_test_X)[:, 1]
        y_pred_prob_bal_bal = clf_balanced.predict_proba(us_test_X)[:, 1]

        overall_res[df_name][i]["y_test"] = deepcopy(test_y)
        overall_res[df_name][i]["y_test_us"] = deepcopy(us_test_y)

        overall_res[df_name][i]["III"] = deepcopy(y_pred_bal_unbal)
        overall_res[df_name][i]["III_prob"] = deepcopy(y_pred_prob_bal_unbal)

        overall_res[df_name][i]["IV"] = deepcopy(y_pred_unbal_unbal)
        overall_res[df_name][i]["IV_prob"] = deepcopy(y_pred_prob_unbal_unbal)

        overall_res[df_name][i]["II"] = deepcopy(y_pred_unbal_bal)
        overall_res[df_name][i]["II_prob"] = deepcopy(y_pred_prob_unbal_bal)

        overall_res[df_name][i]["I"] = deepcopy(y_pred_bal_bal)
        overall_res[df_name][i]["I_prob"] = deepcopy(y_pred_prob_bal_bal)

        overall_res[df_name][i]["I_AUCROC"] = roc_auc_score(us_test_y, y_pred_prob_bal_bal)
        overall_res[df_name][i]["II_AUCROC"] = roc_auc_score(us_test_y, y_pred_prob_unbal_bal)
        overall_res[df_name][i]["III_AUCROC"] = roc_auc_score(test_y, y_pred_prob_bal_unbal)
        overall_res[df_name][i]["IV_AUCROC"] = roc_auc_score(test_y, y_pred_prob_unbal_unbal)

        overall_res[df_name][i]["I_BA"] = balanced_accuracy_score(us_test_y, y_pred_bal_bal)
        overall_res[df_name][i]["II_BA"] = balanced_accuracy_score(us_test_y, y_pred_unbal_bal)
        overall_res[df_name][i]["III_BA"] = balanced_accuracy_score(test_y, y_pred_bal_unbal)
        overall_res[df_name][i]["IV_BA"] = balanced_accuracy_score(test_y, y_pred_unbal_unbal)

        overall_res[df_name][i]["I_PPV"] = precision_score(us_test_y, y_pred_bal_bal)
        overall_res[df_name][i]["II_PPV"] = precision_score(us_test_y, y_pred_unbal_bal)
        overall_res[df_name][i]["III_PPV"] = precision_score(test_y, y_pred_bal_unbal)
        overall_res[df_name][i]["IV_PPV"] = precision_score(test_y, y_pred_unbal_unbal)

        overall_res[df_name][i]["I_BEDROC_20"] = bedroc_score(us_test_y, y_pred_bal_bal)
        overall_res[df_name][i]["II_BEDROC_20"] = bedroc_score(us_test_y, y_pred_unbal_bal)
        overall_res[df_name][i]["III_BEDROC_20"] = bedroc_score(test_y, y_pred_bal_unbal)
        overall_res[df_name][i]["IV_BEDROC_20"] = bedroc_score(test_y, y_pred_unbal_unbal)

        overall_res[df_name][i]["I_BEDROC_100"] = bedroc_score(us_test_y, y_pred_bal_bal, alpha=100)
        overall_res[df_name][i]["II_BEDROC_100"] = bedroc_score(us_test_y, y_pred_unbal_bal, alpha=100)
        overall_res[df_name][i]["III_BEDROC_100"] = bedroc_score(test_y, y_pred_bal_unbal, alpha=100)
        overall_res[df_name][i]["IV_BEDROC_100"] = bedroc_score(test_y, y_pred_unbal_unbal, alpha=100)

        overall_res[df_name][i]['plate_picks_bal'] = overall_res[df_name][i]["y_test"][
            np.argpartition(overall_res[df_name][i]["III_prob"], len(overall_res[df_name][i]["y_test"]) - 128)[-128:]]
        overall_res[df_name][i]['plate_picks_unbal'] = overall_res[df_name][i]["y_test"][
            np.argpartition(overall_res[df_name][i]["IV_prob"], len(overall_res[df_name][i]["y_test"]) - 128)[-128:]]

    for i in ["I", "II", "III", "IV"]:
        overall_res[df_name][f"{i}_avg_AUCROC"] = np.mean([overall_res[df_name][_][f'{i}_AUCROC'] for _ in range(10)])
        overall_res[df_name][f"{i}_avg_BA"] = np.mean([overall_res[df_name][_][f'{i}_BA'] for _ in range(10)])
        overall_res[df_name][f"{i}_avg_PPV"] = np.mean([overall_res[df_name][_][f'{i}_PPV'] for _ in range(10)])
        overall_res[df_name][f"{i}_avg_BEDROC_20"] = np.mean([overall_res[df_name][_][f'{i}_BEDROC_20'] for _ in range(10)])
        overall_res[df_name][f"{i}_avg_BEDROC_100"] = np.mean([overall_res[df_name][_][f'{i}_BEDROC_100'] for _ in range(10)])

        overall_res[df_name][f"{i}_std_AUCROC"] = np.std([overall_res[df_name][_][f'{i}_AUCROC'] for _ in range(10)])
        overall_res[df_name][f"{i}_std_BA"] = np.std([overall_res[df_name][_][f'{i}_BA'] for _ in range(10)])
        overall_res[df_name][f"{i}_std_PPV"] = np.std([overall_res[df_name][_][f'{i}_PPV'] for _ in range(10)])
        overall_res[df_name][f"{i}_std_BEDROC_20"] = np.std([overall_res[df_name][_][f'{i}_BEDROC_20'] for _ in range(10)])
        overall_res[df_name][f"{i}_std_BEDROC_100"] = np.std([overall_res[df_name][_][f'{i}_BEDROC_100'] for _ in range(10)])

    pick_rates_bal = []
    pick_rates_unbal = []
    for fold in range(10):
        pick_rates_bal.append(overall_res[df_name][fold]["plate_picks_bal"].sum() / 128)
        pick_rates_unbal.append(overall_res[df_name][fold]["plate_picks_unbal"].sum() / 128)
    overall_res[df_name]["avg_plate_ppv_bal"] = np.mean(pick_rates_bal)
    overall_res[df_name]["avg_plate_ppv_unbal"] = np.mean(pick_rates_unbal)
    overall_res[df_name]["std_plate_ppv_bal"] = np.std(pick_rates_bal)
    overall_res[df_name]["std_plate_ppv_unbal"] = np.std(pick_rates_unbal)


with open("./results_xgboost_v2_bedroc.pkl", "wb") as f:
    pickle.dump(overall_res, f)
