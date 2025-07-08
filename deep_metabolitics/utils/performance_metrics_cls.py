import os
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import mannwhitneyu
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn_utils.utils import feature_importance_report

from deep_metabolitics.config import outputs_dir
from deep_metabolitics.utils.trainer import cls_results


def mann_whitney_feature_selection(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Her feature için Mann-Whitney U testi uygular.
    
    Parametreler:
        X (pd.DataFrame): Feature'lar (sadece sayısal değerler içermeli)
        y (pd.Series): Binary target (0 ve 1 değerlerini içermeli)

    Dönüş:
        pd.DataFrame: ['feature', 'p_value', 'mean_diff'] sütunlarını içeren sıralı sonuç
    """
    results = []

    for feature in X.columns:
        group1 = X.loc[y == 0, feature].dropna()
        group2 = X.loc[y == 1, feature].dropna()

        # Eğer her iki grup da en az 2 örnek içermiyorsa test yapma
        if len(group1) < 2 or len(group2) < 2:
            continue

        try:
            stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
            mean_diff = np.abs(group1.mean() - group2.mean())
            results.append({
                'feature': feature,
                'pval': p,
                'mean_diff': mean_diff
            })
        except Exception as e:
            print(f"Feature '{feature}' için test yapılamadı: {e}")

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by='pval', ascending=True).reset_index(drop=True)
    result_df = result_df.set_index("feature")
    
    return result_df

def jaccard_similarity(set1, set2):
    set1 = list(set1)
    set1 = [val.replace("_min", "").replace("_max", "") for val in set1]
    set1 = set(set1)
    
    set2 = list(set2)
    set2 = [val.replace("_min", "").replace("_max", "") for val in set2]
    set2 = set(set2)
    
    
    n_intersection = len(
            set1.intersection(
                set2
            )
        )

    n_union = len(
        set1.union(
            set2
        )
    )
    if n_union == 0:
        return 0
    jaccard_sim = n_intersection / n_union
    return jaccard_sim

# NumPy veya PyTorch tensörünü NumPy'ye çevir
def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


# Değerlendirme fonksiyonu (target isimlerine göre hesaplama yapar)
def evaluate_model(
    y_true, y_pred, factors_df, target_names: List[str], dataset_type: str, scaler=None, k_folds=10
):

    
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_indices = list(kf.split(y_true, factors_df))
    df_list = []
    for fold, fold_index in enumerate(fold_indices):
        metrics = {}
        train_indices = fold_index[0]
        test_indices = fold_index[1]

        X_true_train, X_true_test = y_true[train_indices], y_true[test_indices] # for ground truth
        X_pred_train, X_pred_test = y_pred[train_indices], y_pred[test_indices] # for new approach
        
        X_true_train = np.rint(X_true_train).astype(int)
        X_true_test = np.rint(X_true_test).astype(int)
        X_pred_train = np.rint(X_pred_train).astype(int)
        X_pred_test = np.rint(X_pred_test).astype(int)

        y_train, y_test = factors_df.iloc[train_indices], factors_df.iloc[test_indices]
        
        true_metrics, true_feature_importance_df = cls_results(X_train=pd.DataFrame(X_true_train, columns=target_names), y_train=y_train, X_test=pd.DataFrame(X_true_test, columns=target_names), y_test=y_test)
        true_metrics = {f"true_{key}": value for key, value in true_metrics.items()}
        metrics.update(true_metrics)
        pred_metrics, pred_feature_importance_df = cls_results(X_train=pd.DataFrame(X_pred_train, columns=target_names), y_train=y_train, X_test=pd.DataFrame(X_pred_test, columns=target_names), y_test=y_test)
        pred_metrics = {f"pred_{key}": value for key, value in pred_metrics.items()}
        metrics.update(pred_metrics)
        
        jacc_sim = jaccard_similarity(set(true_feature_importance_df.sort_values(by="importance", ascending=False).head(30)["feature"].tolist()), 
                                      set(pred_feature_importance_df.sort_values(by="importance", ascending=False).head(30)["feature"].tolist()))
        metrics["jaccard_similarity_30"] = jacc_sim
        
        jacc_sim = jaccard_similarity(set(true_feature_importance_df.sort_values(by="shap_importance", ascending=False).head(30)["feature"].tolist()), 
                                      set(pred_feature_importance_df.sort_values(by="shap_importance", ascending=False).head(30)["feature"].tolist()))
        metrics["shap_jaccard_similarity_30"] = jacc_sim
        
        jacc_sim = jaccard_similarity(set(true_feature_importance_df.sort_values(by="importance", ascending=False).head(60)["feature"].tolist()), 
                                      set(pred_feature_importance_df.sort_values(by="importance", ascending=False).head(60)["feature"].tolist()))
        metrics["jaccard_similarity_60"] = jacc_sim
        
        jacc_sim = jaccard_similarity(set(true_feature_importance_df.sort_values(by="shap_importance", ascending=False).head(60)["feature"].tolist()), 
                                      set(pred_feature_importance_df.sort_values(by="shap_importance", ascending=False).head(60)["feature"].tolist()))
        metrics["shap_jaccard_similarity_60"] = jacc_sim
        
        jacc_sim = jaccard_similarity(set(true_feature_importance_df.sort_values(by="importance", ascending=False).head(98)["feature"].tolist()), 
                                      set(pred_feature_importance_df.sort_values(by="importance", ascending=False).head(98)["feature"].tolist()))
        metrics["jaccard_similarity_98"] = jacc_sim
        
        jacc_sim = jaccard_similarity(set(true_feature_importance_df.sort_values(by="shap_importance", ascending=False).head(98)["feature"].tolist()), 
                                      set(pred_feature_importance_df.sort_values(by="shap_importance", ascending=False).head(98)["feature"].tolist()))
        metrics["shap_jaccard_similarity_98"] = jacc_sim
        
        
        temp_df = pd.DataFrame([metrics])
        df_list.append(temp_df)
    df = pd.concat(df_list)
    return df, true_feature_importance_df, pred_feature_importance_df


class PerformanceMetrics:
    def __init__(
        self,
        target_names,
        experience_name,
        ds_name,
        scaler=None,
    ):
        self.target_names = target_names
        self.experience_name = experience_name
        self.ds_name = ds_name
        self.scaler = scaler
        self.out_dir = outputs_dir / "cls_metrics_cast_int_shap" / self.experience_name
        self.out_dir.mkdir(exist_ok=True, parents=True)

    def test_metric(self, y_true, y_pred, factors_df):
        from sklearn.preprocessing import StandardScaler
        y_true = to_numpy(y_true)
        y_pred = to_numpy(y_pred)
        
        if self.scaler is not None:
            print(f"{y_true.shape = }")
            print(f"{y_pred.shape = }")
            y_true = self.scaler.inverse_transform(y_true)
            y_pred = self.scaler.inverse_transform(y_pred)

        factors_df["Factors"] = (factors_df["Factors"] != "healthy").astype(int)

        df_cls_metrics, true_feature_importance_df, pred_feature_importance_df = evaluate_model(
            y_true=y_true,
            y_pred=y_pred,
            factors_df=factors_df,
            target_names=self.target_names,
            dataset_type="test",
            scaler=self.scaler,
        )
        true_feature_importance_df.to_csv(self.out_dir / f"featureimportance_cls_{self.ds_name}_true.csv")
        pred_feature_importance_df.to_csv(self.out_dir / f"featureimportance_cls_{self.ds_name}_pred.csv")
        
        
        df_cls_metrics.to_csv(os.path.join(self.out_dir, f"cls_{self.ds_name}.csv"))
        
        try:
            tfi = feature_importance_report(
                pd.DataFrame(y_true, columns=self.target_names),
                factors_df,
                threshold=0,
            )
            tfi.to_csv(self.out_dir / f"featureimportance_{self.ds_name}_true.csv")

            pfi = feature_importance_report(
                pd.DataFrame(y_pred, columns=self.target_names),
                factors_df,
                threshold=0,
            )
            pfi.to_csv(self.out_dir / f"featureimportance_{self.ds_name}_pred.csv")
            
            P_TH = 0.05
            
            true_selected = tfi[tfi["pval"] <= P_TH].index
            pred_selected = pfi[pfi["pval"] <= P_TH].index
            similarity = jaccard_similarity(true_selected, pred_selected)
            sim_df = pd.DataFrame([{"jaccard_similarity": similarity, "true_selected": len(true_selected), "pred_selected": len(pred_selected)}])
            sim_df.to_csv(self.out_dir / f"jaccard_similarity_{self.ds_name}.csv")

            print(f"Jaccard similarity: {similarity}")
            print(f"True selected: {len(true_selected)}")
            print(f"Pred selected: {len(pred_selected)}")
        except Exception as e:
            print(f"{e = }")
            similarity = 0
            sim_df = pd.DataFrame([{"jaccard_similarity": similarity, "true_selected": 0, "pred_selected": 0}])
            sim_df.to_csv(self.out_dir / f"jaccard_similarity_{self.ds_name}.csv")
            
        
        # try:
        tfi = mann_whitney_feature_selection(
            pd.DataFrame(y_true, columns=self.target_names),
            factors_df["Factors"],
        )
        tfi.to_csv(self.out_dir / f"featureimportance_mannwhitneyu_{self.ds_name}_true.csv")

        pfi = mann_whitney_feature_selection(
            pd.DataFrame(y_pred, columns=self.target_names),
            factors_df["Factors"],
        )
        pfi.to_csv(self.out_dir / f"featureimportance_mannwhitneyu_{self.ds_name}_pred.csv")
        
        P_TH = 0.05
        
        true_selected = tfi[tfi["pval"] <= P_TH].index
        pred_selected = pfi[pfi["pval"] <= P_TH].index
        similarity = jaccard_similarity(true_selected, pred_selected)
        sim_df = pd.DataFrame([{"jaccard_similarity": similarity, "true_selected": len(true_selected), "pred_selected": len(pred_selected)}])
        sim_df.to_csv(self.out_dir / f"jaccard_similarity_mannwhitneyu_{self.ds_name}.csv")

        print(f"Jaccard similarity: {similarity}")
        print(f"True selected: {len(true_selected)}")
        print(f"Pred selected: {len(pred_selected)}")
        # except Exception as e:
        #     print(f"{e = }")
        #     similarity = 0
        #     sim_df = pd.DataFrame([{"jaccard_similarity": similarity, "true_selected": 0, "pred_selected": 0}])
        #     sim_df.to_csv(self.out_dir / f"jaccard_similarity_mannwhitneyu_{self.ds_name}.csv")
