import json
import re
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


def clean_for_json(obj: Any) -> Any:
    """
    JSON serileştirme için veriyi temizler.

    Args:
        obj: Temizlenecek veri

    Returns:
        Temizlenmiş veri
    """
    if isinstance(obj, str):
        # Geçersiz karakterleri temizle
        return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", obj)

    if isinstance(obj, (int, float, bool, type(None))):
        return obj

    if isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]

    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}

    return str(obj)


def safe_convert_to_json_serializable(
    obj: Any, max_depth: int = 100
) -> Union[Dict, List, float, str, int]:
    """
    Numpy array'leri ve diğer özel tipleri JSON serileştirilebilir formata çevirir.

    Args:
        obj: Çevrilecek obje
        max_depth: Maksimum özyineleme derinliği

    Returns:
        JSON serileştirilebilir formatta veri
    """
    if max_depth <= 0:
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, dict):
        return {
            str(k): safe_convert_to_json_serializable(v, max_depth - 1)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [safe_convert_to_json_serializable(item, max_depth - 1) for item in obj]

    if hasattr(obj, "__dict__"):
        return safe_convert_to_json_serializable(obj.__dict__, max_depth - 1)

    # Eğer serileştirilemeyen bir tip ise string'e çevir
    try:
        json.dumps(obj)
        return obj
    except:
        return str(obj)


def save_to_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Veriyi JSON dosyasına kaydeder.

    Args:
        data: Kaydedilecek veri
        filepath: Kaydedilecek dosya yolu
        indent: JSON formatındaki girinti sayısı
    """
    # Önce veriyi temizle
    cleaned_data = clean_for_json(data)
    # Sonra JSON serileştirilebilir formata çevir
    json_serializable = safe_convert_to_json_serializable(cleaned_data)

    try:
        with open(filepath, "w") as f:
            json.dump(json_serializable, f, indent=indent)
    except Exception as e:
        print(f"JSON kaydetme hatası: {e}")
        # Hata durumunda daha basit bir format dene
        simplified_data = {
            "error": "Orijinal veri kaydedilemedi",
            "data_str": str(data),
        }
        with open(filepath, "w") as f:
            json.dump(simplified_data, f, indent=indent)


def load_from_json(filepath: str) -> Any:
    """
    JSON dosyasından veri yükler.

    Args:
        filepath: Yüklenecek dosya yolu

    Returns:
        Yüklenen veri
    """
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"JSON okuma hatası: {e}")
        return None


def expand_pathway_metrics(df, metrics_column="pathway_metrics"):
    """
    DataFrame'deki pathway_metrics sütunundaki dictionary'leri ayrı sütunlara ayırır.

    Args:
        df: pandas DataFrame
        metrics_column: Dictionary içeren sütun adı

    Returns:
        Genişletilmiş DataFrame
    """
    # Boş bir DataFrame oluştur
    expanded_df = pd.DataFrame()

    # Her bir pathway için metrikleri topla
    all_metrics = []

    for metrics_list in df[metrics_column]:
        # Her bir pathway için dictionary'leri DataFrame'e çevir
        pathway_df = pd.DataFrame(metrics_list)

        # pathway_idx'e göre pivot
        pivoted = pathway_df.set_index("pathway_idx")

        # Metrikleri listeye ekle
        all_metrics.append(pivoted)

    # Tüm metrikleri birleştir
    metrics_df = pd.concat(all_metrics, axis=0)

    # pathway ve metrik kombinasyonlarını sütun adı olarak kullan
    for col in metrics_df.columns:
        for idx in metrics_df.index.unique():
            col_name = f"pathway_{idx}_{col}"
            expanded_df[col_name] = metrics_df.loc[idx, col]

    # Orijinal DataFrame ile birleştir
    result_df = pd.concat([df.drop(columns=[metrics_column]), expanded_df], axis=1)

    return result_df
