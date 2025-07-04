from setuptools import setup, find_packages

setup(
    name="deep_metabolitics",
    version="0.1",
    packages=find_packages(include=["deep_metabolitics"]),  # İstediğiniz paketleri ekleyin
    # veya: exclude=["logs", "temp", "data", "outputs", "notebooks", "oneleaveout_results"]
    # Diğer ayarlar...
)