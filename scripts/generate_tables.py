# %%
import psycopg

from deep_metabolitics.config import raw_data_dir
from deep_metabolitics.utils.utils import get_queries

conn = psycopg.connect(
    host="localhost", dbname="postgres", user="baris", password="123456"
)

# %%
queries = get_queries(fpath=raw_data_dir / "prepared_ddl.sql")
cur = conn.cursor()
for query in queries:
    cur.execute(query=query)
cur.close()

# %%
conn.commit()

# %%
queries = get_queries(fpath=raw_data_dir / "_user__202408140931.sql")
cur = conn.cursor()
for query in queries:
    cur.execute(query=query)
cur.close()
conn.commit()

# %%
queries = get_queries(fpath=raw_data_dir / "methods_202408140933.sql")
cur = conn.cursor()
for query in queries:
    cur.execute(query=query)
cur.close()
conn.commit()

# %%
queries = get_queries(fpath=raw_data_dir / "diseases_202408140935.sql")
cur = conn.cursor()
for query in queries:
    cur.execute(query=query)
cur.close()
conn.commit()

# %%
queries = get_queries(fpath=raw_data_dir / "metabolomicsdata_202408140934.sql")
cur = conn.cursor()
for query in queries:
    cur.execute(query=query)
cur.close()
conn.commit()

# %%
queries = get_queries(fpath=raw_data_dir / "diseasemodels_202408140935.sql")
cur = conn.cursor()
for query in queries:
    cur.execute(query=query)
cur.close()
conn.commit()

# %%
queries = get_queries(fpath=raw_data_dir / "datasets_202408140935.sql")
cur = conn.cursor()
for query in queries:
    cur.execute(query=query)
cur.close()
conn.commit()

# %%
queries = get_queries(fpath=raw_data_dir / "analysis_202408140936.sql")
cur = conn.cursor()
for query in queries[0].split(";"):
    cur.execute(query=query)
    conn.commit()
cur.close()

# %%
conn.close()
