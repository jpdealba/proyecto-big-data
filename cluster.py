import sys
import os
import logging

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

# ============================
# Logging
# ============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# Parámetros del Job
# ============================
args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "GOLD_PATH",
        "OUTPUT_PATH",
        "K"
    ]
)

GOLD_PATH = args["GOLD_PATH"]
OUTPUT_PATH = args["OUTPUT_PATH"]
K = int(args["K"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

logger.info("=== JOB CLUSTERING GOLD (MX) - INICIO ===")
logger.info(f"GOLD_PATH: {GOLD_PATH}")
logger.info(f"OUTPUT_PATH: {OUTPUT_PATH}")
logger.info(f"K (Clusters): {K}")

# ============================
# Leer Gold MX
# ============================
logger.info("Leyendo datos desde GOLD (MX)...")

# Ejemplo GOLD_PATH:
# s3://bucket/gold/TRAFFIC_DATA_GOLD_TS_PIVOT_MX/
df_gold = spark.read.parquet(GOLD_PATH)

logger.info("Schema de Gold MX:")
df_gold.printSchema()

num_rows = df_gold.count()
logger.info(f"Total de filas en Gold MX: {num_rows}")

# ============================
# Detectar columnas de serie temporal (traffic_XX)
# ============================
feature_cols = [c for c in df_gold.columns if c.startswith("traffic_")]

if not feature_cols:
    raise Exception(
        "ERROR: No se encontraron columnas traffic_XX en la capa Gold MX. "
        "Verifica que estés usando el dataset pivotado en hora local."
    )

logger.info(f"Columnas de serie temporal usadas: {feature_cols}")

# ============================
# Preparar datos para clustering
# ============================
df_features = df_gold.select(
    "id", "Coordx", "Coordy", *feature_cols
)

# Castear a double
for col_name in feature_cols:
    df_features = df_features.withColumn(
        col_name,
        F.col(col_name).cast(T.DoubleType())
    )

# Rellenar nulos con 0
df_features = df_features.fillna(0.0, subset=feature_cols)

# VectorAssembler
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

df_vec = assembler.transform(df_features)

# StandardScaler
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(df_vec)
df_scaled = scaler_model.transform(df_vec)

logger.info("Ejemplo de features normalizados:")
df_scaled.select("id", "features").show(5, truncate=False)

# ============================
# K-Means
# ============================
logger.info("Entrenando modelo K-Means...")

kmeans = KMeans(
    k=K,
    seed=42,
    featuresCol="features",
    predictionCol="cluster_id"
)

kmeans_model = kmeans.fit(df_scaled)
df_clustered = kmeans_model.transform(df_scaled)

logger.info("Clusters asignados (ejemplo):")
df_clustered.select("id", "cluster_id").show(10, truncate=False)

# Mostrar centroides
logger.info("Centroides de clusters (normalizados):")
for i, center in enumerate(kmeans_model.clusterCenters()):
    logger.info(f"Cluster {i}: {center}")

# ============================
# Guardar resultados
# ============================
output_path = os.path.join(OUTPUT_PATH, "TRAFFIC_DATA_GOLD_CLUSTERS_MX")

logger.info(f"Guardando resultados en: {output_path}")

(
    df_clustered
    .select("id", "Coordx", "Coordy", "cluster_id", *feature_cols)
    .write
    .mode("overwrite")
    .parquet(output_path)
)

logger.info("Resultados de clustering guardados correctamente.")

job.commit()
logger.info("=== JOB CLUSTERING GOLD (MX) - FIN ===")
