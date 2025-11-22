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
        # opcional: número de clusters
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

logger.info("=== JOB CLUSTERING GOLD - INICIO ===")
logger.info(f"GOLD_PATH: {GOLD_PATH}")
logger.info(f"OUTPUT_PATH: {OUTPUT_PATH}")
logger.info(f"Clusters (K): {K}")

# ============================
# Leer datos Gold
# ============================
# Ejemplo: GOLD_PATH = s3://bucket/gold/TRAFFIC_DATA_GOLD_TS_PIVOT_ADAPTED
logger.info("Leyendo datos de la capa Gold...")
df_gold = spark.read.parquet(GOLD_PATH)

logger.info("Schema de Gold:")
df_gold.printSchema()

logger.info(f"Total de ubicaciones en Gold: {df_gold.count()}")

# ============================
# Detectar columnas de serie temporal
# ============================
# Intentamos primero columnas que empiecen con 'traffic_'
feature_cols = [c for c in df_gold.columns if c.startswith("traffic_")]

# Si por cualquier cosa tus columnas son solo "1","3","5"... usamos esa lista
if not feature_cols:
    posibles_horas = ["1","3","5","7","13","15","17","19","21","23"]
    feature_cols = [c for c in df_gold.columns if c in posibles_horas]

if not feature_cols:
    raise Exception("No se encontraron columnas de serie temporal (traffic_XX o horas).")

logger.info(f"Columnas de serie temporal usadas como features: {feature_cols}")

# ============================
# Preparar datos para clustering
# ============================
# Nos aseguramos que las features sean Double y sin nulls (rellenamos con 0)
df_features = df_gold.select(
    "id", "Coordx", "Coordy", *feature_cols
)

for col_name in feature_cols:
    df_features = df_features.withColumn(
        col_name,
        F.col(col_name).cast(T.DoubleType())
    )

df_features = df_features.fillna(0.0, subset=feature_cols)

# VectorAssembler para armar el vector de características
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

df_vec = assembler.transform(df_features)

# StandardScaler para normalizar
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(df_vec)
df_scaled = scaler_model.transform(df_vec)

logger.info("Ejemplo de datos con vector de features:")
df_scaled.select("id", "features").show(5, truncate=False)

# ============================
# K-Means Clustering
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

logger.info("Ejemplo de asignación de clusters:")
df_clustered.select("id", "cluster_id", *feature_cols).show(10, truncate=False)

# Centros de cada cluster (para interpretación)
logger.info("Centroides de los clusters (en espacio escalado):")
for idx, center in enumerate(kmeans_model.clusterCenters()):
    logger.info(f"Cluster {idx} center: {center}")

# ============================
# Guardar resultados en Gold
# ============================
output_path = os.path.join(OUTPUT_PATH, "TRAFFIC_DATA_GOLD_CLUSTERS")

logger.info(f"Guardando resultados de clustering en: {output_path}")

(
    df_clustered
    .select("id", "Coordx", "Coordy", "cluster_id", *feature_cols)
    .write
    .mode("overwrite")
    .parquet(output_path)
)

logger.info("Resultados de clustering guardados correctamente.")

job.commit()
logger.info("=== JOB CLUSTERING GOLD - FIN ===")
