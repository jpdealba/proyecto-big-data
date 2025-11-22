import sys
import os
import logging

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

from pyspark.sql import functions as F
from pyspark.sql import types as T

# ============================================
# CONFIGURACIÓN DE LOGGING
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================
# PARÁMETROS DEL JOB
# ============================================
args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "BUCKET",
        "BRONZE_PATH",
        "SILVER_PATH",
    ]
)

BUCKET = args["BUCKET"]
BRONZE_PATH = args["BRONZE_PATH"]
SILVER_PATH = args["SILVER_PATH"]

# ============================================
# CONTEXTO GLUE / SPARK
# ============================================
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

logger.info("=" * 60)
logger.info("JOB SILVER - TRANSFORMACIÓN DE FECHAS (TRÁFICO)")
logger.info("=" * 60)
logger.info(f"Bucket: {BUCKET}")
logger.info(f"Bronze Path: {BRONZE_PATH}")
logger.info(f"Silver Path: {SILVER_PATH}")
logger.info("=" * 60)

# ============================================
# RUTAS DE ENTRADA / SALIDA
# ============================================

# Debe coincidir con lo que usaste en el job Bronze
# FILE_PATH_SAVE = os.path.join(BRONZE_PATH, "TRAFFIC_DATA_BRONZE_PARQUET")
BRONZE_INPUT_PATH = os.path.join(BRONZE_PATH, "TRAFFIC_DATA_BRONZE_PARQUET")
SILVER_OUTPUT_PATH = os.path.join(SILVER_PATH, "TRAFFIC_DATA_SILVER_PARQUET")

logger.info(f"Ruta de lectura Bronze (Parquet): {BRONZE_INPUT_PATH}")
logger.info(f"Ruta de escritura Silver (Parquet): {SILVER_OUTPUT_PATH}")

# ============================================
# LECTURA DE BRONZE
# ============================================
logger.info("Leyendo datos de la capa Bronze (Parquet)...")

df_bronze = (
    spark.read
    .format("parquet")
    .load(BRONZE_INPUT_PATH)
)

logger.info("Schema de Bronze:")
df_bronze.printSchema()

bronze_count = df_bronze.count()
logger.info(f"Registros leídos desde Bronze: {bronze_count}")

# Esperamos columnas: id, predominant_color, exponential_color_weighting,
# linear_color_weighting, diffuse_logic_traffic, Coordx, Coordy, datatime, year_partition (de Bronze)
# Si algo cambia, puedes ajustar.

# ============================================
# TRANSFORMACIÓN DE FECHA/HORA
# ============================================
logger.info("Iniciando transformación de la columna 'datatime' a formato timestamp estándar...")

# 1) Convertimos datatime (string 'yyyyMMddHHmmss') a timestamp
df_silver = df_bronze.withColumn(
    "datatime_ts",
    F.to_timestamp(F.col("datatime"), "yyyyMMddHHmmss")
)

# 2) (Opcional) Creamos un string ISO 8601, si quieres algo listo para consumo externo
df_silver = df_silver.withColumn(
    "datatime_iso",
    F.date_format(F.col("datatime_ts"), "yyyy-MM-dd'T'HH:mm:ss")
)

# 3) Re-creamos la partición de año desde el timestamp limpio
df_silver = df_silver.withColumn(
    "year_partition",
    F.year(F.col("datatime_ts"))
)

logger.info("Schema después de agregar columnas de fecha limpia:")
df_silver.printSchema()

silver_count = df_silver.count()
logger.info(f"Registros después de la transformación de fechas: {silver_count}")

# Sanity check: no deberías perder registros
if silver_count != bronze_count:
    logger.warning(
        f"⚠️ Conteo distinto entre Bronze ({bronze_count}) y Silver ({silver_count}). "
        "Revisar transformaciones."
    )

# ============================================
# ESCRITURA A SILVER (PARQUET)
# ============================================
logger.info("Escribiendo datos transformados a la capa Silver en formato Parquet...")

# Reparticionamos por year_partition para tener archivos ordenados
df_silver_partitioned = df_silver.repartition(30, "year_partition")

dynamic_frame_silver = DynamicFrame.fromDF(
    df_silver_partitioned,
    glueContext,
    "dynamic_frame_silver_trafico"
)

glueContext.write_dynamic_frame.from_options(
    frame=dynamic_frame_silver,
    connection_type="s3",
    connection_options={
        "path": SILVER_OUTPUT_PATH,
        "partitionKeys": ["year_partition"],
    },
    format="parquet",
    format_options={
        "compression": "snappy"
    }
)

logger.info("Datos escritos exitosamente en la capa Silver (Parquet).")
logger.info(f"Ubicación Silver: {SILVER_OUTPUT_PATH}")

job.commit()
logger.info("Job Silver completado exitosamente ✅")
