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
        "SILVER_PATH",
        "GOLD_PATH",
    ]
)

BUCKET = args["BUCKET"]
SILVER_PATH = args["SILVER_PATH"]
GOLD_PATH = args["GOLD_PATH"]

# ============================================
# CONTEXTO GLUE / SPARK
# ============================================
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

logger.info("=" * 60)
logger.info("JOB GOLD - SERIE TEMPORAL Y PIVOT PARA CLUSTERING (HORA LOCAL MX)")
logger.info("=" * 60)
logger.info(f"Bucket: {BUCKET}")
logger.info(f"Silver Path: {SILVER_PATH}")
logger.info(f"Gold Path: {GOLD_PATH}")
logger.info("=" * 60)

# ============================================
# RUTAS DE ENTRADA / SALIDA
# ============================================
SILVER_INPUT_PATH = os.path.join(SILVER_PATH, "TRAFFIC_DATA_SILVER_PARQUET")
GOLD_OUTPUT_PATH = os.path.join(GOLD_PATH, "TRAFFIC_DATA_GOLD_TS_PIVOT_MX")

logger.info(f"Ruta de lectura Silver (Parquet): {SILVER_INPUT_PATH}")
logger.info(f"Ruta de escritura Gold (Parquet): {GOLD_OUTPUT_PATH}")

# ============================================
# LECTURA DE SILVER
# ============================================
logger.info("Leyendo datos de la capa Silver (Parquet)...")

df_silver = (
    spark.read
    .format("parquet")
    .load(SILVER_INPUT_PATH)
)

logger.info("Schema de Silver:")
df_silver.printSchema()

silver_count = df_silver.count()
logger.info(f"Registros leídos desde Silver: {silver_count}")

# ============================================
# ASEGURAR TIMESTAMP (UTC)
# ============================================
# Usamos datatime_ts si existe; si no, lo creamos desde datatime (yyyyMMddHHmmss)
if "datatime_ts" in df_silver.columns:
    df_ts = df_silver
else:
    logger.warning(
        "Columna 'datatime_ts' no encontrada. "
        "Intentando crearla desde 'datatime' con formato 'yyyyMMddHHmmss'. "
        "Se asume que los timestamps están en UTC."
    )
    df_ts = df_silver.withColumn(
        "datatime_ts",
        F.to_timestamp(F.col("datatime"), "yyyyMMddHHmmss")
    )

# ============================================
# CONVERTIR A HORA LOCAL MX (America/Mexico_City)
# ============================================
logger.info("Convirtiendo datatime_ts (UTC) a hora local America/Mexico_City...")

df_ts = df_ts.withColumn(
    "datatime_mx",
    F.from_utc_timestamp(F.col("datatime_ts"), "America/Mexico_City")
)

df_ts = df_ts.withColumn("hour_mx", F.hour(F.col("datatime_mx")))

logger.info("Horas locales (hour_mx) distintas en el dataset (antes del filtro):")
df_ts.select("hour_mx").distinct().orderBy("hour_mx").show(50, truncate=False)

# ============================================
# CREAR SERIE TEMPORAL EN HORA LOCAL
# ============================================
# Horas objetivo en hora local MX, según el enunciado:
# 7 am, 9 am, 11 am, 13 pm, 15 pm, 17 pm, 19 pm, 21 pm, 23 pm y 1 am
ALLOWED_HOURS_MX = [ 7, 9, 11, 13, 15, 17, 19, 21, 23]

logger.info(f"Horas locales MX usadas en la serie temporal: {ALLOWED_HOURS_MX}")

df_filtered = df_ts.filter(F.col("hour_mx").isin(ALLOWED_HOURS_MX))

filtered_count = df_filtered.count()
logger.info(f"Registros después de filtrar a horas locales {ALLOWED_HOURS_MX}: {filtered_count}")

logger.info("Horas locales presentes después del filtro:")
df_filtered.select("hour_mx").distinct().orderBy("hour_mx").show(50, truncate=False)

# ============================================
# AGREGACIÓN POR UBICACIÓN Y HORA LOCAL
# ============================================
# Definimos la métrica de tráfico a usar; aquí usamos diffuse_logic_traffic,
# ajusta si quieres usar otra (linear_color_weighting, etc.)
TRAFFIC_COL = "linear_color_weighting"

if TRAFFIC_COL not in df_filtered.columns:
    raise Exception(
        f"La columna de tráfico '{TRAFFIC_COL}' no existe en el DataFrame. "
        "Ajusta TRAFFIC_COL en el job Gold."
    )

logger.info(
    f"Agregando métrica de tráfico '{TRAFFIC_COL}' por ubicación (id) y hora local..."
)

df_filtered_num = df_filtered.withColumn(
    "traffic_value",
    F.col(TRAFFIC_COL).cast(T.DoubleType())
)

df_grouped = (
    df_filtered_num
    .groupBy("id", "hour_mx")
    .agg(
        F.avg("traffic_value").alias("traffic_mean"),
        F.first("Coordx", ignorenulls=True).alias("Coordx"),
        F.first("Coordy", ignorenulls=True).alias("Coordy"),
    )
)

logger.info("Ejemplo de datos agrupados por id y hour_mx:")
df_grouped.orderBy("id", "hour_mx").show(10, truncate=False)

# ============================================
# PIVOT: UNA FILA POR UBICACIÓN, UNA COLUMNA POR HORA LOCAL
# ============================================
logger.info("Realizando pivot para obtener una fila por ubicación y columnas por hora local...")

df_pivot = (
    df_grouped
    .groupBy("id", "Coordx", "Coordy")
    .pivot("hour_mx", ALLOWED_HOURS_MX)
    .agg(F.first("traffic_mean"))
)

logger.info("Schema después del pivot:")
df_pivot.printSchema()

logger.info("Ejemplo de la tabla pivot (una fila por ubicación, horas locales MX):")
df_pivot.show(10, truncate=False)

# ============================================
# RENOMBRAR COLUMNAS DE HORA A NOMBRES MÁS CLAROS
# ============================================
# De columnas [1,7,9,...] a [traffic_01, traffic_07, ...] (en hora local MX)
rename_expr = {}
for h in ALLOWED_HOURS_MX:
    col_name = str(h)
    if col_name in df_pivot.columns:
        new_name = f"traffic_{h:02d}"
        rename_expr[col_name] = new_name

for old, new in rename_expr.items():
    df_pivot = df_pivot.withColumnRenamed(old, new)

logger.info("Schema final después de renombrar columnas de hora local:")
df_pivot.printSchema()

gold_count = df_pivot.count()
logger.info(f"Total de ubicaciones (filas) en la tabla Gold MX: {gold_count}")

# ===========================================
# ESCRITURA A GOLD (PARQUET)
# ===========================================
logger.info("Escribiendo datos pivotados (hora local MX) a la capa Gold en formato Parquet...")

dynamic_frame_gold = DynamicFrame.fromDF(
    df_pivot,
    glueContext,
    "dynamic_frame_gold_trafico_mx"
)

glueContext.write_dynamic_frame.from_options(
    frame=dynamic_frame_gold,
    connection_type="s3",
    connection_options={
        "path": GOLD_OUTPUT_PATH,
    },
    format="parquet",
    format_options={
        "compression": "snappy"
    }
)

logger.info("Datos escritos exitosamente en la capa Gold (Parquet, hora local MX).")
logger.info(f"Ubicación Gold: {GOLD_OUTPUT_PATH}")

job.commit()
logger.info("Job Gold (MX) completado exitosamente ✅")
logger.info("Dataset listo para clustering de series temporales por ubicación en hora local.")
