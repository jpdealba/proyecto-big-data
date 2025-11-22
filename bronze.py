import sys
import os
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql import types as T

# ============================================
# CONFIGURACIÓN DE LOGGING
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# INICIALIZACIÓN Y LECTURA DE PARÁMETROS
# ============================================
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'BUCKET',
    'BRONZE_PATH'
])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

BUCKET = args['BUCKET']
BRONZE_PATH = args['BRONZE_PATH']

logger.info("=" * 60)
logger.info("JOB BRONZE - CARGA DE DATOS EMPRESARIALES")
logger.info("=" * 60)
logger.info(f"Bucket: {BUCKET}")
logger.info(f"Raw Data Path: {BRONZE_PATH}")
logger.info("=" * 60)

# ============================================
# LEER DATOS CRUDOS (GENÉRICO)
# ============================================

def load_data(path: str):
    try:
        df_empresas = (
            spark.read.format("csv")
            .option("header", "true")
            .option("inferSchema", "false")
            .option("encoding", "UTF-8")
            .option("sep", ",")
            .option("quote", "\"")
            .option("escape", "\"")
            .option("multiLine", "true")
            .option("ignoreLeadingWhiteSpace", "true")
            .option("ignoreTrailingWhiteSpace", "true")
            .option("mode", "PERMISSIVE")
            .option("maxColumns", "50")
            .option("recursiveFileLookup", "true")
            .load(path)
        )
        
        logger.info(f"Datos cargados exitosamente desde CSV: {path}")
        logger.info(f"Columnas detectadas: {df_empresas.columns}")
        return df_empresas
    except Exception as e:
        logger.error(f"Error al leer datos desde {path}: {str(e)}")
        raise e

# ============================================
# VALIDACIONES BÁSICAS
# ============================================

def validate_dataframe(df):
    
    logger.info("Iniciando validaciones basicas de calidad de datos (muestra)")
    
    df_sample = df.sample(False, 0.01, seed=42)
    registros_sample = df_sample.count()
    
    logger.info(f"Validando con muestra de {registros_sample} registros (1% del total)")

    if registros_sample == 0:
        logger.warning("La muestra está vacía, se omiten validaciones.")
        return 0, 0

    logger.info("Conteo de valores NULL por columna (muestra 1%):")
    
    null_counts = df_sample.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c) 
        for c in df_sample.columns
    ]).collect()[0].asDict()
    
    columnas_con_nulos = 0
    for col_name, null_count in null_counts.items():
        if null_count > 0:
            porcentaje = (null_count / registros_sample) * 100
            logger.warning(f"Columna '{col_name}': ~{porcentaje:.2f}% valores NULL (estimado)")
            columnas_con_nulos += 1

    if columnas_con_nulos == 0:
        logger.info("No se detectaron valores NULL significativos")
    else:
        logger.warning(f"Total de columnas con valores NULL: {columnas_con_nulos}")

    logger.info("Analizando duplicados (muestra 1%)")
    registros_unicos_sample = df_sample.dropDuplicates().count()
    duplicados_sample = registros_sample - registros_unicos_sample

    if duplicados_sample > 0:
        porcentaje_dup = (duplicados_sample / registros_sample) * 100
        logger.warning(f"Duplicados en muestra: {duplicados_sample} (~{porcentaje_dup:.2f}%)")
    else:
        logger.info("No se detectaron duplicados en la muestra")
    
    return columnas_con_nulos, duplicados_sample

# ============================================
# GUARDAR EN FORMATO PARQUET (BRONZE)
# ============================================

def save_dataframe(df, path):
    logger.info("Guardando datos en formato Parquet")

    try:
        df_with_partition = df.withColumn(
            'year_partition',
            F.year(F.to_timestamp(F.col('datatime'), 'yyyyMMddHHmmss'))
        )
        
        df_repartitioned = df_with_partition.repartition(30, 'year_partition')
        
        dynamic_frame = DynamicFrame.fromDF(df_repartitioned, glueContext, "dynamic_frame_trafico")
        
        glueContext.write_dynamic_frame.from_options(
            frame=dynamic_frame,
            connection_type="s3",
            connection_options={
                "path": path,
                "partitionKeys": ["year_partition"]
            },
            format="parquet",
            format_options={
                "compression": "snappy"
            }
        )
        
        logger.info("Datos guardados exitosamente en formato Parquet")
        logger.info(f"Ubicación: {path}")
        logger.info("Datos particionados por year_partition (año de datatime)")
        
    except Exception as e:
        logger.error(f"Error al guardar datos en Parquet: {str(e)}")
        raise e

# ============================================
# HELPERS ESPECÍFICOS DE TRÁFICO
# ============================================

def load_traffic_with_datetime(path: str):
    """
    Carga los CSV de histórico de tráfico desde un path
    y genera la columna 'datatime' a partir del nombre del archivo.
    """
    df = load_data(path)

    # Añadimos columna con el nombre de archivo
    df = df.withColumn("_source_file", F.input_file_name())

    # Extraemos 14 dígitos consecutivos (yyyyMMddHHmmss) del nombre del archivo
    df = df.withColumn(
        "datatime",
        F.regexp_extract(F.col("_source_file"), r"([0-9]{14})", 1)
    )

    # Aseguramos que id sea entero (igual que en tu notebook)
    df = df.withColumn("id", F.col("id").cast(T.IntegerType()))

    logger.info(f"Ejemplo de datatime generado en {path}:")
    df.select("id", "datatime", "_source_file").show(5, truncate=False)

    return df


def load_locations_union(path_2024: str, path_2025: str):
    """
    Carga y unifica los locationPoints de 2024 y 2025.
    Usa exactamente la lógica que compartiste para renombrar y castear.
    """

    loc_2024 = load_data(path_2024)
    loc_2025 = load_data(path_2025)

    loc_df = loc_2024.unionByName(loc_2025, allowMissingColumns=True)

    logger.info("Schema original de locationPoints unificados:")
    loc_df.printSchema()

    # === LÓGICA QUE TE FUNCIONABA EN EL NOTEBOOK ===
    renames = {}
    for c in loc_df.columns:
        lc = c.lower()
        if lc in ["coordx", "lon", "longitude", "longitud"]:
            renames[c] = "Coordx"
        if lc in ["coordy", "lat", "latitude", "latitud"]:
            renames[c] = "Coordy"
        if lc != "id" and "id" in lc and "id" not in renames:
            # por si viniera "Id" o similar
            renames[c] = "id"
    for old, new in renames.items():
        loc_df = loc_df.withColumnRenamed(old, new)

    loc_df = loc_df.select([c for c in loc_df.columns if c in ["id", "Coordx", "Coordy"]])
    loc_df = (
        loc_df
        .withColumn("id", F.col("id").cast(T.IntegerType()))
        .withColumn("Coordx", F.col("Coordx").cast(T.DoubleType()))
        .withColumn("Coordy", F.col("Coordy").cast(T.DoubleType()))
    )

    # Consolidar por id tomando el primer valor no-nulo
    loc_df = (
        loc_df
        .groupBy("id")
        .agg(
            F.first("Coordx", ignorenulls=True).alias("Coordx"),
            F.first("Coordy", ignorenulls=True).alias("Coordy"),
        )
    )

    logger.info("Ejemplo de locations consolidados:")
    loc_df.show(5, truncate=False)

    return loc_df

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':

    # Paths específicos según tu estructura
    TRAFFIC_2024_PATH = f"{BRONZE_PATH}/catalogos/AMGTraffic2024/historico/"
    TRAFFIC_2025_PATH = f"{BRONZE_PATH}/catalogos/AMGtraffic2025/historico/"

    LOC_2024_PATH = f"{BRONZE_PATH}/catalogos/AMGTraffic2024/locationPoints.csv"
    LOC_2025_PATH = f"{BRONZE_PATH}/catalogos/AMGtraffic2025/locationPoints.csv"

    logger.info(f"Leyendo tráfico 2024 desde: {TRAFFIC_2024_PATH}")
    traffic_2024_df = load_traffic_with_datetime(TRAFFIC_2024_PATH)

    logger.info(f"Leyendo tráfico 2025 desde: {TRAFFIC_2025_PATH}")
    traffic_2025_df = load_traffic_with_datetime(TRAFFIC_2025_PATH)

    # Unimos histórico 2024 + 2025
    traffic_df = traffic_2024_df.unionByName(traffic_2025_df)

    logger.info("Schema de tráfico unificado (2024 + 2025):")
    traffic_df.printSchema()

    # Cargar y unir locationPoints con la lógica que ya funcionaba
    logger.info(f"Leyendo locationPoints 2024 desde: {LOC_2024_PATH}")
    logger.info(f"Leyendo locationPoints 2025 desde: {LOC_2025_PATH}")
    locations_df = load_locations_union(LOC_2024_PATH, LOC_2025_PATH)

    # Muy importante: asegurar que el tipo de id coincida (IntegerType) en ambos
    traffic_df = traffic_df.withColumn("id", F.col("id").cast(T.IntegerType()))

    # Join tráfico + ubicaciones
    logger.info("Realizando join entre tráfico e información de ubicaciones (locationPoints)")

    final_df = (
        traffic_df.alias("t")
        .join(locations_df.alias("l"), on="id", how="left")
        .select(
            F.col("t.id").alias("id"),
            F.col("t.predominant_color").alias("predominant_color"),
            F.col("t.exponential_color_weighting").alias("exponential_color_weighting"),
            F.col("t.linear_color_weighting").alias("linear_color_weighting"),
            F.col("t.diffuse_logic_traffic").alias("diffuse_logic_traffic"),
            F.col("l.Coordx").alias("Coordx"),
            F.col("l.Coordy").alias("Coordy"),
            F.col("t.datatime").alias("datatime")
        )
    )

    logger.info("Schema final después del join:")
    final_df.printSchema()
    logger.info(f"Total de registros después del join: {final_df.count()}")

    # Validaciones
    validate_dataframe(final_df)

    # Guardar Bronze
    FILE_PATH_SAVE = os.path.join(BRONZE_PATH, "TRAFFIC_DATA_BRONZE_PARQUET")
    save_dataframe(final_df, FILE_PATH_SAVE)

    job.commit()
    logger.info("Job Bronze completado exitosamente")
    logger.info("Los datos están listos para ser procesados en la capa Silver")
