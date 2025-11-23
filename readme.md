# Arquitectura Medallion de Tráfico (Bronce / Plata / Oro + Clustering)

Este proyecto implementa una arquitectura **Medallion (Bronce → Plata → Oro)** en AWS para procesar datos de tráfico (Google Maps / AMGTraffic), generar series temporales por ubicación y aplicar un **análisis de clustering** para encontrar patrones similares de tráfico.

Además, incluye un **notebook local** para visualización en mapa y análisis de las series temporales promedio por clúster.

---

## 1. Arquitectura General

Tecnologías principales:

- **AWS Glue** (ETL: Bronce, Plata, Oro, Clustering)
- **Amazon S3** (almacenamiento de datos en capas)
- **AWS Glue Data Catalog + Athena** (consulta y validación)
- **Python + PySpark** (transformaciones)
- **pyspark.ml** (K-Means)
- **Notebook local**:
  - `pandas`, `pyarrow`, `folium`, `matplotlib`, `seaborn`, `boto3`

### Capas

1. **Bronce**: Ingesta de CSV crudos (histórico de tráfico + locations).
2. **Plata**: Limpieza y tipado (especialmente fechas y timestamp).
3. **Oro**: Serie temporal unificada por ubicación (pivot) + clustering K-Means.
4. **Visualización local**: Mapa con clusters y análisis de series temporales promedio.

---

## 2. Estructura de datos en S3

Ejemplo de estructura en el bucket (ajustar nombres según tu entorno):

```text
s3://glue-bucket-traffic-725895/
  ├── bronze/
  │   ├── catalogos/
  │   │   ├── AMGTraffic2024/
  │   │   │   ├── historico/            # CSVs crudos 2024
  │   │   │   └── locationPoints.csv
  │   │   ├── AMGtraffic2025/
  │   │   │   ├── historico/            # CSVs crudos 2025
  │   │   │   └── locationPoints.csv
  │   └── TRAFFIC_DATA_BRONZE_PARQUET/  # Parquet unificado (job Bronce)
  │
  ├── silver/
  │   └── TRAFFIC_DATA_SILVER_PARQUET/  # Parquet limpio (job Plata)
  │
  ├── gold/
  │   ├── TRAFFIC_DATA_GOLD_TS_PIVOT_MX/   # Serie temporal pivot por ubicación (job Oro)
  │   └── TRAFFIC_DATA_GOLD_CLUSTERS_MX/   # Resultado de clustering (job Clustering)
  │
  └── ...
