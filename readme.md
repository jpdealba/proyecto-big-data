# Arquitectura Medallion para Análisis de Tráfico

## Colaboradores del Proyecto
Santiago Esparragoza - 0161298

Juan Pablo de Alba - 0278627

## Objetivo del Proyecto

Implementar una **arquitectura Medallion en AWS Glue** para procesar datos de tráfico vehicular con las siguientes etapas:

1. **Capa Bronce**: Cargar los CSV de los repositorios de tráfico (AMGTraffic 2024/2025) y unificarlos en un archivo Parquet.
2. **Capa Plata**: Transformar el formato de fecha a uno utilizable (timestamp estándar).
3. **Capa Oro**: Extraer una serie temporal cada 2 horas considerando las horas disponibles: **7am, 9am, 11am, 13pm, 15pm, 17pm, 19pm, 21pm, 23pm y 1am** (excluyendo 18pm y 3-6am que no están en los datos).
4. **Clustering**: Generar un análisis de clustering en series temporales para descubrir qué ubicaciones tienen el mismo patrón de tráfico.

---

## Tecnologías Utilizadas

- **AWS Glue** (Jobs ETL: Bronce, Plata, Oro, Clustering)
- **Amazon S3** (Almacenamiento de datos por capas)
- **AWS Glue Data Catalog + Athena** (Consulta y validación)
- **Python + PySpark** (Transformaciones)
- **pyspark.ml** (K-Means para clustering)
- **Notebooks locales**: `pandas`, `pyarrow`, `folium`, `matplotlib`, `seaborn`, `boto3`

---

## Estructura de Archivos

### Jobs de AWS Glue

| Archivo | Capa | Descripción |
|---------|------|-------------|
| `bronze.py` | Bronce | Ingesta y unificación de datos crudos |
| `silver.py` | Plata | Transformación de fechas |
| `gold.py` | Oro | Extracción de serie temporal cada 2 horas |
| `cluster.py` | Oro | Análisis de clustering K-Means |

### Notebooks Locales

| Archivo | Descripción |
|---------|-------------|
| `graficas.ipynb` | Descarga resultados de S3, genera mapas interactivos y gráficas de series temporales por cluster |
| `validate_csv.ipynb` | Validación exploratoria de los CSV crudos para identificar horas disponibles |

### Carpeta de Figuras

| Archivo | Descripción |
|---------|-------------|
| `figures_clusters/cluster_all.png` | Comparación de series temporales de todos los clusters |
| `figures_clusters/cluster_0.png` | Serie temporal promedio del Cluster 0 |
| `figures_clusters/cluster_1.png` | Serie temporal promedio del Cluster 1 |
| `figures_clusters/cluster_2.png` | Serie temporal promedio del Cluster 2 |
| `figures_clusters/cluster_3.png` | Serie temporal promedio del Cluster 3 |
| `figures_clusters/map1.png`, `map2.png` | Capturas del mapa interactivo |

---

## Descripción Detallada de Cada Archivo

### `bronze.py` - Capa Bronce

**Propósito**: Cargar los archivos CSV crudos de tráfico y unificarlos en formato Parquet.

**Funcionalidades**:
- Lee los CSV de histórico de tráfico de 2024 y 2025 desde S3
- Extrae el timestamp (`datatime`) del nombre del archivo usando regex (formato `yyyyMMddHHmmss`)
- Carga y unifica los archivos `locationPoints.csv` de ambos años (coordenadas de ubicaciones)
- Normaliza nombres de columnas de coordenadas (`Coordx`, `Coordy`)
- Realiza un JOIN entre los datos de tráfico y las ubicaciones
- Ejecuta validaciones básicas de calidad (nulos y duplicados en muestra del 1%)
- Guarda en formato Parquet particionado por año (`year_partition`)

**Parámetros del Job**:
- `JOB_NAME`: Nombre del job
- `BUCKET`: Bucket de S3
- `BRONZE_PATH`: Ruta base para datos bronce

**Salida**: `s3://bucket/bronze/TRAFFIC_DATA_BRONZE_PARQUET/`

---

### `silver.py` - Capa Plata

**Propósito**: Transformar la columna de fecha a formato timestamp estándar.

**Funcionalidades**:
- Lee los datos Parquet de la capa Bronce
- Convierte `datatime` (string `yyyyMMddHHmmss`) a timestamp real (`datatime_ts`)
- Genera formato ISO 8601 (`datatime_iso`) para consumo externo
- Recrea la partición de año desde el timestamp limpio
- Valida que no se pierdan registros en la transformación
- Guarda en formato Parquet particionado por año

**Parámetros del Job**:
- `JOB_NAME`: Nombre del job
- `BUCKET`: Bucket de S3
- `BRONZE_PATH`: Ruta de entrada (Bronce)
- `SILVER_PATH`: Ruta de salida (Plata)

**Salida**: `s3://bucket/silver/TRAFFIC_DATA_SILVER_PARQUET/`

---

### `gold.py` - Capa Oro

**Propósito**: Extraer una serie temporal cada 2 horas por ubicación.

**Funcionalidades**:
- Lee los datos Parquet de la capa Plata
- Convierte timestamps UTC a hora local de México (`America/Mexico_City`)
- Filtra solo las horas objetivo: `[7, 9, 11, 13, 15, 17, 19, 21, 23]` (hora local MX)
- Agrupa por ubicación (`id`) y hora, calculando el promedio de tráfico (`linear_color_weighting`)
- Realiza un **pivot** para obtener una fila por ubicación con columnas por hora (`traffic_07`, `traffic_09`, etc.)
- Guarda en formato Parquet listo para clustering

**Parámetros del Job**:
- `JOB_NAME`: Nombre del job
- `BUCKET`: Bucket de S3
- `SILVER_PATH`: Ruta de entrada (Plata)
- `GOLD_PATH`: Ruta de salida (Oro)

**Salida**: `s3://bucket/gold/TRAFFIC_DATA_GOLD_TS_PIVOT_MX/`

**Esquema de salida**:
```
id | Coordx | Coordy | traffic_07 | traffic_09 | traffic_11 | ... | traffic_23
```

---

### `cluster.py` - Clustering de Series Temporales

**Propósito**: Aplicar K-Means para agrupar ubicaciones con patrones de tráfico similares.

**Funcionalidades**:
- Lee los datos pivotados de la capa Oro
- Detecta automáticamente las columnas de serie temporal (`traffic_XX`)
- Prepara los datos: cast a double, rellena nulos con 0
- Usa `VectorAssembler` para crear el vector de features
- Normaliza con `StandardScaler` (media 0, desviación estándar 1)
- Entrena modelo K-Means con el número de clusters especificado
- Asigna cada ubicación a un cluster (`cluster_id`)
- Guarda resultados con coordenadas y cluster asignado

**Parámetros del Job**:
- `JOB_NAME`: Nombre del job
- `GOLD_PATH`: Ruta de entrada (Oro con pivot)
- `OUTPUT_PATH`: Ruta de salida
- `K`: Número de clusters (ej: 4)

**Salida**: `s3://bucket/gold/TRAFFIC_DATA_GOLD_CLUSTERS_MX/`

---

### `graficas.ipynb` - Visualización y Análisis

**Propósito**: Descargar resultados del clustering y generar visualizaciones.

**Funcionalidades**:
1. **Descarga de datos**: Conecta a S3 con boto3 y descarga los Parquet del clustering
2. **Carga local**: Lee todos los archivos Parquet y los concatena en un DataFrame
3. **Mapa interactivo**: Genera un mapa con Folium donde cada punto representa una ubicación coloreada por su cluster
4. **Series temporales por cluster**: Grafica el promedio de tráfico por hora para cada cluster

**Dependencias**:
```
pandas, numpy, pyarrow, folium, matplotlib, seaborn, boto3, python-dotenv
```

**Salidas**:
- `traffic_clusters_map.html` - Mapa interactivo
- `figures_clusters/*.png` - Gráficas de series temporales

---

### `validate_csv.ipynb` - Validación de Datos Crudos

**Propósito**: Explorar los CSV crudos para identificar las horas disponibles en los datos.

**Funcionalidades**:
- Lista archivos CSV en la carpeta de histórico
- Extrae la hora del timestamp en el nombre del archivo
- Cuenta archivos por hora
- Muestra qué horas están disponibles y cuáles faltan (18pm, 3-6am)

---

## Estructura de Datos en S3

```
s3://glue-bucket-traffic-725895/
├── bronze/
│   ├── catalogos/
│   │   ├── AMGTraffic2024/
│   │   │   ├── historico/           # CSVs crudos 2024
│   │   │   └── locationPoints.csv
│   │   └── AMGtraffic2025/
│   │       ├── historico/           # CSVs crudos 2025
│   │       └── locationPoints.csv
│   └── TRAFFIC_DATA_BRONZE_PARQUET/ # Parquet unificado
│
├── silver/
│   └── TRAFFIC_DATA_SILVER_PARQUET/ # Parquet con fechas limpias
│
└── gold/
    ├── TRAFFIC_DATA_GOLD_TS_PIVOT_MX/   # Serie temporal por ubicación
    └── TRAFFIC_DATA_GOLD_CLUSTERS_MX/   # Resultado del clustering
```

---

## Flujo de Ejecución

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CAPA BRONCE                                 │
│  bronze.py                                                          │
│  • Carga CSV de tráfico 2024 + 2025                                │
│  • Extrae timestamp del nombre de archivo                          │
│  • Une con locationPoints (coordenadas)                            │
│  • Guarda Parquet particionado por año                             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CAPA PLATA                                  │
│  silver.py                                                          │
│  • Transforma string de fecha a timestamp                          │
│  • Genera formato ISO 8601                                         │
│  • Mantiene partición por año                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          CAPA ORO                                   │
│  gold.py                                                            │
│  • Convierte a hora local MX                                       │
│  • Filtra horas: 7,9,11,13,15,17,19,21,23                         │
│  • Agrupa por ubicación + hora                                     │
│  • Pivot: 1 fila por ubicación, columnas por hora                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CLUSTERING                                   │
│  cluster.py                                                         │
│  • Normaliza features (StandardScaler)                             │
│  • K-Means con K clusters                                          │
│  • Asigna cluster_id a cada ubicación                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       VISUALIZACIÓN                                 │
│  graficas.ipynb (local)                                             │
│  • Descarga Parquet desde S3                                       │
│  • Mapa interactivo con Folium                                     │
│  • Gráficas de series temporales por cluster                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Resultados del Clustering

El análisis identificó **4 clusters** de patrones de tráfico:

| Cluster | Color | Descripción |
|---------|-------|-------------|
| 0 | Naranja | Tráfico medio/alto |
| 1 | Rojo | Tráfico alto |
| 2 | Verde | Tráfico muy bajo |
| 3 | Azul | Tráfico medio/bajo |

Las gráficas de series temporales por cluster se encuentran en la carpeta `figures_clusters/`.

---

## Configuración y Ejecución

### Variables de Entorno (para notebooks locales)

Crear archivo `.env`:
```
AWS_ACCESS_KEY_ID=tu_access_key
AWS_SECRET_ACCESS_KEY=tu_secret_key
AWS_DEFAULT_REGION=us-east-1
```

### Ejecución de Jobs en AWS Glue

Cada job requiere parámetros específicos. Ejemplo para el job Bronze:
```
--JOB_NAME=bronze_traffic
--BUCKET=glue-bucket-traffic-725895
--BRONZE_PATH=s3://glue-bucket-traffic-725895/bronze
```

---

## Dependencias

### Jobs de AWS Glue
- PySpark (incluido en Glue)
- pyspark.ml (incluido en Glue)

### Notebooks Locales
```
pandas
numpy
pyarrow
folium
matplotlib
seaborn
boto3
python-dotenv
```

Instalar con:
```bash
pip install pandas numpy pyarrow folium matplotlib seaborn boto3 python-dotenv
```
