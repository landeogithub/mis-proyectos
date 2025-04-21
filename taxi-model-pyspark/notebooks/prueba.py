from pyspark.sql import SparkSession
import os

# ================================
# 1. Crear SparkSession
# ================================
spark = SparkSession.builder \
    .appName("ProcesarDatosNYC") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

print("🚀 SparkSession iniciada")

# ================================
# 2. Cargar datos (ajusta la ruta)
# ================================
input_path = "../data/raw/yellow_tripdata_2019.parquet"

print(f"📂 Leyendo archivo desde: {input_path}")
df = spark.read.parquet(input_path)

# ================================
# 3. Procesar (a modo de ejemplo)
# ================================
df_procesado = df.select("tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", "trip_distance")
print("🛠️ Datos procesados")

# ================================
# 4. Guardar resultado
# ================================
output_path = "../data/processed/yellow_2019_procesado.parquet"

# Crear carpeta si no existe
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f"💾 Guardando en: {output_path}")
df_procesado.write.mode("overwrite").parquet(output_path)

# ================================
# 5. Cerrar Spark
# ================================
spark.stop()
print("✅ SparkSession cerrada correctamente")
