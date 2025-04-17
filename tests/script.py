from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from dotenv import load_dotenv
import os


def initialize_spark_session():
    """Initialize and return a Spark session with S3 configuration"""
    load_dotenv()
    access_key = os.getenv("Access_key_ID")
    secret_key = os.getenv("Secret_access_key")
    bucket = os.getenv("BUCKET_NAME")
    region = os.getenv("REGION_NAME")

    spark = (
        SparkSession.builder.appName("S3DataTransformation")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", access_key)
        .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
        .config("spark.hadoop.fs.s3a.region", region)
        .getOrCreate()
    )

    return spark


def load_data_from_s3(spark, bucket_name, folder_path):
    """Load data from S3 bucket and return as DataFrame"""
    full_path = f"s3a://{bucket_name}/{folder_path}"
    return spark.read.csv(full_path, header=True, inferSchema=True)


def transform_order_items_data(df):
    """Apply transformations to order items data"""
    # Check for missing values
    missing_values = df.select(
        [sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]
    )

    # Add order_time column
    df = df.withColumn("order_time", date_format(col("order_timestamp"), "HH:mm:ss"))

    # Transform reordered column
    df = df.withColumn(
        "reordered", when(col("reordered") == 1, "Reorder").otherwise("Not_Reorder")
    )

    return df, missing_values


def join_datasets(order_items_df, orders_df, products_df):
    """Join order_items, orders, and products DataFrames"""
    joined_df = (
        order_items_df.alias("op")
        .join(orders_df.alias("o"), on="order_id", how="inner")
        .join(products_df.alias("p"), on="product_id", how="inner")
    )

    final_df = joined_df.select(
        "op.id",
        "op.order_id",
        "op.user_id",
        "op.days_since_prior_order",
        "op.product_id",
        "p.product_name",
        "p.department_id",
        "p.department",
        "op.add_to_cart_order",
        "op.reordered",
        "o.order_num",
        "o.total_amount",
        "op.order_timestamp",
        "op.date",
        "op.order_time",
    )

    return final_df


def normalize_data(final_df):
    """Normalize data into Users, Departments, Products, Orders, and Order Items tables"""
    # Users Table
    users_df = final_df.select("user_id").dropDuplicates()

    # Departments Table
    departments_df = final_df.select("department_id", "department").dropDuplicates()

    # Products Table
    products_df = final_df.select(
        "product_id", "product_name", "department_id"
    ).dropDuplicates()

    # Orders Table
    orders_df = final_df.select(
        "order_id",
        "user_id",
        "days_since_prior_order",
        "order_num",
        "total_amount",
        "date",
        "order_time",
    ).dropDuplicates(["order_id"])

    # Order Items Table
    order_items_df = final_df.select(
        "id", "order_id", "product_id", "add_to_cart_order", "reordered", "date"
    )

    return users_df, departments_df, products_df, orders_df, order_items_df


def partition_data(df, partition_by, num_partitions=None, sort_by=None):
    """Partition data by specified column(s) with optional sorting"""
    if num_partitions and sort_by:
        return df.repartition(num_partitions, col(partition_by)).sortWithinPartitions(
            col(sort_by)
        )
    elif num_partitions:
        return df.repartition(num_partitions, col(partition_by))
    else:
        return df.repartition(col(partition_by))


def analyze_data(df):
    """Perform basic data analysis"""
    # Show unique dates
    unique_dates = df.select("date").distinct()

    # Count missing values
    missing_values = df.select(
        [sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]
    )

    return unique_dates, missing_values


def save_normalized_data_to_s3(df, bucket_name, output_path):
    """Save DataFrame to S3 in parquet format, partitioned by date if applicable"""
    full_path = f"s3a://{bucket_name}/{output_path}"
    if "date" in df.columns:
        df.write.partitionBy("date").mode("overwrite").parquet(full_path)
    else:
        df.write.mode("overwrite").parquet(full_path)


def save_master_data_to_s3(df, bucket_name, output_path):
    """Save master DataFrame to S3 in parquet format without additional partitioning"""
    full_path = f"s3a://{bucket_name}/{output_path}"
    df.write.mode("overwrite").parquet(full_path)


def main():
    # Initialize Spark session
    spark = initialize_spark_session()

    try:
        # Load data
        bucket_name = os.getenv("BUCKET_NAME")
        order_items_df = load_data_from_s3(
            spark, bucket_name, "raw-data/order_items_apr_2025/"
        )
        orders_df = load_data_from_s3(spark, bucket_name, "raw-data/orders_apr_2025/")
        products_df = load_data_from_s3(spark, bucket_name, "raw-data/products/")

        # Transform order items data
        transformed_order_items_df, missing_values = transform_order_items_data(
            order_items_df
        )
        transformed_order_items_df.show()
        missing_values.show()

        # Join datasets
        final_df = join_datasets(transformed_order_items_df, orders_df, products_df)
        final_df.show()

        # Partition and sort master data
        final_df = final_df.repartition(15, col("date")).sortWithinPartitions(
            col("department"), col("reordered")
        )
        final_df.show()

        # Save master data without additional partitioning
        save_master_data_to_s3(final_df, bucket_name, "processed-data/master/")

        # Normalize data
        users_df, departments_df, products_df, orders_df, order_items_df = (
            normalize_data(final_df)
        )

        # Show normalized tables
        print("Users Table:")
        users_df.show()
        print("Departments Table:")
        departments_df.show()
        print("Products Table:")
        products_df.show()
        print("Orders Table:")
        orders_df.show()
        print("Order Items Table:")
        order_items_df.show()

        # Save normalized tables to S3 with partitioning by date where applicable
        save_normalized_data_to_s3(users_df, bucket_name, "processed-data/users/")
        save_normalized_data_to_s3(
            departments_df, bucket_name, "processed-data/departments/"
        )
        save_normalized_data_to_s3(products_df, bucket_name, "processed-data/products/")
        save_normalized_data_to_s3(orders_df, bucket_name, "processed-data/orders/")
        save_normalized_data_to_s3(
            order_items_df, bucket_name, "processed-data/order_items/"
        )

        # Partition data
        partitioned_by_date = partition_data(final_df, "date", 15, "order_time")
        partitioned_by_date.show()

        partitioned_by_dept = partition_data(final_df, "department", 6)
        partitioned_by_dept.show()

        # Analyze data
        unique_dates, missing_values_after = analyze_data(final_df)
        unique_dates.show()
        missing_values_after.show()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
