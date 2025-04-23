import sys
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta import configure_spark_with_delta_pip

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_glue_context(job_name):
    """Initialize and return Glue context with Delta Lake configurations"""
    args = getResolvedOptions(sys.argv, ["JOB_NAME", "BUCKET_NAME"])
    sc = SparkContext()
    glue_context = GlueContext(sc)
    spark = (
        SparkSession.builder.appName(job_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.committer.name", "directory")
        .config(
            "spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false"
        )
        .getOrCreate()
    )
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    spark.conf.set("spark.sql.cbo.enabled", "true")
    spark.conf.set("spark.sql.statistics.histogram.enabled", "true")
    job = Job(glue_context)
    job.init(job_name, args)
    return glue_context, spark, args, job  # Return job object


def load_data_from_s3(spark, bucket_name, folder_path):
    """Load data from S3 bucket and return as DataFrame"""
    full_path = f"s3a://{bucket_name}/{folder_path}"
    try:
        df = spark.read.csv(full_path, header=True, inferSchema=True)
        logger.info(f"Successfully loaded data from {full_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {full_path}: {str(e)}")
        raise


def transform_order_items_data(df):
    """Apply transformations to order items data"""
    try:
        df = df.withColumn(
            "order_time", date_format(col("order_timestamp"), "HH:mm:ss")
        )
        # Use withColumn twice to ensure nullability is preserved
        df = df.withColumn(
            "reordered_temp",
            when(col("reordered") == 1, "Reorder").otherwise("Not_Reorder"),
        )
        df = df.withColumn("reordered", col("reordered_temp").cast(StringType())).drop(
            "reordered_temp"
        )
        logger.info("Successfully transformed order items data")
        return df
    except Exception as e:
        logger.error(f"Failed to transform order items data: {str(e)}")
        raise


def join_datasets(order_items_df, orders_df, products_df=None):
    """Join order_items, orders, and products DataFrames"""
    try:
        if products_df is None:
            # Simplified join for testing
            joined_df = order_items_df.alias("op").join(
                orders_df.alias("o"), on="order_id", how="inner"
            )
            final_df = joined_df.select(
                "op.id",
                "op.order_id",
                "op.user_id",
                "op.days_since_prior_order",
                "op.product_id",
                "op.add_to_cart_order",
                "op.reordered",
                "op.order_timestamp",
                "op.date",
                "op.order_time",
                "o.order_num",
                "o.total_amount",
            )
        else:
            # Original join implementation
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
        logger.info("Successfully joined datasets")
        return final_df
    except Exception as e:
        logger.error(f"Failed to join datasets: {str(e)}")
        raise


def normalize_data(final_df):
    """Normalize data into Users, Departments, Products, Orders, and Order Items tables"""
    try:
        users_df = final_df.select("user_id").dropDuplicates()
        departments_df = (
            final_df.select("department_id", "department").dropDuplicates()
            if "department_id" in final_df.columns
            else final_df.createDataFrame(
                [],
                StructType(
                    [
                        StructField("department_id", IntegerType(), True),
                        StructField("department", StringType(), True),
                    ]
                ),
            )
        )
        products_df = (
            final_df.select(
                "product_id", "product_name", "department_id"
            ).dropDuplicates()
            if all(col in final_df.columns for col in ["product_name", "department_id"])
            else final_df.select("product_id").dropDuplicates()
        )
        orders_df = final_df.select(
            "order_id",
            "user_id",
            (
                "days_since_prior_order"
                if "days_since_prior_order" in final_df.columns
                else lit(None).alias("days_since_prior_order")
            ),
            "order_num",
            "total_amount",
            "date",
            (
                "order_time"
                if "order_time" in final_df.columns
                else lit(None).alias("order_time")
            ),
        ).dropDuplicates(["order_id"])
        order_items_df = final_df.select(
            "id", "order_id", "product_id", "add_to_cart_order", "reordered", "date"
        )
        logger.info("Successfully normalized data")
        return users_df, departments_df, products_df, orders_df, order_items_df
    except Exception as e:
        logger.error(f"Failed to normalize data: {str(e)}")
        raise


def partition_data(df, partition_by, num_partitions=None, sort_by=None):
    """Partition data by specified column(s) with optional sorting"""
    try:
        if num_partitions and sort_by:
            sort_columns = [col(c.strip()) for c in sort_by.split(",")]
            df = df.repartition(num_partitions, col(partition_by)).sortWithinPartitions(
                *sort_columns
            )
        elif num_partitions:
            df = df.repartition(num_partitions, col(partition_by))
        else:
            df = df.repartition(col(partition_by))
        logger.info(f"Successfully partitioned data by {partition_by}")
        return df
    except Exception as e:
        logger.error(f"Failed to partition data: {str(e)}")
        raise

# normalize_data function is used to create normalized tables
def save_normalized_data_to_s3(
    df, bucket_name, output_path, archive_path, partition_col=None
):
    """Save DataFrame to S3 in Delta Lake format with optional partitioning, and archive"""
    try:
        primary_full_path = f"s3a://{bucket_name}/{output_path}"
        if partition_col and df.schema.names != [partition_col]:
            df.write.format("delta").partitionBy(partition_col).mode("overwrite").save(
                primary_full_path
            )
        else:
            df.write.format("delta").mode("overwrite").save(primary_full_path)
        logger.info(f"Successfully saved data to {primary_full_path}")

        archive_full_path = f"s3a://{bucket_name}/{archive_path}"
        if partition_col and df.schema.names != [partition_col]:
            df.write.format("delta").partitionBy(partition_col).mode("overwrite").save(
                archive_full_path
            )
        else:
            df.write.format("delta").mode("overwrite").save(archive_full_path)
        logger.info(f"Successfully archived data to {archive_full_path}")
    except Exception as e:
        logger.error(f"Failed to save data to S3: {str(e)}")
        raise

# Save master data to S3 in Delta Lake format, partitioned by date, and archive
def save_master_data_to_s3(df, bucket_name, output_path, archive_path):
    """Save master DataFrame to S3 in Delta Lake format, partitioned by date, and archive"""
    try:
        primary_full_path = f"s3a://{bucket_name}/{output_path}"
        df.write.format("delta").partitionBy("date").mode("overwrite").save(
            primary_full_path
        )
        logger.info(f"Successfully saved master data to {primary_full_path}")

        archive_full_path = f"s3a://{bucket_name}/{archive_path}"
        df.write.format("delta").partitionBy("date").mode("overwrite").save(
            archive_full_path
        )
        logger.info(f"Successfully archived master data to {archive_full_path}")
    except Exception as e:
        logger.error(f"Failed to save master data to S3: {str(e)}")
        raise

# The main function is the entry point for the Glue job
def main():
    """Main ETL job execution"""
    try:
        glue_context, spark, args, job = initialize_glue_context("DeltaLakeETLJob")
        bucket_name = args["BUCKET_NAME"]
        archive_path = "archived-data"

        # Load data
        order_items_df = load_data_from_s3(
            spark, bucket_name, "raw-data/order_items_apr_2025/"
        )
        orders_df = load_data_from_s3(spark, bucket_name, "raw-data/orders_apr_2025/")
        products_df = load_data_from_s3(spark, bucket_name, "raw-data/products/")

        # Transform and join
        transformed_order_items_df = transform_order_items_data(order_items_df)
        final_df = join_datasets(transformed_order_items_df, orders_df, products_df)

        # Partition master data
        final_df = partition_data(
            final_df,
            partition_by="date",
            num_partitions=15,
            sort_by="department,reordered",
        )

        # Save master data
        save_master_data_to_s3(
            final_df, bucket_name, "lakehouse-dwh/master/", f"{archive_path}/master/"
        )

        # Normalize and save dimensional tables
        users_df, departments_df, products_df, orders_df, order_items_df = (
            normalize_data(final_df)
        )

        # Save normalized data
        save_normalized_data_to_s3(
            users_df, bucket_name, "lakehouse-dwh/users/", f"{archive_path}/users/"
        )
        save_normalized_data_to_s3(
            departments_df,
            bucket_name,
            "lakehouse-dwh/departments/",
            f"{archive_path}/departments/",
            "department_id",
        )
        save_normalized_data_to_s3(
            products_df,
            bucket_name,
            "lakehouse-dwh/products/",
            f"{archive_path}/products/",
            "department_id",
        )
        save_normalized_data_to_s3(
            orders_df,
            bucket_name,
            "lakehouse-dwh/orders/",
            f"{archive_path}/orders/",
            "date",
        )
        save_normalized_data_to_s3(
            order_items_df,
            bucket_name,
            "lakehouse-dwh/order_items/",
            f"{archive_path}/order_items/",
            "date",
        )

        # Commit job
        job.commit()  # Use the job object to commit
        logger.info("Job completed successfully")

    except Exception as e:
        logger.error(f"Job failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
