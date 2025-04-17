import pytest
from unittest.mock import patch
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    StringType,
)
import tempfile
import shutil
from datetime import datetime

# Import the functions to test
from script import (
    initialize_spark_session,
    load_data_from_s3,
    transform_order_items_data,
    join_datasets,
    normalize_data,
    partition_data,
    analyze_data,
    save_normalized_data_to_s3,
    save_master_data_to_s3,
)


@pytest.fixture(scope="module")
def spark_session():
    """Create a Spark session for tests"""
    spark = (
        SparkSession.builder.appName("TestSparkDataPipeline")
        .master("local[2]")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing writes"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def order_items_df(spark_session):
    """Create sample order_items DataFrame"""
    data = [
        (1, 101, 201, 7, 301, 1, 1, "2025-04-01 10:00:00", "2025-04-01"),
        (2, 102, 202, 14, 302, 0, 2, "2025-04-02 12:00:00", "2025-04-02"),
    ]
    schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("order_id", IntegerType(), True),
            StructField("user_id", IntegerType(), True),
            StructField("days_since_prior_order", IntegerType(), True),
            StructField("product_id", IntegerType(), True),
            StructField("reordered", IntegerType(), True),
            StructField("add_to_cart_order", IntegerType(), True),
            StructField("order_timestamp", StringType(), True),
            StructField("date", StringType(), True),
        ]
    )
    return spark_session.createDataFrame(data, schema)


@pytest.fixture
def orders_df(spark_session):
    """Create sample orders DataFrame"""
    data = [
        (101, 201, 1, 50.0),
        (102, 202, 2, 75.0),
    ]
    schema = StructType(
        [
            StructField("order_id", IntegerType(), True),
            StructField("user_id", IntegerType(), True),
            StructField("order_num", IntegerType(), True),
            StructField("total_amount", DoubleType(), True),
        ]
    )
    return spark_session.createDataFrame(data, schema)


@pytest.fixture
def products_df(spark_session):
    """Create sample products DataFrame"""
    data = [
        (301, "Apple", 401, "Produce"),
        (302, "Milk", 402, "Dairy"),
    ]
    schema = StructType(
        [
            StructField("product_id", IntegerType(), True),
            StructField("product_name", StringType(), True),
            StructField("department_id", IntegerType(), True),
            StructField("department", StringType(), True),
        ]
    )
    return spark_session.createDataFrame(data, schema)


@patch("os.getenv")
def test_initialize_spark_session(mock_getenv):
    """Test initialize_spark_session"""
    mock_getenv.side_effect = lambda key: {
        "Access_key_ID": "fake_access_key",
        "Secret_access_key": "fake_secret_key",
        "BUCKET_NAME": "fake_bucket",
        "REGION_NAME": "us-east-1",
    }[key]

    spark = initialize_spark_session()
    assert isinstance(spark, SparkSession)
    assert spark.conf.get("spark.app.name") == "S3DataTransformation"
    assert spark.conf.get("spark.jars.packages") == "org.apache.hadoop:hadoop-aws:3.3.1"
    spark.stop()


@patch("pyspark.sql.readwriter.DataFrameReader.csv")
def test_load_data_from_s3(mock_csv, spark_session, order_items_df):
    """Test load_data_from_s3"""
    mock_csv.return_value = order_items_df

    bucket_name = "fake_bucket"
    folder_path = "raw-data/order_items_apr_2025/"
    df = load_data_from_s3(spark_session, bucket_name, folder_path)

    assert df.count() == 2
    assert len(df.columns) == 9  # Updated from 8 to 9
    assert df.schema["order_id"].dataType == IntegerType()
    mock_csv.assert_called_with(
        f"s3a://{bucket_name}/{folder_path}", header=True, inferSchema=True
    )


def test_transform_order_items_data(order_items_df):
    """Test transform_order_items_data"""
    transformed_df, missing_values = transform_order_items_data(order_items_df)

    # Check transformed columns
    assert "order_time" in transformed_df.columns
    assert transformed_df.filter(col("reordered") == "Reorder").count() == 1
    assert transformed_df.filter(col("reordered") == "Not_Reorder").count() == 1

    # Check order_time format
    order_time = transformed_df.select("order_time").first()[0]
    try:
        datetime.strptime(order_time, "%H:%M:%S")
    except ValueError:
        pytest.fail("order_time format is not HH:mm:ss")

    # Check missing values
    assert missing_values.count() == 1  # Single row with null counts
    assert missing_values.first().id == 0  # No nulls in 'id' column


def test_join_datasets(order_items_df, orders_df, products_df):
    """Test join_datasets"""
    transformed_df, _ = transform_order_items_data(order_items_df)
    final_df = join_datasets(transformed_df, orders_df, products_df)

    assert final_df.count() == 2
    expected_columns = [
        "id",
        "order_id",
        "user_id",
        "days_since_prior_order",
        "product_id",
        "product_name",
        "department_id",
        "department",
        "add_to_cart_order",
        "reordered",
        "order_num",
        "total_amount",
        "order_timestamp",
        "date",
        "order_time",
    ]
    assert final_df.columns == expected_columns
    assert final_df.filter(col("product_name") == "Apple").count() == 1


def test_normalize_data(order_items_df, orders_df, products_df):
    """Test normalize_data"""
    transformed_df, _ = transform_order_items_data(order_items_df)
    final_df = join_datasets(transformed_df, orders_df, products_df)
    users_df, departments_df, products_df, orders_df, order_items_df = normalize_data(
        final_df
    )

    # Users table
    assert users_df.count() == 2
    assert users_df.columns == ["user_id"]

    # Departments table
    assert departments_df.count() == 2
    assert departments_df.columns == ["department_id", "department"]

    # Products table
    assert products_df.count() == 2
    assert products_df.columns == ["product_id", "product_name", "department_id"]

    # Orders table
    assert orders_df.count() == 2
    assert orders_df.columns == [
        "order_id",
        "user_id",
        "days_since_prior_order",
        "order_num",
        "total_amount",
        "date",
        "order_time",
    ]

    # Order Items table
    assert order_items_df.count() == 2
    assert order_items_df.columns == [
        "id",
        "order_id",
        "product_id",
        "add_to_cart_order",
        "reordered",
        "date",
    ]


def test_partition_data(order_items_df, orders_df, products_df):
    """Test partition_data"""
    transformed_df, _ = transform_order_items_data(order_items_df)
    final_df = join_datasets(transformed_df, orders_df, products_df)

    # Test with num_partitions and sort_by
    partitioned_df = partition_data(
        final_df, "date", num_partitions=15, sort_by="order_time"
    )
    assert partitioned_df.rdd.getNumPartitions() == 15

    # Test with num_partitions only
    partitioned_df = partition_data(final_df, "date", num_partitions=10)
    assert partitioned_df.rdd.getNumPartitions() == 10

    # Test without num_partitions
    partitioned_df = partition_data(final_df, "date")
    assert partitioned_df.rdd.getNumPartitions() > 0


@patch("pyspark.sql.readwriter.DataFrameWriter.parquet")
def test_save_normalized_data_to_s3(mock_parquet, spark_session, temp_dir):
    """Test save_normalized_data_to_s3"""
    normalized_df = spark_session.createDataFrame(
        [(1, "Apple", "Produce"), (2, "Milk", "Dairy")],
        ["product_id", "product_name", "department"],
    )
    save_normalized_data_to_s3(normalized_df, "fake_bucket", "normalized-data/")

    mock_parquet.assert_called_once_with("s3a://fake_bucket/normalized-data/")


def test_analyze_data(order_items_df):
    """Test analyze_data"""
    unique_dates, missing_values = analyze_data(order_items_df)

    assert unique_dates.count() == 2  # Two distinct dates
    assert missing_values.count() == 1  # Single row with null counts
    assert missing_values.first().id == 0  # No nulls in 'id' column
