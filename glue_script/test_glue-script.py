import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    DoubleType,
)
from pyspark.sql.functions import col, lit
from unittest.mock import patch, MagicMock, PropertyMock
import logging


# Mock `awsglue` to avoid ModuleNotFoundError
class MockGlueModule:
    transforms = MagicMock()
    context = MagicMock()
    job = MagicMock()
    utils = MagicMock()


# Patch `awsglue` in the `ecommerce_delta` module
@pytest.fixture(autouse=True)
def mock_awsglue_imports():
    import sys

    sys.modules["awsglue"] = MockGlueModule()
    sys.modules["awsglue.transforms"] = MockGlueModule.transforms
    sys.modules["awsglue.context"] = MockGlueModule.context
    sys.modules["awsglue.job"] = MockGlueModule.job
    sys.modules["awsglue.utils"] = MockGlueModule.utils


@pytest.fixture(scope="module")
def spark():
    """Set up a PySpark session for testing."""
    spark = (
        SparkSession.builder.appName("TestEcommerceDelta")
        .master("local[2]")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def test_load_data_from_s3_success(spark):
    """Test successful loading of data from S3."""
    from ecommerce_delta import load_data_from_s3

    with patch("pyspark.sql.readwriter.DataFrameReader.csv") as mock_read_csv:
        input_schema = StructType(
            [
                StructField("id", IntegerType(), True),
                StructField("order_id", IntegerType(), True),
                StructField("user_id", IntegerType(), True),
                StructField("product_id", IntegerType(), True),
                StructField("add_to_cart_order", IntegerType(), True),
                StructField("reordered", IntegerType(), True),
                StructField("order_timestamp", StringType(), True),
                StructField("date", StringType(), True),
            ]
        )
        input_data = [(1, 10000, 1990, 988, 1, 0, "2025-04-01T11:27:00", "2025-04-01")]
        mock_df = spark.createDataFrame(input_data, input_schema)
        mock_read_csv.return_value = mock_df
        bucket_name = "test-bucket"
        folder_path = "raw-data/test/"

        result_df = load_data_from_s3(spark, bucket_name, folder_path)

        mock_read_csv.assert_called_once_with(
            f"s3a://{bucket_name}/{folder_path}", header=True, inferSchema=True
        )
        assert result_df.schema == input_schema, f"Schema mismatch: {result_df.schema}"
        assert result_df.collect() == mock_df.collect(), "Data mismatch"


def test_load_data_from_s3_failure(spark, caplog):
    """Test failure handling when loading data from S3."""
    from ecommerce_delta import load_data_from_s3

    with patch("pyspark.sql.readwriter.DataFrameReader.csv") as mock_read_csv:
        with caplog.at_level(logging.ERROR):
            mock_read_csv.side_effect = Exception("S3 read error")
            bucket_name = "test-bucket"
            folder_path = "raw-data/test/"

            with pytest.raises(Exception, match="S3 read error"):
                load_data_from_s3(spark, bucket_name, folder_path)
            assert (
                "Failed to load data from s3a://test-bucket/raw-data/test/: S3 read error"
                in caplog.text
            )


def test_transform_order_items_data(spark):
    """Test the transformation logic of the order_items dataset."""
    from ecommerce_delta import transform_order_items_data

    input_schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("order_id", IntegerType(), True),
            StructField("user_id", IntegerType(), True),
            StructField("days_since_prior_order", IntegerType(), True),
            StructField("product_id", IntegerType(), True),
            StructField("add_to_cart_order", IntegerType(), True),
            StructField("reordered", IntegerType(), True),
            StructField("order_timestamp", StringType(), True),
            StructField("date", StringType(), True),
        ]
    )
    input_data = [
        (1, 10000, 1990, 10, 988, 1, 0, "2025-04-01T11:27:00", "2025-04-01"),
        (2, 10001, 1991, 15, 989, 2, 1, "2025-04-02T12:30:00", "2025-04-02"),
    ]
    input_df = spark.createDataFrame(input_data, input_schema)

    expected_schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("order_id", IntegerType(), True),
            StructField("user_id", IntegerType(), True),
            StructField("days_since_prior_order", IntegerType(), True),
            StructField("product_id", IntegerType(), True),
            StructField("add_to_cart_order", IntegerType(), True),
            StructField("reordered", StringType(), False),
            StructField("order_timestamp", StringType(), True),
            StructField("date", StringType(), True),
            StructField("order_time", StringType(), True),
        ]
    )
    expected_data = [
        (
            1,
            10000,
            1990,
            10,
            988,
            1,
            "Not_Reorder",
            "2025-04-01T11:27:00",
            "2025-04-01",
            "11:27:00",
        ),
        (
            2,
            10001,
            1991,
            15,
            989,
            2,
            "Reorder",
            "2025-04-02T12:30:00",
            "2025-04-02",
            "12:30:00",
        ),
    ]
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    transformed_df = transform_order_items_data(input_df)

    assert (
        transformed_df.schema == expected_schema
    ), f"Schema mismatch: {transformed_df.schema}"
    assert transformed_df.collect() == expected_df.collect(), "Data mismatch"


def test_join_datasets_with_products(spark):
    """Test the join operation with order_items, orders, and products datasets."""
    from ecommerce_delta import join_datasets

    order_items_schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("order_id", IntegerType(), True),
            StructField("user_id", IntegerType(), True),
            StructField("days_since_prior_order", IntegerType(), True),
            StructField("product_id", IntegerType(), True),
            StructField("add_to_cart_order", IntegerType(), True),
            StructField("reordered", StringType(), True),
            StructField("order_timestamp", StringType(), True),
            StructField("date", StringType(), True),
            StructField("order_time", StringType(), True),
        ]
    )
    order_items_data = [
        (
            1,
            1,
            1990,
            10,
            988,
            1,
            "Not_Reorder",
            "2025-04-01T11:27:00",
            "2025-04-01",
            "11:27:00",
        ),
        (
            2,
            2,
            1991,
            15,
            989,
            2,
            "Reorder",
            "2025-04-02T12:30:00",
            "2025-04-02",
            "12:30:00",
        ),
    ]
    order_items_df = spark.createDataFrame(order_items_data, order_items_schema)

    orders_schema = StructType(
        [
            StructField("order_id", IntegerType(), True),
            StructField("order_num", IntegerType(), True),
            StructField("total_amount", DoubleType(), True),
        ]
    )
    orders_data = [(1, 123, 45.6), (2, 124, 67.8)]
    orders_df = spark.createDataFrame(orders_data, orders_schema)

    products_schema = StructType(
        [
            StructField("product_id", IntegerType(), True),
            StructField("product_name", StringType(), True),
            StructField("department_id", IntegerType(), True),
            StructField("department", StringType(), True),
        ]
    )
    products_data = [(988, "Apple", 1, "Produce"), (989, "Bread", 2, "Bakery")]
    products_df = spark.createDataFrame(products_data, products_schema)

    expected_schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("order_id", IntegerType(), True),
            StructField("user_id", IntegerType(), True),
            StructField("days_since_prior_order", IntegerType(), True),
            StructField("product_id", IntegerType(), True),
            StructField("product_name", StringType(), True),
            StructField("department_id", IntegerType(), True),
            StructField("department", StringType(), True),
            StructField("add_to_cart_order", IntegerType(), True),
            StructField("reordered", StringType(), True),
            StructField("order_num", IntegerType(), True),
            StructField("total_amount", DoubleType(), True),
            StructField("order_timestamp", StringType(), True),
            StructField("date", StringType(), True),
            StructField("order_time", StringType(), True),
        ]
    )
    expected_data = [
        (
            1,
            1,
            1990,
            10,
            988,
            "Apple",
            1,
            "Produce",
            1,
            "Not_Reorder",
            123,
            45.6,
            "2025-04-01T11:27:00",
            "2025-04-01",
            "11:27:00",
        ),
        (
            2,
            2,
            1991,
            15,
            989,
            "Bread",
            2,
            "Bakery",
            2,
            "Reorder",
            124,
            67.8,
            "2025-04-02T12:30:00",
            "2025-04-02",
            "12:30:00",
        ),
    ]
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    result_df = join_datasets(order_items_df, orders_df, products_df)

    assert result_df.schema == expected_schema, f"Schema mismatch: {result_df.schema}"
    assert result_df.collect() == expected_df.collect(), "Data mismatch"

# Test the join operation with empty DataFrames
def test_join_datasets_without_products(spark):
    """Test the join operation without products dataset."""
    from ecommerce_delta import join_datasets

    order_items_schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("order_id", IntegerType(), True),
            StructField("user_id", IntegerType(), True),
            StructField("days_since_prior_order", IntegerType(), True),
            StructField("product_id", IntegerType(), True),
            StructField("add_to_cart_order", IntegerType(), True),
            StructField("reordered", StringType(), True),
            StructField("order_timestamp", StringType(), True),
            StructField("date", StringType(), True),
            StructField("order_time", StringType(), True),
        ]
    )
    order_items_data = [
        (
            1,
            1,
            1990,
            10,
            988,
            1,
            "Not_Reorder",
            "2025-04-01T11:27:00",
            "2025-04-01",
            "11:27:00",
        ),
        (
            2,
            2,
            1991,
            15,
            989,
            2,
            "Reorder",
            "2025-04-02T12:30:00",
            "2025-04-02",
            "12:30:00",
        ),
    ]
    order_items_df = spark.createDataFrame(order_items_data, order_items_schema)

    orders_schema = StructType(
        [
            StructField("order_id", IntegerType(), True),
            StructField("order_num", IntegerType(), True),
            StructField("total_amount", DoubleType(), True),
        ]
    )
    orders_data = [(1, 123, 45.6), (2, 124, 67.8)]
    orders_df = spark.createDataFrame(orders_data, orders_schema)

    expected_schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("order_id", IntegerType(), True),
            StructField("user_id", IntegerType(), True),
            StructField("days_since_prior_order", IntegerType(), True),
            StructField("product_id", IntegerType(), True),
            StructField("add_to_cart_order", IntegerType(), True),
            StructField("reordered", StringType(), True),
            StructField("order_timestamp", StringType(), True),
            StructField("date", StringType(), True),
            StructField("order_time", StringType(), True),
            StructField("order_num", IntegerType(), True),
            StructField("total_amount", DoubleType(), True),
        ]
    )
    expected_data = [
        (
            1,
            1,
            1990,
            10,
            988,
            1,
            "Not_Reorder",
            "2025-04-01T11:27:00",
            "2025-04-01",
            "11:27:00",
            123,
            45.6,
        ),
        (
            2,
            2,
            1991,
            15,
            989,
            2,
            "Reorder",
            "2025-04-02T12:30:00",
            "2025-04-02",
            "12:30:00",
            124,
            67.8,
        ),
    ]
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    result_df = join_datasets(order_items_df, orders_df)

    assert result_df.schema == expected_schema, f"Schema mismatch: {result_df.schema}"
    assert result_df.collect() == expected_df.collect(), "Data mismatch"

# Test the join operation with empty DataFrames
def test_normalize_data(spark):
    """Test the normalization logic."""
    from ecommerce_delta import normalize_data

    input_schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("order_id", IntegerType(), True),
            StructField("user_id", IntegerType(), True),
            StructField("days_since_prior_order", IntegerType(), True),
            StructField("product_id", IntegerType(), True),
            StructField("product_name", StringType(), True),
            StructField("department_id", IntegerType(), True),
            StructField("department", StringType(), True),
            StructField("add_to_cart_order", IntegerType(), True),
            StructField("reordered", StringType(), True),
            StructField("order_num", IntegerType(), True),
            StructField("total_amount", DoubleType(), True),
            StructField("order_timestamp", StringType(), True),
            StructField("date", StringType(), True),
            StructField("order_time", StringType(), True),
        ]
    )
    input_data = [
        (
            1,
            1,
            1990,
            10,
            988,
            "Apple",
            1,
            "Produce",
            1,
            "Not_Reorder",
            123,
            45.6,
            "2025-04-01T11:27:00",
            "2025-04-01",
            "11:27:00",
        ),
        (
            2,
            2,
            1991,
            15,
            989,
            "Bread",
            2,
            "Bakery",
            2,
            "Reorder",
            124,
            67.8,
            "2025-04-02T12:30:00",
            "2025-04-02",
            "12:30:00",
        ),
    ]
    input_df = spark.createDataFrame(input_data, input_schema)

    expected_users_data = [(1990,), (1991,)]
    expected_users_schema = StructType([StructField("user_id", IntegerType(), True)])
    expected_users_df = spark.createDataFrame(
        expected_users_data, expected_users_schema
    )

    expected_departments_data = [(1, "Produce"), (2, "Bakery")]
    expected_departments_schema = StructType(
        [
            StructField("department_id", IntegerType(), True),
            StructField("department", StringType(), True),
        ]
    )
    expected_departments_df = spark.createDataFrame(
        expected_departments_data, expected_departments_schema
    )

    expected_products_data = [(988, "Apple", 1), (989, "Bread", 2)]
    expected_products_schema = StructType(
        [
            StructField("product_id", IntegerType(), True),
            StructField("product_name", StringType(), True),
            StructField("department_id", IntegerType(), True),
        ]
    )
    expected_products_df = spark.createDataFrame(
        expected_products_data, expected_products_schema
    )

    expected_orders_data = [
        (1, 1990, 10, 123, 45.6, "2025-04-01", "11:27:00"),
        (2, 1991, 15, 124, 67.8, "2025-04-02", "12:30:00"),
    ]
    expected_orders_schema = StructType(
        [
            StructField("order_id", IntegerType(), True),
            StructField("user_id", IntegerType(), True),
            StructField("days_since_prior_order", IntegerType(), True),
            StructField("order_num", IntegerType(), True),
            StructField("total_amount", DoubleType(), True),
            StructField("date", StringType(), True),
            StructField("order_time", StringType(), True),
        ]
    )
    expected_orders_df = spark.createDataFrame(
        expected_orders_data, expected_orders_schema
    )

    expected_order_items_data = [
        (1, 1, 988, 1, "Not_Reorder", "2025-04-01"),
        (2, 2, 989, 2, "Reorder", "2025-04-02"),
    ]
    expected_order_items_schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("order_id", IntegerType(), True),
            StructField("product_id", IntegerType(), True),
            StructField("add_to_cart_order", IntegerType(), True),
            StructField("reordered", StringType(), True),
            StructField("date", StringType(), True),
        ]
    )
    expected_order_items_df = spark.createDataFrame(
        expected_order_items_data, expected_order_items_schema
    )

    users_df, departments_df, products_df, orders_df, order_items_df = normalize_data(
        input_df
    )

    assert users_df.collect() == expected_users_df.collect(), "Users data mismatch"
    assert (
        departments_df.collect() == expected_departments_df.collect()
    ), "Departments data mismatch"
    assert (
        products_df.collect() == expected_products_df.collect()
    ), "Products data mismatch"
    assert orders_df.collect() == expected_orders_df.collect(), "Orders data mismatch"
    assert (
        order_items_df.collect() == expected_order_items_df.collect()
    ), "Order items data mismatch"


def test_partition_data_with_sort(spark):
    """Test partitioning with sorting."""
    from ecommerce_delta import partition_data

    input_schema = StructType(
        [
            StructField("order_id", IntegerType(), True),
            StructField("user_id", IntegerType(), True),
            StructField("product_id", IntegerType(), True),
            StructField("date", StringType(), True),
        ]
    )
    input_data = [
        (1, 1990, 988, "2025-04-01"),
        (2, 1991, 989, "2025-04-02"),
        (3, 1992, 990, "2025-04-01"),
    ]
    input_df = spark.createDataFrame(input_data, input_schema)

    result_df = partition_data(
        input_df, partition_by="date", num_partitions=2, sort_by="order_id"
    )

    assert result_df.columns == input_df.columns, "Columns mismatch"
    assert sorted(result_df.collect()) == sorted(input_df.collect()), "Data mismatch"
    assert result_df.rdd.getNumPartitions() == 2, "Partition count mismatch"


def test_save_normalized_data_to_s3_with_partition(spark, caplog):
    """Test saving normalized data to S3 with partitioning."""
    from ecommerce_delta import save_normalized_data_to_s3
    import pyspark.sql.dataframe

    # Input schema and data
    input_schema = StructType(
        [
            StructField("order_id", IntegerType(), True),
            StructField("value", StringType(), True),
            StructField("date", StringType(), True),
        ]
    )
    input_data = [(1, "test1", "2025-04-01"), (2, "test2", "2025-04-02")]
    input_df = spark.createDataFrame(input_data, input_schema)

    # Create mock for DataFrameWriter
    mock_writer = MagicMock()
    mock_writer.format.return_value = mock_writer
    mock_writer.partitionBy.return_value = mock_writer
    mock_writer.mode.return_value = mock_writer

    # Patch at module level
    with patch(
        "pyspark.sql.dataframe.DataFrame.write",
        new_callable=PropertyMock,
        return_value=mock_writer,
    ):
        with caplog.at_level(logging.INFO):
            save_normalized_data_to_s3(
                input_df,
                bucket_name="test-bucket",
                output_path="lakehouse-dwh/test/",
                archive_path="archived-data/test/",
                partition_col="date",
            )

            # Verify the calls
            mock_writer.format.assert_called_with("delta")
            mock_writer.partitionBy.assert_called_with("date")
            mock_writer.mode.assert_called_with("overwrite")
            assert mock_writer.save.call_count == 2


def test_save_master_data_to_s3(spark):
    from ecommerce_delta import save_master_data_to_s3

    schema = StructType(
        [
            StructField("order_id", IntegerType(), True),
            StructField("value", StringType(), True),
            StructField("date", StringType(), True),
        ]
    )
    data = [(1, "test1", "2025-04-01"), (2, "test2", "2025-04-02")]
    df = spark.createDataFrame(data, schema)

    mock_writer = MagicMock()
    mock_writer.format.return_value = mock_writer
    mock_writer.partitionBy.return_value = mock_writer
    mock_writer.mode.return_value = mock_writer
    mock_writer.save.return_value = None  # Just in case

    with patch(
        "pyspark.sql.dataframe.DataFrame.write", new_callable=PropertyMock
    ) as mock_write:
        mock_write.return_value = mock_writer

        # 👇 Use mock paths here
        bucket_name = "test-bucket"
        output_path = "s3://test-bucket/output"
        archive_path = "s3://test-bucket/archive"

        save_master_data_to_s3(df, bucket_name, output_path, archive_path)
