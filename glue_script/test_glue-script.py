import pytest
import sys
import types
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    StringType,
    TimestampType,
)
import tempfile
import shutil
from datetime import datetime

# Create mock AWS Glue modules
class MockDynamicFrame:
    def __init__(self, dataframe=None):
        self.dataframe = dataframe

    def toDF(self):
        return self.dataframe

class MockTransforms:
    @staticmethod
    def ApplyMapping(*args, **kwargs):
        return MockDynamicFrame()
    
    @staticmethod
    def ResolveChoice(*args, **kwargs):
        return MockDynamicFrame()
    
    @staticmethod
    def DropNullFields(*args, **kwargs):
        return MockDynamicFrame()

class MockUtils:
    @staticmethod
    def getResolvedOptions(sys_args, options_list):
        return {option: f"mock_{option}" for option in options_list}

class MockContext:
    def __init__(self):
        self.spark_session = None
    
    def getSparkSession(self):
        return self.spark_session
    
    def create_dynamic_frame_from_options(self, *args, **kwargs):
        return MockDynamicFrame()
    
    def create_dynamic_frame_from_catalog(self, *args, **kwargs):
        return MockDynamicFrame()

class MockJob:
    def __init__(self):
        pass
    
    def init(self, *args, **kwargs):
        pass
    
    def commit(self):
        pass

# Create the AWS Glue module structure
mock_glue = types.ModuleType('awsglue')
mock_glue.DynamicFrame = MockDynamicFrame
mock_glue.transforms = types.ModuleType('awsglue.transforms')
mock_glue.transforms.ApplyMapping = MockTransforms.ApplyMapping
mock_glue.transforms.ResolveChoice = MockTransforms.ResolveChoice
mock_glue.transforms.DropNullFields = MockTransforms.DropNullFields
mock_glue.utils = types.ModuleType('awsglue.utils')
mock_glue.utils.getResolvedOptions = MockUtils.getResolvedOptions
mock_glue.context = types.ModuleType('awsglue.context')
mock_glue.context.GlueContext = MockContext
mock_glue.job = types.ModuleType('awsglue.job')
mock_glue.job.Job = MockJob

# Add mock modules to sys.modules
sys.modules['awsglue'] = mock_glue
sys.modules['awsglue.transforms'] = mock_glue.transforms
sys.modules['awsglue.utils'] = mock_glue.utils
sys.modules['awsglue.context'] = mock_glue.context
sys.modules['awsglue.job'] = mock_glue.job

# Mock the delta module (needed for configure_spark_with_delta_pip)
mock_delta = types.ModuleType('delta')
def mock_configure_spark_with_delta_pip(*args, **kwargs):
    return args[0]
mock_delta.configure_spark_with_delta_pip = mock_configure_spark_with_delta_pip
sys.modules['delta'] = mock_delta

# Now import the functions to test from your Glue script
try:
    from ecommerce_delta import (
        initialize_glue_context,
        load_data_from_s3,
        transform_order_items_data,
        join_datasets,
        normalize_data,
        partition_data,
        save_normalized_data_to_s3,
        save_master_data_to_s3,
    )
except ImportError as e:
    print(f"Warning: Could not import from ecommerce_delta. Make sure the file exists and is in the Python path. Error: {e}")
    # Create dummy functions for testing if the actual module can't be imported
    def initialize_glue_context(*args, **kwargs): return MagicMock(), MagicMock(), MagicMock(), MagicMock()
    def load_data_from_s3(*args, **kwargs): return MagicMock()
    def transform_order_items_data(*args, **kwargs): return MagicMock()
    def join_datasets(*args, **kwargs): return MagicMock()
    def normalize_data(*args, **kwargs): return MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
    def partition_data(*args, **kwargs): return MagicMock()
    def save_normalized_data_to_s3(*args, **kwargs): return None
    def save_master_data_to_s3(*args, **kwargs): return None

@pytest.fixture(scope="module")
def spark_session():
    """Create a Spark session for tests"""
    spark = (
        SparkSession.builder
        .appName("TestGlueDataPipeline")
        .master("local[2]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    yield spark
    spark.stop()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing writes"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def glue_context_mock():
    """Mock GlueContext and Job for testing"""
    glue_context = MagicMock()
    job = MagicMock()
    glue_context.get_job = MagicMock(return_value=job)
    return glue_context, job

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

@patch("ecommerce_delta.getResolvedOptions")
@patch("ecommerce_delta.SparkContext")
@patch("ecommerce_delta.GlueContext")
@patch("ecommerce_delta.Job")
def test_initialize_glue_context(mock_job, mock_glue_context, mock_spark_context, mock_getResolvedOptions, spark_session):
    """Test initialize_glue_context"""
    # Mock arguments
    mock_getResolvedOptions.return_value = {
        "JOB_NAME": "TestJob",
        "BUCKET_NAME": "fake_bucket"
    }
    
    # Mock GlueContext and Job
    mock_glue_context_instance = MagicMock()
    mock_glue_context.return_value = mock_glue_context_instance
    mock_job_instance = MagicMock()
    mock_job.return_value = mock_job_instance
    
    # Mock SparkContext
    mock_spark_context_instance = MagicMock()
    mock_spark_context.return_value = mock_spark_context_instance

    # Mock SparkSession (use the fixture's spark_session)
    with patch("ecommerce_delta.SparkSession.builder", spark_session.builder):
        glue_context, spark, args, job = initialize_glue_context("TestJob")

    # Assertions
    assert glue_context == mock_glue_context_instance
    assert isinstance(spark, SparkSession)
    assert args == {"JOB_NAME": "TestJob", "BUCKET_NAME": "fake_bucket"}
    assert job == mock_job_instance
    mock_job_instance.init.assert_called_with("TestJob", {"JOB_NAME": "TestJob", "BUCKET_NAME": "fake_bucket"})

@patch("pyspark.sql.readwriter.DataFrameReader.csv")
def test_load_data_from_s3(mock_csv, spark_session, order_items_df):
    """Test load_data_from_s3"""
    mock_csv.return_value = order_items_df

    bucket_name = "fake_bucket"
    folder_path = "raw-data/order_items_apr_2025/"
    df = load_data_from_s3(spark_session, bucket_name, folder_path)

    assert df.count() == 2
    assert len(df.columns) == 9
    assert df.schema["order_id"].dataType == IntegerType()
    mock_csv.assert_called_with(
        f"s3a://{bucket_name}/{folder_path}", header=True, inferSchema=True
    )

def test_transform_order_items_data(order_items_df):
    """Test transform_order_items_data"""
    transformed_df = transform_order_items_data(order_items_df)

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

def test_join_datasets(order_items_df, orders_df, products_df):
    """Test join_datasets"""
    transformed_df = transform_order_items_data(order_items_df)
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
    assert sorted(final_df.columns) == sorted(expected_columns)
    assert final_df.filter(col("product_name") == "Apple").count() == 1

def test_normalize_data(order_items_df, orders_df, products_df):
    """Test normalize_data"""
    transformed_df = transform_order_items_data(order_items_df)
    final_df = join_datasets(transformed_df, orders_df, products_df)
    users_df, departments_df, products_df_normalized, orders_df_normalized, order_items_df_normalized = normalize_data(final_df)

    # Users table
    assert users_df.count() == 2
    assert users_df.columns == ["user_id"]

    # Departments table
    assert departments_df.count() == 2
    assert sorted(departments_df.columns) == sorted(["department_id", "department"])

    # Products table
    assert products_df_normalized.count() == 2
    assert sorted(products_df_normalized.columns) == sorted(["product_id", "product_name", "department_id"])

    # Orders table
    assert orders_df_normalized.count() == 2
    assert sorted(orders_df_normalized.columns) == sorted([
        "order_id",
        "user_id",
        "days_since_prior_order",
        "order_num",
        "total_amount",
        "date",
        "order_time",
    ])

    # Order Items table
    assert order_items_df_normalized.count() == 2
    assert sorted(order_items_df_normalized.columns) == sorted([
        "id",
        "order_id",
        "product_id",
        "add_to_cart_order",
        "reordered",
        "date",
    ])

def test_partition_data(order_items_df, orders_df, products_df):
    """Test partition_data"""
    transformed_df = transform_order_items_data(order_items_df)
    final_df = join_datasets(transformed_df, orders_df, products_df)

    # Test with num_partitions and sort_by
    partitioned_df = partition_data(
        final_df, "date", num_partitions=15, sort_by="department,reordered"
    )
    assert partitioned_df.rdd.getNumPartitions() == 15

    # Test with num_partitions only
    partitioned_df = partition_data(final_df, "date", num_partitions=10)
    assert partitioned_df.rdd.getNumPartitions() == 10

    # Test without num_partitions
    partitioned_df = partition_data(final_df, "date")
    assert partitioned_df.rdd.getNumPartitions() > 0

@patch("pyspark.sql.readwriter.DataFrameWriter.save")
def test_save_normalized_data_to_s3(mock_save, spark_session, temp_dir):
    """Test save_normalized_data_to_s3"""
    normalized_df = spark_session.createDataFrame(
        [(1, "Apple", 401), (2, "Milk", 402)],
        ["product_id", "product_name", "department_id"],
    )
    bucket_name = "fake_bucket"
    output_path = "normalized-data/products/"
    archive_path = "archived-data/products/"
    partition_col = "department_id"

    save_normalized_data_to_s3(normalized_df, bucket_name, output_path, archive_path, partition_col)

    # Check that save was called twice (primary and archive)
    assert mock_save.call_count == 2
    primary_call = mock_save.call_args_list[0][0][0]
    archive_call = mock_save.call_args_list[1][0][0]
    assert primary_call == f"s3a://{bucket_name}/{output_path}"
    assert archive_call == f"s3a://{bucket_name}/{archive_path}"

@patch("pyspark.sql.readwriter.DataFrameWriter.save")
def test_save_master_data_to_s3(mock_save, spark_session, temp_dir, order_items_df, orders_df, products_df):
    """Test save_master_data_to_s3"""
    transformed_df = transform_order_items_data(order_items_df)
    final_df = join_datasets(transformed_df, orders_df, products_df)
    
    bucket_name = "fake_bucket"
    output_path = "lakehouse-dwh/master/"
    archive_path = "archived-data/master/"

    save_master_data_to_s3(final_df, bucket_name, output_path, archive_path)

    # Check that save was called twice (primary and archive)
    assert mock_save.call_count == 2
    primary_call = mock_save.call_args_list[0][0][0]
    archive_call = mock_save.call_args_list[1][0][0]
    assert primary_call == f"s3a://{bucket_name}/{output_path}"
    assert archive_call == f"s3a://{bucket_name}/{archive_path}"

@patch("ecommerce_delta.initialize_glue_context")
@patch("ecommerce_delta.load_data_from_s3")
@patch("ecommerce_delta.transform_order_items_data")
@patch("ecommerce_delta.join_datasets")
@patch("ecommerce_delta.partition_data")
@patch("ecommerce_delta.save_master_data_to_s3")
@patch("ecommerce_delta.normalize_data")
@patch("ecommerce_delta.save_normalized_data_to_s3")
def test_main(
    mock_save_normalized_data_to_s3,
    mock_normalize_data,
    mock_save_master_data_to_s3,
    mock_partition_data,
    mock_join_datasets,
    mock_transform_order_items_data,
    mock_load_data_from_s3,
    mock_initialize_glue_context,
    spark_session,
    order_items_df,
    orders_df,
    products_df
):
    """Test the main function"""
    # Setup mocks
    mock_glue_context = MagicMock()
    mock_job = MagicMock()
    mock_args = {"BUCKET_NAME": "fake_bucket"}
    mock_initialize_glue_context.return_value = (mock_glue_context, spark_session, mock_args, mock_job)
    
    # Mock load_data_from_s3 to return our test DataFrames
    mock_load_data_from_s3.side_effect = [order_items_df, orders_df, products_df]
    
    # Mock transform_order_items_data
    transformed_df = transform_order_items_data(order_items_df)
    mock_transform_order_items_data.return_value = transformed_df
    
    # Mock join_datasets
    final_df = join_datasets(transformed_df, orders_df, products_df)
    mock_join_datasets.return_value = final_df
    
    # Mock partition_data
    mock_partition_data.return_value = final_df
    
    # Mock normalize_data
    normalized_tables = normalize_data(final_df)
    mock_normalize_data.return_value = normalized_tables
    
    # Import main function
    from ecommerce_delta import main
    
    # Run the main function
    main()
    
    # Verify all the functions were called with correct arguments
    mock_initialize_glue_context.assert_called_once_with("DeltaLakeETLJob")
    
    # Verify load_data_from_s3 was called 3 times
    assert mock_load_data_from_s3.call_count == 3
    mock_load_data_from_s3.assert_any_call(spark_session, "fake_bucket", "raw-data/order_items_apr_2025/")
    mock_load_data_from_s3.assert_any_call(spark_session, "fake_bucket", "raw-data/orders_apr_2025/")
    mock_load_data_from_s3.assert_any_call(spark_session, "fake_bucket", "raw-data/products/")
    
    # Verify transform_order_items_data was called
    mock_transform_order_items_data.assert_called_once()
    
    # Verify join_datasets was called
    mock_join_datasets.assert_called_once()
    
    # Verify partition_data was called
    mock_partition_data.assert_called_once_with(final_df, partition_by="date", num_partitions=15, sort_by="department,reordered")
    
    # Verify save_master_data_to_s3 was called
    mock_save_master_data_to_s3.assert_called_once_with(final_df, "fake_bucket", "lakehouse-dwh/master/", "archived-data/master/")
    
    # Verify normalize_data was called
    mock_normalize_data.assert_called_once()
    
    # Verify save_normalized_data_to_s3 was called 5 times for each normalized table
    assert mock_save_normalized_data_to_s3.call_count == 5
    
    # Verify job.commit() was called
    mock_job.commit.assert_called_once()