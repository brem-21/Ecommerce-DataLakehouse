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
)
import tempfile
import shutil
from datetime import datetime
import os
import logging

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Mock the delta module
mock_delta = types.ModuleType('delta')
def mock_configure_spark_with_delta_pip(spark_builder, *args, **kwargs):
    return spark_builder
mock_delta.configure_spark_with_delta_pip = mock_configure_spark_with_delta_pip
sys.modules['delta'] = mock_delta

# Verify environment setup
def verify_environment():
    try:
        import pyspark
        import delta
        logger.info(f"PySpark version: {pyspark.__version__}")
        logger.info(f"Delta Spark version: {delta.__version__}")
        delta_jar_path = "/home/brempong/Ecommerce-DataLakehouse/glue_script/delta-core_2.12-2.4.0.jar"
        if not os.path.exists(delta_jar_path):
            logger.error(f"Delta JAR file not found at {delta_jar_path}")
            raise FileNotFoundError(f"{delta_jar_path} not found")
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        raise

verify_environment()

# Import functions to test
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
        main,
    )
except ImportError as e:
    logger.error(f"Could not import from ecommerce_delta: {str(e)}")
    raise

@pytest.fixture(scope="module")
def spark_session():
    """Create a Spark session with Delta Lake configuration"""
    delta_jar_path = "/home/brempong/Ecommerce-DataLakehouse/glue_script/delta-core_2.12-2.4.0.jar"
    spark = (
        SparkSession.builder
        .appName("TestGlueDataPipeline")
        .master("local[2]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars", delta_jar_path)
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
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
    mock_getResolvedOptions.return_value = {
        "JOB_NAME": "TestJob",
        "BUCKET_NAME": "fake_bucket"
    }
    mock_glue_context_instance = MagicMock()
    mock_glue_context.return_value = mock_glue_context_instance
    mock_job_instance = MagicMock()
    mock_job.return_value = mock_job_instance
    mock_spark_context_instance = MagicMock()
    mock_spark_context.return_value = mock_spark_context_instance

    with patch("ecommerce_delta.SparkSession.builder", spark_session.builder):
        glue_context, spark, args, job = initialize_glue_context("TestJob")

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
    logger.info("Loaded DataFrame schema:")
    df.printSchema()
    logger.info("Loaded DataFrame data:")
    df.show()
    assert df.count() == 2
    assert len(df.columns) == 9
    assert df.schema["order_id"].dataType == IntegerType()
    mock_csv.assert_called_with(
        f"s3a://{bucket_name}/{folder_path}", header=True, inferSchema=True
    )

def test_transform_order_items_data(order_items_df):
    """Test transform_order_items_data"""
    transformed_df = transform_order_items_data(order_items_df)
    logger.info("Transformed DataFrame schema:")
    transformed_df.printSchema()
    logger.info("Transformed DataFrame data:")
    transformed_df.show()
    assert "order_time" in transformed_df.columns
    assert transformed_df.filter(col("reordered") == "Reorder").count() == 1
    assert transformed_df.filter(col("reordered") == "Not_Reorder").count() == 1
    order_time = transformed_df.select("order_time").first()[0]
    try:
        datetime.strptime(order_time, "%H:%M:%S")
    except ValueError:
        pytest.fail("order_time format is not HH:mm:ss")

def test_join_datasets(order_items_df, orders_df, products_df):
    """Test join_datasets"""
    transformed_df = transform_order_items_data(order_items_df)
    logger.info("Transformed order_items DataFrame:")
    transformed_df.printSchema()
    transformed_df.show()
    logger.info("Orders DataFrame:")
    orders_df.printSchema()
    orders_df.show()
    logger.info("Products DataFrame:")
    products_df.printSchema()
    products_df.show()
    final_df = join_datasets(transformed_df, orders_df, products_df)
    logger.info("Joined DataFrame schema:")
    final_df.printSchema()
    logger.info("Joined DataFrame data:")
    final_df.show()
    assert final_df.count() == 2
    expected_columns = [
        "id", "order_id", "user_id", "days_since_prior_order", "product_id",
        "product_name", "department_id", "department", "add_to_cart_order",
        "re