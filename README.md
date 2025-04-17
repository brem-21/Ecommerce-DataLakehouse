# Ecommerce-DataLakehouse [![Integrating dev changes to main](https://github.com/brem-21/Ecommerce-DataLakehouse/actions/workflows/integrating.yml/badge.svg)](https://github.com/brem-21/Ecommerce-DataLakehouse/actions/workflows/integrating.yml)

A Lakehouse architecture for an e-commerce platform on AWS. The system ingests raw transactional data stored in Amazon S3, cleans and deduplicates it using Delta Lake, and exposes it for downstream analytics through Amazon Athena.

## Project Structure

```
├── scripts/
│   ├── Data/           # Raw data files
│   ├── Dockerfile      # Container configuration
│   ├── Makefile        # Build automation
│   └── requirements.txt
├── notebooks/          # Jupyter notebooks for data exploration
│   ├── product.ipynb
│   ├── order_items.ipynb
│   └── script.ipynb
├── tests/              # Test suite
│   ├── test_script.py
│   └── script.py
└── requirements.txt    # Project dependencies
```

## Features

- Data ingestion from various sources into S3
- Data transformation using PySpark
- Normalized data model with dimension and fact tables
- Automated data quality checks
- Partitioned data storage for optimal query performance

## Data Model

The system implements a star schema with the following tables:

- **Users**: Dimension table for user information
- **Departments**: Dimension table for product departments
- **Products**: Dimension table linking products to departments
- **Orders**: Dimension table for order metadata
- **Order Items**: Fact table connecting orders, products, and users

## Prerequisites

- Python 3.9+
- Docker
- AWS Account with S3 access
- Apache Spark 3.5+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/brem-21/Ecommerce-DataLakehouse.git
cd Ecommerce-DataLakehouse
```

2. Install dependencies:
```bash
make install
```

3. Set up environment variables:
```bash
# Create .env file with:
Access_key_ID=your_aws_access_key
Secret_access_key=your_aws_secret_key
BUCKET_NAME=your_s3_bucket
REGION_NAME=your_aws_region
```

## Usage

### Local Development

1. Format and lint code:
```bash
make refactor
```

2. Run tests:
```bash
make test
```

### Docker Deployment

1. Build the container:
```bash
cd scripts
docker build -t ecommerce-datalake .
```

2. Run the container:
```bash
docker run -e Access_key_ID=your_key \
           -e Secret_access_key=your_secret \
           -e BUCKET_NAME=your_bucket \
           -e REGION_NAME=your_region \
           ecommerce-datalake
```

## Data Pipeline

1. Raw data ingestion to S3
2. Data transformation using PySpark
3. Data quality checks and deduplication
4. Normalized table creation
5. Partitioned storage in processed format

## Testing

Run the test suite:
```bash
make test
```

Tests cover:
- Data loading functionality
- Transformation logic
- Data quality checks
- Storage operations

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
