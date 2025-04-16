import os
import sys
from typing import Dict, Any
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv