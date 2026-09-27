import os
import yaml
from langchain_community.utilities import SQLDatabase
import boto3
from botocore.config import Config
from langchain.prompts import PromptTemplate
from langchain.chains.sql_database.query import create_sql_query_chain
from sqlalchemy import create_engine

"""
Here we will build the required parameter to connect athena and query database.
1. Data is stored in S3 and metadata in Glue metastore.
2. Create a profille which will have access to the required service.
3. if the database exists and s3 buckets exists use them else create.

"""
# Connection details live in athena_config.yaml (git-ignored).
# Copy athena_config.yaml.example to get started.
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "athena_config.yaml")
with open(config_path, "r") as f:
    athena_config = yaml.safe_load(f)

aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
region = athena_config["region"]
athena_url = f"athena.{region}.amazonaws.com"
athena_port = "443"  # Update, if port is different
athena_db = athena_config["athena_db"]
s3stagingathena = athena_config["s3_staging_dir"]
athena_wkgrp = athena_config["workgroup"]
athena_connection_string = f"awsathena+rest://{aws_access_key_id}:{aws_secret_access_key}@{athena_url}:{athena_port}/{athena_db}?s3_staging_dir={s3stagingathena}/&work_group={athena_wkgrp}"

"""
Under the hood, LangChain uses SQLAlchemy to connect to SQL databases. 
The SQLDatabaseChain can therefore be used with any SQL dialect 
supported by SQLAlchemy, such as MS SQL, MySQL, MariaDB, PostgreSQL, 
Oracle SQL, and SQLite. 
"""
athena_engine = create_engine(athena_connection_string, echo=True)
athena_db_connection = SQLDatabase(athena_engine)

"""
Note in athena_connection_string I have not passed the access key or secret.
Here I am using profile which has all these along with access token. 
You might need to pass. for that use below
connection_string = f"awsathena+rest://{aws_access_key_id}:{aws_secret_access_key}@{athena_url}:{athena_port}/{athena_db}?s3_staging_dir={s3stagingathena}/&work_group={athena_wkgrp}";

"""
from langchain.llms import huggingface_pipeline
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline

text_generation_pipeline = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=300,
)

prompt_template = """
### [INST] 
Instruction: Answer the question based on your 
fantasy football knowledge. Here is context to help:

{context}

### QUESTION:
{question} 

[/INST]
 """

mistral_llm = huggingface_pipeline(pipeline=text_generation_pipeline)

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = create_sql_query_chain(
    llm=mistral_llm, db=athena_db_connection, prompt=prompt
)
print(llm_chain(context="context", question="question"))
