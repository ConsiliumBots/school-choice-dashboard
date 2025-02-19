import os
from dotenv import load_dotenv
load_dotenv()
import psycopg2

#Funcion para conectarse a la base de datos core
def conect_bd(database, environment):
    if environment == 'staging':
        host = "staging.cluster-cuasuehzhaay.us-east-1.rds.amazonaws.com"
    elif environment == 'production':
        host = "production.cluster-cuasuehzhaay.us-east-1.rds.amazonaws.com"  
    conn = psycopg2.connect(
        database = database,
        user = os.getenv("db_user"),
        password = os.getenv("db_password"),
        port = os.getenv("db_port"),
        host = host
    )
    return conn

#Funcion para conectarse a la base de datos de users de prod
def conect_bd_users_prod(database):
    conn = psycopg2.connect(
        database = database,
        user = os.getenv("users_user"),
        password = os.getenv("users_password"),
        port = "5432",
        host = "users.cluster-cuasuehzhaay.us-east-1.rds.amazonaws.com"
    )
    return conn
