import os
from dotenv import load_dotenv
load_dotenv()
import psycopg2

#Funcion para conectarse a la base de datos core
def conect_bd(database):
    db_host = os.getenv("db_host")
    conn = psycopg2.connect(
        database = database,
        user = os.getenv("db_user"),
        password = os.getenv("db_password"),
        port = os.getenv("db_port"),
        host = db_host
    )
    return conn

#Funcion para conectarse a la base de datos de users de prod
def conect_bd_users_prod(database):
    db_host = os.getenv("db_users_host")
    port = os.getenv("db_port")
    db_user = os.getenv("db_users_user")
    db_pw = os.getenv("db_users_password")
    conn = psycopg2.connect(
        database = database,
        user = db_user,
        password = db_pw,
        port = port,
        host = db_host
    )
    return conn
