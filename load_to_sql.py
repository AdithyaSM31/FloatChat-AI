# load_to_sql.py

import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

# --- DATABASE CONNECTION DETAILS ---
# Uses environment variables for Railway deployment
# Railway provides DATABASE_PUBLIC_URL (for external access) and DATABASE_URL (internal only)
DATABASE_URL = os.getenv('DATABASE_PUBLIC_URL') or os.getenv('DATABASE_URL')

if DATABASE_URL:
    # Railway uses postgres:// but SQLAlchemy needs postgresql://
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    engine_string = DATABASE_URL
    print(f"Using DATABASE_URL from Railway (public: {bool(os.getenv('DATABASE_PUBLIC_URL'))})")
else:
    # Fallback to manual construction for local development
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'aadu3134')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'argo_db')
    engine_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    print(f"Using manual database configuration for {db_host}:{db_port}/{db_name}")

# --- The name of our data table ---
table_name = 'argo_data'

# --- Path to our data file ---
parquet_file = 'argo_final_data.parquet'

print(f"Reading data from {parquet_file}...")
df = pd.read_parquet(parquet_file)

# The 'platform_number' column was read as a binary object, let's decode it
# This is a common step when moving from certain file formats
if 'platform_number' in df.columns and df['platform_number'].dtype == 'object':
    # Check if the column contains bytes and decode
    if isinstance(df['platform_number'].iloc[0], bytes):
        df['platform_number'] = df['platform_number'].str.decode('utf-8')


print(f"Successfully loaded {len(df)} rows.")

# Create the engine with connection pooling and retry logic
engine = create_engine(
    engine_string,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
    connect_args={
        'connect_timeout': 30,
        'keepalives': 1,
        'keepalives_idle': 30,
        'keepalives_interval': 10,
        'keepalives_count': 5,
    }
)

print(f"Connecting to database and loading data into table '{table_name}'...")
print("This may take several minutes for 1.3 million rows...")

# Use df.to_sql to load the data with progress updates
# if_exists='replace': Deletes the old table and creates a new one. Good for re-running the script.
# chunksize: Loads data in batches of 5000 rows
import time
start_time = time.time()

try:
    df.to_sql(
        table_name,
        con=engine,
        if_exists='replace',
        index=False,
        chunksize=5000,  # Increased chunk size for faster loading
        method='multi'  # Use multi-row insert for better performance
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ SUCCESS: Data has been loaded into the PostgreSQL database!")
    print(f"   Total rows: {len(df):,}")
    print(f"   Time taken: {elapsed_time:.2f} seconds")
    print(f"   Average: {len(df)/elapsed_time:.0f} rows/second")
    
except Exception as e:
    print(f"\n❌ ERROR: Failed to load data: {e}")
    print("\nTroubleshooting:")
    print("1. Check if Railway PostgreSQL is running (not sleeping)")
    print("2. Verify the DATABASE_PUBLIC_URL is correct")
    print("3. Check Railway dashboard for database status")
    raise