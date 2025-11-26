from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware
# import chromadb  # Disabled for Railway deployment (too heavy)
# from sentence_transformers import SentenceTransformer  # Disabled for Railway deployment (too heavy)
import numpy as np
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_PASSWORD = os.getenv("DB_PASSWORD", "YOUR_POSTGRES_PASSWORD")

# --- INITIALIZE VECTOR DB AND EMBEDDING MODEL ON STARTUP ---
# Disabled for Railway deployment to reduce build time and memory usage
collection = None
embedding_model = None
# try:
#     chroma_client = chromadb.PersistentClient(path="chroma_db")
#     collection = chroma_client.get_or_create_collection(name="argo_float_summaries")
#     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#     logger.info("✅ ChromaDB and sentence-transformer model loaded successfully.")
# except Exception as e:
#     logger.error(f"❌ ERROR: Could not load ChromaDB. Run populate_vectordb.py first. Error: {e}")
logger.info("⚠️ ChromaDB disabled for Railway deployment. App will work without vector search.")

app = FastAPI(
    title="OceanGPT API (Groq Edition)",
    description="API for querying ARGO float data using natural language, powered by Groq LPU.",
    version="3.0.0"
)

# Allow CORS for front-end development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. DATABASE CONNECTION ---
# Railway provides DATABASE_PUBLIC_URL and DATABASE_URL
# DATABASE_PUBLIC_URL works from anywhere, DATABASE_URL only works within Railway's private network
DATABASE_URL = os.getenv('DATABASE_PUBLIC_URL') or os.getenv('DATABASE_URL')

if DATABASE_URL:
    # Railway uses postgres:// but SQLAlchemy needs postgresql://
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    engine_string = DATABASE_URL
    logger.info(f"Using DATABASE_URL from Railway (public: {bool(os.getenv('DATABASE_PUBLIC_URL'))})")
else:
    # Fallback to manual construction for local development
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'argo_db')
    engine_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    logger.info("Using manual database configuration")

# Create engine with connection pooling and better timeout settings
engine = create_engine(
    engine_string,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_pre_ping=True,  # Test connections before using them
    connect_args={
        'connect_timeout': 10,
        'keepalives': 1,
        'keepalives_idle': 30,
        'keepalives_interval': 10,
        'keepalives_count': 5,
    }
)

def run_query(query_string: str) -> pd.DataFrame:
    """Executes a SQL query and returns the result as a pandas DataFrame."""
    try:
        with engine.connect() as connection:
            return pd.read_sql(text(query_string), connection)
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database query error: {e}")

# --- 3. ADVANCED AI CONTEXT ---
def get_db_context():
    """Fetches schema, date range, and unique floats to give the AI better context."""
    try:
        # Check if argo_data table exists first
        table_check_query = "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'argo_data');"
        
        with engine.connect() as connection:
            table_exists = pd.read_sql(text(table_check_query), connection).iloc[0, 0]
            
            if not table_exists:
                logger.warning("argo_data table does not exist in database")
                return "Database table 'argo_data' not found. Please ensure data is loaded."
            
            schema_query = "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'argo_data'"
            date_range_query = "SELECT MIN(juld)::date AS min_date, MAX(juld)::date AS max_date FROM argo_data;"
            platform_query = "SELECT DISTINCT platform_number FROM argo_data ORDER BY platform_number LIMIT 10;"
            
            schema_df = pd.read_sql(text(schema_query), connection)
            date_range_df = pd.read_sql(text(date_range_query), connection)
            platform_df = pd.read_sql(text(platform_query), connection)

        schema_info = "\n".join([f"- {row['column_name']} ({row['data_type']})" for _, row in schema_df.iterrows()])
        date_range_info = f"The data covers dates from {date_range_df['min_date'][0]} to {date_range_df['max_date'][0]}."
        platform_info = "Available platform_number values include: " + ", ".join(platform_df['platform_number'].astype(str).tolist()[:10]) + ", among others."

        full_context = f"""
        You are querying a PostgreSQL table named 'argo_data' with the following schema:
        {schema_info}

        Contextual Information:
        - {date_range_info}
        - {platform_info}
        """
        logger.info("✅ Database context loaded successfully")
        return full_context
    except Exception as e:
        logger.error(f"❌ Error fetching DB context: {e}")
        return "Database context could not be loaded. Using fallback context."

# Initialize DB context with error handling
try:
    DB_CONTEXT = get_db_context()
except Exception as e:
    logger.error(f"Failed to initialize DB context on startup: {e}")
    DB_CONTEXT = "Database context initialization failed. Queries may not work correctly."

# --- 4. RAG & CORE AI LOGIC ---
def find_relevant_context(user_question: str) -> str:
    """Searches the vector DB for context relevant to the user's question."""
    if collection is None or embedding_model is None:
        return "Vector database not available. Skipping RAG retrieval."
    try:
        query_embedding = embedding_model.encode(user_question).tolist()
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=3, 
            include=['documents']
        )
        context = "\n---\n".join(results.get('documents', [[]])[0])
        return context if context else "No specific context found in the vector database."
    except Exception as e:
        logger.error(f"Error during vector search: {e}")
        return "Error searching vector database."

def get_sql_query(user_question: str, context: str) -> str:
    """Converts a user question to a SQL query using Groq."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set.")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are an expert PostgreSQL query writer. Your task is to convert a user's question into a single, syntactically correct SQL query. "
             "Use the provided **retrieved context** and **database context** to help you write the most accurate query.\n\n"
             "--- RETRIEVED CONTEXT (from vector search) ---\n{context}\n--------------------------------------------\n\n"
             "Follow these rules precisely:\n"
             "1. **For 'highest'/'lowest'/'latest' records (e.g., 'furthest south'), ALWAYS use `ORDER BY` and `LIMIT 1`.** DO NOT use `GROUP BY`. "
             "   - 'Furthest south' means `ORDER BY latitude ASC LIMIT 1`. 'Furthest west' means `ORDER BY longitude ASC LIMIT 1`.\n"
             "2. **CRITICAL: When using aggregate functions (MAX, MIN, AVG, SUM, COUNT), either:**\n"
             "   - Use only the aggregate function: `SELECT MAX(pressure) AS max_pressure FROM argo_data`\n"
             "   - Or use GROUP BY for non-aggregate columns: `SELECT platform_number, MAX(pressure) AS max_pressure FROM argo_data GROUP BY platform_number`\n"
             "   - For 'deepest measurement location', use a subquery: `SELECT platform_number, latitude, longitude, pressure FROM argo_data WHERE pressure = (SELECT MAX(pressure) FROM argo_data)`\n"
             "3. **For aggregates on a 'top N' subset, ALWAYS use a subquery.**\n"
             "4. **ALWAYS use descriptive aliases for aggregate columns** (e.g., `AVG(temperature) AS average_temperature`).\n"
             "5. **Interpret geographical terms**: 'equator' means `latitude BETWEEN -5 AND 5`.\n"
             "6. **Always include columns mentioned by the user.** If asked 'Which float...', you must select `platform_number`.\n"
             "7. Only output the SQL query. Nothing else. **DO NOT include any explanation or markdown formatting like ```sql...```.**\n\n"
             "--- DATABASE CONTEXT ---\n{db_context}\n-------------------------"),
            ("human", "{question}")
        ]
    )
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
    chain = prompt | llm
    response = chain.invoke({"question": user_question, "context": context, "db_context": DB_CONTEXT})
    sql_query = response.content.strip().replace("```sql", "").replace("```", "").strip()
    return sql_query

def get_natural_language_summary(question: str, results_df: pd.DataFrame) -> str:
    """Generates a natural language summary of the query results using Groq."""
    if results_df.empty:
        return "I couldn't find any data that matches your query. Please try asking in a different way or check the data availability."
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set.")
    
    # Limit rows for summary to avoid Groq token limits
    summary_df = results_df.head(50) if len(results_df) > 50 else results_df
    results_str = summary_df.to_markdown(index=False)
    
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful oceanographic data analyst. The user asked: '{question}'. "
        "The following data was retrieved from the database:\n{results}\n\n"
        "Please provide a concise, natural language summary of the findings."
    )
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
    chain = prompt | llm
    response = chain.invoke({"question": question, "results": results_str})
    return response.content

def decompose_question(user_question: str) -> list[str]:
    """Uses Groq to break a complex question into a list of simple questions."""
    if len(user_question.split()) < 10 and ' and ' not in user_question.lower() and ' vs ' not in user_question.lower() and ' versus ' not in user_question.lower():
        return [user_question]
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert at analyzing user queries. Your task is to decompose a complex question into a list of simple, self-contained questions. "
         "Each simple question must be answerable by a single database query. "
         "Respond ONLY with a valid JSON array of strings. For example, for the input 'Compare the temperature in the Arabian Sea and the salinity in 2023', "
         "you must output: [\"What is the average temperature in the Arabian Sea?\", \"What was the maximum salinity in 2023?\"]"),
        ("human", "{question}")
    ])
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
    chain = prompt | llm
    response = chain.invoke({"question": user_question})
    
    logger.info(f"Decomposer LLM raw output: {response.content}")
    cleaned_content = response.content.strip().removeprefix("```json").removesuffix("```").strip()
    
    try:
        return json.loads(cleaned_content)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to decompose question after cleaning, treating as a single query.")
        return [user_question]

def synthesize_answers(original_question: str, individual_answers: list[dict]) -> str:
    """Uses Groq to combine multiple answers into a single, cohesive response."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set.")

    answers_text = "\n\n".join([f"In response to the sub-question '{ans['question']}', the finding was: {ans['summary']}" for ans in individual_answers])
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful data analyst assistant. The user asked a complex question, which was broken down and answered in parts. "
         "Your job is to synthesize the individual findings into a single, cohesive, and easy-to-read final answer for the user. "
         "Start by acknowledging the user's original question. Do not show the sub-questions in your final response."),
        ("human", 
         "My original question was: '{original_question}'.\n\n"
         "Here are the findings for the different parts of my question:\n{answers_text}\n\n"
         "Please provide a final, combined answer.")
    ])
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
    chain = prompt | llm
    response = chain.invoke({"original_question": original_question, "answers_text": answers_text})
    
    return response.content

def answer_single_question(question: str) -> dict:
    """Refactored core logic to handle one question and return a dictionary."""
    context = find_relevant_context(question)
    sql_query = get_sql_query(question, context)
    full_results_df = run_query(sql_query)
    total_rows = len(full_results_df)

    MAX_ROWS_FOR_SUMMARY = 50  # Reduced for Groq token limits
    MAX_ROWS_FOR_FRONTEND = 100  # Limit data sent to frontend to prevent browser freeze
    
    df_for_summary = full_results_df.head(MAX_ROWS_FOR_SUMMARY)
    summary = get_natural_language_summary(question, df_for_summary)

    if total_rows > MAX_ROWS_FOR_SUMMARY:
        summary += f"\n\n*Note: Summary based on first {MAX_ROWS_FOR_SUMMARY} rows of {total_rows:,} total records.*"

    # Limit data sent to frontend
    results_df = full_results_df.head(MAX_ROWS_FOR_FRONTEND).copy()
    
    for col in results_df.columns:
        if pd.api.types.is_numeric_dtype(results_df[col].dtype):
            results_df[col] = results_df[col].apply(lambda x: float(x) if isinstance(x, np.number) else x)
        else:
            # Handle datetime and other objects safely
            try:
                if pd.api.types.is_datetime64_any_dtype(results_df[col]):
                    results_df[col] = results_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    results_df[col] = results_df[col].astype(str)
            except:
                 results_df[col] = results_df[col].astype(str)
    
    results_df.replace({np.nan: None}, inplace=True)


    return {
        "question": question,
        "summary": summary,
        "data": results_df.to_dict(orient='records'),
        "sql_query": sql_query,
        "total_rows": total_rows,  # Add total row count for info
    }

# --- 6. API ENDPOINT ---
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    """The main endpoint, now with multi-question handling."""
    logger.info(f"Received question: {request.question}")
    
    try:
        simple_questions = decompose_question(request.question)

        if len(simple_questions) <= 1:
            logger.info("Treating as a single question.")
            result = answer_single_question(request.question)
            return {
                "summary": result['summary'],
                "data": result['data'],
                "sql_query": result['sql_query'],
                "total_rows": result.get('total_rows', len(result['data'])),
                "is_multi_part": False,
            }

        logger.info(f"Decomposed into {len(simple_questions)} simple questions.")
        individual_answers = []
        for q in simple_questions:
            try:
                answer_dict = answer_single_question(q)
                individual_answers.append(answer_dict)
            except Exception as e:
                logger.error(f"Error answering sub-question '{q}': {e}")
                individual_answers.append({
                    "question": q,
                    "summary": f"I was unable to answer the question: '{q}'.",
                    "data": [], "sql_query": "Error"
                })
        
        final_summary = synthesize_answers(request.question, individual_answers)
        
        return {
            "summary": final_summary,
            "sub_answers": individual_answers,
            "is_multi_part": True,
        }

    except Exception as e:
        logger.exception(f"An error occurred during query processing for question: {request.question}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# --- 7. HEALTH CHECK ENDPOINT ---
@app.get("/")
def root():
    return {
        "message": "FloatChart AI Backend API",
        "version": "3.0.0",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "ask": "/ask (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    db_status = "OK"
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"Error: {e}"

    rag_status = "OK" if collection and embedding_model else "Warning: RAG components failed to load. Check logs."

    return {
        "status": "online",
        "database_connection": db_status,
        "rag_components": rag_status,
        "db_context_loaded": bool(DB_CONTEXT)
    }
