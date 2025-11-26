# Railway Deployment Setup Guide

## ✅ What's Been Fixed

1. **Dependency Conflict** - Fixed langchain version conflicts in requirements.txt
2. **Database Connection** - Updated api_groq.py to use Railway's DATABASE_URL with proper connection pooling
3. **Connection Pooling** - Added robust retry logic and keepalive settings

## 🔧 Critical: Load Data to Railway PostgreSQL

**The database connection error occurs because your Railway PostgreSQL database is empty!**

You need to load your ARGO float data into the Railway database. Here's how:

### Step 1: Get Railway Database Credentials

1. Go to your Railway dashboard: https://railway.app
2. Select your project (FloatChat-AI)
3. Click on your **PostgreSQL** service
4. Go to the **Variables** tab
5. Copy the `DATABASE_URL` value (it looks like: `postgresql://user:pass@host:port/dbname`)

### Step 2: Set Environment Variable Locally

In PowerShell:
```powershell
$env:DATABASE_URL="paste_your_railway_database_url_here"
```

### Step 3: Load Your Data

Make sure you have the `argo_final_data.parquet` file, then run:
```powershell
python load_to_sql.py
```

This will upload your ARGO data to the Railway PostgreSQL database.

### Step 4: Set Environment Variables in Railway

1. Go to your Railway **backend service** (not the database)
2. Go to the **Variables** tab
3. Add these environment variables:
   - `GROQ_API_KEY` - Your Groq API key
   - `DATABASE_URL` - Railway automatically provides this, but verify it's there
   - If DATABASE_URL is not auto-added, copy it from the PostgreSQL service

### Step 5: Verify Deployment

1. Check Railway logs to ensure the build succeeds
2. Test your health endpoint: `https://your-backend-url.railway.app/health`
3. It should return:
   ```json
   {
     "status": "online",
     "database_connection": "OK",
     "rag_components": "Warning: RAG components failed to load. Check logs.",
     "db_context_loaded": true
   }
   ```

## 🚀 Current Status

- ✅ Fixed dependency conflicts
- ✅ Fixed database connection logic
- ✅ Added connection pooling
- ⚠️ **Action Required**: Load data to Railway database
- ⚠️ **Action Required**: Set GROQ_API_KEY in Railway

## 📝 Notes

- The RAG components (ChromaDB) are disabled on Railway to reduce memory usage
- The app will work without RAG, just without vector search enhancement
- Make sure your Vercel frontend is pointing to the correct Railway backend URL

## 🔗 Update Frontend Backend URL

Update your Vercel environment variables:
1. Go to Vercel dashboard
2. Find your FloatChat-AI project
3. Go to Settings → Environment Variables
4. Set `VITE_BACKEND_URL` to your Railway backend URL (e.g., `https://your-app.railway.app`)
5. Redeploy your Vercel frontend

## 🆘 Troubleshooting

If you still see connection errors:
1. Verify DATABASE_URL is set in Railway backend service
2. Check Railway PostgreSQL is running (not sleeping)
3. Verify data was loaded successfully using Railway's database console
4. Check Railway logs for specific error messages
