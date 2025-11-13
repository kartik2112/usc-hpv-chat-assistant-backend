# HPV Health Assistant - Deployment Guide

## Architecture Overview

```
Frontend (HTML/CSS/JS)
         ↓
    Makes API Call
         ↓
Flask Backend (Python)
    Stores: OpenAI API Key
         ↓
    Makes API Call
         ↓
OpenAI API
```

## Backend Setup Instructions

### 1. Local Development Setup

**Prerequisites:**
- Python 3.8 or higher
- pip (Python package manager)

**Steps:**

```bash
# 1. Create project directory
mkdir hpv-assistant-backend
cd hpv-assistant-backend

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create .env file
cp env_template.txt .env
# Edit .env and add your OpenAI API Key

# 6. Run Flask app
python flask_backend.py
```

Backend will run at: `http://localhost:5000`

### 2. Frontend Configuration

1. Open the HPV assistant website in your browser
2. Click the ⚙️ Settings icon (top-right)
3. Enter Backend API URL: `http://localhost:5000/api/chat`
4. Click "Test Connection" to verify
5. Click "Save"

Now all questions will be proxied through your Flask backend!

---

## Hosting Recommendations

### Best Options for Flask Backend (2025)

#### 1. **Render** ⭐⭐⭐⭐⭐ (Recommended for Beginners)
- **URL:** https://render.com
- **Pros:** 
  - Free tier available (with limitations)
  - Easy deployment (connect GitHub)
  - Auto-redeploys on code push
  - Good documentation
- **Cons:** Free tier has 15-minute inactivity timeout
- **Cost:** Free tier or $7+/month paid
- **Setup:** 
  1. Push code to GitHub
  2. Connect Render to GitHub repo
  3. Set environment variables in Render dashboard
  4. Deploy!

#### 2. **PythonAnywhere** ⭐⭐⭐⭐⭐ (Best for Python)
- **URL:** https://www.pythonanywhere.com
- **Pros:**
  - Python-specific hosting
  - Free tier ($0/month)
  - Paid tier very affordable ($5/month)
  - Great support
- **Cons:** Limited features on free tier
- **Cost:** Free or $5+/month
- **Setup:**
  1. Create account on PythonAnywhere
  2. Upload code via web interface or Git
  3. Configure WSGI file
  4. Set environment variables
  5. Reload app

#### 3. **Railway.app** ⭐⭐⭐⭐
- **URL:** https://railway.app
- **Pros:**
  - Generous free tier ($5 credit)
  - GitHub integration
  - Easy environment variables
  - Modern interface
- **Cons:** Credit runs out after free tier usage
- **Cost:** Free or pay-as-you-go
- **Setup:**
  1. Connect GitHub account
  2. Create new project from repo
  3. Add environment variables
  4. Deploy automatically

#### 4. **DigitalOcean** ⭐⭐⭐⭐
- **URL:** https://www.digitalocean.com
- **Pros:**
  - Very reliable and performant
  - Good documentation
  - App Platform (easy) or Droplets (full control)
  - $5/month starter tier
- **Cons:** Requires more technical setup
- **Cost:** $5+/month
- **Setup:** See DigitalOcean documentation

#### 5. **Heroku** (Traditional, still viable)
- **URL:** https://www.heroku.com
- **Pros:** Historically popular, good documentation
- **Cons:** Removed free tier in late 2022, paid plans start at $7+
- **Cost:** $7+/month

---

## Step-by-Step: Deploy to Render (Easiest)

### 1. Prepare GitHub Repository

```bash
# Create GitHub repo for your Flask backend
# Add these files:
# - flask_backend.py
# - requirements.txt
# - Procfile (create this file with content: web: gunicorn flask_backend:app)
# - .env (add to .gitignore - NEVER commit!)
# - .gitignore (include: .env, venv/, __pycache__/)

# Commit and push to GitHub
git add .
git commit -m "HPV assistant backend"
git push origin main
```

### 2. Create Procfile

Create file `Procfile`:
```
web: gunicorn flask_backend:app
```

### 3. Deploy on Render

1. Go to https://render.com
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Select your GitHub repository
5. Configure:
   - **Name:** hpv-assistant-backend
   - **Environment:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn flask_backend:app`
6. Click "Advanced" and add environment variable:
   - **Key:** `OPENAI_API_KEY`
   - **Value:** Your OpenAI API key
7. Click "Create Web Service"
8. Wait for deployment (2-3 minutes)
9. Note your backend URL (e.g., https://hpv-assistant-backend.onrender.com)

### 4. Update Frontend

In your HPV assistant website:
1. Settings → Backend API URL
2. Enter: `https://hpv-assistant-backend.onrender.com/api/chat`
3. Test Connection
4. Save

Done! Your frontend will now use your hosted backend.

---

## Step-by-Step: Deploy to PythonAnywhere

1. Go to https://www.pythonanywhere.com
2. Create free account
3. Dashboard → Upload files via web interface
4. Upload: `flask_backend.py`, `requirements.txt`, `.env`
5. New console → Bash
   ```bash
   pip install flask flask-cors openai python-dotenv
   python flask_backend.py
   ```
6. Web → Add new web app
7. Select Python 3.x + Flask
8. Edit WSGI file to import your app
9. Set environment variables in Web tab
10. Reload app

---

## Environment Variables (Per Hosting Platform)

### Render
- Dashboard → Environment → Add variable
- Key: `OPENAI_API_KEY`
- Value: Your OpenAI API key

### PythonAnywhere
- Web → Environment variables
- Add: `OPENAI_API_KEY=sk-...`

### Railway
- Project settings → Variables
- Add secret: `OPENAI_API_KEY`

---

## Security Best Practices

✅ **DO:**
- Store API key in environment variables (NOT in code)
- Add `.env` to `.gitignore`
- Use HTTPS for production
- Restrict CORS to your frontend domain
- Set rate limits on backend
- Monitor API usage

❌ **DON'T:**
- Commit `.env` file to GitHub
- Hardcode API keys in Python files
- Expose backend publicly without authentication
- Use weak CORS settings

---

## Troubleshooting

### Backend not reachable from frontend

1. Check backend URL in settings (should be HTTPS in production)
2. Verify backend is running: `curl http://backend-url/api/health`
3. Check CORS settings in `flask_backend.py` - add your frontend domain
4. Check browser console for specific error messages

### "CORS error" message

Add your frontend domain to Flask CORS config:
```python
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://yourdomain.com",  # Add your domain
            "http://localhost:3000"
        ]
    }
})
```

### OpenAI API errors

1. Verify `OPENAI_API_KEY` is set correctly
2. Check API key has sufficient credits
3. Check API key hasn't been revoked
4. See OpenAI dashboard for usage limits

---

## Cost Estimation

**Monthly cost (example):**
- Backend hosting: $5-7/month (Render, PythonAnywhere, Railway)
- OpenAI API: ~$0.10-5.00/month (depends on usage)
- **Total:** ~$5-12/month

**Free tier options:**
- Render free tier (with limitations)
- PythonAnywhere free tier ($5 storage)
- Railway $5 free credit/month

---

## Support & Resources

- OpenAI Documentation: https://platform.openai.com/docs
- Flask Documentation: https://flask.palletsprojects.com
- Render Docs: https://render.com/docs
- PythonAnywhere Help: https://help.pythonanywhere.com
