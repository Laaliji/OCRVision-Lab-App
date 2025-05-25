# Web Deployment Guide for OCR Vision App

This guide provides step-by-step instructions for deploying your OCR Vision application to the web using Render.com.

## Prerequisites

- A [Render.com](https://render.com/) account (free tier available)
- A GitHub, GitLab, or Bitbucket account to host your code repository

## Deployment Steps

### 1. Push Your Code to a Git Repository

1. Create a new repository on GitHub, GitLab, or Bitbucket
2. Initialize Git in your project folder (if not already done):
   ```
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repository-url>
   git push -u origin main
   ```

### 2. Deploy to Render.com

1. Log in to [Render.com](https://render.com/)
2. Click on the "New +" button and select "Web Service"
3. Connect your Git repository
4. Configure the web service:
   - **Name**: `ocr-vision-app` (or any name you prefer)
   - **Runtime**: `Python 3`
   - **Build Command**: `./build.sh && pip install -r requirements.txt && cd frontend/ocr-vision-app && npm install && npm run build && cd ../.. && mkdir -p app/static/build && cp -r frontend/ocr-vision-app/build/* app/static/build/`
   - **Start Command**: `gunicorn app:app`
   - **Advanced Settings** → **Environment Variables**:
     - `TESSERACT_PATH`: `/usr/bin/tesseract`
     - `SECRET_KEY`: (generate a random string or use Render's auto-generated secret)
     - `PYTHON_VERSION`: `3.9.0`

5. Click "Create Web Service"

Render will automatically deploy your application. The initial build may take 5-10 minutes since it needs to install Tesseract OCR and all dependencies.

### 3. Verify Your Deployment

Once the deployment is complete:

1. Click on the URL provided by Render
2. Your OCR Vision App should be fully functional in the browser

## Troubleshooting

If you encounter issues:

1. **Build Errors**: Check the build logs in Render dashboard
2. **Runtime Errors**: Check the logs section in Render dashboard
3. **Tesseract Issues**: Ensure the `build.sh` script executed correctly

## Alternative: Manual Deployment

If you prefer deploying the app separately:

1. Deploy the backend (Flask) to Render.com as a Web Service
2. Deploy the frontend (React) to a static hosting service like Netlify or Vercel
3. Update the API base URL in `frontend/src/utils/api.js` to point to your backend URL

## Important Notes

- The free tier of Render.com has some limitations:
  - Services spin down after 15 minutes of inactivity
  - Limited processing power and memory
- For production use with high traffic, consider upgrading to a paid tier

## Need Help?

If you need assistance with deployment, you can:
- Check Render.com's [documentation](https://render.com/docs)
- Refer to Flask [deployment guide](https://flask.palletsprojects.com/en/2.3.x/deploying/)
- Contact Render.com support 