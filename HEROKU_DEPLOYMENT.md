# Deploying OCR Vision App to Heroku (Free Tier)

This guide will help you deploy your OCR Vision application to Heroku without requiring a credit card.

## Prerequisites

- A [Heroku](https://www.heroku.com/) account (free signup, no credit card required)
- [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) installed on your computer
- [Git](https://git-scm.com/) installed on your computer
- [Netlify](https://www.netlify.com/) account for frontend deployment (free tier, no credit card required)

## Step 1: Prepare Your Backend for Heroku

1. Create a `runtime.txt` file in your project root directory:
   ```
   python-3.9.16
   ```

2. Create an `Aptfile` file in your project root to install Tesseract:
   ```
   tesseract-ocr
   tesseract-ocr-eng
   libsm6
   libxrender1
   libfontconfig1
   libice6
   ```

3. Update your `requirements.txt` to include Heroku-specific packages:
   ```
   # Add these to your existing requirements.txt
   gunicorn==21.2.0
   opencv-python-headless==4.8.1.78
   ```

4. Make sure your `Procfile` is correctly set up (already done):
   ```
   web: gunicorn app:app
   ```

## Step 2: Deploy the Backend to Heroku

1. Login to Heroku from your terminal:
   ```
   heroku login
   ```

2. Create a new Heroku app:
   ```
   heroku create ocr-vision-app-backend
   ```

3. Add Heroku buildpacks (needed for Tesseract and OpenCV):
   ```
   heroku buildpacks:add --index 1 heroku/python
   heroku buildpacks:add --index 2 https://github.com/heroku/heroku-buildpack-apt
   ```

4. Push your code to Heroku:
   ```
   git add .
   git commit -m "Prepare for Heroku deployment"
   git push heroku main
   ```

5. Set environment variables:
   ```
   heroku config:set TESSERACT_PATH=/app/.apt/usr/bin/tesseract
   heroku config:set SECRET_KEY=your_secret_key_here
   ```

6. Open your deployed backend:
   ```
   heroku open
   ```

## Step 3: Prepare Your Frontend for Netlify

1. Update the API utility file to point to your Heroku backend:

   Navigate to `frontend/ocr-vision-app/src/utils/api.js` and update:
   ```javascript
   // Determine the API base URL based on environment
   const getBaseUrl = () => {
     // In production, API requests go to the Heroku backend
     if (process.env.NODE_ENV === 'production') {
       return 'https://ocr-vision-app-backend.herokuapp.com';
     }
     // In development, use the proxy from package.json or explicit URL
     return process.env.REACT_APP_API_URL || 'http://localhost:8080';
   };
   ```

2. Build your React application:
   ```
   cd frontend/ocr-vision-app
   npm install
   npm run build
   ```

## Step 4: Deploy the Frontend to Netlify

1. Sign up for a free Netlify account at [netlify.com](https://www.netlify.com/)

2. Install Netlify CLI:
   ```
   npm install -g netlify-cli
   ```

3. Login to Netlify:
   ```
   netlify login
   ```

4. Deploy to Netlify:
   ```
   cd frontend/ocr-vision-app
   netlify deploy --prod
   ```
   - When prompted, select "Create & configure a new site"
   - Choose your team
   - For the site name, enter "ocr-vision-app" or a name of your choice
   - For the publish directory, enter "build"

5. Once deployed, Netlify will provide you with a URL for your frontend application.

## Step 5: CORS Configuration

1. Update your Flask application to allow requests from your Netlify domain:
   
   Add your Netlify domain to the CORS configuration in `app.py`:
   ```python
   # Near the top of your app.py file
   NETLIFY_URL = os.environ.get('NETLIFY_URL', 'https://your-netlify-app-url.netlify.app')
   CORS(app, origins=[NETLIFY_URL, 'http://localhost:3000'])
   ```

2. Set the environment variable on Heroku:
   ```
   heroku config:set NETLIFY_URL=https://your-netlify-app-url.netlify.app
   ```

3. Push the changes to Heroku:
   ```
   git add .
   git commit -m "Update CORS for Netlify"
   git push heroku main
   ```

## Step 6: Testing Your Deployment

1. Visit your Netlify URL
2. Upload an image for OCR processing
3. Verify that the app works correctly

## Troubleshooting

### Backend Issues

- **Tesseract not found**: Check Heroku logs with `heroku logs --tail`
- **Application crashes**: Review the logs for error messages
- **Memory limits**: Free tier has 512MB RAM limit, which might be insufficient for processing large images

### Frontend Issues

- **API connection errors**: Check browser console for CORS errors
- **Loading issues**: Verify that the API base URL is correctly configured

## Limitations of Free Tier

- Heroku free dynos sleep after 30 minutes of inactivity
- When a sleeping dyno receives traffic, it wakes up, but this can take a few seconds
- Free tier apps are limited to 550-1000 dyno hours per month
- The app will be put to sleep automatically if you exceed your monthly free hours

## Need Help?

- Heroku Documentation: [https://devcenter.heroku.com/](https://devcenter.heroku.com/)
- Netlify Documentation: [https://docs.netlify.com/](https://docs.netlify.com/)
- Flask Deployment Guide: [https://flask.palletsprojects.com/en/2.3.x/deploying/](https://flask.palletsprojects.com/en/2.3.x/deploying/) 