import axios from 'axios';

// Determine the API base URL based on environment
const getBaseUrl = () => {
  // In production, API requests go to the Heroku backend
  if (process.env.NODE_ENV === 'production') {
    // Replace this URL with your actual Heroku app URL once created
    return process.env.REACT_APP_API_URL || 'https://ocr-vision-app-backend.herokuapp.com';
  }
  // In development, use the proxy from package.json or explicit URL
  return process.env.REACT_APP_API_URL || 'http://localhost:8080';
};

const API = axios.create({
  baseURL: getBaseUrl(),
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for API calls
API.interceptors.request.use(
  (config) => {
    // You can add auth tokens here if needed in the future
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for API calls
API.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export default API; 