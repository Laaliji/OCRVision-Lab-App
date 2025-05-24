import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Home from './components/home/Home';
import Process from './components/process/Process';
import { LanguageProvider } from './contexts/LanguageContext';
import './App.css';

const App: React.FC = () => {
  return (
    <LanguageProvider>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/process" element={<Process />} />
          </Routes>
        </Layout>
      </Router>
    </LanguageProvider>
  );
};

export default App;
