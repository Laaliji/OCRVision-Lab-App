import React from 'react';

// Utility to create data URI for SVG images
export const createSvgImage = (content: string): string => {
  const svgContent = `
    <svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300">
      ${content}
    </svg>
  `;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgContent)}`;
};

// Generate an SVG placeholder for grayscale conversion
export const grayscalePlaceholder = createSvgImage(`
  <rect x="0" y="0" width="400" height="300" fill="#f0f0f0" />
  <text x="200" y="140" font-family="Arial" font-size="16" text-anchor="middle" fill="#666">Grayscale Conversion</text>
  <rect x="100" y="160" width="200" height="20" fill="#ddd" />
  <rect x="100" y="190" width="200" height="20" fill="#ccc" />
  <rect x="100" y="220" width="200" height="20" fill="#bbb" />
`);

// Generate an SVG placeholder for Gaussian blur
export const blurPlaceholder = createSvgImage(`
  <rect x="0" y="0" width="400" height="300" fill="#f0f0f0" />
  <text x="200" y="140" font-family="Arial" font-size="16" text-anchor="middle" fill="#666">Gaussian Blur</text>
  <circle cx="200" cy="200" r="60" fill="none" stroke="#999" stroke-width="2" />
  <circle cx="200" cy="200" r="40" fill="none" stroke="#999" stroke-width="2" />
  <circle cx="200" cy="200" r="20" fill="none" stroke="#999" stroke-width="2" />
`);

// Generate an SVG placeholder for thresholding
export const thresholdingPlaceholder = createSvgImage(`
  <rect x="0" y="0" width="400" height="300" fill="#f0f0f0" />
  <text x="200" y="140" font-family="Arial" font-size="16" text-anchor="middle" fill="#666">Adaptive Thresholding</text>
  <rect x="120" y="170" width="160" height="80" fill="#000" />
  <rect x="150" y="180" width="100" height="60" fill="#fff" />
`);

// Generate an SVG placeholder for morphological operations
export const morphologyPlaceholder = createSvgImage(`
  <rect x="0" y="0" width="400" height="300" fill="#f0f0f0" />
  <text x="200" y="140" font-family="Arial" font-size="16" text-anchor="middle" fill="#666">Morphological Operations</text>
  <rect x="120" y="170" width="20" height="20" fill="#000" />
  <rect x="140" y="170" width="20" height="20" fill="#000" />
  <rect x="160" y="170" width="20" height="20" fill="#000" />
  <rect x="180" y="170" width="20" height="20" fill="#000" />
  <rect x="120" y="190" width="20" height="20" fill="#000" />
  <rect x="140" y="190" width="20" height="20" fill="#fff" />
  <rect x="160" y="190" width="20" height="20" fill="#fff" />
  <rect x="180" y="190" width="20" height="20" fill="#000" />
  <rect x="120" y="210" width="20" height="20" fill="#000" />
  <rect x="140" y="210" width="20" height="20" fill="#fff" />
  <rect x="160" y="210" width="20" height="20" fill="#fff" />
  <rect x="180" y="210" width="20" height="20" fill="#000" />
  <rect x="120" y="230" width="20" height="20" fill="#000" />
  <rect x="140" y="230" width="20" height="20" fill="#000" />
  <rect x="160" y="230" width="20" height="20" fill="#000" />
  <rect x="180" y="230" width="20" height="20" fill="#000" />
`);

// Generate an SVG placeholder for contour finding
export const contourPlaceholder = createSvgImage(`
  <rect x="0" y="0" width="400" height="300" fill="#f0f0f0" />
  <text x="200" y="140" font-family="Arial" font-size="16" text-anchor="middle" fill="#666">Contour Detection</text>
  <rect x="150" y="180" width="100" height="60" fill="none" stroke="#333" stroke-width="2" />
  <rect x="160" y="190" width="80" height="40" fill="none" stroke="#333" stroke-width="2" />
`);

// Generate an SVG placeholder for contour filtering
export const filteringPlaceholder = createSvgImage(`
  <rect x="0" y="0" width="400" height="300" fill="#f0f0f0" />
  <text x="200" y="140" font-family="Arial" font-size="16" text-anchor="middle" fill="#666">Contour Filtering</text>
  <rect x="150" y="180" width="30" height="40" fill="none" stroke="#333" stroke-width="2" />
  <rect x="190" y="180" width="30" height="40" fill="none" stroke="#333" stroke-width="2" />
  <rect x="230" y="180" width="30" height="40" fill="none" stroke="#333" stroke-width="2" />
`);

// Generate an SVG placeholder for character extraction
export const extractionPlaceholder = createSvgImage(`
  <rect x="0" y="0" width="400" height="300" fill="#f0f0f0" />
  <text x="200" y="140" font-family="Arial" font-size="16" text-anchor="middle" fill="#666">Character Extraction</text>
  <rect x="150" y="180" width="30" height="40" fill="#fff" stroke="#333" stroke-width="2" />
  <rect x="190" y="180" width="30" height="40" fill="#fff" stroke="#333" stroke-width="2" />
  <rect x="230" y="180" width="30" height="40" fill="#fff" stroke="#333" stroke-width="2" />
`);

// Recognition process placeholders
export const degradationPlaceholder = createSvgImage(`
  <rect x="0" y="0" width="400" height="300" fill="#f0f0f0" />
  <text x="200" y="140" font-family="Arial" font-size="16" text-anchor="middle" fill="#666">Image Degradation</text>
  <rect x="120" y="180" width="160" height="40" fill="#ccc" />
  <rect x="140" y="190" width="40" height="20" fill="#999" />
  <rect x="190" y="190" width="40" height="20" fill="#999" />
  <rect x="240" y="190" width="20" height="20" fill="#999" />
`);

export const preprocessingPlaceholder = createSvgImage(`
  <rect x="0" y="0" width="400" height="300" fill="#f0f0f0" />
  <text x="200" y="140" font-family="Arial" font-size="16" text-anchor="middle" fill="#666">Image Preprocessing</text>
  <rect x="120" y="180" width="160" height="40" fill="#ddd" />
  <line x1="120" y1="200" x2="280" y2="200" stroke="#999" stroke-width="2" />
`);

export const segmentationPlaceholder = createSvgImage(`
  <rect x="0" y="0" width="400" height="300" fill="#f0f0f0" />
  <text x="200" y="140" font-family="Arial" font-size="16" text-anchor="middle" fill="#666">Character Segmentation</text>
  <rect x="150" y="180" width="30" height="40" fill="none" stroke="#333" stroke-width="2" />
  <rect x="190" y="180" width="30" height="40" fill="none" stroke="#333" stroke-width="2" />
  <rect x="230" y="180" width="30" height="40" fill="none" stroke="#333" stroke-width="2" />
`);

export const preparationPlaceholder = createSvgImage(`
  <rect x="0" y="0" width="400" height="300" fill="#f0f0f0" />
  <text x="200" y="120" font-family="Arial" font-size="16" text-anchor="middle" fill="#666">Character Preparation</text>
  <text x="200" y="140" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">32x32 pixels</text>
  <rect x="170" y="160" width="60" height="60" fill="#fff" stroke="#333" stroke-width="2" />
  <text x="200" y="195" font-family="Arial" font-size="24" text-anchor="middle" fill="#333">A</text>
`);

export const predictionPlaceholder = createSvgImage(`
  <rect x="0" y="0" width="400" height="300" fill="#f0f0f0" />
  <text x="200" y="140" font-family="Arial" font-size="16" text-anchor="middle" fill="#666">CNN Prediction</text>
  <rect x="100" y="180" width="80" height="40" fill="#e6f7ff" stroke="#1890ff" stroke-width="2" />
  <text x="140" y="205" font-family="Arial" font-size="14" text-anchor="middle" fill="#1890ff">Input</text>
  <path d="M180,200 L220,200" stroke="#333" stroke-width="2" />
  <polygon points="220,200 215,195 215,205" fill="#333" />
  <rect x="220" y="180" width="80" height="40" fill="#f6ffed" stroke="#52c41a" stroke-width="2" />
  <text x="260" y="205" font-family="Arial" font-size="14" text-anchor="middle" fill="#52c41a">CNN</text>
`);

export const combinationPlaceholder = createSvgImage(`
  <rect x="0" y="0" width="400" height="300" fill="#f0f0f0" />
  <text x="200" y="140" font-family="Arial" font-size="16" text-anchor="middle" fill="#666">Results Combination</text>
  <rect x="120" y="170" width="160" height="30" fill="#e6f7ff" stroke="#1890ff" stroke-width="2" />
  <text x="200" y="190" font-family="Arial" font-size="14" text-anchor="middle" fill="#1890ff">Original: 75%</text>
  <rect x="120" y="210" width="160" height="30" fill="#f6ffed" stroke="#52c41a" stroke-width="2" />
  <text x="200" y="230" font-family="Arial" font-size="14" text-anchor="middle" fill="#52c41a">Degraded: 92%</text>
`); 