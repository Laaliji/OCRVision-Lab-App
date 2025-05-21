document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const imageInput = document.getElementById('image-upload');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const imageName = document.getElementById('image-name');
    const imageSize = document.getElementById('image-size');
    const removeButton = document.getElementById('remove-image');
    
    // Step buttons
    const nextBtn1 = document.getElementById('next-btn-1');
    const nextBtn2 = document.getElementById('next-btn-2');
    const nextBtn3 = document.getElementById('next-btn-3');
    const prevBtn2 = document.getElementById('prev-btn-2');
    const prevBtn3 = document.getElementById('prev-btn-3');
    const prevBtn4 = document.getElementById('prev-btn-4');
    const resultsBtn = document.getElementById('results-btn');
    
    // Steps
    const step1 = document.getElementById('step1');
    const step2 = document.getElementById('step2');
    const step3 = document.getElementById('step3');
    const step4 = document.getElementById('step4');
    const stepperProgress = document.getElementById('stepper-progress');
    
    // Results elements
    const finalResults = document.getElementById('final-results');
    const downloadButton = document.getElementById('download-btn');
    const resetButton = document.getElementById('reset-btn');
    const copyDegradedBtn = document.getElementById('copy-degraded');
    const copyPreprocessedBtn = document.getElementById('copy-preprocessed');
    const copyDegradedFinalBtn = document.getElementById('copy-degraded-final');
    const copyPreprocessedFinalBtn = document.getElementById('copy-preprocessed-final');
    
    // Images and text elements
    const step2Original = document.getElementById('step2-original');
    const step2Degraded = document.getElementById('step2-degraded');
    const step2DegradedPlaceholder = document.getElementById('step2-degraded-placeholder');
    
    const step3Degraded = document.getElementById('step3-degraded');
    const step3Preprocessed = document.getElementById('step3-preprocessed');
    const step3PreprocessedPlaceholder = document.getElementById('step3-preprocessed-placeholder');
    
    const degradedText = document.getElementById('degraded-text');
    const preprocessedText = document.getElementById('preprocessed-text');
    const degradedTextPlaceholder = document.getElementById('degraded-text-placeholder');
    const preprocessedTextPlaceholder = document.getElementById('preprocessed-text-placeholder');
    
    const finalOriginalImage = document.getElementById('original-image');
    const finalDegradedImage = document.getElementById('degraded-image');
    const finalPreprocessedImage = document.getElementById('preprocessed-image');
    const finalDegradedText = document.getElementById('final-degraded-text');
    const finalPreprocessedText = document.getElementById('final-preprocessed-text');
    const improvementText = document.getElementById('improvement-text');
    
    // Store processed data
    let processData = null;

    // Animations using Lottie
    try {
        // Upload animation
        lottie.loadAnimation({
            container: document.getElementById('upload-animation'),
            renderer: 'svg',
            loop: true,
            autoplay: true,
            path: 'https://assets5.lottiefiles.com/packages/lf20_usmfx6bp.json'
        });
        
        // Degradation animation
        lottie.loadAnimation({
            container: document.getElementById('degradation-animation'),
            renderer: 'svg',
            loop: true,
            autoplay: true,
            path: 'https://assets9.lottiefiles.com/private_files/lf30_bn5winlb.json'
        });
        
        // Preprocessing animation
        lottie.loadAnimation({
            container: document.getElementById('preprocessing-animation'),
            renderer: 'svg',
            loop: true,
            autoplay: true,
            path: 'https://assets9.lottiefiles.com/packages/lf20_qjosmr4w.json'
        });
        
        // OCR animation
        lottie.loadAnimation({
            container: document.getElementById('ocr-animation'),
            renderer: 'svg',
            loop: true,
            autoplay: true,
            path: 'https://assets3.lottiefiles.com/packages/lf20_4y2hkdhs.json'
        });
    } catch (error) {
        console.error('Failed to load animations:', error);
    }
    
    // Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Set active step and update progress
    function setActiveStep(stepNum) {
        // Hide all steps
        [step1, step2, step3, step4].forEach(step => {
            step.classList.remove('active');
            step.classList.add('hidden');
        });
        
        // Get all step indicators and update their status
        const stepIndicators = document.querySelectorAll('.step-indicator');
        stepIndicators.forEach((indicator, index) => {
            const stepNumber = index + 1;
            const circle = indicator.querySelector('div');
            const text = indicator.querySelector('span');
            
            // Reset all to inactive
            circle.classList.remove('bg-indigo-500');
            circle.classList.remove('bg-green-500');
            circle.classList.add('bg-gray-300');
            text.classList.remove('text-gray-700');
            text.classList.add('text-gray-500');
            
            // Update completed steps
            if (stepNumber < stepNum) {
                circle.classList.remove('bg-gray-300');
                circle.classList.add('bg-green-500');
                text.classList.remove('text-gray-500');
                text.classList.add('text-gray-700');
            }
            
            // Update active step
            if (stepNumber === stepNum) {
                circle.classList.remove('bg-gray-300');
                circle.classList.add('bg-indigo-500');
                text.classList.remove('text-gray-500');
                text.classList.add('text-gray-700');
            }
        });
        
        // Set active step
        const activeStep = document.getElementById(`step${stepNum}`);
        activeStep.classList.remove('hidden');
        activeStep.classList.add('active');
        
        // Update progress bar (for 4 steps: 0%, 33%, 66%, 100%)
        const progressPercentage = (stepNum - 1) * 33.33;
        stepperProgress.style.width = `${progressPercentage}%`;
    }
    
    // Show toast notification
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        let bgColor = 'bg-blue-600';
        
        if (type === 'success') bgColor = 'bg-green-600';
        if (type === 'error') bgColor = 'bg-red-600';
        
        toast.className = `fixed bottom-4 right-4 ${bgColor} text-white px-4 py-2 rounded-lg shadow-lg z-50 transform transition-all duration-500 translate-y-20 opacity-0`;
        toast.textContent = message;
        document.body.appendChild(toast);
        
        // Show toast
        setTimeout(() => {
            toast.classList.remove('translate-y-20', 'opacity-0');
        }, 10);
        
        // Hide toast
        setTimeout(() => {
            toast.classList.add('translate-y-20', 'opacity-0');
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 500);
        }, 3000);
    }
    
    // Copy text to clipboard
    function copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(
            function() {
                showToast('Texte copié !', 'success');
            },
            function() {
                showToast('Erreur lors de la copie', 'error');
            }
        );
    }
    
    // Analyze improvement
    function analyzeImprovement(degradedText, preprocessedText) {
        if (!degradedText && !preprocessedText) {
            return "Aucun texte n'a été détecté dans les deux images.";
        }
        
        if (!degradedText && preprocessedText) {
            return "Le prétraitement a permis de détecter du texte alors qu'aucun texte n'était détecté sur l'image dégradée. Amélioration significative !";
        }
        
        if (degradedText && !preprocessedText) {
            return "Le prétraitement n'a pas réussi à améliorer la détection de texte. L'image originale fournit de meilleurs résultats.";
        }
        
        const degradedWordCount = degradedText.split(/\s+/).filter(word => word.length > 0).length;
        const preprocessedWordCount = preprocessedText.split(/\s+/).filter(word => word.length > 0).length;
        const wordCountDiff = preprocessedWordCount - degradedWordCount;
        
        let result = "";
        
        if (wordCountDiff > 0) {
            result = `Le prétraitement a permis de détecter ${wordCountDiff} mots supplémentaires (${Math.round((wordCountDiff/degradedWordCount)*100)}% d'amélioration).`;
        } else if (wordCountDiff < 0) {
            result = `Le prétraitement a détecté ${Math.abs(wordCountDiff)} mots de moins que sur l'image dégradée.`;
        } else {
            result = "Le prétraitement a détecté le même nombre de mots que sur l'image dégradée.";
        }
        
        return result;
    }
    
    // Image Upload event
    imageInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imageName.textContent = file.name;
                imageSize.textContent = formatFileSize(file.size);
                previewContainer.classList.remove('hidden');
                nextBtn1.disabled = false;
            }
            
            reader.readAsDataURL(file);
        }
    });
    
    // Remove image
    removeButton.addEventListener('click', function() {
        imageInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        nextBtn1.disabled = true;
    });
    
    // Step 1 to Step 2
    nextBtn1.addEventListener('click', function() {
        const file = imageInput.files[0];
        if (!file) {
            showToast('Veuillez sélectionner une image.', 'error');
            return;
        }
        
        nextBtn1.disabled = true;
        nextBtn1.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Traitement...';
        
        // Process the image
        const formData = new FormData();
        formData.append('image', file);
        
        fetch('/process_image', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Erreur lors du traitement de l\'image.');
            }
            return response.json();
        })
        .then(data => {
            processData = data;
            
            // Update step 2 content
            step2Original.src = data.original_image;
            step2Degraded.src = data.degraded_image;
            
            // Simulate loading
            setTimeout(() => {
                step2DegradedPlaceholder.classList.add('hidden');
                step2Degraded.classList.remove('hidden');
                nextBtn2.disabled = false;
                
                // Move to step 2
                setActiveStep(2);
                nextBtn1.disabled = false;
                nextBtn1.innerHTML = '<span>Continuer</span><i class="fas fa-arrow-right ml-2"></i>';
            }, 1500);
        })
        .catch(error => {
            showToast(error.message, 'error');
            nextBtn1.disabled = false;
            nextBtn1.innerHTML = '<span>Continuer</span><i class="fas fa-arrow-right ml-2"></i>';
        });
    });
    
    // Step 2 to Step 3
    nextBtn2.addEventListener('click', function() {
        if (!processData) {
            showToast('Veuillez d\'abord traiter une image.', 'error');
            return;
        }
        
        // Update step 3 content
        step3Degraded.src = processData.degraded_image;
        step3Preprocessed.src = processData.preprocessed_image;
        
        // Simulate loading
        setTimeout(() => {
            step3PreprocessedPlaceholder.classList.add('hidden');
            step3Preprocessed.classList.remove('hidden');
            nextBtn3.disabled = false;
            
            // Move to step 3
            setActiveStep(3);
        }, 1500);
    });
    
    // Step 3 to Step 4
    nextBtn3.addEventListener('click', function() {
        if (!processData) {
            showToast('Veuillez d\'abord traiter une image.', 'error');
            return;
        }
        
        // Simulate OCR processing
        setTimeout(() => {
            // Show texts
            degradedText.textContent = processData.degraded_text || 'Aucun texte détecté';
            preprocessedText.textContent = processData.preprocessed_text || 'Aucun texte détecté';
            
            degradedTextPlaceholder.classList.add('hidden');
            preprocessedTextPlaceholder.classList.add('hidden');
            degradedText.classList.remove('hidden');
            preprocessedText.classList.remove('hidden');
            
            // Move to step 4
            setActiveStep(4);
        }, 2000);
    });
    
    // Navigation backward
    prevBtn2.addEventListener('click', () => setActiveStep(1));
    prevBtn3.addEventListener('click', () => setActiveStep(2));
    prevBtn4.addEventListener('click', () => setActiveStep(3));
    
    // Show final results
    resultsBtn.addEventListener('click', function() {
        if (!processData) {
            showToast('Veuillez d\'abord traiter une image.', 'error');
            return;
        }
        
        // Update final results
        finalOriginalImage.src = processData.original_image;
        finalDegradedImage.src = processData.degraded_image;
        finalPreprocessedImage.src = processData.preprocessed_image;
        
        finalDegradedText.textContent = processData.degraded_text || 'Aucun texte détecté';
        finalPreprocessedText.textContent = processData.preprocessed_text || 'Aucun texte détecté';
        
        // Analyze improvement
        improvementText.textContent = analyzeImprovement(
            processData.degraded_text, 
            processData.preprocessed_text
        );
        
        // Hide stepper, show final results
        document.getElementById('stepper').classList.add('hidden');
        finalResults.classList.remove('hidden');
        
        // Scroll to results
        finalResults.scrollIntoView({ behavior: 'smooth' });
    });
    
    // Copy text buttons
    copyDegradedBtn.addEventListener('click', function() {
        copyToClipboard(degradedText.textContent);
    });
    
    copyPreprocessedBtn.addEventListener('click', function() {
        copyToClipboard(preprocessedText.textContent);
    });
    
    copyDegradedFinalBtn.addEventListener('click', function() {
        copyToClipboard(finalDegradedText.textContent);
    });
    
    copyPreprocessedFinalBtn.addEventListener('click', function() {
        copyToClipboard(finalPreprocessedText.textContent);
    });
    
    // Download results
    downloadButton.addEventListener('click', function() {
        if (!processData) {
            showToast('Aucun résultat à télécharger.', 'error');
            return;
        }
        
        // Create a text file with the results
        const resultText = `Résultats de l'extraction OCR - OCR Vision Lab\n\n` +
                         `Date: ${new Date().toLocaleString()}\n` +
                         `------------------------------------------------\n\n` +
                         `TEXTE EXTRAIT DE L'IMAGE DÉGRADÉE:\n` +
                         `-----------------------------------\n` +
                         `${processData.degraded_text || 'Aucun texte détecté'}\n\n` +
                         `TEXTE EXTRAIT DE L'IMAGE PRÉTRAITÉE:\n` +
                         `------------------------------------\n` +
                         `${processData.preprocessed_text || 'Aucun texte détecté'}\n\n` +
                         `ANALYSE:\n` +
                         `--------\n` +
                         `${improvementText.textContent}`;
        
        // Create download link
        const blob = new Blob([resultText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `ocr-results-${Date.now()}.txt`;
        link.click();
        
        showToast('Résultats téléchargés avec succès !', 'success');
    });
    
    // Reset button
    resetButton.addEventListener('click', function() {
        // Reset form
        imageInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        nextBtn1.disabled = true;
        
        // Reset step navigation
        nextBtn2.disabled = true;
        nextBtn3.disabled = true;
        
        // Reset placeholders
        step2DegradedPlaceholder.classList.remove('hidden');
        step2Degraded.classList.add('hidden');
        step3PreprocessedPlaceholder.classList.remove('hidden');
        step3Preprocessed.classList.add('hidden');
        degradedTextPlaceholder.classList.remove('hidden');
        preprocessedTextPlaceholder.classList.remove('hidden');
        degradedText.classList.add('hidden');
        preprocessedText.classList.add('hidden');
        
        // Hide results, show stepper
        finalResults.classList.add('hidden');
        document.getElementById('stepper').classList.remove('hidden');
        
        // Reset to step 1
        setActiveStep(1);
        
        // Reset process data
        processData = null;
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}); 