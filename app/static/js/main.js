document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-upload');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeButton = document.getElementById('remove-image');
    const processButton = document.getElementById('process-btn');
    const loader = document.getElementById('loader');
    const results = document.getElementById('results');
    const resetButton = document.getElementById('reset-btn');
    
    // Image preview display
    imageInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.classList.remove('hidden');
                processButton.disabled = false;
            }
            
            reader.readAsDataURL(file);
        }
    });
    
    // Remove selected image
    removeButton.addEventListener('click', function() {
        imageInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        processButton.disabled = true;
    });
    
    // Form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = imageInput.files[0];
        if (!file) {
            alert('Veuillez sélectionner une image.');
            return;
        }
        
        const formData = new FormData();
        formData.append('image', file);
        
        // Show loader, hide form and results
        loader.classList.remove('hidden');
        results.classList.add('hidden');
        
        // Send request to the backend
        fetch('/process', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Erreur lors du traitement de l\'image. Veuillez réessayer.');
            }
            return response.json();
        })
        .then(data => {
            // Hide loader
            loader.classList.add('hidden');
            
            // Display results
            document.getElementById('original-image').src = data.original_image;
            document.getElementById('degraded-image').src = data.degraded_image;
            document.getElementById('preprocessed-image').src = data.preprocessed_image;
            
            document.getElementById('degraded-text').textContent = data.degraded_text || 'Aucun texte détecté';
            document.getElementById('preprocessed-text').textContent = data.preprocessed_text || 'Aucun texte détecté';
            
            // Show results section
            results.classList.remove('hidden');
        })
        .catch(error => {
            loader.classList.add('hidden');
            alert(error.message);
        });
    });
    
    // Reset to upload a new image
    resetButton.addEventListener('click', function() {
        // Reset form
        imageInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        processButton.disabled = true;
        
        // Hide results
        results.classList.add('hidden');
    });
}); 