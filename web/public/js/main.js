document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('imagePreview');
    const resultContainer = document.getElementById('result');
    const predictionElement = document.getElementById('prediction');
    const confidenceElement = document.getElementById('confidence');

    // Preview image when selected
    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                previewContainer.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        }
    });

    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(form);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (data.success) {
                predictionElement.textContent = data.prediction;
                confidenceElement.textContent = (data.confidence * 100).toFixed(1);
                resultContainer.classList.remove('hidden');
                
                // Refresh page after 2 seconds to show new prediction in recent list
                setTimeout(() => {
                    window.location.reload();
                }, 2000);
            } else {
                alert('Error: ' + (data.error || 'Failed to process image'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error processing the image. Please try again.');
        }
    });
});