async function uploadImage() {
    const input = document.getElementById('imageInput');
    const resultDiv = document.getElementById('result-message');
    
    const resultsContainer = document.getElementById('results-container');
    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    
    if (input.files.length === 0) {
        alert("Please select a Retinal Scan first.");
        return;
    }

    const formData = new FormData();
    formData.append('image', input.files[0]);

    // UI Loading Phase
    resultDiv.innerText = "INITIALIZING...";
    resultDiv.style.color = "#00f0ff";
    
    emptyState.classList.add('hidden');
    resultsContainer.classList.add('hidden');
    loadingState.classList.remove('hidden');

    try {
        const response = await fetch('http://127.0.0.1:5000/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.status === 'success') {
            resultDiv.innerText = "✓ ANALYSIS COMPLETE";
            resultDiv.style.color = "#00e676"; // Neon green
            
            document.getElementById('dr-grade').innerText = data.grade;
            document.getElementById('dr-confidence').innerText = data.confidence + '%';
            document.getElementById('ai-report').innerText = data.report;
            
            if (data.enhanced_image_url) {
                document.getElementById('enhanced-image').src = data.enhanced_image_url;
            }
            if (data.heatmap_url) {
                document.getElementById('heatmap-image').src = data.heatmap_url;
            }
            if (data.surf_url) {
                document.getElementById('surf-image').src = data.surf_url;
                document.getElementById('surf-box').classList.remove('hidden');
            }

            loadingState.classList.add('hidden');
            resultsContainer.classList.remove('hidden');

        } else {
            resultDiv.innerText = "⚠ ERROR";
            resultDiv.style.color = "#ff3366";
            
            loadingState.innerHTML = `
                <div style="font-size: 60px; margin-bottom: 20px;">⚠️</div>
                <h3 style="color:#ff3366; letter-spacing: 2px;">SCAN REJECTED</h3>
                <p style="color: #8b9bb4;">${data.message}</p>
            `;
        }
    } catch (error) {
        resultDiv.innerText = "CONNECTION FAILED";
        resultDiv.style.color = "#ff3366";
        
        loadingState.innerHTML = `
            <div style="font-size: 60px; margin-bottom: 20px;">🔌</div>
            <h3 style="color:#ff3366; letter-spacing: 2px;">SERVER OFFLINE</h3>
            <p style="color: #8b9bb4;">The PyTorch backend is not responding. Please run <code>python app.py</code></p>
        `;
    }
}
