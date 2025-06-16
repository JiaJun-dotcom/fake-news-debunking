// script.js

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('analyze-form');
    const contentInput = document.getElementById('content-input');
    const resultOutput = document.getElementById('result-output');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const content = contentInput.value.trim();
        if (!content) {
            alert('Please enter an article URL or text to analyze.');
            return;
        }

        // Clear previous result and show loading
        resultOutput.textContent = 'Analyzing...';

        try {
            const response = await fetch(`/analyze_article/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ content })
            });

            if (!response.ok) {
                // Read the response body as text ONCE.
                const errorText = await response.text();
                let detail = errorText;

                try {
                    const errorJson = JSON.parse(errorText);
                    if (errorJson.detail) {
                        detail = errorJson.detail;
                    }
                } catch (parseError) {
                }

                if (response.status === 502 || response.status === 503) {
                     resultOutput.textContent = `Server is warming up (Status ${response.status}). This is normal for the first request. Please wait 2 minutes and try again.`;
                } else {
                     resultOutput.textContent = `Error: ${detail}`;
                }
                return; 
            }

            const data = await response.json();
            
            resultOutput.textContent = data; 

        } catch (error) {
            resultOutput.textContent = `Network Request Failed: ${error}. Please check your connection.`;
        }
    });
});
