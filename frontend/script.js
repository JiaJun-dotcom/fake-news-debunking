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
                // Try to extract JSON error detail
                let errText = `Status ${response.status}`;
                try {
                    const errJson = await response.json();
                    if (errJson.detail) {
                        errText += `: ${errJson.detail}`;
                    } else {
                        errText += `: ${JSON.stringify(errJson)}`;
                    }
                } catch (_parseErr) {
                    // response not JSON
                    errText += `: ${await response.text()}`;
                }
                resultOutput.textContent = `Error: ${errText}`;
            } else {
                const data = await response.json();
                // Pretty-print JSON
                resultOutput.textContent = JSON.stringify(data, null, 2);
            }
        } catch (error) {
            resultOutput.textContent = `Request failed: ${error}`;
        }
    });
});
