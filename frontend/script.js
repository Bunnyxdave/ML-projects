const API_BASE = 'http://localhost:5000';

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    console.log('Fake News Detection System initialized');
});

// Navigation
function goBack() {
    window.history.back();
}

// File Upload Handling
function triggerFileInput() {
    document.getElementById('fileInput').click();
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        const fileInfo = document.getElementById('fileInfo');
        const fileUploadArea = document.getElementById('fileUploadArea');

        // Validate file type
        const validTypes = ['text/plain', 'application/pdf', 'application/msword',
                           'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
        if (!validTypes.includes(file.type)) {
            alert('Please select a valid file type (TXT, PDF, DOC, DOCX)');
            return;
        }

        // Validate file size (5MB max)
        if (file.size > 5 * 1024 * 1024) {
            alert('File size must be less than 5MB');
            return;
        }

        // Update UI
        fileUploadArea.style.borderColor = '#10b981';
        fileUploadArea.style.background = '#f0fdf4';

        fileInfo.innerHTML = `
            <i class="fas fa-file"></i>
            ${file.name} (${(file.size / 1024).toFixed(1)} KB)
            <button type="button" onclick="removeFile()" style="margin-left: 10px; background: none; border: none; color: #ef4444; cursor: pointer;">
                <i class="fas fa-times"></i>
            </button>
        `;
        fileInfo.style.display = 'block';

        // Read file content for text files
        if (file.type === 'text/plain') {
            const reader = new FileReader();
            reader.onload = function(e) {
                const content = e.target.result;
                document.getElementById('news_text').value = content;
            };
            reader.readAsText(file);
        }
    }
}

function removeFile() {
    document.getElementById('fileInput').value = '';
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('fileUploadArea').style.borderColor = '#cbd5e1';
    document.getElementById('fileUploadArea').style.background = '#f8fafc';
}

// Main Analysis Function
async function analyzeNews() {
    const form = document.getElementById('analysisForm');
    const formData = new FormData(form);

    // Get form values
    const headline = document.getElementById('headline').value.trim();
    const newsText = document.getElementById('news_text').value.trim();
    const url = document.getElementById('url').value.trim();
    const source = document.getElementById('source').value.trim();

    // Validate input
    if (!headline && !newsText && !url) {
        alert('Please provide at least a headline, article content, or URL to analyze.');
        return;
    }

    // Prepare analysis data
    const analysisData = {
        headline: headline,
        news_text: newsText,
        url: url,
        source: source
    };

    // Use the main text content for analysis (priority: news_text > headline > url)
    const analysisText = newsText || headline || url;

    try {
        showLoading(true);

        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: analysisText })
        });

        const result = await response.json();

        if (result.error) {
            throw new Error(result.error);
        }

        // Transform the result to match our new UI format
        const transformedResult = transformResult(result.result, analysisData);
        displayResults(transformedResult);

    } catch (error) {
        console.error('Error analyzing news:', error);
        alert('Error analyzing news: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Transform backend result to frontend format
function transformResult(backendResult, analysisData) {
    const isFake = backendResult.label === 'FAKE';
    const confidence = backendResult.confidence;

    // Calculate risk score based on confidence and prediction
    let riskScore = Math.round(confidence * 100);
    if (!isFake) {
        riskScore = 100 - riskScore; // Invert for real news
    }

    // Determine classification
    let classification, riskLevel;
    if (riskScore < 30) {
        classification = 'Legitimate';
        riskLevel = 'low-risk';
    } else if (riskScore < 70) {
        classification = 'Suspicious';
        riskLevel = 'medium-risk';
    } else {
        classification = isFake ? 'Fraudulent' : 'Verified';
        riskLevel = 'high-risk';
    }

    // Generate key indicators from explanation
    const keyIndicators = backendResult.explanation || ['Analysis based on text patterns'];

    // Generate detailed analysis
    const analysisDetails = {
        source_credibility: analysisData.source ? `Analyzed source: ${analysisData.source}` : 'Source not specified',
        factual_accuracy: isFake ? 'Potential factual inaccuracies detected' : 'Content appears factually consistent',
        emotional_indicators: backendResult.features?.sentiment_compound ?
            `Sentiment analysis: ${getSentimentDescription(backendResult.features.sentiment_compound)}` : 'Neutral emotional tone',
        verification_status: isFake ? 'Requires external verification' : 'Content appears credible',
        bias_indicators: backendResult.features?.fake_indicator_count > 0 ?
            `Detected ${backendResult.features.fake_indicator_count} potential bias indicators` : 'Minimal bias indicators detected'
    };

    return {
        risk_score: riskScore,
        fraud_probability: isFake ? confidence : (1 - confidence),
        classification: classification.toLowerCase().replace(' ', '_'),
        confidence_score: confidence,
        key_indicators: keyIndicators,
        analysis_details: analysisDetails,
        is_fake: isFake,
        risk_level: riskLevel
    };
}

function getSentimentDescription(score) {
    if (score > 0.5) return 'Highly Positive (potential sensationalism)';
    if (score > 0.1) return 'Positive';
    if (score > -0.1) return 'Neutral';
    if (score > -0.5) return 'Negative';
    return 'Highly Negative (potential fear-mongering)';
}

// Display Results
function displayResults(result) {
    const resultsCard = document.getElementById('resultsCard');
    const emptyState = document.getElementById('emptyState');
    const riskBadge = document.getElementById('riskBadge');
    const classificationBadge = document.getElementById('classificationBadge');
    const indicatorsList = document.getElementById('indicatorsList');

    // Show results, hide empty state
    resultsCard.style.display = 'block';
    emptyState.style.display = 'none';

    // Update risk score
    document.getElementById('riskScore').textContent = result.risk_score;
    document.getElementById('confidenceScore').textContent = `${Math.round(result.confidence_score * 100)}%`;

    // Update risk badge
    riskBadge.textContent = `${result.risk_level.replace('-', ' ').toUpperCase()} RISK`;
    riskBadge.className = `risk-badge ${result.risk_level}`;

    // Update classification
    classificationBadge.textContent = result.classification.replace('_', ' ').toUpperCase();

    // Update key indicators
    indicatorsList.innerHTML = '';
    result.key_indicators.forEach(indicator => {
        const indicatorElement = document.createElement('div');
        indicatorElement.className = 'indicator-item';
        indicatorElement.textContent = indicator;
        indicatorsList.appendChild(indicatorElement);
    });

    // Update analysis details
    document.getElementById('sourceCredibility').textContent = result.analysis_details.source_credibility;
    document.getElementById('factualAccuracy').textContent = result.analysis_details.factual_accuracy;
    document.getElementById('emotionalIndicators').textContent = result.analysis_details.emotional_indicators;
    document.getElementById('verificationStatus').textContent = result.analysis_details.verification_status;
    document.getElementById('biasIndicators').textContent = result.analysis_details.bias_indicators;

    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Loading State
function showLoading(show) {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');

    if (show) {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        loadingOverlay.style.display = 'flex';
    } else {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze for Fraud';
        loadingOverlay.style.display = 'none';
    }
}

// Drag and Drop File Upload
document.addEventListener('DOMContentLoaded', function() {
    const fileUploadArea = document.getElementById('fileUploadArea');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileUploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        fileUploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        fileUploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        fileUploadArea.style.borderColor = '#3b82f6';
        fileUploadArea.style.background = '#f0f9ff';
    }

    function unhighlight() {
        fileUploadArea.style.borderColor = '#cbd5e1';
        fileUploadArea.style.background = '#f8fafc';
    }

    fileUploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleFileSelect({ target: { files: files } });
        }
    }
});