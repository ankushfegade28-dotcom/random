import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load Model
MODEL_PATH = 'random.pkl'
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: '{MODEL_PATH}' not found in the current directory.")

# Expected 21 Features in Order
FEATURE_NAMES = [
    'company_name', 'industry', 'country', 'company_size', 'month', 'year',
    'layoffs_count', 'layoff_percentage', 'reason_for_layoffs', 'ai_automation_impact',
    'ai_replacement_risk', 'open_roles', 'hiring_trend', 'remote_jobs_percentage',
    'top_hiring_role', 'stock_growth_percent', 'revenue_growth_percent',
    'salary_budget_change', 'ai_adoption_level', 'employee_sentiment', 'job_security_score'
]

# Single-page HTML Template with Embedded Modern CSS and JS Animations
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market Trend & Economic Outlook Predictor</title>
    <!-- Google Fonts & Font Awesome Icons -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            --bg-dark: #0b0f19;
            --card-bg: rgba(22, 31, 49, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --primary-accent: #6366f1;
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --bull-market: #10b981;
            --stable-market: #3b82f6;
            --recession-market: #ef4444;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.15) 0%, transparent 40%);
        }

        /* Animated Background Particles Grid */
        .bg-grid {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background-image: linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
            background-size: 50px 50px;
            z-index: -1;
        }

        header {
            padding: 2.5rem 1rem 1rem 1rem;
            text-align: center;
        }

        header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(to right, #ffffff, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 1.1rem;
        }

        .container {
            max-width: 1200px;
            margin: 1.5rem auto 3rem auto;
            padding: 0 1.5rem;
            width: 100%;
        }

        .grid-layout {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 992px) {
            .grid-layout {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 1rem;
        }

        .card-header i {
            font-size: 1.5rem;
            color: var(--primary-accent);
        }

        .card-header h2 {
            font-size: 1.25rem;
            font-weight: 700;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: capitalize;
        }

        .form-control {
            background: rgba(11, 15, 25, 0.6);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            transition: all 0.3s ease;
        }

        .form-control:focus {
            outline: none;
            border-color: var(--primary-accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: var(--primary-gradient);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 1rem 2rem;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            box-shadow: 0 10px 25px rgba(99, 102, 241, 0.4);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(99, 102, 241, 0.6);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        /* Results Panel & Animation */
        .result-panel {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            height: 100%;
            min-height: 350px;
        }

        .placeholder-state {
            color: var(--text-muted);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
        }

        .placeholder-state i {
            font-size: 3.5rem;
            opacity: 0.4;
        }

        .prediction-badge {
            font-size: 1.8rem;
            font-weight: 800;
            padding: 1rem 2rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            text-transform: uppercase;
            letter-spacing: 1px;
            animation: pulseIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
        }

        .prediction-badge.Bull {
            background: rgba(16, 185, 129, 0.15);
            color: var(--bull-market);
            border: 2px solid var(--bull-market);
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
        }

        .prediction-badge.Stable {
            background: rgba(59, 130, 246, 0.15);
            color: var(--stable-market);
            border: 2px solid var(--stable-market);
            box-shadow: 0 0 30px rgba(59, 130, 246, 0.3);
        }

        .prediction-badge.Recession {
            background: rgba(239, 68, 68, 0.15);
            color: var(--recession-market);
            border: 2px solid var(--recession-market);
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
        }

        .prob-bar-container {
            width: 100%;
            margin-top: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .prob-item {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
            text-align: left;
        }

        .prob-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .prob-track {
            height: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            width: 0%;
            border-radius: 4px;
            transition: width 1s ease-in-out;
        }

        @keyframes pulseIn {
            0% { transform: scale(0.8); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        .loader {
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top: 3px solid var(--primary-accent);
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            display: none;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="bg-grid"></div>

    <header>
        <h1>Market Sentiment Predictor</h1>
        <p>AI-Powered Economic Outlook & Corporate Trend Classifier</p>
    </header>

    <div class="container">
        <div class="grid-layout">
            <!-- Form Section -->
            <div class="card">
                <div class="card-header">
                    <i class="fa-solid fa-sliders"></i>
                    <h2>Input Parameters</h2>
                </div>
                <form id="predictForm">
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Company Name (Code)</label>
                            <input type="number" name="company_name" class="form-control" value="10" required>
                        </div>
                        <div class="form-group">
                            <label>Industry Code</label>
                            <input type="number" name="industry" class="form-control" value="5" required>
                        </div>
                        <div class="form-group">
                            <label>Country Code</label>
                            <input type="number" name="country" class="form-control" value="2" required>
                        </div>
                        <div class="form-group">
                            <label>Company Size</label>
                            <input type="number" name="company_size" class="form-control" value="500" required>
                        </div>
                        <div class="form-group">
                            <label>Month</label>
                            <input type="number" name="month" class="form-control" min="1" max="12" value="6" required>
                        </div>
                        <div class="form-group">
                            <label>Year</label>
                            <input type="number" name="year" class="form-control" value="2026" required>
                        </div>
                        <div class="form-group">
                            <label>Layoffs Count</label>
                            <input type="number" name="layoffs_count" class="form-control" value="0" required>
                        </div>
                        <div class="form-group">
                            <label>Layoff Percentage (%)</label>
                            <input type="number" step="0.01" name="layoff_percentage" class="form-control" value="0.0" required>
                        </div>
                        <div class="form-group">
                            <label>Reason for Layoffs</label>
                            <input type="number" name="reason_for_layoffs" class="form-control" value="0" required>
                        </div>
                        <div class="form-group">
                            <label>AI Automation Impact</label>
                            <input type="number" step="0.1" name="ai_automation_impact" class="form-control" value="3.5" required>
                        </div>
                        <div class="form-group">
                            <label>AI Replacement Risk</label>
                            <input type="number" step="0.1" name="ai_replacement_risk" class="form-control" value="2.1" required>
                        </div>
                        <div class="form-group">
                            <label>Open Roles</label>
                            <input type="number" name="open_roles" class="form-control" value="45" required>
                        </div>
                        <div class="form-group">
                            <label>Hiring Trend Code</label>
                            <input type="number" name="hiring_trend" class="form-control" value="1" required>
                        </div>
                        <div class="form-group">
                            <label>Remote Jobs (%)</label>
                            <input type="number" step="0.1" name="remote_jobs_percentage" class="form-control" value="65.0" required>
                        </div>
                        <div class="form-group">
                            <label>Top Hiring Role Code</label>
                            <input type="number" name="top_hiring_role" class="form-control" value="3" required>
                        </div>
                        <div class="form-group">
                            <label>Stock Growth (%)</label>
                            <input type="number" step="0.1" name="stock_growth_percent" class="form-control" value="12.4" required>
                        </div>
                        <div class="form-group">
                            <label>Revenue Growth (%)</label>
                            <input type="number" step="0.1" name="revenue_growth_percent" class="form-control" value="8.2" required>
                        </div>
                        <div class="form-group">
                            <label>Salary Budget Change</label>
                            <input type="number" step="0.1" name="salary_budget_change" class="form-control" value="4.0" required>
                        </div>
                        <div class="form-group">
                            <label>AI Adoption Level</label>
                            <input type="number" step="0.1" name="ai_adoption_level" class="form-control" value="7.8" required>
                        </div>
                        <div class="form-group">
                            <label>Employee Sentiment</label>
                            <input type="number" step="0.1" name="employee_sentiment" class="form-control" value="4.2" required>
                        </div>
                        <div class="form-group">
                            <label>Job Security Score</label>
                            <input type="number" step="0.1" name="job_security_score" class="form-control" value="8.0" required>
                        </div>

                        <button type="submit" class="btn-submit">
                            <i class="fa-solid fa-wand-magic-sparkles"></i> Predict Market Trend
                        </button>
                    </div>
                </form>
            </div>

            <!-- Output Display Section -->
            <div class="card">
                <div class="card-header">
                    <i class="fa-solid fa-chart-line"></i>
                    <h2>Prediction Result</h2>
                </div>
                <div class="result-panel" id="resultPanel">
                    <div class="placeholder-state" id="placeholder">
                        <i class="fa-solid fa-chart-pie"></i>
                        <p>Fill in the parameters and click Predict to view market forecast.</p>
                    </div>
                    
                    <div class="loader" id="loader"></div>

                    <div id="outputContent" style="display: none; width: 100%;">
                        <p style="color: var(--text-muted); font-size: 0.9rem;">Predicted Status</p>
                        <div id="badge" class="prediction-badge">---</div>
                        
                        <div class="prob-bar-container" id="probContainer"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('predictForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const loader = document.getElementById('loader');
            const placeholder = document.getElementById('placeholder');
            const outputContent = document.getElementById('outputContent');
            const badge = document.getElementById('badge');
            const probContainer = document.getElementById('probContainer');

            // Reset view
            placeholder.style.display = 'none';
            outputContent.style.display = 'none';
            loader.style.display = 'block';

            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => { data[key] = parseFloat(value); });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                loader.style.display = 'none';

                if (response.ok) {
                    outputContent.style.display = 'block';
                    
                    // Display badge
                    const predClass = result.prediction.split(" ")[0]; // e.g., Bull, Recession, Stable
                    badge.textContent = result.prediction;
                    badge.className = `prediction-badge ${predClass}`;

                    // Display probabilities
                    probContainer.innerHTML = '';
                    if (result.probabilities) {
                        Object.keys(result.probabilities).forEach(cls => {
                            const pct = (result.probabilities[cls] * 100).toFixed(1);
                            let color = 'var(--primary-accent)';
                            if(cls.includes('Bull')) color = 'var(--bull-market)';
                            if(cls.includes('Stable')) color = 'var(--stable-market)';
                            if(cls.includes('Recession')) color = 'var(--recession-market)';

                            probContainer.innerHTML += `
                                <div class="prob-item">
                                    <div class="prob-labels">
                                        <span>${cls}</span>
                                        <span>${pct}%</span>
                                    </div>
                                    <div class="prob-track">
                                        <div class="prob-fill" style="width: ${pct}%; background: ${color}"></div>
                                    </div>
                                </div>
                            `;
                        });
                    }
                } else {
                    alert(result.error || 'Prediction failed');
                    placeholder.style.display = 'flex';
                }
            } catch (err) {
                loader.style.display = 'none';
                placeholder.style.display = 'flex';
                alert('Connection error with server.');
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_LAYOUT)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server.'}), 500

    try:
        data = request.get_json(force=True)
        # Construct feature array strictly following feature order
        feature_values = [float(data.get(feat, 0)) for feat in FEATURE_NAMES]
        features_array = np.array([feature_values])

        # Predict output class
        prediction = model.predict(features_array)[0]

        # Extract Class Probabilities if model supports it
        probabilities = {}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_array)[0]
            classes = getattr(model, "classes_", [f"Class {i}" for i in range(len(proba))])
            for cls, p in zip(classes, proba):
                probabilities[str(cls)] = float(p)

        return jsonify({
            'prediction': str(prediction),
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
