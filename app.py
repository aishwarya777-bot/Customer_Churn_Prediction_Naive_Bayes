import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# Load the trained Naive Bayes model
MODEL_PATH = 'naive_model.pkl'
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    model = None

# HTML Template with an attractive, modern UI using Tailwind CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Prediction Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <style>
        body { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); }
    </style>
</head>
<body class="min-h-screen flex items-center justify-center p-6 text-slate-100 font-sans">

    <div class="w-full max-w-3xl bg-slate-900/80 backdrop-blur-md rounded-2xl shadow-2xl border border-slate-800 p-8">
        <div class="text-center mb-8">
            <h1 class="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
                Customer Analytics Predictor
            </h1>
            <p class="text-slate-400 mt-2">Input customer metrics to run the Naive Bayes classification model.</p>
        </div>

        {% if prediction is not none %}
        <div class="mb-8 p-4 rounded-xl border {% if prediction == 1 %} bg-emerald-950/40 border-emerald-500/30 text-emerald-300 {% else %} bg-indigo-950/40 border-indigo-500/30 text-indigo-300 {% endif %} text-center animate-fade-in">
            <span class="text-sm uppercase tracking-wider font-semibold">Prediction Result</span>
            <div class="text-2xl font-bold mt-1">Class Label: {{ prediction }}</div>
        </div>
        {% endif %}

        <form method="POST" action="/predict" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Age</label>
                    <input type="number" name="age" required min="0" max="120" placeholder="e.g. 34" class="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-white transition">
                </div>

                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Gender</label>
                    <select name="gender" class="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-white transition">
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>

                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">City</label>
                    <select name="city" class="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-white transition">
                        <option value="0">Metro Area A</option>
                        <option value="1">Metro Area B</option>
                        <option value="2">Other</option>
                    </select>
                </div>

                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Tenure (Months)</label>
                    <input type="number" name="tenure_months" required min="0" placeholder="e.g. 12" class="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-white transition">
                </div>

                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Avg Order Value ($)</label>
                    <input type="number" step="0.01" name="avg_order_value" required min="0" placeholder="e.g. 89.50" class="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-white transition">
                </div>

                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Total Orders</label>
                    <input type="number" name="total_orders" required min="0" placeholder="e.g. 5" class="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-white transition">
                </div>

                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Last Purchase (Days Ago)</label>
                    <input type="number" name="last_purchase_days_ago" required min="0" placeholder="e.g. 14" class="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-white transition">
                </div>

                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Support Tickets</label>
                    <input type="number" name="support_tickets" required min="0" placeholder="e.g. 0" class="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-white transition">
                </div>

                <div class="md:col-span-2">
                    <label class="block text-sm font-medium text-slate-300 mb-2">Subscription Type</label>
                    <select name="subscription_type" class="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-white transition">
                        <option value="0">Basic</option>
                        <option value="1">Premium</option>
                        <option value="2">Enterprise</option>
                    </select>
                </div>

            </div>

            <button type="submit" class="w-full mt-4 bg-gradient-to-r from-indigo-500 to-cyan-500 hover:from-indigo-600 hover:to-cyan-600 text-white font-semibold py-3 px-4 rounded-xl shadow-lg transform active:scale-[0.98] transition cursor-pointer">
                Generate Prediction
            </button>
        </form>
    </div>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Model file not found or failed to load.", 500
    
    try:
        # Extract features from form in order 
        # Map or cast inputs appropriately to match your original training encoding
        features = [
            float(request.form['age']),
            float(request.form['gender']),
            float(request.form['city']),
            float(request.form['tenure_months']),
            float(request.form['avg_order_value']),
            float(request.form['total_orders']),
            float(request.form['last_purchase_days_ago']),
            float(request.form['support_tickets']),
            float(request.form['subscription_type'])
        ]
        
        # Convert to 2D numpy array for the scikit-learn model
        final_features = np.array([features])
        
        # Make prediction
        prediction = model.predict(final_features)[0]
        
        # Render page with prediction result
        return render_template_string(HTML_TEMPLATE, prediction=int(prediction))

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
