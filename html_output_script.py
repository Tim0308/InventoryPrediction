import os
import sys
import re
import html
from datetime import datetime, timedelta

def is_valid_future_date(date_str: str) -> bool:
    """Check if the date is in the future (not earlier than today)"""
    try:
        forecast_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        today = datetime.now().date()
        return forecast_date >= today
    except ValueError:
        return False

def is_in_coming_week(date_str: str) -> bool:
    """Check if the date is in the coming week (assuming script runs on Monday)"""
    try:
        forecast_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        today = datetime.now().date()
        
        # Calculate the start of the coming week (next Monday if today is not Monday)
        days_until_monday = (7 - today.weekday()) % 7
        if days_until_monday == 0 and today.weekday() == 0:
            # If today is Monday, coming week starts today
            coming_week_start = today
        else:
            # Otherwise, coming week starts on the next Monday
            coming_week_start = today + timedelta(days=days_until_monday)
        
        coming_week_end = coming_week_start + timedelta(days=6)  # Sunday
        
        return coming_week_start <= forecast_date <= coming_week_end
    except ValueError:
        return False

def parse_hospital_block(block: str):
    """Parse a single hospital block into structured data"""
    lines = block.strip().split('\n')
    if len(lines) < 3:
        return None
    
    hospital_data = {
        'brand': '',
        'hospital': '',
        'last_invoice': '',
        'predicted_interval': '',
        'next_date': '',
        'forecast_dates': []
    }
    
    for line in lines:
        line = line.strip()
        if line.startswith('Brand Detail:'):
            hospital_data['brand'] = line.replace('Brand Detail:', '').strip()
        elif line.startswith('Hospital:'):
            hospital_data['hospital'] = line.replace('Hospital:', '').strip()
        elif line.startswith('Last Invoice Date:'):
            hospital_data['last_invoice'] = line.replace('Last Invoice Date:', '').strip()
        elif line.startswith('Predicted time interval'):
            hospital_data['predicted_interval'] = line.split(':')[1].strip()
        elif line.startswith('Forecasted Next Invoice Date'):
            # Handle format like "Forecasted Next Invoice Date (Stacking): 2025-09-11"
            if ':' in line:
                hospital_data['next_date'] = line.split(':', 1)[1].strip()
            else:
                hospital_data['next_date'] = line.replace('Forecasted Next Invoice Date', '').strip()
        elif re.match(r'\s*\d+\)', line):
            hospital_data['forecast_dates'].append(line.strip())
    
    return hospital_data

def build_html_from_text(txt: str) -> str:
    # Split by separator lines
    blocks = re.split(r'=+\s*\n?', txt)
    hospitals = []
    
    for block in blocks:
        if block.strip():
            hospital_data = parse_hospital_block(block)
            if hospital_data and hospital_data['hospital']:
                hospitals.append(hospital_data)
    
    # Remove duplicates based on hospital name and brand
    unique_hospitals = []
    seen = set()
    for h in hospitals:
        key = (h['hospital'], h['brand'])
        if key not in seen:
            unique_hospitals.append(h)
            seen.add(key)
    
    # Apply date filtering: remove invalid/past dates and non-coming-week dates
    filtered_hospitals = []
    for h in unique_hospitals:
        next_date = h['next_date']
        if next_date:
            # Check if date is valid and in the future
            if is_valid_future_date(next_date):
                # Check if date is in the coming week
                if is_in_coming_week(next_date):
                    filtered_hospitals.append(h)
            # Skip entries with past dates or not in coming week
        else:
            # Skip entries without next date
            continue
    
    # Use filtered hospitals for stats and display
    display_hospitals = filtered_hospitals
    
    # Calculate dynamic stats
    unique_products = len(set(h['brand'] for h in display_hospitals))
    unique_hospital_names = len(set(h['hospital'] for h in display_hospitals))
    
    # Generate HTML cards for each hospital
    cards_html = ""
    
    if not display_hospitals:
        cards_html = """
        <div class="no-results">
            <div class="no-results-icon">📅</div>
            <h3>No forecasts for the coming week</h3>
            <p>All predicted invoice dates are either in the past or not scheduled for the coming week.</p>
        </div>
        """
    else:
        for i, hospital in enumerate(display_hospitals):
            # Determine urgency color based on predicted interval
            try:
                days = int(hospital['predicted_interval'].split()[0])
                if days <= 14:
                    urgency_class = "urgent"
                    urgency_text = "High Priority"
                elif days <= 30:
                    urgency_class = "medium"
                    urgency_text = "Medium Priority"
                else:
                    urgency_class = "low"
                    urgency_text = "Low Priority"
            except:
                urgency_class = "medium"
                urgency_text = "Medium Priority"
            
            forecast_list = ""
            for forecast in hospital['forecast_dates']:
                forecast_list += f"<li>{html.escape(forecast)}</li>"
            
            cards_html += f"""
        <div class="hospital-card {urgency_class}">
            <div class="card-header">
                <h3>{html.escape(hospital['hospital'])}</h3>
            </div>
            <div class="card-content">
                <div class="info-grid">
                    <div class="info-item">
                        <span class="label">Product:</span>
                        <span class="value">{html.escape(hospital['brand'])}</span>
                    </div>
                    <div class="info-item">
                        <span class="label">Last Invoice:</span>
                        <span class="value">{html.escape(hospital['last_invoice'])}</span>
                    </div>
                    <div class="info-item">
                        <span class="label">Predicted Interval:</span>
                        <span class="value interval">{html.escape(hospital['predicted_interval'])}</span>
                    </div>
                    <div class="info-item">
                        <span class="label">Next Invoice Date:</span>
                        <span class="value next-date">{html.escape(hospital['next_date'])}</span>
                    </div>
                </div>
                
            </div>
        </div>
        """
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_hospitals = len(display_hospitals)
    
    # Dynamic stats section based on what data we have
    stats_html = f"""
                <div class="stat-item">
                    <div class="stat-number">{total_hospitals}</div>
                    <div class="stat-label">Total Entries</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{unique_products}</div>
                    <div class="stat-label">Products</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{unique_hospital_names}</div>
                    <div class="stat-label">Hospitals</div>
                </div>
    """
    
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Inventory Forecasts Dashboard</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    
    body {{ 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 20px;
        color: #333;
    }}
    
    .container {{
        max-width: 1200px;
        margin: 0 auto;
        background: white;
        border-radius: 15px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        overflow: hidden;
    }}
    
    .header {{
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 30px;
        text-align: center;
    }}
    
    .header h1 {{
        font-size: 2.5em;
        margin-bottom: 10px;
        font-weight: 300;
    }}
    
    .header .subtitle {{
        font-size: 1.1em;
        opacity: 0.8;
        margin-bottom: 20px;
    }}
    
    .stats {{
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: 15px;
    }}
    
    .stat-item {{
        text-align: center;
    }}
    
    .stat-number {{
        font-size: 2em;
        font-weight: bold;
        color: #3498db;
    }}
    
    .stat-label {{
        font-size: 0.9em;
        opacity: 0.8;
    }}
    
    .content {{
        padding: 40px;
    }}
    
    .hospital-card {{
        background: white;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-left: 5px solid #ddd;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    
    .hospital-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }}
    
    .hospital-card.urgent {{
        border-left-color: #e74c3c;
    }}
    
    .hospital-card.medium {{
        border-left-color: #f39c12;
    }}
    
    .hospital-card.low {{
        border-left-color: #27ae60;
    }}
    
    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 25px;
        border-bottom: 1px solid #eee;
    }}
    
    .card-header h3 {{
        font-size: 1.4em;
        color: #2c3e50;
        font-weight: 600;
    }}
    
    .priority-badge {{
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
        text-transform: uppercase;
    }}
    
    .priority-badge.urgent {{
        background: #ffe6e6;
        color: #e74c3c;
    }}
    
    .priority-badge.medium {{
        background: #fff3e0;
        color: #f39c12;
    }}
    
    .priority-badge.low {{
        background: #e8f5e8;
        color: #27ae60;
    }}
    
    .card-content {{
        padding: 25px;
    }}
    
    .info-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 15px;
        margin-bottom: 25px;
    }}
    
    .info-item {{
        display: flex;
        flex-direction: column;
        gap: 5px;
    }}
    
    .label {{
        font-size: 0.9em;
        color: #7f8c8d;
        font-weight: 500;
    }}
    
    .value {{
        font-size: 1.1em;
        color: #2c3e50;
        font-weight: 600;
    }}
    
    .interval {{
        color: #e74c3c;
        font-size: 1.2em;
    }}
    
    .next-date {{
        color: #3498db;
    }}
    
    .forecast-section {{
        border-top: 1px solid #eee;
        padding-top: 20px;
    }}
    
    .forecast-section h4 {{
        color: #34495e;
        margin-bottom: 15px;
        font-size: 1.1em;
    }}
    
    .forecast-list {{
        list-style: none;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
    }}
    
    .forecast-list li {{
        background: #f8f9fa;
        padding: 10px 15px;
        border-radius: 6px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.9em;
        border-left: 3px solid #3498db;
    }}
    
    .no-results {{
        text-align: center;
        padding: 60px 20px;
        color: #7f8c8d;
    }}
    
    .no-results-icon {{
        font-size: 4em;
        margin-bottom: 20px;
    }}
    
    .no-results h3 {{
        font-size: 1.8em;
        margin-bottom: 15px;
        color: #34495e;
    }}
    
    .no-results p {{
        font-size: 1.1em;
        line-height: 1.6;
        max-width: 500px;
        margin: 0 auto;
    }}
    
    .footer {{
        background: #f8f9fa;
        padding: 20px;
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9em;
        border-top: 1px solid #eee;
    }}
    
    @media (max-width: 768px) {{
        .container {{ margin: 10px; }}
        .header {{ padding: 20px; }}
        .content {{ padding: 20px; }}
        .header h1 {{ font-size: 2em; }}
        .stats {{ flex-direction: column; gap: 15px; }}
        .info-grid {{ grid-template-columns: 1fr; }}
        .forecast-list {{ grid-template-columns: 1fr; }}
    }}
</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Inventory Forecasts Dashboard</h1>
            <p class="subtitle">AI-Powered Hospital Inventory Predictions</p>
            <div class="stats">
                {stats_html}
            </div>
        </div>
        
        <div class="content">
            {cards_html}
        </div>
        
        <div class="footer">
            Generated on {current_time} | Powered by Machine Learning Stacking Model
        </div>
    </div>
</body>
</html>"""

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    base_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(base_dir, "Email_display.txt")

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            txt = f.read()
    except FileNotFoundError:
        txt = "No content found. Email_display.txt is missing."

    html_out = build_html_from_text(txt)
    sys.stdout.write(html_out)

if __name__ == "__main__":
    main()