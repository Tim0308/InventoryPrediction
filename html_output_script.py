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

def is_in_coming_3_weeks(date_str: str) -> bool:
    """Check if the date is in the coming 3 weeks (assuming script runs on Monday)"""
    try:
        forecast_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        today = datetime.now().date()
        
        # Calculate the start of the coming week (next Monday if today is not Monday)
        days_until_monday = (7 - today.weekday()) % 7
        if days_until_monday == 0 and today.weekday() == 0:
            # If today is Monday, coming 3 weeks starts today
            coming_period_start = today
        else:
            # Otherwise, coming 3 weeks starts on the next Monday
            coming_period_start = today + timedelta(days=days_until_monday)
        
        coming_period_end = coming_period_start + timedelta(days=100)  # 3 weeks (21 days - 1)
        
        return coming_period_start <= forecast_date <= coming_period_end
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
    
    # Apply date filtering: remove invalid/past dates and non-coming-3-weeks dates
    filtered_hospitals = []
    for h in unique_hospitals:
        next_date = h['next_date']
        if next_date:
            # Check if date is valid and in the future
            if is_valid_future_date(next_date):
                # Check if date is in the coming 3 weeks
                if is_in_coming_3_weeks(next_date):
                    filtered_hospitals.append(h)
            # Skip entries with past dates or not in coming 3 weeks
        else:
            # Skip entries without next date
            continue
    
    # Use filtered hospitals for stats and display
    display_hospitals = filtered_hospitals
    
    # Group hospitals by product for stacking card layout
    products_dict = {}
    for h in display_hospitals:
        product = h['brand']
        if product not in products_dict:
            products_dict[product] = []
        products_dict[product].append(h)
    
    # Calculate dynamic stats
    unique_products_list = list(products_dict.keys())
    unique_hospital_names = len(set(h['hospital'] for h in display_hospitals))
    
    # Create product names display (limit to reasonable length for display)
    if len(unique_products_list) <= 3:
        products_display = "<br>".join(unique_products_list)
    else:
        products_display = f"{len(unique_products_list)} Products:<br>" + "<br>".join(unique_products_list[:2]) + f"<br>... and {len(unique_products_list)-2} more"
    
    # Generate HTML cards grouped by product
    cards_html = ""
    
    if not display_hospitals:
        cards_html = """
        <div class="no-results">
            <div class="no-results-icon">📅</div>
            <h3>No forecasts for the coming 3 weeks</h3>
            <p>All predicted invoice dates are either in the past or not scheduled for the coming 3 weeks.</p>
        </div>
        """
    else:
        for product_name, hospitals in products_dict.items():
            # Create product header
            cards_html += f"""
            <div class="product-section">
                <div class="product-header">
                    <h2>📦 {html.escape(product_name)}</h2>
                </div>
                <div class="hospitals-stack">
            """
            
            # Add hospital cards for this product
            for i, hospital in enumerate(hospitals):
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
                
                cards_html += f"""
                <div class="hospital-card {urgency_class}">
                    <div class="card-content">
                        <div class="hospital-name">{html.escape(hospital['hospital'])}</div>
                        <div class="info-row">
                            <span class="info-label">Last Invoice:</span>
                            <span class="info-value">{html.escape(hospital['last_invoice'])}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Next Invoice:</span>
                            <span class="info-value next-date">{html.escape(hospital['next_date'])}</span>
                        </div>
                    </div>
                </div>
                """
            
            cards_html += """
                </div>
            </div>
            """
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_hospitals = len(display_hospitals)
    
    # Dynamic stats section based on what data we have
    stats_html = f"""
                <div class="stat-item">
                    <div class="stat-number" style="font-size: 0.9em; line-height: 1.2;">{products_display}</div>
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
        background: white;
        padding: 20px;
        color: #333;
        line-height: 1.4;
    }}
    
    .container {{
        max-width: 800px;
        margin: 0 auto;
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
    }}
    
    .header {{
        background: white;
        border-bottom: 2px solid #e1e5e9;
        padding: 25px;
        text-align: center;
    }}
    
    .header h1 {{
        font-size: 1.8em;
        margin-bottom: 8px;
        color: #2c3e50;
        font-weight: 600;
    }}
    
    .header .subtitle {{
        font-size: 1em;
        color: #6c757d;
        margin-bottom: 15px;
    }}
    
    .stats {{
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-top: 10px;
    }}
    
    .stat-item {{
        text-align: center;
    }}
    
    .stat-number {{
        font-size: 1.2em;
        font-weight: bold;
        color: #2c3e50;
        line-height: 1.2;
    }}
    
    .stat-label {{
        font-size: 0.85em;
        color: #6c757d;
        margin-top: 5px;
    }}
    
    .content {{
        padding: 20px;
    }}
    
    .product-section {{
        margin-bottom: 30px;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        overflow: hidden;
    }}
    
    .product-header {{
        background: #e9ecef;
        padding: 15px 20px;
        border-bottom: 1px solid #dee2e6;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    
    .product-header h2 {{
        font-size: 1.3em;
        color: #2c3e50;
        font-weight: 600;
        margin: 0;
    }}
    
    .hospital-count {{
        background: #6c757d;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
    }}
    
    .hospitals-stack {{
        background: white;
    }}
    
    .hospital-card {{
        border-bottom: 1px solid #f1f3f4;
        transition: background-color 0.2s ease;
    }}
    
    .hospital-card:last-child {{
        border-bottom: none;
    }}
    
    .hospital-card:hover {{
        background: #f8f9fa;
    }}
    
    .hospital-card.urgent {{
        border-left: 4px solid #dc3545;
    }}
    
    .hospital-card.medium {{
        border-left: 4px solid #fd7e14;
    }}
    
    .hospital-card.low {{
        border-left: 4px solid #28a745;
    }}
    
    .card-content {{
        padding: 15px 20px;
    }}
    
    .hospital-name {{
        font-size: 1.1em;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 8px;
    }}
    
    .info-row {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
        align-items: center;
    }}
    
    .info-row:last-child {{
        margin-bottom: 0;
    }}
    
    .info-label {{
        font-size: 0.9em;
        color: #6c757d;
        font-weight: 500;
    }}
    
    .info-value {{
        font-size: 0.95em;
        color: #2c3e50;
        font-weight: 600;
    }}
    
    .next-date {{
        color: #007bff !important;
        background: #e3f2fd;
        padding: 2px 8px;
        border-radius: 4px;
    }}
    
    .no-results {{
        text-align: center;
        padding: 40px 20px;
        color: #6c757d;
    }}
    
    .no-results-icon {{
        font-size: 3em;
        margin-bottom: 15px;
    }}
    
    .no-results h3 {{
        font-size: 1.5em;
        margin-bottom: 10px;
        color: #495057;
    }}
    
    .no-results p {{
        font-size: 1em;
        line-height: 1.5;
        max-width: 400px;
        margin: 0 auto;
    }}
    
    .footer {{
        background: #f8f9fa;
        padding: 15px;
        text-align: center;
        color: #6c757d;
        font-size: 0.85em;
        border-top: 1px solid #e1e5e9;
    }}
    
    /* Outlook-specific fixes */
    table {{ border-collapse: collapse; }}
    .outlook-table {{ width: 100%; }}
    
    @media (max-width: 600px) {{
        .container {{ margin: 10px; border-radius: 4px; }}
        .header {{ padding: 15px; }}
        .content {{ padding: 15px; }}
        .header h1 {{ font-size: 1.5em; }}
        .stats {{ flex-direction: column; gap: 20px; }}
        .product-header {{ flex-direction: column; gap: 10px; text-align: center; }}
        .info-row {{ flex-direction: column; align-items: flex-start; gap: 3px; }}
    }}
</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Weekly Inventory Forecasts</h1>
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