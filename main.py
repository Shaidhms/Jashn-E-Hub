import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime
import random
import os
from pathlib import Path
import calendar
import csv
import re
import json
import io
import time
import qrcode
from PIL import Image


# Set page configuration
st.set_page_config(
    page_title="EventEase Planner",
    layout="wide"
)

# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
USER_DATA_FILE = DATA_DIR / "user_data.csv"

# Function to validate email
def is_valid_email(email):
    """Validate email format using regex"""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

# Function to validate phone number (Indian format)
def is_valid_phone(phone):
    """Validate Indian phone number format"""
    # Accept 10 digits with optional +91 prefix
    pattern = r"^(?:\+91)?[6-9]\d{9}$"
    return re.match(pattern, phone) is not None

# Function to save user data to CSV
def save_user_data(user_data):
    file_exists = os.path.isfile(USER_DATA_FILE)
    
    # Create headers for the CSV if file doesn't exist
    fieldnames = ['name', 'email', 'phone', 'city', 'event_type', 'budget', 'location', 
                 'event_date', 'selected_services', 'num_guests', 'eco_preference', 'timestamp']
    
    with open(USER_DATA_FILE, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write header if file is being created for the first time
        if not file_exists:
            writer.writeheader()
        
        # Add timestamp and write the data
        user_data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow(user_data)
    
    return True

# Load ML model
@st.cache_resource
def load_model():
    model_path = os.path.join("artifacts", "model.joblib")
    
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            st.warning(f"Model file not found at {model_path}. Using rule-based recommendations instead.")
            return None
            
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.warning(f"Could not load ML model: {e}. Using rule-based recommendations instead.")
        return None

# Sample data for demonstration
service_providers = {
    'Arjun': {'services': ['catering', 'decoration'], 'rating': 4.7, 'location': 'Mumbai', 'budget_category': '50K-1L', 'eco_friendly': True},
    'Saanvi': {'services': ['catering', 'venue'], 'rating': 4.9, 'location': 'Delhi', 'budget_category': '2L+', 'eco_friendly': True},
    'Fiona': {'services': ['photography', 'venue'], 'rating': 4.5, 'location': 'Bangalore', 'budget_category': '2L+', 'eco_friendly': False},
    'Diya': {'services': ['decoration', 'entertainment'], 'rating': 4.8, 'location': 'Chennai', 'budget_category': '50K-1L', 'eco_friendly': True},
    'Priya': {'services': ['catering', 'venue', 'decoration'], 'rating': 4.6, 'location': 'Pune', 'budget_category': '50K-1L', 'eco_friendly': False},
    'Rahul': {'services': ['photography', 'entertainment'], 'rating': 4.3, 'location': 'Mumbai', 'budget_category': '10K-30K', 'eco_friendly': False},
    'Aarav': {'services': ['transportation', 'venue'], 'rating': 4.4, 'location': 'Delhi', 'budget_category': '10K-30K', 'eco_friendly': True},
    'Kabir': {'services': ['wedding cards', 'decoration'], 'rating': 4.5, 'location': 'Chennai', 'budget_category': '10K-30K', 'eco_friendly': True},
    'George': {'services': ['priest', 'catering'], 'rating': 4.7, 'location': 'Bangalore', 'budget_category': '50K-1L', 'eco_friendly': False},
    'Hannah': {'services': ['entertainment', 'photography'], 'rating': 4.8, 'location': 'Pune', 'budget_category': '2L+', 'eco_friendly': True},
    'Ananya': {'services': ['venue', 'catering', 'decoration'], 'rating': 4.9, 'location': 'Mumbai', 'budget_category': '2L+', 'eco_friendly': True},
    'Isha': {'services': ['wedding cards', 'transportation'], 'rating': 4.2, 'location': 'Delhi', 'budget_category': '10K-30K', 'eco_friendly': True},
    'Vivaan': {'services': ['priest', 'decoration', 'catering'], 'rating': 4.6, 'location': 'Bangalore', 'budget_category': '50K-1L', 'eco_friendly': True},
    'Meera': {'services': ['catering', 'decoration'], 'rating': 4.5, 'location': 'Mumbai', 'budget_category': '1L-2L', 'eco_friendly': True},
    'Rohan': {'services': ['venue', 'decoration'], 'rating': 4.6, 'location': 'Delhi', 'budget_category': '1L-2L', 'eco_friendly': True},
    'Tara': {'services': ['photography', 'entertainment'], 'rating': 4.7, 'location': 'Bangalore', 'budget_category': '1L-2L', 'eco_friendly': True},
    'Vikram': {'services': ['catering', 'transportation'], 'rating': 4.8, 'location': 'Chennai', 'budget_category': '1L-2L', 'eco_friendly': False},
    'Neha': {'services': ['venue', 'decoration', 'photography'], 'rating': 4.9, 'location': 'Pune', 'budget_category': '1L-2L', 'eco_friendly': True},
}

# Define budget ranges for display
budget_ranges = {
    '50K-1L': 'Economy (₹50,000-₹1,00,000)',
    '1L-2L': 'Standard (₹1,00,000-₹2,00,000)',
    '2L-3L': 'Premium (₹2,00,000-₹3,00,000)',
    '3L-5L': 'Luxury (₹3,00,000-₹5,00,000)',  
    '5L-10L': 'Ultra Luxury (₹5,00,000-₹10,00,000)',
    '10L+': 'Elite (₹10,00,000+)'
}

budget_category_mapping = {
    '50K-1L': 'low',
    '1L-2L': 'medium',
    '2L-3L': 'medium+', 
    '3L-5L': 'high',
    '5L-10L': 'high',
    '10L+': 'very_high'
}

# Define service costs by budget category
service_costs = {
    'catering': {'10K-30K': 15000, '30K-50K': 30000, '50K-1L': 40000, '1L-2L': 60000, '2L+': 100000, '5L+': 150000},
    'venue': {'10K-30K': 30000, '30K-50K': 75000, '50K-1L': 100000, '1L-2L': 150000, '2L+': 250000, '5L+': 500000},
    'decoration': {'10K-30K': 10000, '30K-50K': 25000, '50K-1L': 35000, '1L-2L': 50000, '2L+': 100000, '5L+': 200000},
    'photography': {'10K-30K': 15000, '30K-50K': 35000, '50K-1L': 50000, '1L-2L': 70000, '2L+': 120000, '5L+': 200000},
    'entertainment': {'10K-30K': 10000, '30K-50K': 25000, '50K-1L': 35000, '1L-2L': 50000, '2L+': 100000, '5L+': 200000},
    'transportation': {'10K-30K': 5000, '30K-50K': 15000, '50K-1L': 20000, '1L-2L': 30000, '2L+': 50000, '5L+': 100000},
    'wedding cards': {'10K-30K': 5000, '30K-50K': 10000, '50K-1L': 15000, '1L-2L': 20000, '2L+': 35000, '5L+': 50000},
    'priest': {'10K-30K': 5000, '30K-50K': 10000, '50K-1L': 12000, '1L-2L': 15000, '2L+': 25000, '5L+': 40000}
}

# Define possible event services based on event type
event_services = {
    'wedding': ['catering', 'venue', 'decoration', 'photography', 'entertainment', 'transportation', 'wedding cards', 'priest'],
    'engagement': ['catering', 'venue', 'decoration', 'photography', 'entertainment'],
    'corporate': ['catering', 'venue', 'decoration', 'photography', 'transportation'],
    'house party': ['catering', 'decoration', 'entertainment'],
    'birthday': ['catering', 'venue', 'decoration', 'photography', 'entertainment']
}

# Function to calculate match score based on rules
def get_provider_match_score(event_type, budget, location, services, provider):
    score = 0
    
    # Location match
    if provider['location'] == location:
        score += 3
    
    # Budget match
    if provider['budget_category'] == budget:
        score += 2
    
    # Services match
    service_match = sum(1 for s in services if s in provider['services'])
    score += service_match
    
    # Rating bonus
    score += (provider['rating'] - 4) * 2  # 0.1 in rating = 0.2 in score
    
    # Eco-friendly bonus
    if provider['eco_friendly']:
        score += 1
    
    return score

# Function to predict vendor suitability using ML model
def predict_vendor_suitability(model, event_type, budget, location, services, provider):
    if model is None:
        # Fallback to rule-based matching if model isn't loaded
        return get_provider_match_score(event_type, budget, location, services, provider)
    
    try:
        # Map the new budget format to the old one for model compatibility
        budget_for_model = budget_category_mapping.get(budget.upper(), 'medium')
        
        # Add the event_services column that was missing
        features = {
            'event_type': event_type,
            'budget': budget_for_model,
            'location': location,
            'service_match': 0,          # dummy
            'provider_rating': provider['rating'],  # Actual rating from provider
            'eco_friendly': 1 if provider['eco_friendly'] else 0,  # Convert boolean to integer
            'rating': provider['rating'],  # The model specifically requires this column
            'registration_number': 12345,  # Placeholder value
            'service': services[0] if services else 'general',  # Use the first selected service as a placeholder
            'event_services': ', '.join(services)  # Add the missing column
        }
        
        # Convert to DataFrame for model prediction
        df = pd.DataFrame([features], columns=[
            'event_type',
            'budget',
            'location',
            'service_match',
            'provider_rating',
            'eco_friendly',
            'rating',
            'registration_number',
            'service',
            'event_services'  # Include in the columns list
        ])
        
        # Predict suitability
        suitability_score = model.predict(df)[0]
        return suitability_score
    except Exception as e:
        st.error(f"Error using model for prediction: {e}")
        # Fallback to rule-based matching
        return get_provider_match_score(event_type, budget, location, services, provider)

# Function to get providers based on services, budget, and location
# Replace the get_providers_recommendation function with this fixed version:

def get_providers_recommendation(event_type, budget, location, selected_services):
    matching_providers = []
    model = load_model()
    
    for name, info in service_providers.items():
        # Check if any of the requested services are offered by this provider
        service_match = any(service in info['services'] for service in selected_services)
        
        if service_match:
            # Calculate match score using ML model or rule-based matching
            if model is not None:
                match_score = predict_vendor_suitability(model, event_type, budget, location, selected_services, info)
            else:
                match_score = get_provider_match_score(event_type, budget, location, selected_services, info)
            
            # Ensure match_score is numeric
            try:
                match_score = float(match_score)
            except (ValueError, TypeError):
                match_score = 5.0  # Default score if conversion fails
            
            # Create provider entry with score
            provider_entry = {
                'name': name,
                'services': info['services'],
                'rating': info['rating'],
                'location': info['location'],
                'budget': info['budget_category'],
                'eco_friendly': info['eco_friendly'],
                'match_score': match_score
            }
            
            matching_providers.append(provider_entry)
    
    # Sort by match score
    matching_providers.sort(key=lambda x: x['match_score'], reverse=True)
    
    return matching_providers

# Also add this helper function to safely format scores throughout the app:

def safe_format_score(score, decimal_places=1):
    """Safely format a score value for display"""
    try:
        return f"{float(score):.{decimal_places}f}"
    except (ValueError, TypeError):
        return str(score) if score is not None else "N/A"

# Then replace ALL instances of score formatting with this function:
# Instead of: f"{provider['match_score']:.1f}"
# Use: safe_format_score(provider['match_score'], 1)

# Fixed cost breakdown function
def calculate_cost_breakdown(selected_services, budget):
    cost_breakdown = {}
    total_cost = 0
    
    for service in selected_services:
        service_cost = service_costs.get(service, {}).get(budget, 0)
        cost_breakdown[service] = service_cost
        total_cost += service_cost
    
    return cost_breakdown, total_cost

# FIXED: ML-based upgrade suggestions function
def get_ml_based_upgrade_suggestions(event_type, location, selected_services, current_budget):
    """Generate upgrade suggestions using ML model predictions"""
    model = load_model()
    budget_levels = ['50K-1L', '1L-2L', '2L-3L', '3L-5L', '5L-10L', '10L+']

    
    try:
        current_index = budget_levels.index(current_budget)
    except ValueError:
        current_index = 0
    
    suggestions = []
    
    # Check all higher budget levels
    for next_index in range(current_index + 1, len(budget_levels)):
        next_budget = budget_levels[next_index]
        
        # Get providers for the upgraded budget
        upgraded_providers = []
        
        for name, info in service_providers.items():
            # Check if provider matches the upgraded budget and has required services
            if (info['budget_category'] == next_budget and 
                any(service in info['services'] for service in selected_services)):
                
                # Use ML model to predict suitability for upgraded budget
                if model is not None:
                    match_score = predict_vendor_suitability(
                        model, event_type, next_budget, location, selected_services, info
                    )
                else:
                    match_score = get_provider_match_score(
                        event_type, next_budget, location, selected_services, info
                    )
                
                upgraded_providers.append({
                    'name': name,
                    'services': info['services'],
                    'rating': info['rating'],
                    'location': info['location'],
                    'budget': info['budget_category'],
                    'eco_friendly': info['eco_friendly'],
                    'match_score': match_score
                })
        
        # Sort by match score
        upgraded_providers.sort(key=lambda x: x['match_score'], reverse=True)
        
        if upgraded_providers:
            # Calculate cost difference
            current_cost = sum(service_costs.get(service, {}).get(current_budget, 0) 
                             for service in selected_services)
            upgraded_cost = sum(service_costs.get(service, {}).get(next_budget, 0) 
                              for service in selected_services)
            additional_cost = upgraded_cost - current_cost
            
            # Create detailed suggestion with ML predictions
            # FIXED: Safely calculate ML satisfaction score - only use numeric match_score values
            valid_scores = []
            for p in upgraded_providers[:5]:
                try:
                    score = float(p['match_score'])
                    valid_scores.append(score)
                except (ValueError, TypeError):
                    # Skip invalid scores
                    continue
            
            ml_satisfaction = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            
            suggestion = {
                'budget_level': budget_ranges[next_budget],
                'budget_key': next_budget,
                'additional_cost': additional_cost,
                'total_cost': upgraded_cost,
                'num_providers': len(upgraded_providers),
                'top_providers': upgraded_providers[:3],  # Top 3 providers
                'ml_predicted_satisfaction': ml_satisfaction,
                'service_improvements': [],
                'benefits': []
            }
            
            # Add specific service improvements
            for service in selected_services:
                current_service_cost = service_costs.get(service, {}).get(current_budget, 0)
                upgraded_service_cost = service_costs.get(service, {}).get(next_budget, 0)
                
                if upgraded_service_cost > current_service_cost:
                    improvement = upgraded_service_cost - current_service_cost
                    suggestion['service_improvements'].append({
                        'service': service,
                        'improvement': improvement,
                        'details': f"Upgrade {service} for ₹{improvement:,} more"
                    })
            
            # Add benefits based on top providers
            if upgraded_providers:
                top_provider = upgraded_providers[0]
                # FIXED: Safely calculate average rating - only use numeric rating values
                valid_ratings = []
                for p in upgraded_providers:
                    try:
                        rating = float(p['rating'])
                        valid_ratings.append(rating)
                    except (ValueError, TypeError):
                        continue
                
                avg_rating = sum(valid_ratings) / len(valid_ratings) if valid_ratings else 0.0
                
                # Safely format the match scorer
                try:
                    match_score_formatted = f"{float(top_provider['match_score']):.2f}"
                except (ValueError, TypeError):
                    match_score_formatted = str(top_provider['match_score'])

                suggestion['benefits'].extend([
                    f"Access to {len(upgraded_providers)} premium vendors",
                    f"Top recommended: {top_provider['name']} (rated {top_provider['rating']})",
                    f"ML prediction score: {match_score_formatted}/10",
                    f"Average provider rating: {avg_rating:.1f}"
                ])
                                # Add eco-friendly benefits if applicable
                eco_providers = [p for p in upgraded_providers if p['eco_friendly']]
                if eco_providers:
                    suggestion['benefits'].append(f"{len(eco_providers)} eco-friendly options available")
            
            suggestions.append(suggestion)
    
    return suggestions

# Function to generate calendar with availability
def generate_availability_calendar(provider_name, month=None, year=None):
    if month is None:
        month = datetime.date.today().month
    if year is None:
        year = datetime.date.today().year
    
    # Get the calendar for the specified month and year
    cal = calendar.monthcalendar(year, month)
    
    # Generate random availability (in a real app, this would come from a database)
    available_dates = []
    partially_available_dates = []
    booked_dates = []
    
    for week in cal:
        for day in week:
            if day != 0:  # Skip days that belong to other months
                current_date = datetime.date(year, month, day)
                if current_date >= datetime.date.today():  # Only future dates
                    random_value = random.random()
                    if random_value > 0.6:  # 40% chance of being available
                        available_dates.append(day)
                    elif random_value > 0.3:  # 30% chance of being partially available
                        partially_available_dates.append(day)
                    else:  # 30% chance of being booked
                        booked_dates.append(day)
    
    return cal, available_dates, partially_available_dates, booked_dates

# Main function to create the app
def main():
    st.title("Jashn-E-Hub")
    st.markdown("### Find the perfect service providers for your event!")
    
    # Initialize model in session state
    if 'model' not in st.session_state:
        st.session_state['model'] = load_model()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Plan Your Event", "Service Providers", "Cost Breakdown", "ML Insights"])
    
    with tab1:
        st.header("Event Details")
        
        # User information collection
        with st.expander("Your Information", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                user_name = st.text_input("Your Name")
                user_email = st.text_input("Email Address")
                if user_email and not is_valid_email(user_email):
                    st.error("Please enter a valid email address.")
            with col2:
                user_phone = st.text_input("Phone Number (with or without +91)")
                if user_phone and not is_valid_phone(user_phone):
                    st.error("Please enter a valid 10-digit Indian phone number (with optional +91 prefix).")
                user_city = st.text_input("City of Residence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Event type selection
            event_type = st.selectbox(
                "Select Event Type",
                ['wedding', 'engagement', 'corporate', 'house party', 'birthday']
            )
            
            # Budget selection using the new currency format
            budget = st.selectbox(
                "Select Budget Category",
                list(budget_ranges.values()),
                format_func=lambda x: x
            )
            # Convert display budget to internal format
            selected_budget = next((k for k, v in budget_ranges.items() if v == budget), '50K-1L')
            
            # Location selection
            location = st.selectbox(
                "Select Location",
                ['Mumbai', 'Chennai', 'Pune', 'Delhi', 'Bangalore']
            )
            
        with col2:
            # Date selection
            event_date = st.date_input(
                "Select Event Date",
                datetime.date.today() + datetime.timedelta(days=30),
                min_value=datetime.date.today()
            )
            
            # Services selection based on event type
            available_services = event_services.get(event_type, [])
            selected_services = st.multiselect(
                "Select Required Services",
                available_services,
                default=available_services[:3]  # Default to first 3 services
            )
            
            # Number of guests
            num_guests = st.slider("Number of Guests", 10, 500, 100)
            
            # Eco-friendly preference
            eco_preference = st.checkbox("Prefer eco-friendly options", value=True)
        
        # Store selections in session state
        if st.button("Find Providers & Generate ML Insights"):
            # Form validation
            validation_error = False
            
            if not user_name:
                st.error("Please enter your name.")
                validation_error = True
            
            if not user_email:
                st.error("Please enter your email address.")
                validation_error = True
            elif not is_valid_email(user_email):
                st.error("Please enter a valid email address.")
                validation_error = True
                
            if not user_phone:
                st.error("Please enter your phone number.")
                validation_error = True
            elif not is_valid_phone(user_phone):
                st.error("Please enter a valid 10-digit Indian phone number.")
                validation_error = True
                
            if not validation_error:
                # Save user data to CSV
                user_data = {
                    'name': user_name,
                    'email': user_email,
                    'phone': user_phone,
                    'city': user_city,
                    'event_type': event_type,
                    'budget': selected_budget,
                    'location': location,
                    'event_date': event_date.strftime('%Y-%m-%d'),
                    'selected_services': ', '.join(selected_services),
                    'num_guests': num_guests,
                    'eco_preference': eco_preference
                }
                
                save_success = save_user_data(user_data)
                if save_success:
                    st.success("Your information has been saved!")
                
                # Store in session state for current session
                st.session_state['user_name'] = user_name
                st.session_state['user_email'] = user_email
                st.session_state['user_phone'] = user_phone
                st.session_state['user_city'] = user_city
                st.session_state['event_type'] = event_type
                st.session_state['budget'] = selected_budget
                st.session_state['budget_display'] = budget
                st.session_state['location'] = location
                st.session_state['event_date'] = event_date
                st.session_state['selected_services'] = selected_services
                st.session_state['num_guests'] = num_guests
                st.session_state['eco_preference'] = eco_preference
                
                # Calculate cost breakdown
                cost_breakdown, total_cost = calculate_cost_breakdown(selected_services, selected_budget)
                st.session_state['cost_breakdown'] = cost_breakdown
                st.session_state['total_cost'] = total_cost
                
                # Get provider recommendations
                providers = get_providers_recommendation(event_type, selected_budget, location, selected_services)
                st.session_state['providers'] = providers
                
                # FIXED: Get ML-based upgrade suggestions
                upgrade_suggestions = get_ml_based_upgrade_suggestions(
                    event_type, location, selected_services, selected_budget
                )
                st.session_state['ml_upgrade_suggestions'] = upgrade_suggestions
                
                st.success("Event details saved! Check all tabs for detailed analysis and ML-powered recommendations.")
    
    with tab2:
        st.header("Recommended Service Providers")
        
        if 'providers' in st.session_state:
            providers = st.session_state['providers']
            
            # FIXED: Show ML-based upgrade suggestions
            if 'ml_upgrade_suggestions' in st.session_state and st.session_state['ml_upgrade_suggestions']:
                st.subheader("🤖 ML-Powered Budget Upgrade Recommendations")
                
                for i, suggestion in enumerate(st.session_state['ml_upgrade_suggestions'][:2]):  # Show top 2 suggestions
                    with st.expander(
                        f"Upgrade to {suggestion['budget_level']} - ML Satisfaction Score: {suggestion['ml_predicted_satisfaction']:.1f}/10", 
                        expanded=(i==0)
                    ):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Additional Cost:** ₹{suggestion['additional_cost']:,}")
                            st.markdown(f"**Total Budget:** ₹{suggestion['total_cost']:,}")
                            st.markdown(f"**Available Providers:** {suggestion['num_providers']}")
                            
                            st.markdown("**🎯 Why our ML model recommends this upgrade:**")
                            for benefit in suggestion['benefits']:
                                st.write(f"• {benefit}")
                            
                            if suggestion['service_improvements']:
                                st.markdown("**Service Upgrades:**")
                                for imp in suggestion['service_improvements']:
                                    st.write(f"• {imp['details']}")
                        
                        with col2:
                            st.markdown("**Top ML-Recommended Providers:**")
                            for provider in suggestion['top_providers']:
                                st.markdown(f"**{provider['name']}**")
                                st.markdown(f"Rating: {'⭐' * int(provider['rating'])}")
                                try:
                                    ml_score_formatted = f"{float(provider['match_score']):.1f}"
                                except (ValueError, TypeError):
                                    ml_score_formatted = str(provider['match_score'])
                                st.markdown(f"ML Score: {ml_score_formatted}/10")
                                if provider['eco_friendly']:
                                    st.markdown("🌱 Eco-friendly")
                                st.markdown("---")
                        
                        if st.button(f"Upgrade to {suggestion['budget_level']}", key=f"upgrade_{i}"):
                            # Update budget and recalculate everything
                            new_budget = suggestion['budget_key']
                            new_budget_display = suggestion['budget_level']
                            
                            st.session_state['budget'] = new_budget
                            st.session_state['budget_display'] = new_budget_display
                            
                            # Recalculate with new budget
                            cost_breakdown, total_cost = calculate_cost_breakdown(
                                st.session_state['selected_services'], new_budget
                            )
                            st.session_state['cost_breakdown'] = cost_breakdown
                            st.session_state['total_cost'] = total_cost
                            
                            providers = get_providers_recommendation(
                                st.session_state['event_type'], 
                                new_budget, 
                                st.session_state['location'], 
                                st.session_state['selected_services']
                            )
                            st.session_state['providers'] = providers
                            
                            # Update upgrade suggestions
                            upgrade_suggestions = get_ml_based_upgrade_suggestions(
                                st.session_state['event_type'], 
                                st.session_state['location'], 
                                st.session_state['selected_services'], 
                                new_budget
                            )
                            st.session_state['ml_upgrade_suggestions'] = upgrade_suggestions
                            
                            st.success(f"✨ Budget upgraded to {new_budget_display}! ML analysis updated.")
                            st.rerun()
            
            # Show current providers
            if not providers:
                st.warning("No service providers found for your criteria. Try adjusting your requirements.")
            else:
                st.write(f"**Found {len(providers)} providers matching your criteria for {st.session_state.get('budget_display', 'your budget')}:**")
                
                # Group providers by budget category
                budget_categories = sorted(set(p['budget'] for p in providers))
                
                for budget_category in budget_categories:
                    budget_providers = [p for p in providers if p['budget'] == budget_category]
                    
                    # Sort by match score (ML prediction)
                    budget_providers.sort(key=lambda x: x['match_score'], reverse=True)
                    
                    # Display budget category header
                    display_budget = next((v for k, v in budget_ranges.items() if k == budget_category), budget_category)
                    st.subheader(f"{display_budget} Providers")
                    
                    for i, provider in enumerate(budget_providers):
                        provider_key = f"{provider['name']}_{budget_category}"
                        
                        # Enhanced provider display with ML scores
                        with st.expander(f"🏆 {provider['name']} - ML Score: {float(provider['match_score']):.1f}/10 | Rating: {provider['rating']} {'🌱' if provider['eco_friendly'] else ''}"):

                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Location:** {provider['location']}")
                                st.write(f"**Budget Category:** {display_budget}")
                                st.write(f"**Services Offered:** {', '.join(provider['services'])}")
                                st.write(f"**ML Compatibility Score:** {provider['match_score']:.1f}/10")
                                if provider['eco_friendly']:
                                    st.write("**Eco-Friendly:** Yes 🌱")
                                
                                # ML-based recommendation reason
                                if provider['match_score'] > 8:
                                    st.success("🎯 Highly recommended by our ML model!")
                                elif provider['match_score'] > 6:
                                    st.info("👍 Good match according to ML analysis")
                                else:
                                    st.warning("⚠️ Moderate match - consider other options")
                                
                                if st.button(f"Contact {provider['name']}", key=f"contact_{provider_key}"):
                                    # Save contact request to CSV
                                    contact_data = {
                                        'name': st.session_state.get('user_name', 'Unknown'),
                                        'email': st.session_state.get('user_email', 'Unknown'),
                                        'phone': st.session_state.get('user_phone', 'Unknown'),
                                        'city': st.session_state.get('user_city', 'Unknown'),
                                        'event_type': st.session_state.get('event_type', 'Unknown'),
                                        'budget': st.session_state.get('budget', 'Unknown'),
                                        'location': st.session_state.get('location', 'Unknown'),
                                        'provider': provider['name'],
                                        'ml_score': provider['match_score'],
                                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    
                                    # Create contacts directory if it doesn't exist
                                    contacts_dir = DATA_DIR / "contacts"
                                    contacts_dir.mkdir(exist_ok=True)
                                    contact_file = contacts_dir / "contact_requests.csv"
                                    
                                    # Check if file exists to determine if header is needed
                                    file_exists = os.path.isfile(contact_file)
                                    
                                    with open(contact_file, 'a', newline='') as f:
                                        writer = csv.DictWriter(f, fieldnames=contact_data.keys())
                                        if not file_exists:
                                            writer.writeheader()
                                        writer.writerow(contact_data)
                                    
                                    st.success(f"Contact request sent to {provider['name']}! They will get in touch with you soon.")
                            
                            with col2:
                                # Show calendar availability
                                st.write("**Availability Calendar:**")
                                month = st.session_state.get('event_date', datetime.date.today()).month
                                year = st.session_state.get('event_date', datetime.date.today()).year
                                
                                cal, available_dates, partially_available, booked_dates = generate_availability_calendar(
                                    provider['name'], month, year
                                )
                                
                                # Create calendar display
                                cal_html = "<table class='calendar'><tr><th>Mon</th><th>Tue</th><th>Wed</th><th>Thu</th><th>Fri</th><th>Sat</th><th>Sun</th></tr>"
                                
                                for week in cal:
                                    cal_html += "<tr>"
                                    for day in week:
                                        if day == 0:
                                            cal_html += "<td></td>"
                                        else:
                                            if day in available_dates:
                                                cal_html += f"<td class='available'>{day}</td>"
                                            elif day in partially_available:
                                                cal_html += f"<td class='partial'>{day}</td>"
                                            elif day in booked_dates:
                                                cal_html += f"<td class='booked'>{day}</td>"
                                            else:
                                                cal_html += f"<td>{day}</td>"
                                    cal_html += "</tr>"
                                
                                cal_html += "</table>"
                                cal_html += """
                                <div class='legend'>
                                    <span class='available-box'></span> Available
                                    <span class='partial-box'></span> Limited Availability
                                    <span class='booked-box'></span> Booked
                                </div>
                                <style>
                                    .calendar {width: 100%; border-collapse: collapse;}
                                    .calendar td, .calendar th {padding: 5px; text-align: center; border: 1px solid #ddd;}
                                    .available {background-color: #c8e6c9;}
                                    .partial {background-color: #fff9c4;}
                                    .booked {background-color: #ffcdd2;}
                                    .legend {margin-top: 10px; display: flex; gap: 10px; align-items: center;}
                                    .available-box {width: 15px; height: 15px; background-color: #c8e6c9; display: inline-block;}
                                    .partial-box {width: 15px; height: 15px; background-color: #fff9c4; display: inline-block;}
                                    .booked-box {width: 15px; height: 15px; background-color: #ffcdd2; display: inline-block;}
                                </style>
                                """
                                st.markdown(cal_html, unsafe_allow_html=True)
        else:
            st.info("Please fill in your event details in the Plan Your Event tab to see recommended providers.")
    
    with tab3:
        st.header("Cost Breakdown & Payment")

        if 'cost_breakdown' in st.session_state:
            cost_breakdown = st.session_state['cost_breakdown']
            total_cost = st.session_state['total_cost']

            st.write(f"**Budget Category:** {st.session_state.get('budget_display', 'Unknown')}")
            st.write(f"**Total Estimated Cost:** ₹{total_cost:,}")

            # Cost breakdown charts (keep your original code)
            services = list(cost_breakdown.keys())
            costs = list(cost_breakdown.values())

            if services and costs:
                col1, col2 = st.columns([3, 2])

                with col1:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    colormap = plt.cm.viridis
                    colors = colormap(np.linspace(0, 0.8, len(services)))
                    bars = ax.bar(services, costs, color=colors, edgecolor='white', linewidth=1)
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                f'₹{int(height):,}',
                                ha='center', va='bottom', rotation=0, fontsize=9)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#DDDDDD')
                    ax.spines['bottom'].set_color('#DDDDDD')
                    ax.tick_params(bottom=False, left=False)
                    ax.set_axisbelow(True)
                    ax.yaxis.grid(True, color='#EEEEEE')
                    plt.title('Cost Breakdown by Service', fontsize=12, pad=15)
                    plt.xlabel('Services', fontsize=10)
                    plt.ylabel('Cost (₹)', fontsize=10)
                    plt.xticks(rotation=45, ha='right', fontsize=9)
                    plt.yticks(fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)

                with col2:
                    fig, ax = plt.subplots(figsize=(5, 5))
                    colors = plt.cm.tab10(np.linspace(0, 1, len(services)))
                    wedges, texts, autotexts = ax.pie(
                        costs,
                        labels=None,
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=colors,
                        wedgeprops={'width': 0.6, 'edgecolor': 'w', 'linewidth': 1},
                        textprops={'fontsize': 9}
                    )
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    ax.axis('equal')
                    plt.title('Cost Distribution', fontsize=12)
                    plt.legend(
                        wedges,
                        services,
                        title='Services',
                        loc='center left',
                        bbox_to_anchor=(0.85, 0, 0.5, 1),
                        fontsize=8
                    )
                    plt.tight_layout()
                    st.pyplot(fig)

                st.subheader("Detailed Cost Breakdown")
                cost_data = pd.DataFrame({
                    'Service': services,
                    'Cost (₹)': [f'₹{cost:,}' for cost in costs],
                    'Percentage': [f'{(cost/total_cost)*100:.1f}%' for cost in costs]
                })
                st.table(cost_data)

            # --- Payment Breakdown Section ---
            st.subheader("💳 Payment Schedule & Invoice")
            st.markdown("**Payment Method:** 25% now, 50% later, 25% after completion")

            # Calculate payment amounts
            advance_amt = round(total_cost * 0.25)
            mid_amt = round(total_cost * 0.50)
            final_amt = total_cost - advance_amt - mid_amt  # ensures total matches

            # Animation function for payments
            def payment_animation(label):
                with st.spinner(label):
                    progress_bar = st.progress(0)
                    for i in range(0, 101, 10):
                        progress_bar.progress(i)
                        time.sleep(0.07)
                    progress_bar.empty()

            # Function to create downloadable invoice file
            def generate_invoice_file(invoice):
                invoice_json = json.dumps(invoice, indent=4)
                invoice_bytes = invoice_json.encode('utf-8')
                buffer = io.BytesIO(invoice_bytes)
                return buffer

            # Function to generate invoice dictionary
            def generate_invoice(payment_type, amount, payment_method):
                import uuid, datetime
                user_name = st.session_state.get('user_name', 'Unknown')
                user_email = st.session_state.get('user_email', 'Unknown')
                user_phone = st.session_state.get('user_phone', 'Unknown')
                event_type = st.session_state.get('event_type', 'Unknown')
                event_date = st.session_state.get('event_date', 'Unknown')
                selected_services = st.session_state.get('selected_services', [])
                invoice_id = f"JASHN-{str(uuid.uuid4())[:8].upper()}"
                date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                invoice = {
                    "Invoice ID": invoice_id,
                    "Date": date_now,
                    "Name": user_name,
                    "Email": user_email,
                    "Phone": user_phone,
                    "Event Type": event_type,
                    "Event Date": str(event_date),
                    "Services": selected_services,
                    "Payment For": payment_type,
                    "Payment Method": payment_method,
                    "Amount Paid (₹)": amount,
                    "Total Event Cost (₹)": total_cost
                }
                return invoice

            # Demo UPI and Bank details
            UPI_ID = "demo@upi"
            BANK_DETAILS = {
                "Account Name": "Jashn Events Pvt Ltd",
                "Account Number": "1234567890",
                "IFSC Code": "DEMO0001234",
                "Bank Name": "Demo Bank",
                "Branch": "Main Branch"
            }

            # Payment buttons and invoice display
            colA, colB, colC = st.columns(3)
            for col, label, amt, pay_type, key in zip(
                [colA, colB, colC],
                ["Advance Payment (25%)", "Mid Payment (50%)", "Final Payment (25%)"],
                [advance_amt, mid_amt, final_amt],
                ["Advance (25%)", "Mid Payment (50%)", "Final Payment (25%)"],
                ["advance", "mid", "final"]
            ):
                with col:
                    if label.startswith("Advance"):
                        st.info(f"**{label}**")
                    elif label.startswith("Mid"):
                        st.warning(f"**{label}**")
                    else:
                        st.error(f"**{label}**")
                    st.write(f"Amount: ₹{amt:,}")
                    payment_method = st.radio(
                        "Select Payment Method",
                        ["UPI", "Bank Transfer"],
                        key=f"{key}_payment_method"
                    )
                    pay_btn = st.button(f"Pay {label.split()[0]}", key=f"{key}_pay_btn")
                    if pay_btn:
                        payment_animation(f"Processing {label}...")
                        invoice = generate_invoice(pay_type, amt, payment_method)
                        st.success(f"{label} successful via {payment_method}!")
                        st.balloons()
                        # UPI Option
                        if payment_method == "UPI":
                            st.markdown(f"**Scan this UPI QR to (pretend to) pay {amt:,} to {UPI_ID}:**")
                            upi_url = f"upi://pay?pa={UPI_ID}&pn=JashnEvents&am={amt}&cu=INR"
                            qr = qrcode.make(upi_url)
                            buf = io.BytesIO()
                            qr.save(buf, format="PNG")
                            st.image(buf.getvalue(), width=200)
                            st.info("If you really scan this, it will say: **Nice try!** 😄")
                        # Bank Transfer Option
                        elif payment_method == "Bank Transfer":
                            st.markdown("**Bank Transfer Details:**")
                            for k, v in BANK_DETAILS.items():
                                st.write(f"**{k}:** {v}")
                            st.info("Please use the above details to make your transfer.")
                        # Download invoice
                        buffer = generate_invoice_file(invoice)
                        st.download_button(
                            label="Download Invoice",
                            data=buffer,
                            file_name=f"invoice_{invoice['Invoice ID']}.json",
                            mime="application/json"
                        )

        else:
            st.info("Please fill in your event details in the Plan Your Event tab to see cost breakdown.")

    # ML Insights tab    
    with tab4:
        st.header("🤖 ML-Powered Insights & Recommendations")
        
        if 'ml_upgrade_suggestions' in st.session_state:
            # ML Model Status
            model = load_model()
            if model is not None:
                st.success("✅ ML Model Active - Providing intelligent recommendations")
            else:
                st.warning("⚠️ ML Model not available - Using rule-based recommendations")
            
            # Current selection analysis
            st.subheader("📊 Your Current Selection Analysis")
            
            if 'providers' in st.session_state:
                providers = st.session_state['providers']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # FIXED: Safely calculate average score
                    valid_scores = []
                    for p in providers:
                        try:
                            score = float(p['match_score'])
                            valid_scores.append(score)
                        except (ValueError, TypeError):
                            continue
                    
                    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                    st.metric(
                        "Average ML Compatibility Score", 
                        f"{avg_score:.1f}/10",
                        help="Higher scores indicate better matches according to our ML model"
                    )
                
                with col2:
                    high_score_providers = 0
                    for p in providers:
                        try:
                            if float(p['match_score']) > 7:
                                high_score_providers += 1
                        except (ValueError, TypeError):
                            continue
                    
                    st.metric(
                        "High-Match Providers", 
                        f"{high_score_providers}/{len(providers)}",
                        help="Providers with ML score > 7.0"
                    )
                
                with col3:
                    eco_providers = len([p for p in providers if p['eco_friendly']])
                    st.metric(
                        "Eco-Friendly Options", 
                        f"{eco_providers}/{len(providers)}",
                        help="Providers offering sustainable services"
                    )
            
            # Budget optimization insights
            st.subheader("💰 Budget Optimization Insights")
            
            if 'ml_upgrade_suggestions' in st.session_state and st.session_state['ml_upgrade_suggestions']:
                # Create a comparison chart
                current_budget = st.session_state.get('budget', '50K-1L')
                current_cost = st.session_state.get('total_cost', 0)
                current_providers = len(st.session_state.get('providers', []))
                
                # FIXED: Calculate current ML satisfaction score safely
                current_ml_score = 0.0
                current_providers_data = st.session_state.get('providers', [])
                if current_providers_data:
                    valid_scores = []
                    for p in current_providers_data:
                        try:
                            score = float(p['match_score'])
                            valid_scores.append(score)
                        except (ValueError, TypeError):
                            continue
                    current_ml_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                
                budget_comparison = [
                    {
                        'Budget Level': budget_ranges.get(current_budget, current_budget),
                        'Total Cost': current_cost,
                        'Providers Available': current_providers,
                        'ML Satisfaction Score': current_ml_score,
                        'Status': 'Current'
                    }
                ]
                
                for suggestion in st.session_state['ml_upgrade_suggestions'][:3]:
                    budget_comparison.append({
                        'Budget Level': suggestion['budget_level'],
                        'Total Cost': suggestion['total_cost'],
                        'Providers Available': suggestion['num_providers'],
                        'ML Satisfaction Score': suggestion['ml_predicted_satisfaction'],
                        'Status': 'Upgrade Option'
                    })
                
                comparison_df = pd.DataFrame(budget_comparison)
                
                # Display comparison table
                st.write("**Budget vs. Value Analysis:**")
                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # ML Satisfaction Score vs Budget
                budget_levels = comparison_df['Budget Level']
                ml_scores = comparison_df['ML Satisfaction Score']
                colors = ['#FF6B6B' if status == 'Current' else '#4ECDC4' for status in comparison_df['Status']]
                
                bars1 = ax1.bar(range(len(budget_levels)), ml_scores, color=colors, alpha=0.7)
                ax1.set_title('ML Satisfaction Score by Budget Level')
                ax1.set_ylabel('ML Score (out of 10)')
                ax1.set_xticks(range(len(budget_levels)))
                ax1.set_xticklabels([bl.split('(')[0].strip() for bl in budget_levels], rotation=45, ha='right')
                ax1.set_ylim(0, 10)
                
                # Add value labels on bars
                for bar, score in zip(bars1, ml_scores):
                    height = bar.get_height()
                    if height > 0:  # Only add label if height is positive
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{height:.1f}', ha='center', va='bottom')
                
                # Cost vs Providers Available
                costs = comparison_df['Total Cost'] / 1000  # Convert to thousands
                providers_count = comparison_df['Providers Available']
                
                ax2.scatter(costs, providers_count, c=colors, s=100, alpha=0.7)
                ax2.set_title('Cost vs Provider Options')
                ax2.set_xlabel('Total Cost (₹ thousands)')
                ax2.set_ylabel('Number of Providers')
                
                # Add labels for each point
                for i, (cost, providers, level) in enumerate(zip(costs, providers_count, budget_levels)):
                    ax2.annotate(level.split('(')[0].strip(), 
                               (cost, providers), 
                               xytext=(5, 5), 
                               textcoords='offset points',
                               fontsize=8)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # ML Prediction Confidence
            st.subheader("🎯 ML Model Confidence Analysis")
            
            if model is not None and 'providers' in st.session_state:
                providers = st.session_state['providers']
                
                if providers:
                    # FIXED: Analyze prediction confidence - safely handle scores
                    valid_scores = []
                    for p in providers:
                        try:
                            score = float(p['match_score'])
                            valid_scores.append(score)
                        except (ValueError, TypeError):
                            continue
                    
                    if valid_scores:  # Only proceed if we have valid scores
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Score distribution
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.hist(valid_scores, bins=5, color='skyblue', alpha=0.7, edgecolor='black')
                            ax.set_title('Distribution of ML Compatibility Scores')
                            ax.set_xlabel('ML Score')
                            ax.set_ylabel('Number of Providers')
                            avg_score = sum(valid_scores) / len(valid_scores)
                            ax.axvline(avg_score, color='red', linestyle='--', label=f'Average: {avg_score:.1f}')
                            ax.legend()
                            st.pyplot(fig)
                        
                        with col2:
                            st.write("**ML Model Insights:**")
                            
                            avg_score = sum(valid_scores) / len(valid_scores)
                            if avg_score > 8:
                                st.success("🎉 Excellent matches found! The ML model is highly confident in these recommendations.")
                            elif avg_score > 6:
                                st.info("👍 Good matches available. Consider the top-rated providers.")
                            else:
                                st.warning("⚠️ Limited optimal matches. Consider adjusting your criteria or budget.")
                            
                            # FIXED: Top recommendation - find provider with highest valid score
                            best_provider = None
                            best_score = -1
                            
                            for provider in providers:
                                try:
                                    score = float(provider['match_score'])
                                    if score > best_score:
                                        best_score = score
                                        best_provider = provider
                                except (ValueError, TypeError):
                                    continue
                            
                            if best_provider:
                                st.write(f"**Top ML Recommendation:** {best_provider['name']}")
                                st.write(f"**Confidence Score:** {best_provider['match_score']:.1f}/10")
                                st.write(f"**Why recommended:** High compatibility with your event type, location, and service requirements.")
                            
                            # Improvement suggestions
                            st.write("**To improve recommendations:**")
                            if st.session_state.get('eco_preference', False):
                                eco_providers = [p for p in providers if p['eco_friendly']]
                                if eco_providers:
                                    st.write("✅ Great choice on eco-friendly preference!")
                                else:
                                    st.write("🌱 Consider expanding location search for eco-friendly options")
                            
                            location_matches = [p for p in providers if p['location'] == st.session_state.get('location')]
                            if len(location_matches) < len(providers) * 0.5:
                                st.write("📍 Consider nearby cities for more options")
                    else:
                        st.warning("No valid ML scores available for analysis.")
            
            # Future predictions
            st.subheader("🔮 Predictive Insights")
            
            # Predict busy seasons
            current_date = st.session_state.get('event_date', datetime.date.today())
            if current_date:
                month = current_date.month
                busy_months = [11, 12, 1, 2, 4, 5]  # Wedding season months
                
                if month in busy_months:
                    st.warning(f"📅 Peak season detected! Events in {calendar.month_name[month]} tend to have:")
                    st.write("• Higher prices (15-25% premium)")
                    st.write("• Limited vendor availability")
                    st.write("• Recommendation: Book 3-4 months in advance")
                else:
                    st.info(f"📅 Off-peak season advantage! Events in {calendar.month_name[month]} typically offer:")
                    st.write("• Better pricing (10-20% savings possible)")
                    st.write("• More vendor options")
                    st.write("• Flexible scheduling")
        else:
            st.info("Please fill in your event details to see ML-powered insights and recommendations.")
    
    # Add enhanced styling
    st.markdown("""
    <style>
    .main .block-container {
        max-width: 95%;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 4px 4px 0px 0px;
    }
    .stExpander {
        border: 1px solid #f0f0f0;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
