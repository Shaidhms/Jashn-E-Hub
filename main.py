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
    '10K-30K': 'Economy (₹10,000-₹30,000)',
    '30K-50K': 'Standard (₹30,000-₹50,000)',
    '50K-1L': 'Premium (₹50,000-₹1,00,000)',
    '1L-2L': 'Luxury (₹1,00,000-₹2,00,000)',  
    '2L+': 'Ultra Luxury (₹2,00,000+)',
    '5L+': 'Elite (₹5,00,000+)'
}

# Budget mapping (for compatibility with existing logic)
budget_category_mapping = {
    '10K-30K': 'low',
    '30K-50K': 'medium',
    '50K-1L': 'medium', 
    '1L-2L': 'medium+',
    '2L+': 'high',
    '5L+': 'high'
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

# Sustainable alternatives for services
sustainable_alternatives = {
    'catering': [
        'Use locally sourced, seasonal ingredients to reduce carbon footprint',
        'Offer plant-based menu options which require less resources to produce',
        'Use reusable or biodegradable serving ware instead of plastic',
        'Plan precise quantities to minimize food waste'
    ],
    'venue': [
        'Choose venues with natural lighting to reduce electricity usage',
        'Select venues that have sustainability certifications',
        'Opt for venues that use renewable energy sources',
        'Consider outdoor locations during appropriate seasons'
    ],
    'decoration': [
        'Use potted plants instead of cut flowers (can be given as gifts later)',
        'Rent decorations instead of buying new ones',
        'Use LED lights instead of traditional bulbs for lighting',
        'Choose natural, biodegradable materials for decor'
    ],
    'transportation': [
        'Arrange shared transportation to reduce individual carbon footprints',
        'Select electric or hybrid vehicles where available',
        'Choose venues that are centrally located to minimize travel distance',
        'Offer virtual attendance options for distant guests'
    ],
    'wedding cards': [
        'Use digital invitations to eliminate paper waste',
        'If physical cards are needed, choose recycled paper options',
        'Select cards embedded with seeds that can be planted after use',
        'Use soy-based inks instead of petroleum-based inks'
    ]
}

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

# Function to parse the services from the original data file
def parse_services_from_data():
    # Extract the services from the paste.txt data
    services_data = []
    for line in st.session_state.get('raw_data', []):
        # Clean the string and extract services
        cleaned = line.replace("'", "").strip()
        services = [s.strip() for s in cleaned.split(',')]
        services_data.append(services)
    return services_data

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

# Fixed cost breakdown function
def calculate_cost_breakdown(selected_services, budget):
    cost_breakdown = {}
    total_cost = 0
    
    for service in selected_services:
        service_cost = service_costs.get(service, {}).get(budget, 0)
        cost_breakdown[service] = service_cost
        total_cost += service_cost
    
    return cost_breakdown, total_cost

# Function to get upgrade suggestions
def get_upgrade_suggestions(providers, budget, selected_services):
    # Define the budget levels in order from lowest to highest
    budget_levels = ['10K-30K', '30K-50K', '50K-1L', '1L-2L', '2L+', '5L+']
    
    try:
        current_index = budget_levels.index(budget)
    except ValueError:
        # If budget not found in the list, default to the first one
        current_index = 0
    
    suggestions = []
    
    # If not already at the highest budget level
    if current_index < len(budget_levels) - 1:
        next_budget = budget_levels[current_index + 1]
        
        # Calculate additional cost
        current_cost = sum(service_costs.get(service, {}).get(budget, 0) for service in selected_services)
        upgraded_cost = sum(service_costs.get(service, {}).get(next_budget, 0) for service in selected_services)
        
        additional_cost = upgraded_cost - current_cost
        
        # Find upgraded providers
        upgraded_providers = [p for p in providers if p['budget'] == next_budget]
        
        if upgraded_providers:
            top_provider = upgraded_providers[0]
            suggestions.append({
                'budget_level': budget_ranges[next_budget],
                'additional_cost': additional_cost,
                'benefits': f"Access to {len(upgraded_providers)} premium vendors like {top_provider['name']} (rated {top_provider['rating']})",
                'service_improvements': []
            })
            
            # Add specific service improvements
            for service in selected_services:
                current_service_cost = service_costs.get(service, {}).get(budget, 0)
                upgraded_service_cost = service_costs.get(service, {}).get(next_budget, 0)
                
                if upgraded_service_cost > current_service_cost:
                    improvement = upgraded_service_cost - current_service_cost
                    suggestions[0]['service_improvements'].append({
                        'service': service,
                        'improvement': improvement,
                        'details': f"Upgrade {service} for ₹{improvement:,} more"
                    })
    
    return suggestions

# Add raw data to session state (simulating the input from paste.txt)
if 'raw_data' not in st.session_state:
    raw_data = [
        "wedding cards, entertainment, decoration",
        "entertainment, photography, catering, venue",
        "catering, venue, entertainment, transportation",
        # Add more entries as needed
    ]
    st.session_state['raw_data'] = raw_data

# Main function to create the app
def main():
    st.title("Jashn-E-Hub")
    st.markdown("### Find the perfect service providers for your event!")
    
    # Initialize model in session state
    if 'model' not in st.session_state:
        st.session_state['model'] = load_model()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Plan Your Event", "Service Providers", "Cost Breakdown", "Sustainability"])
    
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
        if st.button("Find Providers"):
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
                
                # Get upgrade suggestions
                upgrade_suggestions = get_upgrade_suggestions(providers, selected_budget, selected_services)
                st.session_state['upgrade_suggestions'] = upgrade_suggestions
                
                st.success("Event details saved! Check the Service Providers and Cost Breakdown tabs for more information.")
    
    with tab2:
        st.header("Recommended Service Providers")
        
        if 'providers' in st.session_state:
            providers = st.session_state['providers']
            
            # Show upgrade suggestions if available
            if 'upgrade_suggestions' in st.session_state and st.session_state['upgrade_suggestions']:
                with st.expander("Upgrade your experience!", expanded=True):
                    suggestion = st.session_state['upgrade_suggestions'][0]
                    st.write(f"**Upgrade to {suggestion['budget_level']} for just ₹{suggestion['additional_cost']:,} more!**")
                    st.write(suggestion['benefits'])
                    st.write("**What you'll get:**")
                    for imp in suggestion['service_improvements']:
                        st.write(f"- {imp['details']}")
                    
                    if st.button("Upgrade Budget"):
                        # Get the next budget level
                        budget_levels = ['10K-30K', '30K-50K', '50K-1L', '1L-2L', '2L+', '5L+']
                        current_index = budget_levels.index(st.session_state['budget'])
                        if current_index < len(budget_levels) - 1:
                            new_budget = budget_levels[current_index + 1]
                            new_budget_display = budget_ranges[new_budget]
                            
                            st.session_state['budget'] = new_budget
                            st.session_state['budget_display'] = new_budget_display
                            
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
                            
                            upgrade_suggestions = get_upgrade_suggestions(
                                providers, new_budget, st.session_state['selected_services']
                            )
                            st.session_state['upgrade_suggestions'] = upgrade_suggestions
                            
                            st.success(f"Budget upgraded to {new_budget_display}! The service providers list has been updated.")
                            st.rerun()
            
            if not providers:
                st.warning("No service providers found for your criteria. Try adjusting your requirements.")
            else:
                st.write(f"Found {len(providers)} providers matching your criteria:")
                
                # Group providers by budget category
                budget_categories = sorted(set(p['budget'] for p in providers))
                
                for budget_category in budget_categories:
                    budget_providers = [p for p in providers if p['budget'] == budget_category]
                    
                    # Sort by rating within category
                    budget_providers.sort(key=lambda x: x['rating'], reverse=True)
                    
                    # Display budget category header
                    display_budget = next((v for k, v in budget_ranges.items() if k == budget_category), budget_category)
                    st.subheader(f"{display_budget} Providers")
                    
                    for i, provider in enumerate(budget_providers):
                        provider_key = f"{provider['name']}_{budget_category}"
                        with st.expander(f"{provider['name']} - Rating: {provider['rating']} {'🌱' if provider['eco_friendly'] else ''}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Location:** {provider['location']}")
                                st.write(f"**Budget Category:** {display_budget}")
                                st.write(f"**Services Offered:** {', '.join(provider['services'])}")
                                if provider['eco_friendly']:
                                    st.write("**Eco-Friendly:** Yes 🌱")
                                
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
                                
                                # Add a check to show reviews
                                if st.checkbox(f"Show reviews for {provider['name']}", key=f"reviews_{provider_key}"):
                                    # In a real app, this would come from a database
                                    sample_reviews = [
                                        {"user": "Priya S.", "rating": 5, "text": "Amazing service! Very professional and delivered beyond expectations."},
                                        {"user": "Rahul K.", "rating": 4, "text": "Good service overall. Minor delays but the quality was great."},
                                        {"user": "Simran M.", "rating": 5, "text": "Outstanding! Made our event truly special."}
                                    ]
                                    
                                    st.write("**Client Reviews:**")
                                    for review in sample_reviews:
                                        st.markdown(f"**{review['user']}** - {'⭐' * review['rating']}")
                                        st.markdown(f"> {review['text']}")
        else:
            st.info("Please fill in your event details in the Plan Your Event tab to see recommended providers.")
    
    with tab3:
        st.header("Cost Breakdown")
        
        if 'cost_breakdown' in st.session_state:
            cost_breakdown = st.session_state['cost_breakdown']
            total_cost = st.session_state['total_cost']
            
            st.write(f"**Budget Category:** {st.session_state.get('budget_display', 'Unknown')}")
            st.write(f"**Total Estimated Cost:** ₹{total_cost:,}")
            
            # Create columns for services and costs
            services = list(cost_breakdown.keys())
            costs = list(cost_breakdown.values())
            
            # Create more appealing and smaller charts
            if services and costs:
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Create a more attractive bar chart
                    fig, ax = plt.subplots(figsize=(7, 4))
                    
                    # Use a color gradient for bars
                    colormap = plt.cm.viridis
                    colors = colormap(np.linspace(0, 0.8, len(services)))
                    
                    bars = ax.bar(services, costs, color=colors, edgecolor='white', linewidth=1)
                    
                    # Add value labels on top of the bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                f'₹{int(height):,}',
                                ha='center', va='bottom', rotation=0, fontsize=9)
                    
                    # Enhance the chart appearance
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
                    # Create a more attractive pie chart
                    fig, ax = plt.subplots(figsize=(5, 5))
                    
                    # Use a consistent color scheme and better styling
                    colors = plt.cm.tab10(np.linspace(0, 1, len(services)))
                    wedges, texts, autotexts = ax.pie(
                        costs, 
                        labels=None,  # We'll add a legend instead
                        autopct='%1.1f%%', 
                        startangle=90, 
                        colors=colors,
                        wedgeprops={'width': 0.6, 'edgecolor': 'w', 'linewidth': 1},
                        textprops={'fontsize': 9}
                    )
                    
                    # Improve the appearance of percentage labels
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                    plt.title('Cost Distribution', fontsize=12)
                    
                    # Add a legend for clearer labeling
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
                
                # Detailed breakdown table
                st.subheader("Detailed Cost Breakdown")
                
                # Create a table for cost breakdown
                cost_data = pd.DataFrame({
                    'Service': services,
                    'Cost (₹)': [f'₹{cost:,}' for cost in costs],
                    'Percentage': [f'{(cost/total_cost)*100:.1f}%' for cost in costs]
                })
                
                st.table(cost_data)
                
                # Payment schedule suggestion
                st.subheader("Suggested Payment Schedule")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Booking Amount (25%)", f"₹{total_cost * 0.25:,.0f}")
                    st.write("Due: Immediately")
                
                with col2:
                    st.metric("Midway Payment (50%)", f"₹{total_cost * 0.5:,.0f}")
                    # Calculate midway date (halfway between today and event)
                    if 'event_date' in st.session_state:
                        today = datetime.date.today()
                        event_date = st.session_state['event_date']
                        days_diff = (event_date - today).days
                        midway_date = today + datetime.timedelta(days=days_diff // 2)
                        st.write(f"Due: {midway_date.strftime('%d %b %Y')}")
                    else:
                        st.write("Due: 1 month before event")
                
                with col3:
                    st.metric("Final Payment (25%)", f"₹{total_cost * 0.25:,.0f}")
                    if 'event_date' in st.session_state:
                        event_date = st.session_state['event_date']
                        before_event = event_date - datetime.timedelta(days=7)
                        st.write(f"Due: {before_event.strftime('%d %b %Y')}")
                    else:
                        st.write("Due: 1 week before event")
                
                # Add payment method options with better UI
                st.subheader("Payment Methods")
                payment_col1, payment_col2 = st.columns(2)
                
                with payment_col1:
                    st.markdown("**Online Payment Options:**")
                    payment_methods = {
                        "Credit Card": st.checkbox("Credit Card", value=True),
                        "Debit Card": st.checkbox("Debit Card", value=True),
                        "UPI (Google Pay/PhonePe/Paytm)": st.checkbox("UPI (Google Pay/PhonePe/Paytm)", value=True),
                        "Net Banking": st.checkbox("Net Banking", value=True),
                        "EMI (3/6/12 months available)": st.checkbox("EMI (3/6/12 months available)", value=False)
                    }
                
                with payment_col2:
                    st.markdown("**Offline Payment Options:**")
                    payment_methods.update({
                        "Bank Transfer": st.checkbox("Bank Transfer", value=True),
                        "Cash": st.checkbox("Cash", value=False),
                        "Cheque": st.checkbox("Cheque", value=True)
                    })
                
                # Payment terms
                st.markdown("**Payment Terms:**")
                st.markdown("- Booking amount must be paid to confirm reservation")
                st.markdown("- All payments are non-refundable after 7 days of booking")
                st.markdown("- GST @18% applicable on all services")
                
                # Create a container for the payment animation
                payment_animation_container = st.empty()
                
                # Create a payment button with a prominent style
                st.markdown("### ")  # Add space
                payment_button_col1, payment_button_col2, payment_button_col3 = st.columns([1, 2, 1])
                with payment_button_col2:
                    payment_button_clicked = st.button(
                        "Make Payment Now", 
                        type="primary",
                        key="payment_button",
                    )
                
                # Payment processing logic
                if payment_button_clicked:
                    selected_methods = [method for method, selected in payment_methods.items() if selected]
                    
                    if not selected_methods:
                        st.error("Please select at least one payment method")
                    else:
                        # Show processing animation
                        with st.spinner("Processing payment..."):
                            import time
                            time.sleep(2)  # Simulate payment processing
                        
                        # Success message with celebration animation
                        st.success("Payment successful! Booking confirmed!")
                        
                        # Display celebration animation using HTML/CSS
                        celebration_html = """
                        <style>
                        /* Confetti animation */
                        .confetti {
                          position: fixed;
                          width: 10px;
                          height: 10px;
                          background-color: #f2d74e;
                          opacity: 0;
                          animation: confetti 5s ease-in-out infinite;
                        }
                        
                        .confetti.red {
                          background-color: #ef3c58;
                        }
                        
                        .confetti.blue {
                          background-color: #3c9aef;
                        }
                        
                        .confetti.green {
                          background-color: #44cf6c;
                        }
                        
                        @keyframes confetti {
                          0% {
                            transform: translateY(0) rotate(0deg);
                            opacity: 0;
                          }
                          10% {
                            opacity: 1;
                          }
                          100% {
                            transform: translateY(100vh) rotate(720deg);
                            opacity: 0;
                          }
                        }
                        
                        /* Balloons animation */
                        .balloon {
                          position: fixed;
                          bottom: -100px;
                          width: 40px;
                          height: 65px;
                          background-color: #ff5252;
                          border-radius: 50%;
                          animation: float 10s ease-in-out infinite;
                        }
                        
                        .balloon:before {
                          content: "";
                          position: absolute;
                          bottom: -10px;
                          left: 20px;
                          height: 20px;
                          width: 1px;
                          background-color: #ff5252;
                        }
                        
                        .balloon.blue {
                          background-color: #448aff;
                          animation-delay: 0.5s;
                          left: 40%;
                        }
                        
                        .balloon.green {
                          background-color: #4caf50;
                          animation-delay: 1s;
                          left: 60%;
                        }
                        
                        .balloon.yellow {
                          background-color: #ffd740;
                          animation-delay: 1.5s;
                          left: 80%;
                        }
                        
                        .balloon.red {
                          left: 20%;
                        }
                        
                        @keyframes float {
                          0% {
                            transform: translateY(0) rotate(5deg);
                            opacity: 0;
                          }
                          10% {
                            opacity: 1;
                          }
                          100% {
                            transform: translateY(-100vh) rotate(-5deg);
                            opacity: 0;
                          }
                        }
                        </style>
                        
                        <!-- Confetti elements -->
                        <div class="confetti" style="left:10%; animation-delay: 0s;"></div>
                        <div class="confetti red" style="left:20%; animation-delay: 0.2s;"></div>
                        <div class="confetti blue" style="left:30%; animation-delay: 0.4s;"></div>
                        <div class="confetti green" style="left:40%; animation-delay: 0.6s;"></div>
                        <div class="confetti" style="left:50%; animation-delay: 0.8s;"></div>
                        <div class="confetti red" style="left:60%; animation-delay: 1s;"></div>
                        <div class="confetti blue" style="left:70%; animation-delay: 1.2s;"></div>
                        <div class="confetti green" style="left:80%; animation-delay: 1.4s;"></div>
                        <div class="confetti" style="left:90%; animation-delay: 1.6s;"></div>
                        
                        <!-- Balloon elements -->
                        <div class="balloon red"></div>
                        <div class="balloon blue"></div>
                        <div class="balloon green"></div>
                        <div class="balloon yellow"></div>
                        
                        <div style="text-align: center; margin-top: 30px; font-size: 24px; font-weight: bold; color: #3c9aef;">
                            🎉 Congratulations! Your booking is confirmed! 🎉
                        </div>
                        """
                        
                        payment_animation_container.markdown(celebration_html, unsafe_allow_html=True)
                        
                        # Show booking details
                        st.markdown("### Booking Summary")
                        st.markdown(f"**Event Type:** {st.session_state.get('event_type', 'Unknown')}")
                        st.markdown(f"**Event Date:** {st.session_state.get('event_date', 'Unknown')}")
                        st.markdown(f"**Location:** {st.session_state.get('location', 'Unknown')}")
                        st.markdown(f"**Total Amount Paid:** ₹{st.session_state.get('total_cost', 0):,}")
                        st.markdown(f"**Payment Method:** {', '.join(selected_methods)}")
                        
                        # Display next steps
                        st.markdown("### Next Steps")
                        st.markdown("1. You will receive a confirmation email shortly")
                        st.markdown("2. Our team will contact you within 24 hours")
                        st.markdown("3. A detailed receipt will be sent to your registered email")
                        
                        # Add download invoice button
                        st.download_button(
                            label="Download Invoice",
                            data="This is a placeholder for the actual invoice PDF data",
                            file_name=f"EventEase_Invoice_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
        else:
            st.info("Please fill in your event details in the Plan Your Event tab to see cost breakdown.")
    
    with tab4:
        st.header("Sustainability Options")
        
        if 'selected_services' in st.session_state:
            selected_services = st.session_state['selected_services']
            
            st.markdown("""
            ### Make Your Event Eco-Friendly
            
            Small choices can make a big difference to the environmental impact of your event.
            Here are some sustainable alternatives for your selected services:
            """)
            
            for service in selected_services:
                if service in sustainable_alternatives:
                    with st.expander(f"Eco-friendly options for {service.title()}", expanded=True):
                        alternatives = sustainable_alternatives[service]
                        for i, alt in enumerate(alternatives):
                            st.markdown(f"- {alt}")
                            
                        # Find eco-friendly providers for this service
                        eco_providers = [p for p in service_providers.items() 
                                       if service in p[1]['services'] and p[1]['eco_friendly']]
                        
                        if eco_providers:
                            st.write("**Eco-friendly providers available:**")
                            for name, info in eco_providers:
                                st.write(f"- {name} (Rating: {info['rating']})")
            
            # Calculate carbon footprint based on event choices
            st.subheader("Estimated Carbon Footprint")
            
            # Calculate baseline footprint based on guests
            num_guests = st.session_state.get('num_guests', 100)
            baseline_footprint = num_guests * 7.5  # kg CO2 per guest for a standard event
            
            # Adjustments based on services
            service_footprints = {
                'catering': num_guests * 3.0,  # Food production and waste
                'venue': 250 + (num_guests * 0.5),  # Venue energy use
                'decoration': 100 + (num_guests * 0.2),  # Production of decorations
                'transportation': num_guests * 2.5,  # Guest transportation
                'photography': 20,  # Equipment and editing energy
                'entertainment': 50 + (num_guests * 0.1),  # Equipment energy use
                'wedding cards': num_guests * 0.1,  # Paper production
                'priest': 10  # Minimal impact
            }
            
            total_footprint = sum(service_footprints.get(service, 0) for service in selected_services)
            
            # Eco-preference reduction
            if st.session_state.get('eco_preference', False):
                total_footprint *= 0.7  # 30% reduction with eco-friendly choices
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Carbon Footprint", f"{total_footprint:.1f} kg CO2e")
                st.write(f"Equivalent to driving approximately {(total_footprint/0.12):.1f} km in an average car")
                
                if st.session_state.get('eco_preference', False):
                    st.success(f"By choosing eco-friendly options, you're saving approximately {(total_footprint * 0.3):.1f} kg CO2e!")
                else:
                    st.info(f"By choosing eco-friendly options, you could save approximately {(total_footprint * 0.3):.1f} kg CO2e.")
            
            with col2:
                # Create a gauge chart for carbon footprint - make it more appealing and smaller
                fig = plt.figure(figsize=(3.5, 3.5))
                ax = fig.add_subplot(111, polar=True)
                
                # Determine footprint level (Low, Medium, High)
                if total_footprint < num_guests * 5:  # Less than 5kg per guest
                    footprint_level = "Low"
                    color = "#4CAF50"  # A nicer green
                    theta = 0.25 * 2 * np.pi
                    percentage = 25
                elif total_footprint < num_guests * 10:  # Less than 10kg per guest
                    footprint_level = "Medium"
                    color = "#FF9800"  # A nicer orange
                    theta = 0.5 * 2 * np.pi
                    percentage = 50
                else:
                    footprint_level = "High"
                    color = "#F44336"  # A nicer red
                    theta = 0.75 * 2 * np.pi
                    percentage = 75
                
                # Plot the gauge with better styling
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add a background arc
                ax.barh(0, 2 * np.pi, color="#EEEEEE", alpha=0.5, linewidth=0)
                
                # Add the colored arc
                arc = ax.barh(0, theta, color=color, alpha=0.8, linewidth=0)
                
                # Add a nice center circle
                center_circle = plt.Circle((0, 0), 0.5, color='white')
                ax.add_artist(center_circle)
                
                # Add text in the center
                ax.text(0, 0, f"{percentage}%", ha='center', va='center', fontsize=14, fontweight='bold')
                ax.text(0, -0.2, footprint_level, ha='center', va='center', fontsize=12)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add recommendations to reduce footprint
                st.write("**Recommendations to reduce footprint:**")
                if 'catering' in selected_services:
                    st.write("- Consider a plant-based menu option")
                if 'transportation' in selected_services:
                    st.write("- Arrange shared transportation for guests")
                if 'venue' in selected_services:
                    st.write("- Choose a venue with natural lighting")
                if 'decoration' in selected_services:
                    st.write("- Opt for reusable or biodegradable decorations")
                
                # Add carbon offset options
                offset_cost = total_footprint * 0.02  # Approximately ₹2 per kg CO2
                st.write(f"**Offset your carbon footprint for approximately ₹{offset_cost:.0f}**")
                
                if st.button("Add Carbon Offset to Event"):
                    st.session_state['carbon_offset'] = True
                    st.session_state['carbon_offset_cost'] = offset_cost
                    st.success(f"Carbon offset added to your event planning! ₹{offset_cost:.0f} will be contributed to reforestation projects.")
        else:
            st.info("Please fill in your event details in the Plan Your Event tab to see sustainability options.")
    
    # Add some styling
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
    </style>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()