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

# Set page configuration
st.set_page_config(
    page_title="EventEase Planner",
    layout="wide"
)

# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
USER_DATA_FILE = DATA_DIR / "user_data.csv"

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
    model_path = "C:\\Users\\ayush\\OneDrive\\Desktop\\EventEase-main\\artifacts\\model.joblib"
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Sample data for demonstration
service_providers = {
    'Arjun': {'services': ['catering', 'decoration'], 'rating': 4.7, 'location': 'Mumbai', 'budget_category': 'medium', 'eco_friendly': True},
    'Saanvi': {'services': ['catering', 'venue'], 'rating': 4.9, 'location': 'Delhi', 'budget_category': 'high', 'eco_friendly': True},
    'Fiona': {'services': ['photography', 'venue'], 'rating': 4.5, 'location': 'Bangalore', 'budget_category': 'high', 'eco_friendly': False},
    'Diya': {'services': ['decoration', 'entertainment'], 'rating': 4.8, 'location': 'Chennai', 'budget_category': 'medium', 'eco_friendly': True},
    'Priya': {'services': ['catering', 'venue', 'decoration'], 'rating': 4.6, 'location': 'Pune', 'budget_category': 'medium', 'eco_friendly': False},
    'Rahul': {'services': ['photography', 'entertainment'], 'rating': 4.3, 'location': 'Mumbai', 'budget_category': 'low', 'eco_friendly': False},
    'Aarav': {'services': ['transportation', 'venue'], 'rating': 4.4, 'location': 'Delhi', 'budget_category': 'low', 'eco_friendly': True},
    'Kabir': {'services': ['wedding cards', 'decoration'], 'rating': 4.5, 'location': 'Chennai', 'budget_category': 'low', 'eco_friendly': True},
    'George': {'services': ['priest', 'catering'], 'rating': 4.7, 'location': 'Bangalore', 'budget_category': 'medium', 'eco_friendly': False},
    'Hannah': {'services': ['entertainment', 'photography'], 'rating': 4.8, 'location': 'Pune', 'budget_category': 'high', 'eco_friendly': True},
    'Ananya': {'services': ['venue', 'catering', 'decoration'], 'rating': 4.9, 'location': 'Mumbai', 'budget_category': 'high', 'eco_friendly': True},
    'Isha': {'services': ['wedding cards', 'transportation'], 'rating': 4.2, 'location': 'Delhi', 'budget_category': 'low', 'eco_friendly': True},
    'Vivaan': {'services': ['priest', 'decoration', 'catering'], 'rating': 4.6, 'location': 'Bangalore', 'budget_category': 'medium', 'eco_friendly': True},
    'Meera': {'services': ['catering', 'decoration'], 'rating': 4.5, 'location': 'Mumbai', 'budget_category': 'medium+', 'eco_friendly': True},
    'Rohan': {'services': ['venue', 'decoration'], 'rating': 4.6, 'location': 'Delhi', 'budget_category': 'medium+', 'eco_friendly': True},
    'Tara': {'services': ['photography', 'entertainment'], 'rating': 4.7, 'location': 'Bangalore', 'budget_category': 'medium+', 'eco_friendly': True},
    'Vikram': {'services': ['catering', 'transportation'], 'rating': 4.8, 'location': 'Chennai', 'budget_category': 'medium+', 'eco_friendly': False},
    'Neha': {'services': ['venue', 'decoration', 'photography'], 'rating': 4.9, 'location': 'Pune', 'budget_category': 'medium+', 'eco_friendly': True},
}

# Define service costs by budget category
service_costs = {
    'catering': {'low': 15000, 'medium': 30000, 'medium+': 40000, 'high': 60000},
    'venue': {'low': 30000, 'medium': 75000, 'medium+': 100000, 'high': 150000},
    'decoration': {'low': 10000, 'medium': 25000, 'medium+': 35000, 'high': 50000},
    'photography': {'low': 15000, 'medium': 35000, 'medium+': 50000, 'high': 70000},
    'entertainment': {'low': 10000, 'medium': 25000, 'medium+': 35000, 'high': 50000},
    'transportation': {'low': 5000, 'medium': 15000, 'medium+': 20000, 'high': 30000},
    'wedding cards': {'low': 5000, 'medium': 10000, 'medium+': 15000, 'high': 20000},
    'priest': {'low': 5000, 'medium': 10000, 'medium+': 12000, 'high': 15000}
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
    for week in cal:
        for day in week:
            if day != 0:  # Skip days that belong to other months
                current_date = datetime.date(year, month, day)
                if current_date >= datetime.date.today():  # Only future dates
                    if random.random() > 0.4:  # 60% chance of being available
                        available_dates.append(day)
    
    return cal, available_dates

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

# Function to predict vendor suitability using ML model
def predict_vendor_suitability(model, event_type, budget, location, services, provider):
    if model is None:
        # Fallback to rule-based matching if model isn't loaded
        return get_provider_match_score(event_type, budget, location, services, provider)
    
    # Function to predict vendor suitability using ML model
# Function to predict vendor suitability using ML model
def predict_vendor_suitability(model, event_type, budget, location, services, provider):
    if model is None:
        # Fallback to rule-based matching if model isn't loaded
        return get_provider_match_score(event_type, budget, location, services, provider)
    
    try:
        # Include the three additional required columns that were missing
        features = {
            'event_type': event_type,
            'budget': budget.lower(),
            'location': location,
            'service_match': 0,          # dummy
            'provider_rating': provider['rating'],  # Actual rating from provider
            'eco_friendly': 1 if provider['eco_friendly'] else 0,  # Convert boolean to integer
            'rating': provider['rating'],  # The model specifically requires this column
            'registration_number': 12345,  # Placeholder value since we don't have this in provider data
            'service': services[0] if services else 'general'  # Use the first selected service as a placeholder
        }

        
        # Convert to appropriate format for your model - make sure all required columns are included
        df = pd.DataFrame([features], columns=[
            'event_type',
            'budget',
            'location',
            'service_match',
            'provider_rating',
            'eco_friendly',
            'rating',
            'registration_number',
            'service'
        ])
        
        # Predict suitability
        suitability_score = model.predict(df)[0]
        return suitability_score
    except Exception as e:
        st.error(f"Error using model for prediction: {e}")
        # Fallback to rule-based matching
        return get_provider_match_score(event_type, budget, location, services, provider)

# Function to calculate match score based on rules
def get_provider_match_score(event_type, budget, location, services, provider):
    score = 0
    
    # Location match
    if provider['location'] == location:
        score += 3
    
    # Budget match
    if provider['budget_category'] == budget.lower():
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

# Function to get providers based on services, budget, and location
def get_providers_recommendation(event_type, budget, location, selected_services):
    # For actual implementation, this would use the ML model
    # For now, we'll filter based on matching criteria
    
    matching_providers = []
    
    budget_category = budget.lower()
    
    # Get all providers that have at least one of the selected services
    for name, info in service_providers.items():
        if any(service in info['services'] for service in selected_services):
            # Calculate match score for sorting
            if model:
                match_score = predict_vendor_suitability(model, event_type, budget, location, selected_services, info)
            else:
                match_score = get_provider_match_score(event_type, budget, location, selected_services, info)
            
            matching_providers.append({
                'name': name,
                'services': info['services'],
                'rating': info['rating'],
                'location': info['location'],
                'budget': info['budget_category'],
                'eco_friendly': info['eco_friendly'],
                'match_score': match_score
            })
    
    # Sort by match score
    matching_providers.sort(key=lambda x: x['match_score'], reverse=True)
    return matching_providers

# Function to calculate cost breakdown
def calculate_cost_breakdown(selected_services, budget):
    cost_breakdown = {}
    total_cost = 0
    
    for service in selected_services:
        service_cost = service_costs.get(service, {}).get(budget.lower(), 0)
        cost_breakdown[service] = service_cost
        total_cost += service_cost
    
    return cost_breakdown, total_cost

# Function to get upgrade suggestions
def get_upgrade_suggestions(providers, budget, selected_services):
    current_budget = budget.lower()
    
    # Define the next budget level
    budget_levels = ['low', 'medium', 'medium+', 'high']
    current_index = budget_levels.index(current_budget)
    
    suggestions = []
    
    # If not already at the highest budget level
    if current_index < len(budget_levels) - 1:
        next_budget = budget_levels[current_index + 1]
        
        # Calculate additional cost
        current_cost = sum(service_costs.get(service, {}).get(current_budget, 0) for service in selected_services)
        upgraded_cost = sum(service_costs.get(service, {}).get(next_budget, 0) for service in selected_services)
        
        additional_cost = upgraded_cost - current_cost
        
        # Find upgraded providers
        upgraded_providers = [p for p in providers if p['budget'] == next_budget]
        
        if upgraded_providers:
            top_provider = upgraded_providers[0]
            suggestions.append({
                'budget_level': next_budget.capitalize(),
                'additional_cost': additional_cost,
                'benefits': f"Access to {len(upgraded_providers)} premium vendors like {top_provider['name']} (rated {top_provider['rating']})",
                'service_improvements': []
            })
            
            # Add specific service improvements
            for service in selected_services:
                current_service_cost = service_costs.get(service, {}).get(current_budget, 0)
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
            with col2:
                user_phone = st.text_input("Phone Number")
                user_city = st.text_input("City of Residence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Event type selection
            event_type = st.selectbox(
                "Select Event Type",
                ['wedding', 'engagement', 'corporate', 'house party', 'birthday']
            )
            
            # Budget selection
            budget = st.selectbox(
                "Select Budget Category",
                ['Low', 'Medium', 'Medium+', 'High']
            )
            
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
            if not user_name or not user_phone:
                st.error("Please provide your name and phone number to proceed.")
            else:
                # Save user data to CSV
                user_data = {
                    'name': user_name,
                    'email': user_email,
                    'phone': user_phone,
                    'city': user_city,
                    'event_type': event_type,
                    'budget': budget,
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
                st.session_state['budget'] = budget
                st.session_state['location'] = location
                st.session_state['event_date'] = event_date
                st.session_state['selected_services'] = selected_services
                st.session_state['num_guests'] = num_guests
                st.session_state['eco_preference'] = eco_preference
                
                # Calculate cost breakdown
                cost_breakdown, total_cost = calculate_cost_breakdown(selected_services, budget)
                st.session_state['cost_breakdown'] = cost_breakdown
                st.session_state['total_cost'] = total_cost
                
                # Get provider recommendations
                providers = get_providers_recommendation(event_type, budget, location, selected_services)
                st.session_state['providers'] = providers
                
                # Get upgrade suggestions
                upgrade_suggestions = get_upgrade_suggestions(providers, budget, selected_services)
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
                    st.write(f"**Upgrade to {suggestion['budget_level']} budget for just ₹{suggestion['additional_cost']:,} more!**")
                    st.write(suggestion['benefits'])
                    st.write("**What you'll get:**")
                    for imp in suggestion['service_improvements']:
                        st.write(f"- {imp['details']}")
                    
                    if st.button("Upgrade Budget"):
                        # Update the budget and recalculate everything
                        next_budget_index = ['low', 'medium', 'medium+', 'high'].index(st.session_state['budget'].lower()) + 1
                        new_budget = ['Low', 'Medium', 'Medium+', 'High'][next_budget_index]
                        
                        st.session_state['budget'] = new_budget
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
                        
                        st.success(f"Budget upgraded to {new_budget}! The service providers list has been updated.")
                        st.rerun()
            
            if not providers:
                st.warning("No service providers found for your criteria. Try adjusting your requirements.")
            else:
                st.write(f"Found {len(providers)} providers matching your criteria:")
                
                # First sort by budget category to group providers
                budget_order = {'low': 0, 'medium': 1, 'medium+': 2, 'high': 3}
                providers.sort(key=lambda x: (budget_order.get(x['budget'], 0), -x['rating']))
                
                current_budget = None
                
                for i, provider in enumerate(providers):
                    # Add budget category header if changed
                    if provider['budget'] != current_budget:
                        current_budget = provider['budget']
                        st.subheader(f"{current_budget.capitalize()} Budget Providers")
                    
                    with st.expander(f"{provider['name']} - Rating: {provider['rating']} {'🌱' if provider['eco_friendly'] else ''}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Location:** {provider['location']}")
                            st.write(f"**Budget Category:** {provider['budget'].capitalize()}")
                            st.write(f"**Services Offered:** {', '.join(provider['services'])}")
                            if provider['eco_friendly']:
                                st.write("**Eco-Friendly:** Yes 🌱")
                            
                            if st.button(f"Contact {provider['name']}", key=f"contact_{i}"):
                                # Save contact request to CSV
                                contact_data = {
                                    'name': st.session_state.get('user_name', 'Unknown'),
                                    'email': st.session_state.get('user_email', 'Unknown'),
                                    'phone': st.session_state.get('user_phone', 'Unknown'),
                                    'city': st.session_state.get('user_city', 'Unknown'),
                                    'event_type': st.session_state.get('event_type', 'Unknown'),
                                    'budget': st.session_state.get('budget', 'Unknown'),
                                    'location': st.session_state.get('location', 'Unknown'),
                                    'event_date': st.session_state.get('event_date', datetime.date.today()).strftime('%Y-%m-%d'),
                                    'selected_services': ', '.join(st.session_state.get('selected_services', [])),
                                    'num_guests': st.session_state.get('num_guests', 0),
                                    'eco_preference': st.session_state.get('eco_preference', False),
                                    'contacted_provider': provider['name']
                                }
                                
                                # Create contacts directory if it doesn't exist
                                CONTACTS_DIR = DATA_DIR / "contacts"
                                CONTACTS_DIR.mkdir(exist_ok=True)
                                CONTACTS_FILE = CONTACTS_DIR / f"contact_requests_{datetime.date.today().strftime('%Y%m%d')}.csv"
                                
                                contact_file_exists = os.path.isfile(CONTACTS_FILE)
                                with open(CONTACTS_FILE, mode='a', newline='') as file:
                                    writer = csv.DictWriter(file, fieldnames=contact_data.keys())
                                    if not contact_file_exists:
                                        writer.writeheader()
                                    writer.writerow(contact_data)
                                
                                st.info(f"Request sent to {provider['name']}. They will contact you soon!")
                        
                        with col2:
                            # Show live calendar
                            st.write("**Availability Calendar:**")
                            
                            # Let user select month/year for the calendar
                            today = datetime.date.today()
                            months = list(range(1, 13))
                            month_names = [calendar.month_name[m] for m in months]
                            
                            col_month, col_year = st.columns(2)
                            with col_month:
                                selected_month_name = st.selectbox(
                                    "Month", 
                                    month_names[today.month-1:] + month_names[:today.month-1],
                                    key=f"month_{i}"
                                )
                                selected_month = month_names.index(selected_month_name) + 1
                            
                            with col_year:
                                selected_year = st.selectbox(
                                    "Year", 
                                    [today.year, today.year + 1],
                                    key=f"year_{i}"
                                )
                            
                            # Generate calendar with availability
                            cal, available_dates = generate_availability_calendar(
                                provider['name'], selected_month, selected_year
                            )
                            
                            # Display calendar
                            cal_html = "<table class='calendar'>"
                            cal_html += "<tr><th>Mon</th><th>Tue</th><th>Wed</th><th>Thu</th><th>Fri</th><th>Sat</th><th>Sun</th></tr>"
                            
                            for week in cal:
                                cal_html += "<tr>"
                                for day in week:
                                    if day == 0:
                                        cal_html += "<td></td>"  # Empty cell for days not in month
                                    else:
                                        # Check if date is available
                                        current_date = datetime.date(selected_year, selected_month, day)
                                        is_available = day in available_dates
                                        is_today = current_date == today
                                        is_selected = current_date == st.session_state.get('event_date')
                                        
                                        # Define cell style based on availability
                                        if is_selected:
                                            cell_style = "selected-date"
                                        elif is_today:
                                            cell_style = "today"
                                        elif is_available:
                                            cell_style = "available"
                                        else:
                                            cell_style = "unavailable"
                                        
                                        cal_html += f"<td class='{cell_style}'>{day}</td>"
                                
                                cal_html += "</tr>"
                            
                            cal_html += "</table>"
                            
                            # Add CSS for calendar
                            st.markdown("""
                            <style>
                            .calendar {
                                width: 100%;
                                border-collapse: collapse;
                            }
                            .calendar th, .calendar td {
                                border: 1px solid #ddd;
                                padding: 8px;
                                text-align: center;
                            }
                            .calendar th {
                                background-color: #f2f2f2;
                            }
                            .available {
                                background-color: #d4edda;
                                cursor: pointer;
                            }
                            .unavailable {
                                background-color: #f8d7da;
                                color: #999;
                            }
                            .today {
                                border: 2px solid blue;
                            }
                            .selected-date {
                                background-color: #007bff;
                                color: white;
                                font-weight: bold;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(cal_html, unsafe_allow_html=True)
                            
                            if st.button(f"Book {provider['name']}", key=f"book_{i}"):
                                # Save booking to CSV
                                booking_data = {
                                    'name': st.session_state.get('user_name', 'Unknown'),
                                    'email': st.session_state.get('user_email', 'Unknown'),
                                    'phone': st.session_state.get('user_phone', 'Unknown'),
                                    'event_type': st.session_state.get('event_type', 'Unknown'),
                                    'budget': st.session_state.get('budget', 'Unknown'),
                                    'location': st.session_state.get('location', 'Unknown'),
                                    'event_date': st.session_state.get('event_date', datetime.date.today()).strftime('%Y-%m-%d'),
                                    'selected_services': ', '.join(st.session_state.get('selected_services', [])),
                                    'num_guests': st.session_state.get('num_guests', 0),
                                    'provider_name': provider['name'],
                                    'booking_timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                
                                # Create bookings directory if it doesn't exist
                                BOOKINGS_DIR = DATA_DIR / "bookings"
                                BOOKINGS_DIR.mkdir(exist_ok=True)
                                BOOKINGS_FILE = BOOKINGS_DIR / f"bookings_{datetime.date.today().strftime('%Y%m%d')}.csv"
                                
                                booking_file_exists = os.path.isfile(BOOKINGS_FILE)
                                with open(BOOKINGS_FILE, mode='a', newline='') as file:
                                    writer = csv.DictWriter(file, fieldnames=booking_data.keys())
                                    if not booking_file_exists:
                                        writer.writeheader()
                                    writer.writerow(booking_data)
                                
                                st.success(f"Booking with {provider['name']} confirmed for {st.session_state.get('event_date').strftime('%B %d, %Y')}!")
    
    with tab3:
        st.header("Cost Breakdown")
        
        if 'cost_breakdown' in st.session_state:
            cost_breakdown = st.session_state['cost_breakdown']
            total_cost = st.session_state['total_cost']
            
            st.subheader(f"Estimated Budget: {st.session_state.get('budget', '')} Category")
            
            # Create a pie chart of costs
            if cost_breakdown:
                fig, ax = plt.subplots(figsize=(10, 6))
                labels = list(cost_breakdown.keys())
                sizes = list(cost_breakdown.values())
                
                # Only include non-zero values
                non_zero_labels = [label for label, size in zip(labels, sizes) if size > 0]
                non_zero_sizes = [size for size in sizes if size > 0]
                
                if non_zero_sizes:  # Check if there's any data to plot
                    colors = plt.cm.viridis(np.linspace(0, 0.9, len(non_zero_labels)))
                    wedges, texts, autotexts = ax.pie(
                        non_zero_sizes, 
                        labels=None,  # We'll add our own labels
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=colors
                    )
                    
                    # Equal aspect ratio ensures that pie is drawn as a circle
                    ax.axis('equal')
                    
                    # Add legend
                    ax.legend(
                        wedges, 
                        [f"{label} (₹{size:,})" for label, size in zip(non_zero_labels, non_zero_sizes)],
                        title="Services",
                        loc="center left",
                        bbox_to_anchor=(1, 0, 0.5, 1)
                    )
                    
                    plt.title(f"Cost Breakdown - Total: ₹{total_cost:,}")
                    st.pyplot(fig)
                    
                    # Table with detailed breakdown
                    st.subheader("Service Cost Details")
                    
                    # Prepare the data for the table
                    table_data = []
                    for service, cost in cost_breakdown.items():
                        if cost > 0:  # Only include services with costs
                            table_data.append({
                                "Service": service.capitalize(),
                                "Cost (₹)": f"₹{cost:,}",
                                "Percentage": f"{(cost/total_cost)*100:.1f}%"
                            })
                    
                    # Add total row
                    table_data.append({
                        "Service": "**Total**",
                        "Cost (₹)": f"**₹{total_cost:,}**",
                        "Percentage": "**100.0%**"
                    })
                    
                    # Display the table
                    st.table(table_data)
                    
                    # Payment section
                    st.subheader("Payment Options")
                    payment_method = st.radio(
                        "Select Payment Method",
                        ["Credit Card", "Debit Card", "UPI", "Bank Transfer", "Pay Later (EMI)"]
                    )
                    
                    # Calculate booking amount (10% of total)
                    booking_amount = total_cost * 0.1
                    
                    st.write(f"**Booking Amount (10%):** ₹{booking_amount:,.2f}")
                    st.write(f"**Balance Amount:** ₹{total_cost - booking_amount:,.2f}")
                    
                    if st.button("Pay Booking Amount"):
                        st.success(f"Payment of ₹{booking_amount:,.2f} processed successfully!")
                        st.balloons()
                else:
                    st.warning("No services selected with costs to display.")
            else:
                st.warning("Please select services to see the cost breakdown.")
        else:
            st.info("Please go to the 'Plan Your Event' tab and fill in your event details to see the cost breakdown.")
    
    with tab4:
        st.header("Sustainable Event Planning")
        
        if 'selected_services' in st.session_state:
            selected_services = st.session_state['selected_services']
            eco_preference = st.session_state.get('eco_preference', False)
            
            st.write("Making your event eco-friendly not only helps the environment but can also create a memorable experience for your guests.")
            
            if eco_preference:
                st.success("Thank you for choosing eco-friendly options! Here are some sustainable alternatives for your selected services:")
            else:
                st.info("Consider these eco-friendly alternatives for your event planning:")
            
            # Show sustainable alternatives for selected services
            for service in selected_services:
                if service in sustainable_alternatives:
                    with st.expander(f"Sustainable {service.capitalize()} Options", expanded=eco_preference):
                        for alternative in sustainable_alternatives[service]:
                            st.write(f"- {alternative}")
            
            # Show carbon footprint estimation
            st.subheader("Carbon Footprint Estimation")
            
            # Calculate a rough estimate based on number of guests and services
            base_carbon_per_guest = 7.5  # kg CO2 per guest for standard event
            eco_reduction_factor = 0.6 if eco_preference else 1.0
            
            # Additional factors based on services
            service_carbon_factors = {
                'catering': 2.5,
                'venue': 1.0,
                'decoration': 0.8,
                'photography': 0.2,
                'entertainment': 0.5,
                'transportation': 3.0,
                'wedding cards': 0.3,
                'priest': 0.1
            }
            
            total_carbon = 0
            num_guests = st.session_state.get('num_guests', 100)
            
            # Calculate base carbon footprint
            base_carbon = num_guests * base_carbon_per_guest * eco_reduction_factor
            
            # Add service-specific carbon
            for service in selected_services:
                service_factor = service_carbon_factors.get(service, 0.5)
                service_carbon = num_guests * service_factor * eco_reduction_factor
                total_carbon += service_carbon
            
            total_carbon += base_carbon
            
            # Create a gauge chart for carbon footprint
            fig, ax = plt.subplots(figsize=(10, 2))
            
            # Define ranges
            low_range = num_guests * 5  # Good
            medium_range = num_guests * 10  # Average
            high_range = num_guests * 15  # High
            
            # Plot the gauge
            gauge_range = np.linspace(0, high_range * 1.2, 100)
            ax.barh([0], [high_range * 1.2], color='#f8d7da', height=0.3)
            ax.barh([0], [medium_range], color='#fff3cd', height=0.3)
            ax.barh([0], [low_range], color='#d4edda', height=0.3)
            
            # Add the pointer (triangle)
            pointer_height = 0.4
            pointer_x = total_carbon
            triangle_y = [-pointer_height/2, 0, pointer_height/2]
            triangle_x = [pointer_x, pointer_x + pointer_height, pointer_x]
            ax.fill(triangle_x, triangle_y, color='black')
            
            # Add labels
            ax.text(low_range/2, -0.5, 'Low Impact', ha='center')
            ax.text((medium_range + low_range)/2, -0.5, 'Medium Impact', ha='center')
            ax.text((high_range + medium_range)/2, -0.5, 'High Impact', ha='center')
            
            # Format the chart
            ax.set_xlim(0, high_range * 1.2)
            ax.set_ylim(-1, 1)
            ax.set_yticks([])
            ax.set_xticks([0, low_range, medium_range, high_range])
            ax.set_xticklabels(['0', f'{low_range:,.0f} kg', f'{medium_range:,.0f} kg', f'{high_range:,.0f} kg'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            plt.title(f'Estimated Carbon Footprint: {total_carbon:,.0f} kg CO2')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Carbon offset suggestion
            offset_cost_per_kg = 0.5  # Cost in ₹ to offset 1 kg of CO2
            offset_cost = total_carbon * offset_cost_per_kg
            
            st.write(f"**Offset your event's carbon footprint for approximately ₹{offset_cost:,.0f}**")
            
            if st.button("Add Carbon Offset to Package"):
                st.success(f"Carbon offset added to your package! Thank you for making your event carbon-neutral.")
                
            # Sustainable vendor recommendations
            st.subheader("Sustainable Vendor Recommendations")
            
            if 'providers' in st.session_state:
                eco_providers = [p for p in st.session_state['providers'] if p['eco_friendly']]
                
                if eco_providers:
                    st.write(f"We found {len(eco_providers)} eco-friendly providers for your event:")
                    
                    for i, provider in enumerate(eco_providers[:3]):  # Show top 3
                        st.write(f"**{provider['name']}** - {', '.join(provider['services'])}")
                else:
                    st.warning("No eco-friendly providers found for your criteria. Consider adjusting your requirements.")

    # Add a footer
    st.markdown("---")
    st.markdown("### Jashn-E-Hub - Your One-Stop Event Planning Solution")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Contact Us**")
        st.markdown("Email: support@jashnehub.in")
        st.markdown("Phone: +91 989XXXXXXX")
    with col2:
        st.markdown("**Follow Us**")
        st.markdown("Instagram | Facebook | Twitter")
    with col3:
        st.markdown("**About Us**")
        st.markdown("Jashn-E-Hub is committed to making event planning easy, affordable, and sustainable.")
        
if __name__ == "__main__":
    main()  