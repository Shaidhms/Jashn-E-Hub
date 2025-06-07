# 🎉 EventEase Planner

**EventEase Planner** is an intelligent event planning application built using **Python** and **Streamlit**. It allows users to create, customize, and manage events like weddings, birthdays, or corporate parties—while leveraging **Machine Learning** to recommend the best vendors based on budget, availability, and compatibility.

---

## 🚀 Features

### ✅ User Registration
- Collects user details: Name, Email, Phone, and Address
- Data stored for future event planning and personalization

### 🎯 Event Customization
- Select Event Type: `Wedding`, `Engagement`, `Corporate Party`, `House Party`, `Birthday`
- Choose Budget Tier: `Economy`, `Standard`, `Premium`, `Luxury`, `Ultra Luxury` (₹50,000+)
- Pick Event Date & City: Mumbai, Chennai, Pune, Delhi, Bengaluru
- Select Required Services: Catering, Venue, Decoration, Photography, Entertainment, Transportation, Wedding Card, etc.
- Specify Number of Guests

### 🤖 ML-Powered Vendor Recommendation
- ML model recommends the best vendors based on:
  - Selected services
  - Budget and rating match
  - Vendor availability on selected date
- Displays:
  - Vendor Name
  - Ratings
  - Compatibility Score
  - Availability Calendar
- Smart logic: More budget unlocks premium vendors
- One-click vendor contact with request packet sent

### 💰 Cost Breakdown Section
- Dynamically calculates total event cost based on services
- Transparent budget allocation:
  - e.g., ₹15,000 for catering, ₹30,000 for venue, etc.
- Real-time insights on budget utilization

### 💳 Smart Payment Gateway
- Payment options:
  - UPI: QR Code Generation
  - Bank Transfer: IFSC, Account Number, Branch Info
- Payment Schedule:
  - 25% Advance
  - 50% Midway
  - 25% After Event
- Option to download invoice

### 📊 ML-Based Event Insights
- Insights dashboard powered by the ML model:
  - Vendor availability charts
  - Budget vs service visualization
  - Compatibility scores across service categories
- Email Recommendation:
  - Top vendor recommendation based on smart logic
  - Explanation of why the vendor is the best match
  - Sent directly to the user's email

---

## 🧠 Tech Stack

- **Frontend & Backend**: [Streamlit](https://streamlit.io/)
- **Programming Language**: Python
- **ML Libraries**: Scikit-learn, Pandas, NumPy
- **Visualization**: Streamlit native charts
- **Email Service**: SMTP (or similar, optional)
- **Payment Logic**: Simulated with UPI/Bank options

---

## 📌 How It Works

1. User registers and logs in
2. Selects and customizes the event
3. ML model generates top vendor matches
4. User views vendor ratings, availability, and score
5. Cost breakdown and payment options shown
6. ML insights and vendor recommendations sent via email

---



## 📈 Future Enhancements

- 🔐 Firebase/MongoDB integration for persistent user/vendor data
- 📱 Mobile responsive UI
- 📧 Built-in email invite system for guests
- 🧾 Auto-generated invoice PDF
- 📆 Google Calendar integration
- 🧠 Improved ML model using real vendor datasets
- 🛠️ Admin dashboard for managing vendors

---

## 👨‍💻 Author

**Ayush Mishra**  
B.Tech CSE | AI/ML Enthusiast  
📧 [ayushmishra7548@gmail.com](mailto:ayushmishra7548@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourgithub)  
🎓 Raj Kumar Goel Institute of Technology (RKGIT), 2027

---

## 📃 License

This project is open source and free to use for learning purposes.  
Feel free to fork it and build on it!

