# MCP AutoAdvisor Server

## Overview

**MCP AutoAdvisor Server** is a custom Model Context Protocol (MCP) server built for the course project.  
It provides a set of tools to explore and analyze a dataset of **23,000 used cars** with attributes such as make, model, year, mileage, price, fuel type, transmission, condition, and accident history.

The server demonstrates a **non-trivial use case** for MCP by exposing both data filtering utilities and a simple machine learning model for **price estimation**.

---

## Dataset

- **File:** `data/Updated_Car_Sales_Data.csv`  
- **Rows:** 23,000  
- **Columns:**

  | Column            | Description                                              |
  |-------------------|----------------------------------------------------------|
  | Car Make          | Manufacturer (e.g., Honda, Toyota, BMW)                  |
  | Car Model         | Model name (e.g., Civic, Camry)                          |
  | Year              | Year of manufacture                                      |
  | Mileage           | Mileage driven (in kilometers)                           |
  | Price             | Price of the car                                         |
  | Fuel Type         | Fuel type (Gasoline, Diesel, Hybrid, Electric, Petrol)   |
  | Color             | Exterior color                                           |
  | Transmission      | Transmission type (Automatic, Manual)                    |
  | Options/Features  | Extra features (GPS, Leather seats, Sunroof, etc.)       |
  | Condition         | New, Like New, Used                                      |
  | Accident          | Accident history (Yes / No)                              |

---

## Installation

Clone the repository and install dependencies inside a Python virtual environment:

```bash
git clone https://github.com/EstebanZG999/MCP_AutoAdvisor_Server.git
cd MCP_AutoAdvisor_Server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Requirements:**

```
pandas==2.2.2
scikit-learn==1.5.2
numpy==1.26.4
mcp>=1.12,<2
```

---

## Running the Server

Run the server directly with:

```bash
python server.py
```

Normally, the server should not be started manually — the **MCP Host** launches it via stdio according to the configuration.

---

## MCP Configuration (Host)

In your **MCP Host** repository, add the following entry to `servers.yaml`:

```yaml
servers:
  auto_advisor:
    command: "/absolute/path/to/MCP_AutoAdvisor_Server/.venv/bin/python"
    args: ["/absolute/path/to/MCP_AutoAdvisor_Server/server.py"]
    env: {}
```

---

## Tools

This server exposes 5 tools through MCP:

### 1. `filter_cars`

**Description:** Filter cars by criteria such as make, model, year, price, mileage, fuel, transmission, condition, accident.

**Input schema:**
```json
{
  "Car Make": "Toyota",
  "Year_min": 2019,
  "Price_max": 25000,
  "Transmission": "Automatic",
  "Condition": "Used",
  "Accident": "No",
  "limit": 5
}
```

**Example call:**
```bash
/mcp call auto_advisor filter_cars {...}
```

---

### 2. `recommend`

**Description:** Recommend cars within a budget and preferences. Sorted by ascending price.

**Example input:**
```json
{
  "budget_max": 20000,
  "Fuel Type": "Gasoline",
  "Transmission": "Automatic",
  "Condition": "Used",
  "Accident": "No",
  "Year_min": 2017,
  "limit": 5
}
```

---

### 3. `estimate_price`

**Description:** Estimate the price of a car using a linear regression model.

**Example input:**
```json
{
  "Car Make": "Honda",
  "Car Model": "Civic",
  "Year": 2020,
  "Mileage": 40000,
  "Fuel Type": "Gasoline",
  "Transmission": "Automatic",
  "Condition": "Like New",
  "Accident": "No"
}
```

**Example output:**
```json
{
  "input": {...},
  "estimated_price": 28188.75
}
```

---

### 4. `average_price`

**Description:** Compute the average price for cars filtered by make, model, fuel type, year range, etc.

**Example input:**
```json
{
  "Car Make": "BMW",
  "Fuel Type": "Diesel",
  "Year_min": 2018
}
```

---

### 5. `top_cars`

**Description:** Return top N cars sorted by price (cheap or expensive).

**Example input:**
```json
{
  "n": 5,
  "sort_order": "expensive",
  "Car Make": "Audi"
}
```

---

## Demo

From the host CLI:

```bash
/mcp tools auto_advisor
/mcp call auto_advisor filter_cars {"Car Make":"Toyota","Year_min":2019,"Price_max":25000,"Transmission":"Automatic","Condition":"Used","Accident":"No","limit":5}
/mcp call auto_advisor estimate_price {"Car Make":"Honda","Car Model":"Civic","Year":2020,"Mileage":40000,"Fuel Type":"Gasoline","Transmission":"Automatic","Condition":"Like New","Accident":"No"}
/mcp call auto_advisor average_price {"Car Make":"BMW","Fuel Type":"Diesel","Year_min":2018}
/mcp call auto_advisor top_cars {"n":5,"sort_order":"expensive","Car Make":"Audi"}
/mcp call auto_advisor recommend {"budget_max":20000,"Fuel Type":"Gasoline","Transmission":"Automatic","Condition":"Used","Accident":"No","Year_min":2017,"limit":5}
```

---

## Project Structure

```
MCP_AutoAdvisor_Server/
├── README.md
├── requirements.txt
├── data_check.py
├── server.py              # MCP entrypoint: defines tools, routes calls, stdio runner
├── mcp_server/
│   ├── __init__.py
│   └── tools.py           # Implementation of filtering, recommendation, ML price estimator
└── data/
    └── Updated_Car_Sales_Data.csv
```

---
