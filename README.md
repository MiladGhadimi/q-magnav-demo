# Q-MagNav Demo

**AI-assisted magnetic anomaly navigation under GNSS denial and spoofing**

Q-MagNav is a small technical prototype that demonstrates robust localization in GPS-denied or GPS-spoofed environments using magnetic anomaly maps, sensor fusion, and particle-filter-based localization.

The goal of this demo is to show how an autonomous system can detect when GNSS/GPS becomes unreliable and switch toward magnetic-map-aided navigation while keeping uncertainty visible.

## Overview

The demo simulates a vehicle or drone moving through a 2D environment with a synthetic magnetic anomaly map.

During the mission:

* GNSS/GPS is initially reliable.
* After a certain point, the GPS signal becomes spoofed.
* Dead reckoning begins to drift over time.
* A magnetic anomaly map is used for map-aided localization.
* A particle filter estimates the vehicle position.
* A simple OODA-style decision layer detects GNSS inconsistency and switches trust toward magnetic navigation.

## Key Features

* Synthetic magnetic anomaly map
* GPS spoofing simulation
* Dead reckoning / INS drift simulation
* Magnetic-map-aided localization
* Particle-filter sensor fusion
* GNSS residual monitoring
* Uncertainty visualization
* OODA-style decision layer: Observe → Orient → Decide → Act
* Streamlit dashboard for interactive exploration

## Important Note

This demo does not claim access to a real quantum magnetometer.

The “Quantum-grade magnetometer” mode simulates a lower-noise magnetic sensor in order to study how improved magnetic sensitivity could affect localization robustness under GNSS spoofing.

A precise way to describe the demo is:

> This prototype simulates quantum-magnetometer-grade sensitivity and studies its effect on magnetic-map-aided localization under GNSS spoofing.

## Why This Matters

GNSS/GPS can become unreliable in environments affected by jamming, spoofing, signal blockage, or intentional denial. This is important for autonomous systems, drones, aerospace platforms, robotics, and resilient navigation systems.

Magnetic anomaly navigation provides an alternative localization signal by comparing measured magnetic-field values with a known magnetic anomaly map.

This demo explores the algorithmic side of that idea using sensor fusion and probabilistic localization.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/q-magnav-demo.git
cd q-magnav-demo
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the demo:

```bash
streamlit run app.py
```

## Project Structure

```text
q-magnav-demo/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Demo Components

### 1. Magnetic Anomaly Map

A synthetic 2D magnetic anomaly map is generated to represent spatial magnetic-field variations.

### 2. Vehicle Trajectory

A simulated vehicle follows a ground-truth path across the map.

### 3. GPS Spoofing

GPS starts as a reliable measurement, then becomes gradually spoofed after a configurable point in the mission.

### 4. Dead Reckoning

Dead reckoning uses noisy odometry and slowly drifts away from the true trajectory.

### 5. Q-MagNav Estimation

A particle filter propagates candidate vehicle positions using odometry and scores them using magnetic-field map matching.

### 6. GNSS Trust Monitor

The system compares GPS measurements with the fused estimate. If the residual becomes too large for multiple steps, GNSS is marked as unreliable.

### 7. OODA Decision Layer

The demo includes a simple decision layer inspired by the OODA loop:

* **Observe:** collect GNSS, odometry, and magnetic measurements
* **Orient:** perform map matching and sensor fusion
* **Decide:** evaluate whether GNSS is trustworthy
* **Act:** switch toward magnetic-map-aided navigation when GNSS is denied or spoofed

## Possible Extensions

* Replace the synthetic magnetic map with public airborne magnetic navigation data
* Add real IMU models with heading, velocity, and bias states
* Add Kalman filter, EKF, or UKF baselines
* Add learned magnetic compensation for vehicle-induced magnetic noise
* Add real-time animation
* Add geospatial coordinates and map tiles
* Benchmark different sensor-noise regimes

## Technologies Used

* Python
* Streamlit
* NumPy
* Pandas
* Plotly

## Author

Milad Ghadimi

Background: quantum information, quantum algorithms, communication networks, and applied machine learning.

This demo was built as a small prototype connecting quantum-enabled sensing ideas with robust AI-assisted navigation.
