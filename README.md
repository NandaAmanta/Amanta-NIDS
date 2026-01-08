# Amanta NIDS: AI-Powered Network Intrusion Detection System
Amanta NIDS is a real-time Network Intrusion Detection System (NIDS) that leverages Machine Learning to identify and classify network attacks. It integrates NFStream for deep packet inspection and a Random Forest model trained on the CIC-IDS2017 dataset to provide live security monitoring through a Web Dashboard.

## 🚀 Features
- Real-time Traffic Analysis: Captures and processes live network flows using NFStreamer.

- AI-Driven Detection: Detects various attack types including DDoS, Brute Force, Botnets, and Infiltration.

- Comprehensive Dashboard: Visualizes attack statistics, traffic volume, and threat levels via Grafana.

- Automated Logging: Stores detected anomalies in a SQLite database for forensic analysis.

- Containerized Deployment: Easy to deploy anywhere using Docker and Docker Compose.

## 🛠️ How It Works
- Ingestion: The Engine monitors the network interface (NIC) and uses NFStream to aggregate raw packets into bi-directional flows.

- Feature Extraction: The engine extracts 52 statistical features (e.g., Flow Duration, Packet Length Variance, IAT Mean) to match the CIC-IDS2017 format.

- Inference: Extracted features are scaled and passed through a pre-trained Random Forest model to predict the traffic label (Normal or Attack type).

- Storage: Results, including timestamps, source/destination IPs, and confidence scores, are saved to a SQLite database.

- Visualization: Grafana reads the SQLite database to display real-time analytics and alerts.

## 📁 Repository Structure
```
.
├── amanta_engine.py       # Main AI processing engine
├── nids_model.pkl         # Pre-trained Random Forest model
├── scaler.pkl             # StandardScaler for feature normalization
├── label_encoder.pkl      # Encoder for attack labels
├── feature_names.pkl      # List of required ML features
├── Dockerfile             # Container configuration for the engine
├── docker-compose.yml     # Orchestration for Engine, DB, and Grafana
└── data/                  # Persistent storage for SQLite DB
```

## 🔧 Installation & Setup

Prerequisites :
- Docker installed.
- Docker Compose installed.

1. Create `docker-compose.yml` file
```
services:
  # --- Service 1: Amanta NIDS Engine ---
  amanta-nids-engine:
    image: nandaamanta/amanta-nids:0.1.1-alpha
    container_name: amanta-nids-engine
    network_mode: "host"  # Perlu host mode untuk sniffing
    restart: always
    privileged: true       # Agar bisa akses interface network (tcpdump, pyshark, dll)
    environment:
      - PYTHONUNBUFFERED=1   # Log engine keluar real-time
    volumes:
      - /var/lib/amanta-nids/data:/app/data
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```
or if you want to include the grafana. you can copy paste down here :

```
services:
  # --- Service 1: Amanta NIDS Engine ---
  amanta-nids-engine:
    image: nandaamanta/amanta-nids:0.1.1-alpha
    container_name: amanta-nids-engine
    network_mode: "host"  # Perlu host mode untuk sniffing
    restart: always
    privileged: true       # Agar bisa akses interface network (tcpdump, pyshark, dll)
    environment:
      - PYTHONUNBUFFERED=1   # Log engine keluar real-time
    volumes:
      - /var/lib/amanta-nids/data:/app/data
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  # --- Service 2: Grafana Dashboards ---
  grafana:
    image: grafana/grafana:latest
    container_name: amanta-grafana
    restart: always
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=frser-sqlite-datasource
    depends_on:
      - amanta-nids-engine
    volumes:
      - /var/lib/amanta-nids/data:/var/lib/amanta-nids/data:ro
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:

```

2. Run `docker-compose up -d`. Now Grafana is accessible at http://localhost:3000.

## 🛡️ Supported Attack Detections
The system is trained to recognize the following categories from the CIC-IDS2017 dataset:
- DDoS / DoS (Slowloris, Hulk, GoldenEye, etc.)

- Brute Force (FTP/SSH Patator)

- Port Scanning

- Botnets

- Web Attacks (SQL Injection, XSS)

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.