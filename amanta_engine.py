import pandas as pd
import joblib
import sqlite3
import numpy as np
from nfstream import NFStreamer
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

print("Loading model & assets...")
try:
    model = joblib.load('nids_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("Assets loaded successfully.")
except Exception as e:
    print(f"Error loading assets: {e}")
    exit()

DB_PATH = "/app/data/amanta_nids.db"

MAPS = {
    # --- Basic Flow Info ---
    'Destination Port': 'dst_port',
    'Flow Duration': 'bidirectional_duration_ms', 
    
    # --- Forward Packets ---
    'Total Fwd Packets': 'src2dst_packets',
    'Total Length of Fwd Packets': 'src2dst_bytes',
    'Fwd Packet Length Max': 'src2dst_max_ps',
    'Fwd Packet Length Min': 'src2dst_min_ps',
    'Fwd Packet Length Mean': 'src2dst_mean_ps',
    'Fwd Packet Length Std': 'src2dst_stddev_ps',
    
    # --- Backward Packets ---
    'Bwd Packet Length Max': 'dst2src_max_ps',
    'Bwd Packet Length Min': 'dst2src_min_ps',
    'Bwd Packet Length Mean': 'dst2src_mean_ps',
    'Bwd Packet Length Std': 'dst2src_stddev_ps',
    
    # --- Flow Rates ---
    'Flow Bytes/s': 'bidirectional_bytes',   
    'Flow Packets/s': 'bidirectional_packets', 
    
    # --- Flow IAT (Inter Arrival Time) ---
    'Flow IAT Mean': 'bidirectional_mean_piat_ms',
    'Flow IAT Std': 'bidirectional_stddev_piat_ms',
    'Flow IAT Max': 'bidirectional_max_piat_ms',
    'Flow IAT Min': 'bidirectional_min_piat_ms',
    
    # --- Forward IAT ---
    'Fwd IAT Total': 'src2dst_duration_ms',
    'Fwd IAT Mean': 'src2dst_mean_piat_ms',
    'Fwd IAT Std': 'src2dst_stddev_piat_ms',
    'Fwd IAT Max': 'src2dst_max_piat_ms',
    'Fwd IAT Min': 'src2dst_min_piat_ms',
    
    # --- Backward IAT ---
    'Bwd IAT Total': 'dst2src_duration_ms',
    'Bwd IAT Mean': 'dst2src_mean_piat_ms',
    'Bwd IAT Std': 'dst2src_stddev_piat_ms',
    'Bwd IAT Max': 'dst2src_max_piat_ms',
    'Bwd IAT Min': 'dst2src_min_piat_ms',
    
    # --- Headers ---
    'Fwd Header Length': 'src2dst_header_size',
    'Bwd Header Length': 'dst2src_header_size',
    'Fwd Packets/s': 'src2dst_packets', # Approximation
    'Bwd Packets/s': 'dst2src_packets', # Approximation
    
    # --- Packet Stats ---
    'Min Packet Length': 'bidirectional_min_ps',
    'Max Packet Length': 'bidirectional_max_ps',
    'Packet Length Mean': 'bidirectional_mean_ps',
    'Packet Length Std': 'bidirectional_stddev_ps',
    'Packet Length Variance': 'bidirectional_stddev_ps', # Logic: Std^2
    
    # --- Flags ---
    'FIN Flag Count': 'bidirectional_fin_packets',
    'PSH Flag Count': 'bidirectional_psh_packets',
    'ACK Flag Count': 'bidirectional_ack_packets',
    
    # --- Others ---
    'Average Packet Size': 'bidirectional_mean_ps',
    'Subflow Fwd Bytes': 'src2dst_bytes',
    'Init_Win_bytes_forward': 'src2dst_init_window_size',
    'Init_Win_bytes_backward': 'dst2src_init_window_size',
    'act_data_pkt_fwd': 'src2dst_packets',
    'min_seg_size_forward': 'src2dst_min_ps' 
}

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                src_ip TEXT,
                dst_ip TEXT,
                attack_type TEXT,
                confidence REAL
            )
        """)
        conn.commit()
        return conn
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return None

db_conn = init_db()

def save_log_entry(conn, data):
    if conn is None: return
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (timestamp, src_ip, dst_ip, attack_type, confidence) VALUES (?, ?, ?, ?, ?)",
            (data['timestamp'], data['src_ip'], data['dst_ip'], data['attack_type'], data['confidence'])
        )
        conn.commit()
    except Exception: pass

streamer = NFStreamer(
    source="any",
    statistical_analysis=True,
    idle_timeout=1,
    active_timeout=1,
    promiscuous_mode=True
)
print("🛡️ Amanta NIDS Engine is LIVE (Dataset: CICIDS2017 Cleaned)...")

for flow in streamer:
    try:
        features_dict = {}
        
        for col in feature_names:
            if col == 'Attack Type': continue 

            attr_name = MAPS.get(col, None)
            
            if attr_name:
                val = getattr(flow, attr_name, 0)
            else:
                val = 0 

            if "Duration" in col or "IAT" in col:
                val = val * 1000
            
            if col == 'Packet Length Variance':
                std_val = getattr(flow, 'bidirectional_stddev_ps', 0)
                val = std_val ** 2

            features_dict[col] = [val]

        X_df = pd.DataFrame(features_dict)
        X_scaled = scaler.transform(X_df)
        probs = model.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        label = le.inverse_transform([pred_idx])[0]
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': now, 
            'src_ip': flow.src_ip, 
            'dst_ip': flow.dst_ip, 
            'attack_type': label, 
            'confidence': float(confidence)
        }
        save_log_entry(db_conn, log_entry)

        # icon = "🟢" if label == "Normal Traffic" else "⚠️"
        # pkt_info = f"Pkts: {flow.bidirectional_packets}"
        # if flow.bidirectional_packets > 1000:
        #     pkt_info = f"\033[91mPkts: {flow.bidirectional_packets}\033[0m"
        # print(f"{icon} [{now}] {label:20} | {flow.src_ip:15} -> {flow.dst_ip:15} | Conf: {confidence*100:.1f}% | {pkt_info}")

    except Exception as e:
        print(f"Error: {e}") 
        continue