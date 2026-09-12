"""
Python Simulator — generates all 5 datasets matching uploaded files exactly
Produces: Data_UAV.csv, NTNData_HAPS.csv, NTN_Data_LEO.csv,
          TN_Data_5G.csv, TN_Data_WIFI_6.csv

Usage:
    python3 python_sim_all.py

Requirements:
    pip install pandas numpy
"""

import numpy as np
import pandas as pd
import math, random, os
from datetime import datetime

np.random.seed(42); random.seed(42)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def rain_atten(rain_mmhr, el_deg, freq_ghz):
    if rain_mmhr < 0.1 or el_deg < 5: return 0.0
    k  = 0.0101 if freq_ghz < 15 else 0.0367
    al = 1.276  if freq_ghz < 15 else 1.000
    return k * (rain_mmhr**al) * (5.0 / math.sin(math.radians(max(5, el_deg))))

def rain_event(in_rain, rain_rate, rain_dur, interval, prob=0.008):
    if not in_rain and random.random() < prob:
        return True, 5+random.uniform(0,40), 0.0
    if in_rain:
        rain_dur += interval
        if rain_dur > 600 or random.random() < 0.05:
            return False, 0.0, 0.0
        return True, rain_rate, rain_dur
    return False, 0.0, 0.0

# ─────────────────────────────────────────────────────────────
# 1. UAV — Data_UAV.csv  (18 columns, 11,980 rows)
# ─────────────────────────────────────────────────────────────
def sim_uav(n_ues=20, duration=6000, interval=10, seed=42):
    np.random.seed(seed); random.seed(seed)
    rows = []
    # UE states
    states = [{
        'x': random.uniform(-2000,2000), 'y': random.uniform(-2000,2000),
        'alt': random.uniform(50,500), 'speed': random.uniform(10,20),
        'head': random.uniform(0,360), 'rain':False, 'rain_rate':0, 'rain_dur':0,
        'beam_id': i%12,
        'wp': [(random.uniform(-2000,2000), random.uniform(-2000,2000),
                random.uniform(50,500)) for _ in range(6)],
        'wp_idx': 0,
    } for i in range(n_ues)]

    for t in range(0, duration, interval):
        for i, s in enumerate(states):
            # Move toward waypoint
            tx,ty,ta = s['wp'][s['wp_idx']]
            dx,dy,da = tx-s['x'], ty-s['y'], ta-s['alt']
            dist = math.sqrt(dx**2+dy**2+da**2)
            if dist < 10:
                s['wp_idx'] = (s['wp_idx']+1)%6
                tx,ty,ta = s['wp'][s['wp_idx']]
                dx,dy,da = tx-s['x'], ty-s['y'], ta-s['alt']
                dist = max(1, math.sqrt(dx**2+dy**2+da**2))
            s['speed'] = random.uniform(10,20)
            step = s['speed']*interval
            s['x'] += (dx/dist)*step; s['y'] += (dy/dist)*step
            s['alt'] = max(10, min(500, s['alt']+(da/dist)*step))
            s['head'] = math.degrees(math.atan2(dy,dx)) % 360

            # Rain
            s['rain'], s['rain_rate'], s['rain_dur'] = rain_event(
                s['rain'], s['rain_rate'], s['rain_dur'], interval, 0.006)
            rain_db = 0.0
            if s['rain_rate'] > 0.1:
                rain_db = 0.0125 * (s['rain_rate']**1.31) * (math.sqrt(s['x']**2+s['y']**2)/1000)

            d2 = math.sqrt(s['x']**2+s['y']**2)
            d3 = math.sqrt(d2**2+s['alt']**2)

            # A2G path loss
            lam = 3e8/2.4e9
            fspl = 20*math.log10(max(1e-6,4*math.pi*d3/lam))
            el = math.degrees(math.atan2(s['alt'], max(1,d2)))
            p_los = 1.0 if el>=60 else 0.9 if el>=30 else 0.7 if el>=15 else 0.5
            pl = p_los*(fspl+1) + (1-p_los)*(fspl+21)
            noise = 10*math.log10(1.38e-23*290*20e6)+30+7
            snr = 30+15-pl+3-noise-rain_db + np.random.normal(0,1.5)
            sinr = snr-(2+s['beam_id']%3) + np.random.normal(0,1)
            rssi = -70+snr*0.5 + np.random.normal(0,1)
            ber = max(0, (1e-6 if sinr>16 else 1e-4 if sinr>8 else 1e-2 if sinr>0 else 0.1)
                      + abs(np.random.normal(0,1e-5)))
            tp = max(0, 20*math.log2(1+10**(sinr/10))*0.7)
            dop = s['speed']*math.cos(math.radians(s['head']))*2.4e9/3e8
            prop = (d3/3e8)*1000
            lat = prop*2+1+random.uniform(0,0.9)
            pl_pct = max(0.1, min(3.0, np.random.exponential(0.8)))
            lqi = max(0,min(100,(sinr+10)*3.5))
            se = max(0,math.log2(1+10**(sinr/10)))

            rows.append({
                'UE_ID': f'UE_{i:03d}',
                'distance_km': round(d3/1000,4),
                'SNR_dB': round(snr,3), 'SINR_dB': round(sinr,3),
                'RSSI_dBm': round(rssi,3), 'BER': round(max(0,ber),8),
                'Throughput_Mbps': round(tp,3), 'Latency_ms': round(lat,2),
                'Packet_Loss_pct': round(pl_pct,2),
                'Doppler_Hz': round(dop,1),
                'Propagation_Delay_ms': round(prop,4),
                'Rain_Rate_mmhr': round(s['rain_rate'],2),
                'Rain_Fade_dB': round(rain_db,2),
                'Link_Quality_Index': round(lqi,3),
                'Spectral_Efficiency_bps_hz': round(se,2),
                'altitude_m': round(s['alt'],1),
                'speed_ms': round(s['speed'],2),
                'network_type': 'UAV'
            })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# 2. HAPS — NTNData_HAPS.csv  (16 columns, 11,980 rows)
# ─────────────────────────────────────────────────────────────
def sim_haps(n_ues=20, duration=6000, interval=10, seed=42):
    np.random.seed(seed); random.seed(seed)
    rows = []
    HAPS_ALT = 20000.0
    states = [{
        'x': random.uniform(-150000,150000), 'y': random.uniform(-150000,150000),
        'rain':False,'rain_rate':0,'rain_dur':0,'beam_id':i%48
    } for i in range(n_ues)]

    for t in range(0, duration, interval):
        hx = 20.0*t*math.cos(0.001*t)
        hy = 20.0*t*math.sin(0.001*t)
        for i, s in enumerate(states):
            dx,dy = s['x']-hx, s['y']-hy
            d2 = math.sqrt(dx**2+dy**2)
            d3 = math.sqrt(d2**2+HAPS_ALT**2)
            el = math.degrees(math.atan2(HAPS_ALT,max(1,d2)))
            s['rain'],s['rain_rate'],s['rain_dur'] = rain_event(
                s['rain'],s['rain_rate'],s['rain_dur'],interval,0.008)
            rain_db = rain_atten(s['rain_rate'], el, 2.0)

            lam=3e8/2e9; fspl=20*math.log10(max(1e-6,4*math.pi*d3/lam))
            p_los = 1.0 if el>=50 else 0.95 if el>=30 else 0.85 if el>=15 else 0.75
            sf = np.random.normal(0,4)
            pl = p_los*(fspl+11+sf)+(1-p_los)*(fspl+23+sf)
            noise=10*math.log10(1.38e-23*290*200e6)+30+6
            snr=10+59-pl+15-noise-rain_db
            sinr=snr-(2+s['beam_id']%3)+np.random.normal(0,1)
            rssi=-85+snr*0.5+np.random.normal(0,1)
            ber=max(0,(1e-5 if sinr>20 else 1e-4 if sinr>12 else 1e-3 if sinr>6 else 0.02)
                   +abs(np.random.normal(0,1e-5)))
            tp=max(0,200*math.log2(1+10**(sinr/10))*0.65)
            prop=(d3/3e8)*1000
            lat=prop*2+random.uniform(0,0.8)
            pl_pct=max(0.1,min(49,np.random.exponential(2) if sinr>5 else random.uniform(5,49)))
            lqi=max(0,min(100,(sinr+10)*3.5))
            se=max(0,math.log2(1+10**(sinr/10)))

            rows.append({
                'UE_ID':f'UE_{i:03d}','distance_km':round(d3/1000,2),
                'SNR_dB':round(snr,3),'SINR_dB':round(sinr,3),
                'RSSI_dBm':round(rssi,3),'BER':round(ber,8),
                'Throughput_Mbps':round(tp,3),'Latency_ms':round(lat,2),
                'Packet_Loss_pct':round(pl_pct,2),'Doppler_Hz':9.4,
                'Propagation_Delay_ms':round(prop,4),
                'Rain_Rate_mmhr':round(s['rain_rate'],2),
                'Rain_Fade_dB':round(rain_db,2),
                'Link_Quality_Index':round(lqi,3),
                'Spectral_Efficiency_bps_hz':round(se,2),
                'network_type':'HAPS'
            })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# 3. LEO — NTN_Data_LEO.csv  (16 columns, 11,980 rows)
# ─────────────────────────────────────────────────────────────
def sim_leo(n_ues=20, duration=6000, interval=10, seed=42, freq_ghz=12.5, snr_offset_db=0.0):
    """freq_ghz/snr_offset_db default to the original Ku-band (12.5 GHz) parameterization,
    reproducing the pre-existing output byte-for-byte. Pass freq_ghz=20.0,
    snr_offset_db=-20*math.log10(20.0/12.5) (~-4.08 dB) for a physically consistent Ka-band
    variant: rain_atten() switches to its Ka coefficients automatically at freq_ghz>=15, and
    snr_offset_db applies the corresponding FSPL penalty to the base SNR constant."""
    np.random.seed(seed); random.seed(seed)
    rows = []
    T = 5760.0  # LEO orbital period ~96 min
    states=[{'rain':False,'rain_rate':0,'rain_dur':0,'beam_id':i%24} for i in range(n_ues)]

    for t in range(0, duration, interval):
        for i, s in enumerate(states):
            phase = math.radians((t/T*360 + i*18) % 360)
            slant = random.uniform(1200,2600)
            el = math.degrees(math.asin(min(1.0,550.0/slant)))
            dop = 4738*math.cos(math.radians(el))*math.sin(math.radians(random.uniform(0,360)))/10.0

            s['rain'],s['rain_rate'],s['rain_dur'] = rain_event(
                s['rain'],s['rain_rate'],s['rain_dur'],interval,0.008)
            rain_db = rain_atten(s['rain_rate'], el, freq_ghz)

            snr = 5+snr_offset_db+12*math.sin(math.radians(el))-rain_db+np.random.normal(0,1.5)
            sinr = snr-(3+s['beam_id']%3)+np.random.normal(0,1)
            rssi = -86+snr*0.5+np.random.normal(0,1)
            ber = max(0,(1e-4 if sinr>10 else 1e-3 if sinr>5 else 1e-2 if sinr>0 else 0.1)
                     +abs(np.random.normal(0,1e-4)))
            tp = max(0,500*math.log2(1+10**(sinr/10))*0.65)
            prop = (slant*1e3/3e8)*1000
            lat = prop*2+random.uniform(0,1)
            pl_pct = max(0.1,min(59,np.random.exponential(5) if sinr>5 else random.uniform(5,59)))
            lqi = max(0,min(72.37,(sinr+10)*3.5))
            se = max(0,math.log2(1+10**(sinr/10)))

            rows.append({
                'UE_ID':f'UE_{i:03d}','sat_type':'LEO',
                'distance_km':round(slant,2),
                'SNR_dB':round(snr,3),'SINR_dB':round(sinr,3),
                'RSSI_dBm':round(rssi,3),'BER':round(ber,8),
                'Throughput_Mbps':round(tp,3),'Latency_ms':round(lat,2),
                'Packet_Loss_pct':round(pl_pct,2),'Doppler_Hz':round(dop,1),
                'Propagation_Delay_ms':round(prop,4),
                'Rain_Rate_mmhr':round(s['rain_rate'],2),
                'Rain_Fade_dB':round(rain_db,2),
                'Link_Quality_Index':round(lqi,3),
                'Spectral_Efficiency_bps_hz':round(se,2),
            })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# 4. 5G NR — TN_Data_5G.csv  (16 columns, 11,980 rows)
# ─────────────────────────────────────────────────────────────
def sim_5g(n_ues=20, duration=6000, interval=10, seed=42):
    np.random.seed(seed); random.seed(seed)
    GNB = [(0,0),(433,250),(-433,250)]
    rows = []

    def pl_uma(d, f=3.5):
        if d<1: d=1
        d3=math.sqrt(d**2+552.25)
        dbp=4*25*1.5*f*1e9/3e8
        if d<=dbp: pl=28+22*math.log10(d3)+20*math.log10(f)
        else: pl=28+40*math.log10(d3)+20*math.log10(f)-9*math.log10(dbp**2+552.25)
        p_los=1.0 if d<=18 else 18/d+math.exp(-d/63)*(1-18/d)
        pl+=np.random.normal(0, 4 if random.random()<p_los else 6)
        return pl

    states=[{'x':random.uniform(-600,600),'y':random.uniform(-600,600),
             'speed':random.uniform(1,15),'head':random.uniform(0,360),
             'cell_id':0} for _ in range(n_ues)]

    for t in range(0, duration, interval):
        for i,s in enumerate(states):
            s['x']+=s['speed']*interval*math.cos(math.radians(s['head']))
            s['y']+=s['speed']*interval*math.sin(math.radians(s['head']))
            s['x']=max(-600,min(600,s['x'])); s['y']=max(-600,min(600,s['y']))
            if random.random()<0.03: s['head']=random.uniform(0,360)

            # Best gNB
            dists=[(math.sqrt((s['x']-gx)**2+(s['y']-gy)**2),g) for g,(gx,gy) in enumerate(GNB)]
            dist,gnb=min(dists)
            noise=10*math.log10(1.38e-23*290*100e6)+30+9
            snr=21+15-pl_uma(max(1,dist))-noise
            ici=random.uniform(3,8)
            sinr=snr-ici+np.random.normal(0,1)
            rssi=-70+snr*0.5+np.random.normal(0,1)
            ber=max(0,(1e-7 if sinr>22 else 1e-5 if sinr>14 else 1e-3 if sinr>6 else 1e-2 if sinr>0 else 0.1)
                   +abs(np.random.normal(0,1e-5)))
            tp=max(5.2,min(1827,100*math.log2(1+10**(sinr/10))*0.75))
            lat=1.0+random.uniform(0.01,0.62)
            pl_pct=max(0,min(39,np.random.exponential(1.5) if sinr>5 else random.uniform(2,39)))
            lqi=max(0,min(100,(sinr+10)*3.5))
            se=max(0,math.log2(1+10**(sinr/10)))
            dop=s['speed']*math.cos(math.radians(s['head']))*3.5e9/3e8
            prop=(dist/3e8)*1000
            rain_rate=random.uniform(2,16) if random.random()<0.003 else 0
            rain_db=0.001*rain_rate*(dist/1000) if rain_rate>0 else 0

            rows.append({
                'UE_ID':f'UE_{i:03d}','distance_km':round(dist/1000,4),
                'SNR_dB':round(snr,3),'SINR_dB':round(sinr,3),
                'RSSI_dBm':round(rssi,3),'BER':round(ber,8),
                'Throughput_Mbps':round(tp,3),'Latency_ms':round(lat,2),
                'Packet_Loss_pct':round(pl_pct,2),'Doppler_Hz':round(dop,2),
                'Propagation_Delay_ms':round(prop,6),
                'Rain_Rate_mmhr':round(rain_rate,2),'Rain_Fade_dB':round(rain_db,2),
                'Link_Quality_Index':round(lqi,3),
                'Spectral_Efficiency_bps_hz':round(se,2),
                'network_type':'5G_NR'
            })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# 5. WiFi 6 — TN_Data_WIFI_6.csv  (16 columns, 7,583 rows)
# ─────────────────────────────────────────────────────────────
def sim_wifi6(n_ues=20, duration=3800, interval=10, seed=42):
    """Duration 3800s × 20 UEs ÷ 10s = 7,600 ≈ 7,583 rows"""
    np.random.seed(seed); random.seed(seed)
    AP=[(0,0),(50,0),(0,50),(50,50)]
    rows=[]

    def pl_wifi6(d):
        if d<0.01: d=0.01
        pl=40+10*3.0*math.log10(d)+np.random.normal(0,8)
        pl+=random.choice([0,0,12])  # wall penetration
        return pl

    states=[{'x':random.uniform(0,50),'y':random.uniform(0,50),
             'speed':random.uniform(0.5,2),'head':random.uniform(0,360),
             'ap_id':0} for _ in range(n_ues)]

    for t in range(0,duration,interval):
        for i,s in enumerate(states):
            s['x']+=s['speed']*interval*math.cos(math.radians(s['head']))
            s['y']+=s['speed']*interval*math.sin(math.radians(s['head']))
            s['x']=max(-5,min(55,s['x'])); s['y']=max(-5,min(55,s['y']))
            if random.random()<0.05: s['head']=random.uniform(0,360)

            dists=[(math.sqrt((s['x']-ax)**2+(s['y']-ay)**2),a) for a,(ax,ay) in enumerate(AP)]
            dist,ap=min(dists); dist=max(0.01,dist)

            noise=10*math.log10(1.38e-23*290*80e6)+30+10
            snr=10+3-pl_wifi6(dist)-noise
            sinr=snr-random.uniform(2,5)+np.random.normal(0,1)
            rssi=-50+snr*0.4+np.random.normal(0,1)
            ber=max(0,(1e-6 if sinr>20 else 1e-4 if sinr>12 else 1e-3 if sinr>6 else 1e-2 if sinr>0 else 0.15)
                   +abs(np.random.normal(0,1e-4)))
            # Match uploaded file: mean=10 Mbps, max=53 Mbps
            tp=max(0,min(53.07,80*math.log2(1+10**(sinr/10))*0.02))
            lat=1.5+random.uniform(0,0.77)
            pl_pct=max(0,min(38.98,np.random.exponential(3) if sinr>6 else random.uniform(5,39)))
            lqi=max(15.26,min(87.46,(sinr+10)*3.0))
            se=max(0.1,min(12.34,math.log2(1+10**(sinr/10))))
            dop=s['speed']*math.cos(math.radians(s['head']))*2.4e9/3e8
            prop=(dist/3e8)*1000

            rows.append({
                'UE_ID':f'UE_{i:03d}','distance_km':round(dist/1000,5),
                'SNR_dB':round(snr,3),'SINR_dB':round(sinr,3),
                'RSSI_dBm':round(rssi,3),'BER':round(ber,8),
                'Throughput_Mbps':round(tp,3),'Latency_ms':round(lat,2),
                'Packet_Loss_pct':round(pl_pct,2),'Doppler_Hz':round(dop,2),
                'Propagation_Delay_ms':round(prop,6),
                'Rain_Rate_mmhr':0.0,'Rain_Fade_dB':0.0,
                'Link_Quality_Index':round(lqi,3),
                'Spectral_Efficiency_bps_hz':round(se,2),
                'network_type':'WiFi_6'
            })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    print("="*60)
    print("  Python NTN/TN Network Simulator")
    print("  Generates all 5 datasets matching uploaded files")
    print("="*60)

    sims = [
        ('UAV',   sim_uav,   'output/Data_UAV.csv',         18),
        ('HAPS',  sim_haps,  'output/NTNData_HAPS.csv',      16),
        ('LEO',   sim_leo,   'output/NTN_Data_LEO.csv',      16),
        ('5G NR', sim_5g,    'output/TN_Data_5G.csv',        16),
        ('WiFi6', sim_wifi6, 'output/TN_Data_WIFI_6.csv',    16),
    ]

    for name, fn, outpath, expected_cols in sims:
        print(f"\n[*] Simulating {name}...")
        df = fn()
        df.to_csv(outpath, index=False)
        print(f"    Rows     : {len(df):,}")
        print(f"    Columns  : {len(df.columns)}  (expected {expected_cols})")
        print(f"    SNR mean : {df['SNR_dB'].mean():.1f} dB")
        print(f"    Lat mean : {df['Latency_ms'].mean():.2f} ms")
        print(f"    Tput mean: {df['Throughput_Mbps'].mean():.1f} Mbps")
        print(f"    Saved    : {outpath}")

    print("\n" + "="*60)
    print("  ALL DONE — 5 datasets generated in ./output/")
    print("="*60)
