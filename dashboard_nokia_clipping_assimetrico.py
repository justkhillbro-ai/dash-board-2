import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 0. CONFIGURAZIONE PAGINA E COSTANTI
# ==========================================
st.set_page_config(page_title="Nokia FD - Caso 3 (FWA Asimmetrico)", layout="wide")

# Costanti di sistema
fc = 145e9      # 145 GHz
c = 3e8         # m/s
Gtx_dBi = 40    # dBi
Grx_dBi = 40    # dBi
T0 = 290        # K
k_B = 1.38e-23  # J/K
B_MHz = 1000    # 1 GHz Banda
B_Hz = B_MHz * 1e6
M = 16          # 16-QAM

# ==========================================
# 1. SIDEBAR: CONTROLLI INTERATTIVI
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/Nokia_wordmark.svg/1200px-Nokia_wordmark.svg.png", width=150)
st.sidebar.header("⚙️ Rete FWA Asimmetrica")

d_m = st.sidebar.slider("Distanza Link (m)", 100, 1000, 300, step=50)
iso_dB = st.sidebar.slider("Isolamento Antenna (dB)", 40, 80, 55, step=1)

st.sidebar.markdown("---")
st.sidebar.header("📡 Nodo 1: HUB (Premium)")
Ptx_Hub_dBm = st.sidebar.slider("Potenza TX Hub (dBm)", 0, 20, 10, step=1)
NF_Hub_dB = st.sidebar.number_input("Noise Figure Hub (dB)", value=6)
st.sidebar.caption("L'Hub usa un PA altamente lineare. Distorsione trascurabile.")

st.sidebar.markdown("---")
st.sidebar.header("🏠 Nodo 2: CPE (Low-Cost)")
Psat_CPE_dBm = st.sidebar.number_input("Potenza Saturazione CPE [dBm]", value=10)
IBO_CPE_dB = st.sidebar.slider("Input Back-Off CPE (IBO) [dB]", 0, 15, 4, step=1)
NF_CPE_dB = st.sidebar.number_input("Noise Figure CPE (dB)", value=10)
st.sidebar.caption("Il CPE subisce clipping se l'IBO è basso.")

# Calcoli base
Ptx_Hub_W = 10 ** ((Ptx_Hub_dBm - 30) / 10)
Ptx_CPE_dBm = Psat_CPE_dBm - IBO_CPE_dB
Ptx_CPE_W = 10 ** ((Ptx_CPE_dBm - 30) / 10)

FSPL_dB = 20 * np.log10(d_m) + 20 * np.log10(fc) + 20 * np.log10(4 * np.pi / c)
FSPL_lin = 10 ** (FSPL_dB / 10)
G_tot = 10 ** ((Gtx_dBi + Grx_dBi) / 10)

# Potenze Ricevute Utili
Prx_Hub_W = Ptx_CPE_W * G_tot / FSPL_lin  # Uplink
Prx_CPE_W = Ptx_Hub_W * G_tot / FSPL_lin  # Downlink

# Rumore Termico
Pn_Hub_W = k_B * T0 * B_Hz * (10 ** (NF_Hub_dB / 10))
Pn_CPE_W = k_B * T0 * B_Hz * (10 ** (NF_CPE_dB / 10))

# Interferenza da Leakage
Pi_Hub_W = 10 ** ((Ptx_Hub_dBm - iso_dB - 30) / 10)  # L'Hub interferisce se stesso
Pi_CPE_W = 10 ** ((Ptx_CPE_dBm - iso_dB - 30) / 10)  # Il CPE interferisce se stesso

# ==========================================
# SIMULAZIONE DISTORSIONE (Solo per il CPE)
# ==========================================
n_sym = 4000
np.random.seed(42)
I_ideal = np.random.choice([-3, -1, 1, 3], n_sym)
Q_ideal = np.random.choice([-3, -1, 1, 3], n_sym)
sym_tx = (I_ideal + 1j*Q_ideal)
sym_tx_norm = sym_tx / np.sqrt(np.mean(np.abs(sym_tx)**2))

def calc_dist_var_cpe(ibo):
    A_sat = 10 ** (-ibo / 20) 
    amp = np.abs(sym_tx_norm)
    clip_mask = amp > A_sat
    sym_clip = np.copy(sym_tx_norm)
    sym_clip[clip_mask] = sym_tx_norm[clip_mask] * (A_sat / amp[clip_mask])
    return np.mean(np.abs(sym_tx_norm - sym_clip)**2), sym_clip

P_dist_norm_CPE, sym_clip_CPE = calc_dist_var_cpe(IBO_CPE_dB)

# La distorsione viaggia nell'Uplink verso l'Hub:
P_dist_rx_Hub_W = P_dist_norm_CPE * Prx_Hub_W
# La distorsione trapela nel Downlink come interferenza locale del CPE:
P_dist_int_CPE_W = P_dist_norm_CPE * Pi_CPE_W

# ==========================================
# HEADER NARRATIVO
# ==========================================
st.title("⚖️ Caso 3: Asimmetria di Rete FWA (Hub vs CPE)")
st.markdown("Nelle reti **Fixed Wireless Access (FWA)** reali, la Base Station (Hub) ha hardware premium e lineare, mentre il router domestico (CPE) ha componenti a basso costo soggetti a **saturazione (clipping)** e maggiore rumore termico. Analizziamo l'impatto di questa asimmetria sulle prestazioni Full-Duplex.")
st.markdown("---")

# ==========================================
# SEZIONE 1: Costellazioni Asimmetriche
# ==========================================
st.header("1. Analisi Fisica dell'Asimmetria (Downlink vs Uplink)")
st.markdown("Nota la differenza: Il segnale in Downlink (ricevuto dal CPE) viaggia pulito ma è sommerso dal rumore del CPE stesso. Il segnale in Uplink (ricevuto dall'Hub) appare **schiacciato e compresso** perché viene trasmesso da un hardware (il CPE) già in saturazione.")

col1, col2 = st.columns(2)

# Costellazione DOWNLINK (Riceve il CPE dal PA pulito dell'Hub)
with col1:
    rx_points_dl = sym_tx_norm * np.sqrt(Prx_CPE_W) # Segnale utile pulito
    # Sommo rumore e interferenza (che include la distorsione del leakage)
    n_dl = (np.random.randn(n_sym) + 1j*np.random.randn(n_sym)) * np.sqrt((Pn_CPE_W + Pi_CPE_W + P_dist_int_CPE_W)/2)
    rx_points_dl_tot = (rx_points_dl + n_dl) / np.sqrt(Prx_CPE_W)
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=rx_points_dl_tot.real, y=rx_points_dl_tot.imag, mode='markers', marker=dict(size=3, color='rgba(31, 119, 180, 0.6)'), name="Ricevuti (CPE)"))
    I_ref = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    Q_ref = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    ref_pts = [complex(i, q) for i in I_ref for q in Q_ref]
    fig1.add_trace(go.Scatter(x=[p.real for p in ref_pts], y=[p.imag for p in ref_pts], mode='markers', marker=dict(symbol='cross', size=12, color='black'), showlegend=False))
    fig1.update_layout(title="Ricevitore CPE (Downlink)", xaxis_title="I", yaxis_title="Q", height=450, template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

# Costellazione UPLINK (Riceve l'Hub dal PA distorto del CPE)
with col2:
    rx_points_ul = sym_clip_CPE * np.sqrt(Prx_Hub_W) # Segnale utile distorto!
    # Sommo rumore e interferenza pulita dell'Hub
    n_ul = (np.random.randn(n_sym) + 1j*np.random.randn(n_sym)) * np.sqrt((Pn_Hub_W + Pi_Hub_W)/2)
    rx_points_ul_tot = (rx_points_ul + n_ul) / np.sqrt(Prx_Hub_W)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=rx_points_ul_tot.real, y=rx_points_ul_tot.imag, mode='markers', marker=dict(size=3, color='rgba(214, 39, 40, 0.6)'), name="Ricevuti (Hub)"))
    fig2.add_trace(go.Scatter(x=[p.real for p in ref_pts], y=[p.imag for p in ref_pts], mode='markers', marker=dict(symbol='cross', size=12, color='black'), showlegend=False))
    fig2.update_layout(title="Ricevitore HUB (Uplink)", xaxis_title="I", yaxis_title="Q", height=450, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ==========================================
# CALCOLI VETTORIALI PER GRAFICI DI SISTEMA
# ==========================================
ibo_vec = np.linspace(0, 15, 40)
cap_ul_vec, cap_dl_vec = [], []

for ibo in ibo_vec:
    ptx_cpe_temp = Psat_CPE_dBm - ibo
    ptx_cpe_w_temp = 10 ** ((ptx_cpe_temp - 30) / 10)
    
    prx_hub_w_t = ptx_cpe_w_temp * G_tot / FSPL_lin
    pi_cpe_w_t = 10 ** ((ptx_cpe_temp - iso_dB - 30) / 10)
    
    dist_norm_t, _ = calc_dist_var_cpe(ibo)
    
    # UPLINK (Hub riceve): Segnale utile con distorsione
    sndr_ul = prx_hub_w_t / (Pn_Hub_W + Pi_Hub_W + (dist_norm_t * prx_hub_w_t))
    cap_ul_vec.append((B_Hz * np.log2(1 + sndr_ul)) / 1e9)
    
    # DOWNLINK (CPE riceve): Interferenza locale con distorsione
    sndr_dl = Prx_CPE_W / (Pn_CPE_W + pi_cpe_w_t + (dist_norm_t * pi_cpe_w_t))
    cap_dl_vec.append((B_Hz * np.log2(1 + sndr_dl)) / 1e9)

# ==========================================
# SEZIONE 2: Capacità vs IBO CPE
# ==========================================
st.header("2. L'Effetto dell'IBO Domestico sull'Intera Rete")
st.markdown("Variando l'IBO del router di casa, notiamo un fenomeno affascinante. Abbassare l'IBO distrugge l'Uplink (rosso) a causa del segnale distorto trasmesso verso l'Hub, ma **migliora il Downlink** (blu) perché il CPE, trasmettendo meno potenza per colpa della saturazione, si auto-interferisce di meno!")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=ibo_vec, y=cap_ul_vec, mode='lines', name="Uplink (CPE ➔ Hub)", line=dict(color='red', width=3)))
fig3.add_trace(go.Scatter(x=ibo_vec, y=cap_dl_vec, mode='lines', name="Downlink (Hub ➔ CPE)", line=dict(color='blue', width=3)))
# Aggiungiamo la capacità totale come metrica combinata
fig3.add_trace(go.Scatter(x=ibo_vec, y=np.array(cap_ul_vec)+np.array(cap_dl_vec), mode='lines', name="Capacità Totale (UL + DL)", line=dict(color='green', width=3, dash='dash')))

fig3.add_vline(x=IBO_CPE_dB, line=dict(color="gray", dash="dot"), annotation_text="IBO Attuale")
fig3.update_layout(xaxis_title="Input Back-Off del CPE (dB)", yaxis_title="Capacità Netta (Gbps)", height=500, template="plotly_white", hovermode="x unified")
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ==========================================
# SEZIONE 3: Capacità vs Isolamento
# ==========================================
st.header("3. Asimmetria del Link: Il divario tra Uplink e Downlink")
st.markdown("In uno scenario FWA reale con CPE a basso costo, la rete sarà inevitabilmente sbilanciata. L'Uplink saturerà a livelli molto più bassi rispetto al Downlink a causa dei limiti fisici del trasmettitore economico. Per Nokia, questo giustifica l'uso del FD per fornire servizi asimmetrici (es. molto download, poco upload) tipici degli utenti residenziali.")

iso_array = np.linspace(40, 80, 40)
c_ul_iso, c_dl_iso = [], []

for iso_val in iso_array:
    pi_h_t = 10 ** ((Ptx_Hub_dBm - iso_val - 30) / 10)
    pi_c_t = 10 ** ((Ptx_CPE_dBm - iso_val - 30) / 10)
    
    sndr_u = Prx_Hub_W / (Pn_Hub_W + pi_h_t + P_dist_rx_Hub_W)
    c_ul_iso.append((B_Hz * np.log2(1 + sndr_u)) / 1e9)
    
    dist_int = P_dist_norm_CPE * pi_c_t
    sndr_d = Prx_CPE_W / (Pn_CPE_W + pi_c_t + dist_int)
    c_dl_iso.append((B_Hz * np.log2(1 + sndr_d)) / 1e9)

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=iso_array, y=c_dl_iso, mode='lines', name="Downlink Capacity (Hub ➔ CPE)", line=dict(color='blue', width=3)))
fig4.add_trace(go.Scatter(x=iso_array, y=c_ul_iso, mode='lines', name="Uplink Capacity (CPE ➔ Hub)", line=dict(color='red', width=3)))

fig4.update_layout(xaxis_title="Isolamento Antenna (dB)", yaxis_title="Capacità di Canale (Gbps)", height=500, template="plotly_white")
fig4.add_vline(x=iso_dB, line=dict(color="gray", dash="dot"), annotation_text="Isolamento Attuale")
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ==========================================
# SEZIONE 4: Heatmap Ottimizzazione
# ==========================================
st.header("4. Heatmap di Ottimizzazione Rete (Capacità Aggregata Totale)")
st.markdown("Questa è la mappa del tesoro per gli ingegneri di sistema. Mostra la **Capacità Totale (Uplink + Downlink)** in base all'isolamento dell'antenna e all'IBO del modem domestico. Permette di calibrare perfettamente il router prima di venderlo al cliente, bilanciando il guadagno con il costo dei componenti RF.")

Ibo_m, Iso_m = np.meshgrid(np.linspace(0, 15, 30), np.linspace(40, 80, 30))

# Vettorizzazione veloce
dist_v = np.array([calc_dist_var_cpe(i)[0] for i in np.linspace(0, 15, 30)])
P_dist_m = np.interp(Ibo_m, np.linspace(0, 15, 30), dist_v)

Ptx_CPE_m = 10 ** ((Psat_CPE_dBm - Ibo_m - 30) / 10)

Prx_Hub_m = Ptx_CPE_m * G_tot / FSPL_lin
Pi_Hub_m = 10 ** ((Ptx_Hub_dBm - Iso_m - 30) / 10)
SNDR_UL_m = Prx_Hub_m / (Pn_Hub_W + Pi_Hub_m + (P_dist_m * Prx_Hub_m))
Cap_UL_m = (B_Hz * np.log2(1 + SNDR_UL_m)) / 1e9

Pi_CPE_m = Ptx_CPE_m * (10 ** (-Iso_m / 10))
SNDR_DL_m = Prx_CPE_W / (Pn_CPE_W + Pi_CPE_m + (P_dist_m * Pi_CPE_m))
Cap_DL_m = (B_Hz * np.log2(1 + SNDR_DL_m)) / 1e9

Total_Cap_m = Cap_UL_m + Cap_DL_m

fig5 = go.Figure(data=go.Heatmap(
    z=Total_Cap_m, x=np.linspace(0, 15, 30), y=np.linspace(40, 80, 30), 
    colorscale='Viridis', colorbar=dict(title="Tot Gbps")
))
fig5.update_layout(xaxis_title="IBO del CPE (dB)", yaxis_title="Isolamento Antenna (dB)", height=600, template="plotly_white")
fig5.add_trace(go.Scatter(x=[IBO_CPE_dB], y=[iso_dB], mode='markers+text', text=["Punto Operativo"], marker=dict(size=12, color='white', symbol='diamond'), textposition="top center", showlegend=False))
st.plotly_chart(fig5, use_container_width=True)