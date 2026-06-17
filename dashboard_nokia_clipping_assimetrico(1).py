import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 0. PAGE CONFIGURATION AND CONSTANTS
# ==========================================
st.set_page_config(page_title="Nokia FD - Case 3 (Asymmetric FWA)", layout="wide")

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
# 1. SIDEBAR: INTERACTIVE CONTROLS
# ==========================================
st.sidebar.header("Asymmetric FWA Network")

d_m = st.sidebar.slider("Link Distance (m)", 100, 1000, 300, step=50)
iso_dB = st.sidebar.slider("Antenna Isolation (dB)", 40, 80, 55, step=1)

st.sidebar.markdown("---")
st.sidebar.header("Node 1: HUB (Premium)")
Ptx_Hub_dBm = st.sidebar.slider("Hub TX Power (dBm)", 0, 20, 10, step=1)
NF_Hub_dB = st.sidebar.number_input("Hub Noise Figure (dB)", value=6)
st.sidebar.caption("The Hub utilizes a highly linear PA. Negligible distortion.")

st.sidebar.markdown("---")
st.sidebar.header("Node 2: CPE (Low-Cost)")
Psat_CPE_dBm = st.sidebar.number_input("CPE Saturation Power (Psat, dBm)", value=10)
IBO_CPE_dB = st.sidebar.slider("CPE Input Back-Off (IBO, dB)", 0, 15, 4, step=1)
NF_CPE_dB = st.sidebar.number_input("CPE Noise Figure (dB)", value=10)
st.sidebar.caption("The CPE suffers clipping at low IBO.")

# Calcoli base Link Budget
Ptx_Hub_W = 10 ** ((Ptx_Hub_dBm - 30) / 10)
Ptx_CPE_dBm = Psat_CPE_dBm - IBO_CPE_dB
Ptx_CPE_W = 10 ** ((Ptx_CPE_dBm - 30) / 10)

# Mostriamo dinamicamente la potenza TX calcolata del CPE
st.sidebar.info(f"**Calculated CPE TX Power:** {Ptx_CPE_dBm} dBm")

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

# Normalizzazione deterministica esatta per 16-QAM
I_ideal = np.random.choice([-3, -1, 1, 3], n_sym) / np.sqrt(10)
Q_ideal = np.random.choice([-3, -1, 1, 3], n_sym) / np.sqrt(10)
sym_tx_norm = (I_ideal + 1j*Q_ideal)

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
# NARRATIVE HEADER
# ==========================================
st.title("Case 3: FWA Network Asymmetry (Hub vs. CPE)")
st.markdown("In real-world **Fixed Wireless Access (FWA)** deployments, the Base Station (Hub) relies on premium, highly linear hardware, whereas the home router (CPE) utilizes low-cost components prone to **saturation (clipping)** and higher thermal noise. Let's analyze how this hardware asymmetry impacts Full-Duplex performance.")
st.markdown("---")

# ==========================================
# SEZIONE 1: Costellazioni Asimmetriche
# ==========================================
st.header("1. Physical Analysis of Asymmetry (Downlink vs. Uplink)")
st.markdown("Notice the physical difference at the receiver. The **Downlink** signal (received by the CPE) travels cleanly but is degraded by the CPE's high internal noise. Conversely, the **Uplink** signal (received by the Hub) appears radially **squashed and compressed** because it was transmitted by a CPE already operating in saturation.")

col1, col2 = st.columns(2)

# Punti ideali di riferimento comuni
I_ref = np.array([-3, -1, 1, 3]) / np.sqrt(10)
Q_ref = np.array([-3, -1, 1, 3]) / np.sqrt(10)
ref_pts = [complex(i, q) for i in I_ref for q in Q_ref]

# Costellazione DOWNLINK (Riceve il CPE dal PA pulito dell'Hub)
with col1:
    rx_points_dl = sym_tx_norm * np.sqrt(Prx_CPE_W) 
    # Sommo rumore e interferenza (inclusa la distorsione del leakage locale)
    n_dl = (np.random.randn(n_sym) + 1j*np.random.randn(n_sym)) * np.sqrt((Pn_CPE_W + Pi_CPE_W + P_dist_int_CPE_W)/2)
    rx_points_dl_tot = (rx_points_dl + n_dl) / np.sqrt(Prx_CPE_W)
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=rx_points_dl_tot.real, y=rx_points_dl_tot.imag, mode='markers', 
                              marker=dict(size=3, color='rgba(31, 119, 180, 0.5)'), 
                              name="Received (CPE)"))
    
    fig1.add_trace(go.Scatter(x=[p.real for p in ref_pts], y=[p.imag for p in ref_pts], mode='markers', 
                              marker=dict(symbol='cross', size=10, color='black'), 
                              showlegend=False))
    
    fig1.update_layout(
        title="CPE Receiver (Downlink)", 
        xaxis_title="In-Phase (I)", 
        yaxis_title="Quadrature (Q)", 
        height=500, 
        template="plotly_white",
        yaxis=dict(scaleanchor="x", scaleratio=1, zeroline=True, zerolinewidth=1.5, zerolinecolor='black'),
        xaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor='black'),
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

# Costellazione UPLINK (Riceve l'Hub dal PA distorto del CPE)
with col2:
    rx_points_ul = sym_clip_CPE * np.sqrt(Prx_Hub_W) 
    # Sommo rumore e interferenza pulita dell'Hub locale
    n_ul = (np.random.randn(n_sym) + 1j*np.random.randn(n_sym)) * np.sqrt((Pn_Hub_W + Pi_Hub_W)/2)
    rx_points_ul_tot = (rx_points_ul + n_ul) / np.sqrt(Prx_Hub_W)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=rx_points_ul_tot.real, y=rx_points_ul_tot.imag, mode='markers', 
                              marker=dict(size=3, color='rgba(214, 39, 40, 0.5)'), 
                              name="Received (Hub)"))
    
    fig2.add_trace(go.Scatter(x=[p.real for p in ref_pts], y=[p.imag for p in ref_pts], mode='markers', 
                              marker=dict(symbol='cross', size=10, color='black'), 
                              showlegend=False))
    
    fig2.update_layout(
        title="HUB Receiver (Uplink)", 
        xaxis_title="In-Phase (I)", 
        yaxis_title="Quadrature (Q)", 
        height=500, 
        template="plotly_white",
        yaxis=dict(scaleanchor="x", scaleratio=1, zeroline=True, zerolinewidth=1.5, zerolinecolor='black'),
        xaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor='black'),
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ==========================================
# VECTORIZED SYSTEM CALCULATIONS
# ==========================================
ibo_vec = np.linspace(0, 15, 40)

# Vettorizzazione veloce della distorsione
dist_norm_vec = np.array([calc_dist_var_cpe(i)[0] for i in ibo_vec])

ptx_cpe_temp_vec = Psat_CPE_dBm - ibo_vec
ptx_cpe_w_temp_vec = 10 ** ((ptx_cpe_temp_vec - 30) / 10)

prx_hub_w_t_vec = ptx_cpe_w_temp_vec * G_tot / FSPL_lin
pi_cpe_w_t_vec = 10 ** ((ptx_cpe_temp_vec - iso_dB - 30) / 10)

# UPLINK (Hub riceve): Segnale utile attenuato + distorsione
sndr_ul_vec = prx_hub_w_t_vec / (Pn_Hub_W + Pi_Hub_W + (dist_norm_vec * prx_hub_w_t_vec))
cap_ul_vec = (B_Hz * np.log2(1 + sndr_ul_vec)) / 1e9

# DOWNLINK (CPE riceve): Interferenza locale + distorsione
sndr_dl_vec = Prx_CPE_W / (Pn_CPE_W + pi_cpe_w_t_vec + (dist_norm_vec * pi_cpe_w_t_vec))
cap_dl_vec = (B_Hz * np.log2(1 + sndr_dl_vec)) / 1e9

# Capacità Aggregata
cap_tot_vec = cap_ul_vec + cap_dl_vec

# ==========================================
# SEZIONE 2: Capacità vs IBO CPE
# ==========================================
st.header("2. System-Wide Impact of the CPE's IBO")
st.markdown("Sweeping the home router's IBO reveals the true physical trade-off of asymmetric Full-Duplex. **Increasing the IBO** (moving right) forces the CPE to reduce its transmit power. This causes the **Uplink to collapse** (red curve) as the signal drowns in thermal noise at the Hub. However, transmitting with less power drastically reduces the CPE's local self-interference: as a result, the **Downlink skyrockets** (blue curve).")
st.markdown("The **Aggregate Capacity** (dashed green line) proves a counter-intuitive point: in this setup, choking the Uplink to clear the Downlink from local leakage actually maximizes the total network throughput.")

fig3 = go.Figure()

fig3.add_trace(go.Scatter(x=ibo_vec, y=cap_ul_vec, mode='lines', 
                          name="Uplink (CPE ➔ Hub)", 
                          line=dict(color='#d62728', width=3),
                          hovertemplate="Uplink: %{y:.2f} Gbps<extra></extra>"))

fig3.add_trace(go.Scatter(x=ibo_vec, y=cap_dl_vec, mode='lines', 
                          name="Downlink (Hub ➔ CPE)", 
                          line=dict(color='#1f77b4', width=3),
                          hovertemplate="Downlink: %{y:.2f} Gbps<extra></extra>"))

fig3.add_trace(go.Scatter(x=ibo_vec, y=cap_tot_vec, mode='lines', 
                          name="Aggregate Capacity (UL + DL)", 
                          line=dict(color='#2ca02c', width=3, dash='dash'),
                          hovertemplate="Total: %{y:.2f} Gbps<extra></extra>"))

fig3.add_vline(x=IBO_CPE_dB, line=dict(color="gray", dash="dot"), annotation_text="Current CPE IBO", annotation_position="bottom right")

fig3.update_layout(
    xaxis_title="CPE Input Back-Off (IBO) [dB]", 
    yaxis_title="Net Capacity (Gbps)", 
    height=500, 
    template="plotly_white", 
    hovermode="x unified"
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ==========================================
# SEZIONE 3: Capacità vs Isolamento
# ==========================================
st.header("3. Link Asymmetry: The Uplink vs. Downlink Gap")
st.markdown("In a real-world FWA scenario utilizing a low-cost CPE, the network becomes inherently unbalanced. Due to the physical limitations of the cheaper transmitter, the **Uplink saturates much earlier** and provides lower capacity than the Downlink. For Nokia, this physical reality perfectly justifies targeting Full-Duplex at residential users, whose traffic profiles are naturally asymmetric (heavy download, light upload).")

iso_array = np.linspace(40, 80, 50)

# Calcoli Vettorializzati (Senza ciclo for)
pi_h_array = 10 ** ((Ptx_Hub_dBm - iso_array - 30) / 10)
pi_c_array = 10 ** ((Ptx_CPE_dBm - iso_array - 30) / 10)

# Uplink (Hub riceve il segnale distorto dal CPE)
sndr_u_array = Prx_Hub_W / (Pn_Hub_W + pi_h_array + P_dist_rx_Hub_W)
c_ul_iso = (B_Hz * np.log2(1 + sndr_u_array)) / 1e9

# Downlink (CPE riceve il segnale pulito ma subisce il proprio leakage distorto)
dist_int_array = P_dist_norm_CPE * pi_c_array
sndr_d_array = Prx_CPE_W / (Pn_CPE_W + pi_c_array + dist_int_array)
c_dl_iso = (B_Hz * np.log2(1 + sndr_d_array)) / 1e9

fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=iso_array, 
    y=c_dl_iso, 
    mode='lines', 
    name="Downlink (Hub ➔ CPE)", 
    line=dict(color='#1f77b4', width=3),
    hovertemplate="Downlink: %{y:.2f} Gbps<extra></extra>"
))

fig4.add_trace(go.Scatter(
    x=iso_array, 
    y=c_ul_iso, 
    mode='lines', 
    name="Uplink (CPE ➔ Hub)", 
    line=dict(color='#d62728', width=3),
    hovertemplate="Uplink: %{y:.2f} Gbps<extra></extra>"
))

fig4.update_layout(
    xaxis_title="Antenna Isolation (dB)", 
    yaxis_title="Channel Capacity (Gbps)", 
    height=500, 
    template="plotly_white",
    hovermode="x unified"
)

fig4.add_vline(x=iso_dB, line=dict(color="gray", dash="dot"), annotation_text="Current Isolation", annotation_position="top left")

st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ==========================================
# SEZIONE 4: Heatmap Ottimizzazione
# ==========================================
st.header("4. Network Optimization Heatmap (Total Aggregate Capacity)")
st.markdown("This is the ultimate system engineering map. It displays the **Total Capacity (Uplink + Downlink)** as a function of antenna isolation and the home modem's IBO. It allows Nokia to perfectly calibrate the router prior to deployment, balancing net throughput against the cost of RF components.")

Ibo_m, Iso_m = np.meshgrid(np.linspace(0, 15, 30), np.linspace(40, 80, 30))

# Fast vectorization for the UI
dist_v = np.array([calc_dist_var_cpe(i)[0] for i in np.linspace(0, 15, 30)])
P_dist_m = np.interp(Ibo_m, np.linspace(0, 15, 30), dist_v)

Ptx_CPE_m = 10 ** ((Psat_CPE_dBm - Ibo_m - 30) / 10)

# Matrix Link Budget
Prx_Hub_m = Ptx_CPE_m * G_tot / FSPL_lin
Pi_Hub_m = 10 ** ((Ptx_Hub_dBm - Iso_m - 30) / 10)
SNDR_UL_m = Prx_Hub_m / (Pn_Hub_W + Pi_Hub_m + (P_dist_m * Prx_Hub_m))
Cap_UL_m = (B_Hz * np.log2(1 + SNDR_UL_m)) / 1e9

Pi_CPE_m = Ptx_CPE_m * (10 ** (-Iso_m / 10))
SNDR_DL_m = Prx_CPE_W / (Pn_CPE_W + Pi_CPE_m + (P_dist_m * Pi_CPE_m))
Cap_DL_m = (B_Hz * np.log2(1 + SNDR_DL_m)) / 1e9

Total_Cap_m = Cap_UL_m + Cap_DL_m

fig5 = go.Figure(data=go.Heatmap(
    z=Total_Cap_m, 
    x=np.linspace(0, 15, 30), 
    y=np.linspace(40, 80, 30), 
    colorscale='Viridis', 
    colorbar=dict(title="Total (Gbps)"),
    hovertemplate="<b>CPE IBO:</b> %{x:.1f} dB<br><b>Isolation:</b> %{y:.1f} dB<br><b>Total Capacity:</b> %{z:.2f} Gbps<extra></extra>"
))

fig5.update_layout(
    xaxis_title="CPE Input Back-Off (IBO) [dB]", 
    yaxis_title="Antenna Isolation (dB)", 
    height=600, 
    template="plotly_white"
)

# Marker per identificare il punto di lavoro attuale
fig5.add_trace(go.Scatter(
    x=[IBO_CPE_dB], 
    y=[iso_dB], 
    mode='markers+text', 
    text=["Operating Point"], 
    marker=dict(size=12, color='white', symbol='diamond', line=dict(color='black', width=1.5)), 
    textposition="top center", 
    showlegend=False,
    hoverinfo="skip"
))

st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")
st.link_button( "Dashboard 1",  "https://nokia-band-analysis.streamlit.app/")
st.link_button( "Dashboard 2",  "https://dash-board-3-simmetrico.streamlit.app/")