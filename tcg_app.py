import streamlit as st
import pandas as pd
import math
from PIL import Image

@st.cache_data
def safe_comb(n, k):
    if k < 0 or n < k or n < 0:
        return 0
    try:
        return math.comb(n, k)
    except ValueError:
        return 0

@st.cache_data
def calculate_single_prob(K, N, n):
    if n == 0:
        return 0.0
    if K < 0:
        K = 0
    if K > N:
        K = N
        
    total_combinations = safe_comb(N, n)
    if total_combinations == 0:
        return 0.0

    num_non_starters = N - K
    ways_to_draw_zero_starters = safe_comb(num_non_starters, n)
        
    prob_zero_starters = ways_to_draw_zero_starters / total_combinations
    return 1.0 - prob_zero_starters

@st.cache_data
def calculate_exact_prob(i, K, N, n):
    if n == 0:
        return 0.0
    if K < i:
        return 0.0
    
    total_combinations = safe_comb(N, n)
    if total_combinations == 0:
        return 0.0
    
    non_starters = N - K
    draw_non_starters = n - i
    if draw_non_starters < 0:
        return 0.0
    
    ways_to_draw_exact = safe_comb(K, i) * safe_comb(non_starters, draw_non_starters)
    
    return ways_to_draw_exact / total_combinations

@st.cache_data
def get_starter_probability_data(N, n):
    
    plot_k_col = list(range(N + 1))
    
    P_values_full_ge_1 = [calculate_single_prob(k, N, n) for k in range(-1, N + 2)]
    plot_p_ge_1 = P_values_full_ge_1[1 : N + 2]
    
    plot_p_eq_2 = [calculate_exact_prob(2, k, N, n) for k in plot_k_col]
    plot_p_eq_3 = [calculate_exact_prob(3, k, N, n) for k in plot_k_col]
    
    df_plot = pd.DataFrame({
        "K (Starters)": plot_k_col,
        "P(X >= 1)": plot_p_ge_1,
        "P(X = 2)": plot_p_eq_2,
        "P(X = 3)": plot_p_eq_3
    }).set_index("K (Starters)")

    table_K_col = list(range(1, N + 1))
    table_P_col = P_values_full_ge_1[2 : N + 2]
    table_D_col = [P_values_full_ge_1[i+2] - P_values_full_ge_1[i+1] for i in range(len(table_K_col))]
    table_C_col = [P_values_full_ge_1[i+3] - 2*P_values_full_ge_1[i+2] + P_values_full_ge_1[i+1] for i in range(len(table_K_col))]

    df_table = pd.DataFrame({
        "K (Starters)": table_K_col,
        f"P(X>=1) (N={N}, n={n})": table_P_col,
        "P(K) - P(K-1) (Marginal)": table_D_col,
        "P(K+1)-2P(K)+P(K-1) (Curvature)": table_C_col
    }).set_index("K (Starters)") 
    
    return df_plot, df_table

def calculate_combo_prob_single(A, D, n, K_fixed, total_comb, comb_not_K):
    if A < 0:
        return 0.0
    if total_comb == 0:
        return 0.0
    comb_not_A = safe_comb(D - A, n)
    comb_not_A_and_not_K = safe_comb(D - K_fixed - A, n)
    prob_A_is_0_or_K_is_0_num = (comb_not_A + comb_not_K - comb_not_A_and_not_K)
    prob_A_is_0_or_K_is_0 = prob_A_is_0_or_K_is_0_num / total_comb
    return 1.0 - prob_A_is_0_or_K_is_0

@st.cache_data
def get_combo_probability_data(D, n, K_fixed):
    max_A = D - K_fixed
    total_comb = safe_comb(D, n)
    comb_not_K = safe_comb(D - K_fixed, n)
    P_values_full = [calculate_combo_prob_single(a, D, n, K_fixed, total_comb, comb_not_K) for a in range(-1, max_A + 2)]

    plot_A_col = list(range(max_A + 1))
    plot_P_col = P_values_full[1 : max_A + 2]
    df_plot = pd.DataFrame({
        "A (Insecticides)": plot_A_col,
        "Probability": plot_P_col
    }).set_index("A (Insecticides)")

    table_A_col = list(range(max_A + 1))
    table_P_col = P_values_full[1 : max_A + 2]
    table_D_col = [P_values_full[i+2] - P_values_full[i+1] for i in range(len(table_A_col))]
    table_C_col = [P_values_full[i+2] - 2*P_values_full[i+1] + P_values_full[i] for i in range(len(table_A_col))]
    
    df_table = pd.DataFrame({
        "A (Insecticides)": table_A_col,
        "Probability": table_P_col,
        "P(A+1) - P(A) (Marginal)": table_D_col,
        "P(A+1)-2P(A)+P(A-1) (Curvature)": table_C_col
    }).set_index("A (Insecticides)")
    return df_plot, df_table

@st.cache_data
def calculate_part3_prob_single(NE, D, K_fixed, i):
    n = 5
    n_plus_1 = 6
    
    if K_fixed == 0 and i != 7:
        return 0.0 
    if NE < i:
        return 0.0
    
    Trash = D - K_fixed - NE
    if Trash < 0:
        return 0.0
        
    total_deck_draw_5 = safe_comb(D, n)
    if total_deck_draw_5 == 0:
        return 0.0
    
    non_NE_cards = D - NE
    K_plus_Trash = D - NE
    P_A_i_num = safe_comb(NE, i) * safe_comb(non_NE_cards, n - i)
    P_A_i = P_A_i_num / total_deck_draw_5
    if P_A_i == 0:
        return 0.0

    P_k5_is_0_num = safe_comb(Trash, n - i)
    P_k5_is_0_den = safe_comb(K_plus_Trash, n - i)
    if P_k5_is_0_den == 0:
        P_k5_is_0 = 1.0 if P_k5_is_0_num == 0 else 0.0
    else:
        P_k5_is_0 = P_k5_is_0_num / P_k5_is_0_den

    P_6th_not_K_num = (D - n) - K_fixed
    P_6th_not_K_den = D - n
    if P_6th_not_K_den == 0:
        P_6th_not_K = 1.0
    else:
        P_6th_not_K = P_6th_not_K_num / P_6th_not_K_den

    P_NOT_B_given_A_i = P_k5_is_0 * P_6th_not_K
    
    P_B_given_A_i = 1.0 - P_NOT_B_given_A_i
    
    final_prob = P_A_i * P_B_given_A_i
    
    if i == 7:
        P_A_5_num = safe_comb(NE, 5) * safe_comb(non_NE_cards, 0)
        P_A_5 = P_A_5_num / total_deck_draw_5
        if P_A_5 == 0:
            return 0.0
            
        P_k5_is_0_num_c7 = safe_comb(Trash, 0)
        P_k5_is_0_den_c7 = safe_comb(K_plus_Trash, 0)
        P_k5_is_0_c7 = 1.0
        
        P_NOT_B_given_A_5 = P_k5_is_0_c7 * P_6th_not_K
        
        return P_A_5 * P_NOT_B_given_A_5

    return final_prob

@st.cache_data
def get_part3_data(D, K_fixed):
    max_NE = D - K_fixed
    
    P_full = [[] for _ in range(8)]

    for ne_val in range(-1, max_NE + 2):
        for i in range(0, 5):
            P_full[i].append(calculate_part3_prob_single(ne_val, D, K_fixed, i))
        P_full[5].append(calculate_part3_prob_single(ne_val, D, K_fixed, 5))
        P_full[7].append(calculate_part3_prob_single(ne_val, D, K_fixed, 7))

    plot_NE_col = list(range(max_NE + 1))
    df_plot = pd.DataFrame({"NE (Non-Engine)": plot_NE_col})
    df_plot[f"C0 (i=0 NE)"] = P_full[0][1 : max_NE + 2]
    df_plot[f"C1 (i=1 NE)"] = P_full[1][1 : max_NE + 2]
    df_plot[f"C2 (i=2 NE)"] = P_full[2][1 : max_NE + 2]
    df_plot[f"C3 (i=3 NE)"] = P_full[3][1 : max_NE + 2]
    df_plot[f"C4 (i=4 NE)"] = P_full[4][1 : max_NE + 2]
    df_plot[f"C6 (i=5 NE, >=1 K)"] = P_full[5][1 : max_NE + 2]
    df_plot[f"C7 (i=5 NE, 0 K)"] = P_full[7][1 : max_NE + 2]
    
    df_plot = df_plot.set_index("NE (Non-Engine)")

    all_tables = []
    curve_names = [
        "C0: P(0 NE in 5, >=1 K in 6)",
        "C1: P(1 NE in 5, >=1 K in 6)",
        "C2: P(2 NE in 5, >=1 K in 6)",
        "C3: P(3 NE in 5, >=1 K in 6)",
        "C4: P(4 NE in 5, >=1 K in 6)",
        "C6: P(5 NE in 5, >=1 K in 6)",
        "C7: P(5 NE in 5, 0 K in 6)"
    ]
    
    for i_curve_internal in [0, 1, 2, 3, 4, 5, 7]:
        
        table_NE_col = list(range(max_NE + 1))
        P_curve = P_full[i_curve_internal]
        
        table_P_col = P_curve[1 : max_NE + 2]
        table_D_col = [P_curve[j+2] - P_curve[j+1] for j in range(len(table_NE_col))]
        table_C_col = [P_curve[j+2] - 2*P_curve[j+1] + P_curve[j] for j in range(len(table_NE_col))]
        
        df_table = pd.DataFrame({
            "NE (Non-Engine)": table_NE_col,
            "Probability": table_P_col,
            "Marginal": table_D_col,
            "Curvature": table_C_col
        })
        df_table = df_table.set_index("NE (Non-Engine)")
        
        df_display = df_table.copy()
        df_display["Probability"] = df_display["Probability"].map('{:.4%}'.format)
        df_display["Marginal"] = df_display["Marginal"].map('{:+.4%}'.format)
        df_display["Curvature"] = df_display["Curvature"].map('{:+.4%}'.format)
        
        table_name = curve_names[5] if i_curve_internal == 5 else (curve_names[6] if i_curve_internal == 7 else curve_names[i_curve_internal])
        all_tables.append((table_name, df_display))

    return df_plot, all_tables


st.set_page_config(layout="wide")

st.sidebar.markdown("Made by mikhaElise")

try:
    img = Image.open("avatar.png") 
    target_width = 150
    w_percent = (target_width / float(img.size[0]))
    target_height = int((float(img.size[1]) * float(w_percent)))
    img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    st.sidebar.image(img_resized)
except FileNotFoundError:
    st.sidebar.caption("avatar.png not found. (Place it in the same folder as the script)")
except Exception as e:
    st.sidebar.error(f"Error loading image: {e}")

st.sidebar.markdown("Bilibili: https://b23.tv/9aM3G4T")
st.sidebar.header("Parameters")

DECK_SIZE = st.sidebar.number_input(
    "1. Total Deck Size (D)", 
    min_value=40, 
    max_value=60, 
    value=60,
    step=1,
    help="Set your total deck size (40-60)"
)
HAND_SIZE = st.sidebar.number_input(
    "2. Opening Hand Size (n)", 
    min_value=0, 
    max_value=10, 
    value=5,
    step=1,
    help="Set your opening hand size (0-10). Note: Part 3 calculations are hard-coded for opening 5, drawing 1."
)
STARTER_COUNT_K = st.sidebar.number_input(
    "3. Fixed Starter Count (K)",
    min_value=0,
    max_value=60,
    value=24,
    step=1,
    help="Set the FIXED number of starters (K) for Part 2 and Part 3 calculations."
)

st.title("TCG Opening Hand Probability Calculator")
st.write(f"Current Settings: **{DECK_SIZE}** Card Deck, **{HAND_SIZE}** Card Hand")
st.caption(f"Part 2 & 3 Fixed Starter Count (K) = **{STARTER_COUNT_K}**")


st.header("Part 1: P(At least 1 Starter)")
st.write("This chart shows the probability of drawing at least one 'Starter' card (K), as K (the X-axis) increases.")
df_plot_1, df_table_1 = get_starter_probability_data(DECK_SIZE, HAND_SIZE)
st.line_chart(df_plot_1)
st.header(f"📊 Probability Table (K=1 to {DECK_SIZE})")
df_display_1 = df_table_1.copy()
prob_col_name_1 = f"P(X>=1) (N={DECK_SIZE}, n={HAND_SIZE})"
df_display_1[prob_col_name_1] = df_display_1[prob_col_name_1].map('{:.4%}'.format)
df_display_1["P(K) - P(K-1) (Marginal)"] = df_display_1["P(K) - P(K-1) (Marginal)"].map('{:+.4%}'.format)
df_display_1["P(K+1)-2P(K)+P(K-1) (Curvature)"] = df_display_1["P(K+1)-2P(K)+P(K-1) (Curvature)"].map('{:+.4%}'.format)
st.dataframe(df_display_1, use_container_width=True, height=300) 


st.divider()
st.header("Part 2: P(At least 1 Starter AND At least 1 'Insecticide')")
st.write(f"This chart uses the Fixed Starter (K) count of **{STARTER_COUNT_K}** and shows how the probability changes as the 'Insecticide' (A) count (the X-axis) increases.")
st.caption("⚠️ Assumption: This calculation assumes 'Starters' (K) and 'Insecticides' (A) are separate, non-overlapping sets of cards.")

if STARTER_COUNT_K >= DECK_SIZE:
    st.error(f"Error: Fixed Starter Count (K={STARTER_COUNT_K}) must be less than Total Deck Size (D={DECK_SIZE}).")
else:
    df_plot_2, df_table_2 = get_combo_probability_data(DECK_SIZE, HAND_SIZE, STARTER_COUNT_K)
    st.line_chart(df_plot_2)
    st.header(f"📊 Probability Table (A=0 to {DECK_SIZE - STARTER_COUNT_K})")
    df_display_2 = df_table_2.copy()
    df_display_2["Probability"] = df_display_2["Probability"].map('{:.4%}'.format)
    df_display_2["P(A+1) - P(A) (Marginal)"] = df_display_2["P(A+1) - P(A) (Marginal)"].map('{:+.4%}'.format)
    df_display_2["P(A+1)-2P(A)+P(A-1) (Curvature)"] = df_display_2["P(A+1)-2P(A)+P(A-1) (Curvature)"].map('{:+.4%}'.format)
    
    st.dataframe(df_display_2, use_container_width=True, height=300)


st.divider()
st.header("Part 3: P(Draw `i` Non-Engine in 5 AND >= 1 Starter in 6)")
st.write(f"This chart uses the Fixed Starter (K) count of **{STARTER_COUNT_K}**. The X-axis is the **Non-Engine (NE) count**.")
st.write(f"Deck = `{STARTER_COUNT_K}` (K) + `X-axis` (NE) + `Remainder` (Trash)")
st.caption("Note: Calculations for this part are hard-coded for an opening hand of 5, drawing a 6th card.")

if STARTER_COUNT_K >= DECK_SIZE:
    st.error(f"Error: Fixed Starter Count (K={STARTER_COUNT_K}) must be less than Total Deck Size (D={DECK_SIZE}).")
else:
    max_NE = DECK_SIZE - STARTER_COUNT_K
    df_plot_3, all_tables_3 = get_part3_data(DECK_SIZE, STARTER_COUNT_K)
    
    st.line_chart(df_plot_3)
    
    st.header(f"📊 Probability Tables (X-axis = NE, from 0 to {max_NE})")
    st.write("Tables show Probability, Marginal (P(NE+1) - P(NE)), and Curvature (P(NE+1) - 2P(NE) + P(NE-1)).")

    for (table_name, table_data) in all_tables_3:
        with st.expander(f"**{table_name}**"):
            st.dataframe(table_data, use_container_width=True)
