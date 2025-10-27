import streamlit as st
import streamlit.components.v1 as components
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
    
    p_ge_1_list = []
    p_ge_2_list = []
    p_ge_3_list = []
    p_ge_4_list = []
    p_eq_5_list = []
    
    p_ge_1_full_for_table = [calculate_single_prob(k, N, n) for k in range(-1, N + 2)] 

    for k_val in plot_k_col:
        p_eq_0 = calculate_exact_prob(0, k_val, N, n)
        p_eq_1 = calculate_exact_prob(1, k_val, N, n)
        p_eq_2 = calculate_exact_prob(2, k_val, N, n)
        p_eq_3 = calculate_exact_prob(3, k_val, N, n)
        p_eq_4 = calculate_exact_prob(4, k_val, N, n)
        p_eq_5 = calculate_exact_prob(5, k_val, N, n) 

        p_lt_1 = p_eq_0
        p_lt_2 = p_eq_0 + p_eq_1
        p_lt_3 = p_eq_0 + p_eq_1 + p_eq_2
        p_lt_4 = p_eq_0 + p_eq_1 + p_eq_2 + p_eq_3
        
        p_ge_1 = 1.0 - p_lt_1
        p_ge_2 = 1.0 - p_lt_2
        p_ge_3 = 1.0 - p_lt_3
        p_ge_4 = 1.0 - p_lt_4
        
        p_ge_1_list.append(p_ge_1)
        p_ge_2_list.append(p_ge_2)
        p_ge_3_list.append(p_ge_3)
        p_ge_4_list.append(p_ge_4)
        p_eq_5_list.append(p_eq_5)

    df_plot = pd.DataFrame({
        "K (Starters)": plot_k_col,
        "P(X >= 1)": p_ge_1_list,
        "P(X >= 2)": p_ge_2_list,
        "P(X >= 3)": p_ge_3_list,
        "P(X >= 4)": p_ge_4_list,
        "P(X = 5)": p_eq_5_list
    }).set_index("K (Starters)")

    table_K_col = list(range(1, N + 1))
    table_P_col = p_ge_1_full_for_table[2 : N + 2] 
    table_D_col = [p_ge_1_full_for_table[i+2] - p_ge_1_full_for_table[i+1] for i in range(len(table_K_col))]
    table_C_col = [p_ge_1_full_for_table[i+3] - 2*p_ge_1_full_for_table[i+2] + p_ge_1_full_for_table[i+1] for i in range(len(table_K_col))]

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
    if NE < i and i != 7 : 
        return 0.0
    if NE < 5 and i == 7: 
        return 0.0

    Trash = D - K_fixed - NE
    if Trash < 0 and i!=7: 
        if NE >= (D-K_fixed): pass 
        else: return 0.0
    elif Trash < 0 and i==7: 
         return 0.0

    total_deck_draw_5 = safe_comb(D, n)
    if total_deck_draw_5 == 0:
        return 0.0
    
    non_NE_cards = D - NE
    K_plus_Trash = D - NE

    if i != 7: 
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
        if P_6th_not_K_den <= 0: 
             P_6th_not_K = 1.0
        else:
             P_6th_not_K = max(0.0, P_6th_not_K_num / P_6th_not_K_den) 


        P_NOT_B_given_A_i = P_k5_is_0 * P_6th_not_K
        P_B_given_A_i = 1.0 - P_NOT_B_given_A_i
        final_prob = P_A_i * P_B_given_A_i
    
    else: 
        i_actual = 5 
        P_A_5_num = safe_comb(NE, i_actual) * safe_comb(non_NE_cards, n - i_actual)
        P_A_5 = P_A_5_num / total_deck_draw_5
        if P_A_5 == 0:
            return 0.0
            
        P_k5_is_0_num_c7 = safe_comb(Trash, n - i_actual) 
        P_k5_is_0_den_c7 = safe_comb(K_plus_Trash, n - i_actual) 
        if P_k5_is_0_den_c7 == 0:
             P_k5_is_0_c7 = 1.0 if P_k5_is_0_num_c7 == 0 else 0.0
        else:
             P_k5_is_0_c7 = P_k5_is_0_num_c7 / P_k5_is_0_den_c7


        P_6th_not_K_num = (D - n) - K_fixed
        P_6th_not_K_den = D - n
        if P_6th_not_K_den <= 0:
             P_6th_not_K = 1.0
        else:
             P_6th_not_K = max(0.0, P_6th_not_K_num / P_6th_not_K_den)

        P_NOT_B_given_A_5 = P_k5_is_0_c7 * P_6th_not_K
        
        # Original code had P_A_5 * P_NOT_B_given_A_5 which seemed incorrect based on description
        # Assuming the description "0 K in 6" means P(NOT B) overall for the condition A=5
        final_prob = P_A_5 * P_NOT_B_given_A_5 # Keep calculation as P(A=5 AND Not B)

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

@st.cache_data
def get_part3_cumulative_data(D, K_fixed):
    max_NE = D - K_fixed
    
    P_exact_full = [[calculate_part3_prob_single(ne_val, D, K_fixed, i) for ne_val in range(-1, max_NE + 2)] for i in range(6)] 

    P_cumulative_full = [[0.0] * (max_NE + 3) for _ in range(5)] 

    for ne_idx in range(max_NE + 3): 
        p0 = P_exact_full[0][ne_idx]
        p1 = P_exact_full[1][ne_idx]
        p2 = P_exact_full[2][ne_idx]
        p3 = P_exact_full[3][ne_idx]
        p4 = P_exact_full[4][ne_idx]
        p5 = P_exact_full[5][ne_idx]
        
        P_cumulative_full[0][ne_idx] = p1 + p2 + p3 + p4 + p5 
        P_cumulative_full[1][ne_idx] = p2 + p3 + p4 + p5      
        P_cumulative_full[2][ne_idx] = p3 + p4 + p5           
        P_cumulative_full[3][ne_idx] = p4 + p5                
        P_cumulative_full[4][ne_idx] = p5                     

    plot_NE_col = list(range(max_NE + 1))
    df_plot = pd.DataFrame({"NE (Non-Engine)": plot_NE_col})
    df_plot[f"C_ge1 (>=1 NE)"] = P_cumulative_full[0][1 : max_NE + 2]
    df_plot[f"C_ge2 (>=2 NE)"] = P_cumulative_full[1][1 : max_NE + 2]
    df_plot[f"C_ge3 (>=3 NE)"] = P_cumulative_full[2][1 : max_NE + 2]
    df_plot[f"C_ge4 (>=4 NE)"] = P_cumulative_full[3][1 : max_NE + 2]
    df_plot[f"C_ge5 (>=5 NE)"] = P_cumulative_full[4][1 : max_NE + 2]
    df_plot = df_plot.set_index("NE (Non-Engine)")

    all_tables = []
    curve_names = [
        "C_ge1: P(>=1 NE in 5, >=1 K in 6)",
        "C_ge2: P(>=2 NE in 5, >=1 K in 6)",
        "C_ge3: P(>=3 NE in 5, >=1 K in 6)",
        "C_ge4: P(>=4 NE in 5, >=1 K in 6)",
        "C_ge5: P(>=5 NE in 5, >=1 K in 6)",
    ]

    for i_curve in range(5): 
        table_NE_col = list(range(max_NE + 1))
        P_curve = P_cumulative_full[i_curve] 
        
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
        
        all_tables.append((curve_names[i_curve], df_display))

    return df_plot, all_tables

@st.cache_data
def calculate_part4_prob_single(NE, D, K_fixed, i):
    n_draw = 6
    j = n_draw - i 
    
    if i < 0 or j < 0:
        return 0.0
    if K_fixed < j: 
        return 0.0
    if NE < i:    
        return 0.0
        
    Trash = D - K_fixed - NE
    
    total_combinations = safe_comb(D, n_draw)
    if total_combinations == 0:
        return 0.0

    ways_to_draw_exact = safe_comb(NE, i) * safe_comb(K_fixed, j) * safe_comb(Trash, 0)
    
    return ways_to_draw_exact / total_combinations

@st.cache_data
def get_part4_data(D, K_fixed):
    max_NE = D - K_fixed
    
    P_full = [[] for _ in range(7)] 

    for ne_val in range(-1, max_NE + 2):
        for i in range(1, 7): 
            P_full[i].append(calculate_part4_prob_single(ne_val, D, K_fixed, i))

    plot_NE_col = list(range(max_NE + 1))
    df_plot = pd.DataFrame({"NE (Non-Engine)": plot_NE_col})
    df_plot[f"C1 (1NE, 5K)"] = P_full[1][1 : max_NE + 2]
    df_plot[f"C2 (2NE, 4K)"] = P_full[2][1 : max_NE + 2]
    df_plot[f"C3 (3NE, 3K)"] = P_full[3][1 : max_NE + 2]
    df_plot[f"C4 (4NE, 2K)"] = P_full[4][1 : max_NE + 2]
    df_plot[f"C5 (5NE, 1K)"] = P_full[5][1 : max_NE + 2]
    df_plot[f"C6 (6NE, 0K)"] = P_full[6][1 : max_NE + 2]
    
    df_plot = df_plot.set_index("NE (Non-Engine)")

    all_tables = []
    curve_names = [
        "", 
        "C1: P(1 NE, 5 K in 6)",
        "C2: P(2 NE, 4 K in 6)",
        "C3: P(3 NE, 3 K in 6)",
        "C4: P(4 NE, 2 K in 6)",
        "C5: P(5 NE, 1 K in 6)",
        "C6: P(6 NE, 0 K in 6)"
    ]
    
    for i_curve in range(1, 7): 
        
        table_NE_col = list(range(max_NE + 1))
        P_curve = P_full[i_curve] 
        
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
        
        all_tables.append((curve_names[i_curve], df_display))

    return df_plot, all_tables

st.set_page_config(layout="wide")

# ===== GoatCounter 代码 =====
GOATCOUNTER_SCRIPT = """
<script data-goatcounter="https://mikhaelise.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>
"""
# 使用 'gc_injected' 作为键
if 'gc_injected' not in st.session_state:
    st.session_state.gc_injected = False

if not st.session_state.gc_injected:
    components.html(GOATCOUNTER_SCRIPT, height=0)
    st.session_state.gc_injected = True
# ===== GoatCounter 结束 =====


# ===== Google Analytics 代码 =====
GA_ID = "G-NKZ1V5K6B3"

# 使用 'ga_injected' 作为键 (确保与上面不同)
if 'ga_injected' not in st.session_state:
    st.session_state.ga_injected = False

if not st.session_state.ga_injected:
    GA_SCRIPT = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', '{GA_ID}', {{
        'page_title': 'YGO Probability Calculator',
        'page_location': window.location.href
      }});
    </script>
    """
    components.html(GA_SCRIPT, height=0)
    st.session_state.ga_injected = True # <-- 确保这里设置的是 'ga_injected'
# ===== Google Analytics 结束 =====

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
    help="Set your opening hand size (0-10). Note: Part 3&4 calculations assume opening 5, drawing 1 (total 6 cards)."
)
STARTER_COUNT_K = st.sidebar.number_input(
    "3. Fixed Starter Count (K)",
    min_value=0,
    max_value=60,
    value=24,
    step=1,
    help="Set the FIXED number of starters (K) for Part 2, 3 and 4 calculations."
)

st.title("YGO Opening Hand Probability Calculator")
st.write(f"Current Settings: **{DECK_SIZE}** Card Deck, **{HAND_SIZE}** Card Hand")
st.caption(f"Part 2, 3 & 4 Fixed Starter Count (K) = **{STARTER_COUNT_K}**")


st.header("Part 1: P(At least X Starter)")
st.write("This chart shows the probability of drawing specific numbers of 'Starter' cards (K) in your opening hand (n cards), as K (the X-axis) increases.")
df_plot_1, df_table_1 = get_starter_probability_data(DECK_SIZE, HAND_SIZE)
st.line_chart(df_plot_1)
st.header(f"📊 Probability Table for P(X>=1) (K=1 to {DECK_SIZE})")
df_display_1 = df_table_1.copy()
prob_col_name_1 = f"P(X>=1) (N={DECK_SIZE}, n={HAND_SIZE})"
df_display_1[prob_col_name_1] = df_display_1[prob_col_name_1].map('{:.4%}'.format)
df_display_1["P(K) - P(K-1) (Marginal)"] = df_display_1["P(K) - P(K-1) (Marginal)"].map('{:+.4%}'.format)
df_display_1["P(K+1)-2P(K)+P(K-1) (Curvature)"] = df_display_1["P(K+1)-2P(K)+P(K-1) (Curvature)"].map('{:+.4%}'.format)
st.dataframe(df_display_1, use_container_width=True, height=300) 


st.divider()
st.header("Part 2: P(At least 1 Starter AND At least 1 'Insecticide')")
st.write(f"This chart uses the Fixed Starter (K) count of **{STARTER_COUNT_K}** and shows how the probability changes as the 'Insecticide' (A) count (the X-axis) increases in your opening hand (n cards).")
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
st.header("Part 3, Chart 1: P(Draw `i` Non-Engine in 5 AND >= 1 Starter in 6)")
st.write(f"This chart uses the Fixed Starter (K) count of **{STARTER_COUNT_K}**. The X-axis is the **Non-Engine (NE) count**.")
st.write(f"Deck = `{STARTER_COUNT_K}` (K) + `X-axis` (NE) + `Remainder` (Trash)")
st.caption("Note: Calculations are for opening 5, drawing 1 (total 6 cards).")

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

st.divider()
st.header("Part 3, Chart 2: P(Draw `>= i` Non-Engine in 5 AND >= 1 Starter in 6)")
st.write(f"This chart shows the cumulative probability. It uses the Fixed Starter (K) count of **{STARTER_COUNT_K}**. The X-axis is the **Non-Engine (NE) count**.")
st.caption("Note: Calculations assume opening 5, drawing 1 (total 6 cards).")

if STARTER_COUNT_K >= DECK_SIZE:
    st.error(f"Error: Fixed Starter Count (K={STARTER_COUNT_K}) must be less than Total Deck Size (D={DECK_SIZE}).")
else:
    max_NE_2 = DECK_SIZE - STARTER_COUNT_K
    df_plot_3_cumulative, all_tables_3_cumulative = get_part3_cumulative_data(DECK_SIZE, STARTER_COUNT_K)
    
    st.line_chart(df_plot_3_cumulative)
    
    st.header(f"📊 Cumulative Probability Tables (X-axis = NE, from 0 to {max_NE_2})")
    st.write("Tables show Cumulative Probability, Marginal (P(NE+1) - P(NE)), and Curvature (P(NE+1) - 2P(NE) + P(NE-1)).")

    for (table_name, table_data) in all_tables_3_cumulative:
        with st.expander(f"**{table_name}**"):
            st.dataframe(table_data, use_container_width=True)

st.divider()
st.header("Part 4: P(Draw `i` Non-Engine AND `6-i` Starters in 6 cards)")
st.write(f"This chart analyzes the exact hand composition after drawing 6 cards (going second). It uses the Fixed Starter (K) count of **{STARTER_COUNT_K}**. The X-axis is the **Non-Engine (NE) count**.")
st.write(f"Deck = `{STARTER_COUNT_K}` (K) + `X-axis` (NE) + `Remainder` (Trash)")
st.caption("Note: Calculations are for drawing exactly 6 cards.")

if STARTER_COUNT_K >= DECK_SIZE:
    st.error(f"Error: Fixed Starter Count (K={STARTER_COUNT_K}) must be less than Total Deck Size (D={DECK_SIZE}).")
else:
    max_NE_4 = DECK_SIZE - STARTER_COUNT_K
    df_plot_4, all_tables_4 = get_part4_data(DECK_SIZE, STARTER_COUNT_K)
    
    st.line_chart(df_plot_4)
    
    st.header(f"📊 Probability Tables (X-axis = NE, from 0 to {max_NE_4})")
    st.write("Tables show Probability, Marginal (P(NE+1) - P(NE)), and Curvature (P(NE+1) - 2P(NE) + P(NE-1)).")

    for (table_name, table_data) in all_tables_4:
        with st.expander(f"**{table_name}**"):
            st.dataframe(table_data, use_container_width=True)
