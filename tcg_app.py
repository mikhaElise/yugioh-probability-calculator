# -*- coding: utf-8 -*-
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import math
from PIL import Image
import altair as alt  # 新增: 导入 Altair 图表库

# --- 页面配置 (包含主题设置) ---
st.set_page_config(
    layout="wide",
    page_title="YGO Prob Calc",
    page_icon="🎲",
    initial_sidebar_state="auto" # 让侧边栏状态更稳定
)
# --- 页面配置结束 ---

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
    
    P_exact_full = [[calculate_exact_prob(i, k, N, n) for k in range(-1, N + 2)] for i in range(n + 1)]
    P_cumulative_full = [[0.0] * (N + 3) for _ in range(n + 1)]

    for k_idx in range(N + 3):
        p_sum = 0.0
        for i in range(n, -1, -1):
             p_sum += P_exact_full[i][k_idx]
             P_cumulative_full[i][k_idx] = p_sum

    df_plot = pd.DataFrame({"K (Starters)": plot_k_col})
    df_plot["P(X >= 1)"] = P_cumulative_full[1][1 : N + 2]
    df_plot["P(X >= 2)"] = P_cumulative_full[2][1 : N + 2]
    df_plot["P(X >= 3)"] = P_cumulative_full[3][1 : N + 2]
    df_plot["P(X >= 4)"] = P_cumulative_full[4][1 : N + 2]
    df_plot["P(X = 5)"] = P_exact_full[5][1 : N + 2]
    df_plot = df_plot.set_index("K (Starters)")

    all_tables = []
    curve_names = [
        "P(X >= 1) / 至少1张动点",
        "P(X >= 2) / 至少2张动点",
        "P(X >= 3) / 至少3张动点",
        "P(X >= 4) / 至少4张动点",
        "P(X = 5) / 正好5张动点"
    ]
    
    data_sources = [
        P_cumulative_full[1],
        P_cumulative_full[2],
        P_cumulative_full[3],
        P_cumulative_full[4],
        P_exact_full[5]
    ]

    turning_points = {} # 新增: 用于存储转折点

    for i_curve in range(5):
        table_K_col = list(range(1, N + 1))
        P_curve = data_sources[i_curve]
        table_P_col = P_curve[2 : N + 2]
        table_D_col = [P_curve[k+2] - P_curve[k+1] for k in range(len(table_K_col))]
        table_C_col = [P_curve[k+3] - 2*P_curve[k+2] + P_curve[k+1] for k in range(len(table_K_col))]
        
        # 新增: 计算转折点
        if table_D_col:
            try:
                max_marginal_gain = max(table_D_col)
                # 找到最大边际收益对应的索引
                turning_point_idx = table_D_col.index(max_marginal_gain)
                # 对应的 K 值
                turning_point_k = table_K_col[turning_point_idx]
                curve_name = df_plot.columns[i_curve]
                turning_points[curve_name] = turning_point_k
            except (ValueError, IndexError):
                pass # 如果列表为空或找不到，则忽略

        df_table = pd.DataFrame({
            "K (Starters / 动点)": table_K_col,
            "Probability / 概率": table_P_col,
            "Marginal / 边际": table_D_col,
            "Curvature / 曲率": table_C_col
        }).set_index("K (Starters / 动点)")
        
        df_display = df_table.copy()
        df_display["Probability / 概率"] = df_display["Probability / 概率"].map('{:.4%}'.format)
        df_display["Marginal / 边际"] = df_display["Marginal / 边际"].map('{:+.4%}'.format)
        df_display["Curvature / 曲率"] = df_display["Curvature / 曲率"].map('{:+.4%}'.format)
        
        all_tables.append((curve_names[i_curve], df_display))

    return df_plot, all_tables, turning_points # 修改: 返回转折点

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
    
    # 新增: 计算转折点
    turning_point = None
    if table_D_col:
        try:
            max_marginal_gain = max(table_D_col)
            turning_point_idx = table_D_col.index(max_marginal_gain)
            turning_point = table_A_col[turning_point_idx]
        except (ValueError, IndexError):
            turning_point = None # 找不到则为 None

    df_table = pd.DataFrame({
        "A (Insecticides / 杀虫剂)": table_A_col, 
        "Probability / 概率": table_P_col,
        "P(A+1) - P(A) (Marginal / 边际)": table_D_col,
        "P(A+1)-2P(A)+P(A-1) (Curvature / 曲率)": table_C_col
    }).set_index("A (Insecticides / 杀虫剂)")

    return df_plot, df_table, turning_point # 修改: 返回转折点

@st.cache_data
def calculate_part3_prob_single(NE, D, K_fixed, i):
    """
    计算：5张手牌有i张系统外，6张牌中至少1张动点
    """
    n = 5
    if i < 0 or i > 5 or NE < 0 or K_fixed < 0 or D < n:
        return 0.0
    Trash = D - NE - K_fixed
    if Trash < 0:
        return 0.0
    total_comb_5 = safe_comb(D, n)
    if total_comb_5 == 0:
        return 0.0
    prob_case1 = 0.0
    for k in range(1, min(K_fixed, 5-i) + 1):
        if i + k <= 5:
            ways = (safe_comb(NE, i) * safe_comb(K_fixed, k) * safe_comb(Trash, 5-i-k))
            prob_case1 += ways / total_comb_5
    if 5-i >= 0 and Trash >= 5-i:
        ways_5cards = (safe_comb(NE, i) * safe_comb(K_fixed, 0) * safe_comb(Trash, 5-i))
        prob_5cards = ways_5cards / total_comb_5 if total_comb_5 > 0 else 0.0
        remaining_cards = D - 5
        remaining_K = K_fixed
        if remaining_cards > 0 and remaining_K > 0:
            prob_6th_K = remaining_K / remaining_cards
            prob_case2 = prob_5cards * prob_6th_K
        else:
            prob_case2 = 0.0
    else:
        prob_case2 = 0.0
    return prob_case1 + prob_case2

@st.cache_data
def calculate_part3_prob_single_case7(NE, D, K_fixed):
    """
    特殊情况：5张手牌有5张系统外，6张牌中0张动点
    """
    n = 5
    if NE < 5 or K_fixed == 0: return 0.0
    Trash = D - NE - K_fixed
    if Trash < 0: return 0.0
    total_comb_5 = safe_comb(D, n)
    if total_comb_5 == 0: return 0.0
    ways_5cards = safe_comb(NE, 5) * safe_comb(K_fixed, 0) * safe_comb(Trash, 0)
    prob_5cards = ways_5cards / total_comb_5
    remaining_cards = D - 5
    remaining_K = K_fixed
    if remaining_cards > 0:
        prob_6th_not_K = (remaining_cards - remaining_K) / remaining_cards
    else:
        prob_6th_not_K = 1.0
    return prob_5cards * prob_6th_not_K

@st.cache_data
def get_part3_data(D, K_fixed):
    max_NE = D - K_fixed
    P_full = [[] for _ in range(8)]
    for ne_val in range(-1, max_NE + 2):
        for i in range(0, 6): P_full[i].append(calculate_part3_prob_single(ne_val, D, K_fixed, i))
        P_full[7].append(calculate_part3_prob_single_case7(ne_val, D, K_fixed))
    plot_NE_col = list(range(max_NE + 1))
    df_plot = pd.DataFrame({"NE (Non-Engine)": plot_NE_col}) 
    df_plot["C0 (i=0 NE)"] = P_full[0][1 : max_NE + 2] 
    df_plot["C1 (i=1 NE)"] = P_full[1][1 : max_NE + 2] 
    df_plot["C2 (i=2 NE)"] = P_full[2][1 : max_NE + 2] 
    df_plot["C3 (i=3 NE)"] = P_full[3][1 : max_NE + 2] 
    df_plot["C4 (i=4 NE)"] = P_full[4][1 : max_NE + 2] 
    df_plot["C6 (i=5 NE, >=1 K)"] = P_full[5][1 : max_NE + 2] 
    df_plot["C7 (i=5 NE, 0 K)"] = P_full[7][1 : max_NE + 2] 
    df_plot = df_plot.set_index("NE (Non-Engine)")
    all_tables = []
    curve_names = ["C0: P(0 NE in 5, >=1 K in 6) / 抽5张含0系统外, 抽6张含>=1动点", "C1: P(1 NE in 5, >=1 K in 6) / 抽5张含1系统外, 抽6张含>=1动点", "C2: P(2 NE in 5, >=1 K in 6) / 抽5张含2系统外, 抽6张含>=1动点", "C3: P(3 NE in 5, >=1 K in 6) / 抽5张含3系统外, 抽6张含>=1动点", "C4: P(4 NE in 5, >=1 K in 6) / 抽5张含4系统外, 抽6张含>=1动点", "C6: P(5 NE in 5, >=1 K in 6) / 抽5张含5系统外, 抽6张含>=1动点", "C7: P(5 NE in 5, 0 K in 6) / 抽5张含5系统外, 抽6张含0动点"]
    
    turning_points = {} # 新增
    
    internal_indices_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:7} # Map curve index to P_full index
    
    for i_curve in range(len(df_plot.columns)):
        i_curve_internal = internal_indices_map.get(i_curve, i_curve)
        table_NE_col = list(range(max_NE + 1))
        P_curve = P_full[i_curve_internal]
        table_P_col = P_curve[1 : max_NE + 2]
        table_D_col = [P_curve[j+2] - P_curve[j+1] for j in range(len(table_NE_col))]
        table_C_col = [P_curve[j+2] - 2*P_curve[j+1] + P_curve[j] for j in range(len(table_NE_col))]
        
        # 新增: 计算转折点
        if table_D_col:
            try:
                max_marginal_gain = max(table_D_col)
                turning_point_idx = table_D_col.index(max_marginal_gain)
                turning_point_ne = table_NE_col[turning_point_idx]
                curve_name = df_plot.columns[i_curve]
                turning_points[curve_name] = turning_point_ne
            except (ValueError, IndexError):
                pass

        df_table = pd.DataFrame({"NE (Non-Engine / 系统外)": table_NE_col, "Probability / 概率": table_P_col, "Marginal / 边际": table_D_col, "Curvature / 曲率": table_C_col}).set_index("NE (Non-Engine / 系统外)")
        df_display = df_table.copy()
        df_display["Probability / 概率"] = df_display["Probability / 概率"].map('{:.4%}'.format)
        df_display["Marginal / 边际"] = df_display["Marginal / 边际"].map('{:+.4%}'.format)
        df_display["Curvature / 曲率"] = df_display["Curvature / 曲率"].map('{:+.4%}'.format)
        table_name = curve_names[5] if i_curve_internal == 5 else (curve_names[6] if i_curve_internal == 7 else curve_names[i_curve_internal])
        all_tables.append((table_name, df_display))

    return df_plot, all_tables, turning_points # 修改

@st.cache_data
def get_part3_cumulative_data(D, K_fixed):
    max_NE = D - K_fixed
    P_exact_full = [[calculate_part3_prob_single(ne_val, D, K_fixed, i) for ne_val in range(-1, max_NE + 2)] for i in range(6)] 
    P_cumulative_full = [[0.0] * (max_NE + 3) for _ in range(5)] 
    for ne_idx in range(max_NE + 3): 
        p_sum = sum(P_exact_full[i][ne_idx] for i in range(6))
        for i in range(5):
            P_cumulative_full[i][ne_idx] = sum(P_exact_full[j][ne_idx] for j in range(i + 1, 6))

    plot_NE_col = list(range(max_NE + 1))
    df_plot = pd.DataFrame({"NE (Non-Engine)": plot_NE_col}) 
    df_plot["C_ge1 (>=1 NE)"] = P_cumulative_full[0][1 : max_NE + 2] 
    df_plot["C_ge2 (>=2 NE)"] = P_cumulative_full[1][1 : max_NE + 2] 
    df_plot["C_ge3 (>=3 NE)"] = P_cumulative_full[2][1 : max_NE + 2] 
    df_plot["C_ge4 (>=4 NE)"] = P_cumulative_full[3][1 : max_NE + 2] 
    df_plot["C_ge5 (>=5 NE)"] = P_cumulative_full[4][1 : max_NE + 2] 
    df_plot = df_plot.set_index("NE (Non-Engine)")

    all_tables = []
    curve_names = ["C_ge1: P(>=1 NE in 5, >=1 K in 6) / 抽5张含>=1系统外, 抽6张含>=1动点", "C_ge2: P(>=2 NE in 5, >=1 K in 6) / 抽5张含>=2系统外, 抽6张含>=1动点", "C_ge3: P(>=3 NE in 5, >=1 K in 6) / 抽5张含>=3系统外, 抽6张含>=1动点", "C_ge4: P(>=4 NE in 5, >=1 K in 6) / 抽5张含>=4系统外, 抽6张含>=1动点", "C_ge5: P(>=5 NE in 5, >=1 K in 6) / 抽5张含>=5系统外, 抽6张含>=1动点"]
    
    turning_points = {} # 新增

    for i_curve in range(5): 
        table_NE_col = list(range(max_NE + 1))
        P_curve = P_cumulative_full[i_curve] 
        table_P_col = P_curve[1 : max_NE + 2] 
        table_D_col = [P_curve[j+2] - P_curve[j+1] for j in range(len(table_NE_col))]
        table_C_col = [P_curve[j+2] - 2*P_curve[j+1] + P_curve[j] for j in range(len(table_NE_col))]
        
        # 新增
        if table_D_col:
            try:
                max_marginal_gain = max(table_D_col)
                turning_point_idx = table_D_col.index(max_marginal_gain)
                turning_point_ne = table_NE_col[turning_point_idx]
                curve_name = df_plot.columns[i_curve]
                turning_points[curve_name] = turning_point_ne
            except (ValueError, IndexError):
                pass
        
        df_table = pd.DataFrame({"NE (Non-Engine / 系统外)": table_NE_col, "Probability / 概率": table_P_col, "Marginal / 边际": table_D_col, "Curvature / 曲率": table_C_col}).set_index("NE (Non-Engine / 系统外)")
        df_display = df_table.copy()
        df_display["Probability / 概率"] = df_display["Probability / 概率"].map('{:.4%}'.format)
        df_display["Marginal / 边际"] = df_display["Marginal / 边际"].map('{:+.4%}'.format)
        df_display["Curvature / 曲率"] = df_display["Curvature / 曲率"].map('{:+.4%}'.format)
        all_tables.append((curve_names[i_curve], df_display))

    return df_plot, all_tables, turning_points # 修改


@st.cache_data
def calculate_part4_prob_single(NE, D, K_fixed, i):
    n_draw = 6
    j = n_draw - i 
    if i < 0 or j < 0 or K_fixed < j or NE < i: return 0.0
    Trash = D - K_fixed - NE
    total_combinations = safe_comb(D, n_draw)
    if total_combinations == 0: return 0.0
    ways_to_draw_exact = safe_comb(NE, i) * safe_comb(K_fixed, j) * safe_comb(Trash, 0)
    return ways_to_draw_exact / total_combinations

@st.cache_data
def get_part4_data(D, K_fixed):
    max_NE = D - K_fixed
    P_full = [[] for _ in range(7)] 
    for ne_val in range(-1, max_NE + 2):
        for i in range(1, 7): P_full[i].append(calculate_part4_prob_single(ne_val, D, K_fixed, i))
    plot_NE_col = list(range(max_NE + 1))
    df_plot = pd.DataFrame({"NE (Non-Engine)": plot_NE_col}) 
    df_plot["C1 (1NE, 5K)"] = P_full[1][1 : max_NE + 2] 
    df_plot["C2 (2NE, 4K)"] = P_full[2][1 : max_NE + 2] 
    df_plot["C3 (3NE, 3K)"] = P_full[3][1 : max_NE + 2] 
    df_plot["C4 (4NE, 2K)"] = P_full[4][1 : max_NE + 2] 
    df_plot["C5 (5NE, 1K)"] = P_full[5][1 : max_NE + 2] 
    df_plot["C6 (6NE, 0K)"] = P_full[6][1 : max_NE + 2] 
    df_plot = df_plot.set_index("NE (Non-Engine)")
    all_tables = []
    curve_names = ["", "C1: P(1 NE, 5 K in 6) / 抽6张含1系统外, 5动点", "C2: P(2 NE, 4 K in 6) / 抽6张含2系统外, 4动点", "C3: P(3 NE, 3 K in 6) / 抽6张含3系统外, 3动点", "C4: P(4 NE, 2 K in 6) / 抽6张含4系统外, 2动点", "C5: P(5 NE, 1 K in 6) / 抽6张含5系统外, 1动点", "C6: P(6 NE, 0 K in 6) / 抽6张含6系统外, 0动点"]
    
    turning_points = {} # 新增

    for i_curve in range(1, 7): 
        table_NE_col = list(range(max_NE + 1))
        P_curve = P_full[i_curve] 
        table_P_col = P_curve[1 : max_NE + 2] 
        table_D_col = [P_curve[j+2] - P_curve[j+1] for j in range(len(table_NE_col))]
        table_C_col = [P_curve[j+2] - 2*P_curve[j+1] + P_curve[j] for j in range(len(table_NE_col))]
        
        # 新增
        if table_D_col:
            try:
                max_marginal_gain = max(table_D_col)
                turning_point_idx = table_D_col.index(max_marginal_gain)
                turning_point_ne = table_NE_col[turning_point_idx]
                curve_name = df_plot.columns[i_curve-1]
                turning_points[curve_name] = turning_point_ne
            except (ValueError, IndexError):
                pass
        
        df_table = pd.DataFrame({"NE (Non-Engine / 系统外)": table_NE_col, "Probability / 概率": table_P_col, "Marginal / 边际": table_D_col, "Curvature / 曲率": table_C_col}).set_index("NE (Non-Engine / 系统外)")
        df_display = df_table.copy()
        df_display["Probability / 概率"] = df_display["Probability / 概率"].map('{:.4%}'.format)
        df_display["Marginal / 边际"] = df_display["Marginal / 边际"].map('{:+.4%}'.format)
        df_display["Curvature / 曲率"] = df_display["Curvature / 曲率"].map('{:+.4%}'.format)
        all_tables.append((curve_names[i_curve], df_display))

    return df_plot, all_tables, turning_points # 修改


# ===== GoatCounter & Google Analytics (No changes) =====
GOATCOUNTER_SCRIPT = """
<script data-goatcounter="https://mikhaelise.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
"""
if 'gc_injected' not in st.session_state:
    st.session_state.gc_injected = False
if not st.session_state.gc_injected:
    components.html(GOATCOUNTER_SCRIPT, height=0)
    st.session_state.gc_injected = True

GA_ID = "G-NKZ1V5K6B3"
if 'ga_injected' not in st.session_state:
    st.session_state.ga_injected = False
if not st.session_state.ga_injected:
    GA_SCRIPT = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', '{GA_ID}', {{ 'page_title': 'YGO Probability Calculator', 'page_location': window.location.href }});
    </script>
    """
    components.html(GA_SCRIPT, height=0)
    st.session_state.ga_injected = True
# ===== End of Analytics scripts =====

# --- Sidebar (No changes) ---
try:
    img = Image.open("avatar.png") 
    target_width=150; w_percent=(target_width/float(img.size[0])); target_height=int((float(img.size[1])*float(w_percent)))
    img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    st.sidebar.image(img_resized)
except FileNotFoundError: st.sidebar.caption("avatar.png not found.")
except Exception as e: st.sidebar.error(f"Error loading image: {e}")

st.sidebar.markdown("Made by mikhaElise")
st.sidebar.markdown("Bilibili: https://b23.tv/9aM3G4T")
st.sidebar.header("Parameters / 参数")

DECK_SIZE = st.sidebar.number_input("1. Total Deck Size (D) / 卡组总数", min_value=40, max_value=60, value=40, step=1, help="设置卡组总数 (40-60)")
HAND_SIZE = st.sidebar.number_input("2. Opening Hand Size (n) / 起手数", min_value=0, max_value=10, value=5, step=1, help="设置起手抽几张牌 (0-10)。注意: Part 3 & 4 计算固定为起手5张，抽第6张。")
STARTER_COUNT_K = st.sidebar.number_input("3. Starter Size (K) / 动点数", min_value=0, max_value=DECK_SIZE, value=min(17, DECK_SIZE), step=1, help="为 Part 2, 3 和 4 的计算设置固定的动点 (K) 数量。")
K_HIGHLIGHT = st.sidebar.number_input("4. Highlight Starter Value (K) / 高亮动点数 (用于 Part 1)", min_value=0, max_value=DECK_SIZE, value=min(17, DECK_SIZE), step=1, help=f"输入一个 K 值 (0 到 {DECK_SIZE})，将在 Part 1 图表下方显示该点的精确概率。")

max_ne_possible = DECK_SIZE - STARTER_COUNT_K
max_ne_possible = max(0, max_ne_possible) 
NE_HIGHLIGHT = st.sidebar.number_input("5. Non-engine Size（NE）/系统外数量", min_value=0, max_value=max_ne_possible, value=min(20, max_ne_possible), step=1, help=f"输入一个 NE 值 (0 到 {max_ne_possible})，将在 Part 3 和 4 图表下方显示该点的精确概率。")
# --- End of Sidebar ---

st.title("YGO Opening Hand Probability Calculator / YGO起手概率计算器")
st.write(f"Current Settings / 当前设置: **{DECK_SIZE}** Card Deck / 卡组总数, **{HAND_SIZE}** Card Hand / 起手卡数")
st.caption(f"Part 2, 3 & 4 Fixed Starter Count (K) / Part 2, 3 & 4 固定动点数 = **{STARTER_COUNT_K}**")


# =================================================================================
# Part 1
# =================================================================================
st.header("Part 1: P(At least X Starter) / Part 1: 起手至少X张动点概率")
st.write("This chart shows the probability of drawing specific numbers of 'Starter' cards (K) in your opening hand (n cards), as K (the X-axis) increases. / 此图表显示随着卡组中动点 (K) 数量 (X轴) 的增加，起手手牌 (n张) 中抽到特定数量动点的概率。")
st.subheader("Probability Formulas / 概率公式")
st.latex(r"P(X \geq x) = 1 - \sum_{i=0}^{x-1} \frac{\binom{K}{i} \binom{D-K}{n-i}}{\binom{D}{n}}")

df_plot_1, all_tables_1, turning_points_1 = get_starter_probability_data(DECK_SIZE, HAND_SIZE) 

# --- 修改: 使用 Altair 绘制图表 ---
df_plot_1_melted = df_plot_1.reset_index().melt('K (Starters)', var_name='Curve', value_name='Probability')
base_chart_1 = alt.Chart(df_plot_1_melted).encode(
    x=alt.X('K (Starters):Q', title='K (Number of Starters in Deck)'),
    y=alt.Y('Probability:Q', axis=alt.Axis(format='%'), title='Probability'),
    color='Curve:N',
    tooltip=['K (Starters)', 'Curve', alt.Tooltip('Probability', format='.4%')]
)
lines_1 = base_chart_1.mark_line()

# 创建转折点的数据
tp_data_1 = [{'K (Starters)': v, 'label': f'TP @ {v}'} for v in turning_points_1.values()]
if tp_data_1:
    tp_df_1 = pd.DataFrame(tp_data_1)
    rules_1 = alt.Chart(tp_df_1).mark_rule(color='red', strokeDash=[5,5], size=2).encode(x='K (Starters):Q')
    st.altair_chart((lines_1 + rules_1).interactive(), use_container_width=True)
else:
    st.altair_chart(lines_1.interactive(), use_container_width=True)

# --- 新增: 边际效益分析 ---
st.write("📈 **边际效益分析 (Marginal Utility Analysis):**")
st.write("上图中红色虚线标示出了每条曲线上边际效益最高点 (The point of maximum marginal gain)。这代表在该点（K值）增加一张动点带来的概率提升是最大的。超过这个点后，每再增加一张动点，其带来的概率提升将开始减少（收益递减）。各曲线的转折点如下：")
if turning_points_1:
    tp_cols_1 = st.columns(len(turning_points_1))
    i = 0
    for curve, k_val in turning_points_1.items():
        with tp_cols_1[i]:
            st.metric(label=f"转折点: {curve.split('/')[0].strip()}", value=f"K = {k_val}")
        i += 1
# --- 边际效益分析结束 ---

if K_HIGHLIGHT in df_plot_1.index:
    # ... (rest of Part 1 remains the same)
    highlight_data_1 = df_plot_1.loc[K_HIGHLIGHT]
    st.write(f"**Probabilities for K = {K_HIGHLIGHT} / K = {K_HIGHLIGHT} 时的概率:**")
    valid_cols_1 = [col for col in highlight_data_1.index if not pd.isna(highlight_data_1[col])]
    cols_1 = st.columns(len(valid_cols_1))
    col_idx_1 = 0
    for col_name, prob in highlight_data_1.items():
        if not pd.isna(prob): 
            with cols_1[col_idx_1]:
                st.metric(label=col_name.split('/')[0].strip(), value=f"{prob:.2%}") 
                col_idx_1 += 1
else:
    st.caption(f"Value for K={K_HIGHLIGHT} not available in this chart (max K is {DECK_SIZE}).")
st.header(f"📊 Probability Tables (K=1 to {DECK_SIZE}) / 概率表") 
for (table_name, table_data) in all_tables_1:
    with st.expander(f"**{table_name}**"): st.dataframe(table_data, use_container_width=True)


# =================================================================================
# Part 2
# =================================================================================
st.divider()
st.header("Part 2: P(At least 1 Starter AND At least 1 'Insecticide') / Part 2: P(至少1动点 且 至少1杀虫剂)")
st.write(f"This chart uses the Fixed Starter (K) count of **{STARTER_COUNT_K}** and shows how the probability changes as the 'Insecticide' (A) count (the X-axis) increases in your opening hand (n cards). / 此图表使用固定的动点数 K=**{STARTER_COUNT_K}**，显示随着卡组中‘杀虫剂’(A) 数量 (X轴) 的增加，起手手牌 (n张) 中同时抽到至少1动点和至少1杀虫剂的概率变化。")
st.caption("Assumption: This calculation assumes 'Starters' (K) and 'Insecticides' (A) are separate, non-overlapping sets of cards. / 注：此计算假设动点 (K) 和杀虫剂 (A) 是完全不重叠的两组卡。")

if STARTER_COUNT_K >= DECK_SIZE:
    st.error(f"Error: Fixed Starter Count (K={STARTER_COUNT_K}) must be less than Total Deck Size (D={DECK_SIZE}).")
else:
    max_A_part2 = DECK_SIZE - STARTER_COUNT_K
    if max_A_part2 < 0:
         st.warning("Warning: K is larger than Deck Size.")
    else:
        st.subheader("Probability Formula / 概率公式")
        st.latex(r"P(\text{...}) = 1 - \frac{\binom{D-A}{n} + \binom{D-K}{n} - \binom{D-K-A}{n}}{\binom{D}{n}}")
        
        df_plot_2, df_table_2, turning_point_2 = get_combo_probability_data(DECK_SIZE, HAND_SIZE, STARTER_COUNT_K)
        
        # --- 修改: 使用 Altair 绘制图表 ---
        base_chart_2 = alt.Chart(df_plot_2.reset_index()).encode(
            x=alt.X('A (Insecticides):Q', title='A (Number of Insecticides in Deck)'),
            y=alt.Y('Probability:Q', axis=alt.Axis(format='%'), title='Probability'),
            tooltip=['A (Insecticides)', alt.Tooltip('Probability', format='.4%')]
        )
        lines_2 = base_chart_2.mark_line()
        if turning_point_2 is not None:
            tp_df_2 = pd.DataFrame([{'A (Insecticides)': turning_point_2}])
            rule_2 = alt.Chart(tp_df_2).mark_rule(color='red', strokeDash=[5,5], size=2).encode(x='A (Insecticides):Q')
            st.altair_chart((lines_2 + rule_2).interactive(), use_container_width=True)
        else:
            st.altair_chart(lines_2.interactive(), use_container_width=True)
        
        # --- 新增: 边际效益分析 ---
        st.write("📈 **边际效益分析 (Marginal Utility Analysis):**")
        if turning_point_2 is not None:
            st.write(f"上图中红色虚线（A = {turning_point_2}）标示出边际效益最高点。在此之后，每增加一张'杀虫剂'带来的概率提升将开始减少。")
        else:
            st.write("未找到明显的收益递减转折点。")

        # ... (rest of Part 2 remains the same)
        st.header(f"📊 Probability Table (A=0 to {max_A_part2}) / 概率表")
        df_display_2 = df_table_2.copy()
        df_display_2["Probability / 概率"] = df_display_2["Probability / 概率"].map('{:.4%}'.format)
        df_display_2["P(A+1) - P(A) (Marginal / 边际)"] = df_display_2["P(A+1) - P(A) (Marginal / 边际)"].map('{:+.4%}'.format)
        df_display_2["P(A+1)-2P(A)+P(A-1) (Curvature / 曲率)"] = df_display_2["P(A+1)-2P(A)+P(A-1) (Curvature / 曲率)"].map('{:+.4%}'.format)
        st.dataframe(df_display_2, use_container_width=True, height=300)


# =================================================================================
# Part 3, Chart 1
# =================================================================================
st.divider()
st.header("Part 3, Chart 1: P(Draw `i` Non-Engine in 5 AND >= 1 Starter in 6) / Part 3, 图1: P(抽5张含i张系统外 且 抽6张含>=1动点)")
st.write(f"This chart uses the Fixed Starter (K) count of **{STARTER_COUNT_K}**. The X-axis is the **Non-Engine (NE) count**. / 此图表使用固定的动点数 K=**{STARTER_COUNT_K}**。X轴是卡组中系统外卡牌 (NE) 的数量。")

if STARTER_COUNT_K >= DECK_SIZE:
    st.error(f"Error: Fixed Starter Count (K={STARTER_COUNT_K}) must be less than Total Deck Size (D={DECK_SIZE}).")
elif max_ne_possible < 0:
     st.warning(f"Warning: K ({STARTER_COUNT_K}) + Highlighted NE ({NE_HIGHLIGHT}) cannot exceed Deck Size ({DECK_SIZE}).")
else:
    max_NE = max_ne_possible
    df_plot_3, all_tables_3, turning_points_3 = get_part3_data(DECK_SIZE, STARTER_COUNT_K)
    
    # --- 修改: 使用 Altair 绘制图表 ---
    df_plot_3_melted = df_plot_3.reset_index().melt('NE (Non-Engine)', var_name='Curve', value_name='Probability')
    base_chart_3 = alt.Chart(df_plot_3_melted).encode(
        x=alt.X('NE (Non-Engine):Q', title='NE (Number of Non-Engine cards in Deck)'),
        y=alt.Y('Probability:Q', axis=alt.Axis(format='%'), title='Probability'),
        color='Curve:N',
        tooltip=['NE (Non-Engine)', 'Curve', alt.Tooltip('Probability', format='.4%')]
    )
    lines_3 = base_chart_3.mark_line()
    tp_data_3 = [{'NE (Non-Engine)': v} for v in turning_points_3.values()]
    if tp_data_3:
        tp_df_3 = pd.DataFrame(tp_data_3)
        rules_3 = alt.Chart(tp_df_3).mark_rule(color='red', strokeDash=[5,5], size=2).encode(x='NE (Non-Engine):Q')
        st.altair_chart((lines_3 + rules_3).interactive(), use_container_width=True)
    else:
        st.altair_chart(lines_3.interactive(), use_container_width=True)

    # --- 新增: 边际效益分析 ---
    st.write("📈 **边际效益分析 (Marginal Utility Analysis):**")
    st.write("红色虚线标示了每条曲线收益递减的转折点。各曲线转折点如下：")
    if turning_points_3:
        tp_cols_3 = st.columns(min(len(turning_points_3), 5)) # Avoid too many columns
        i = 0
        for curve, ne_val in turning_points_3.items():
            with tp_cols_3[i % 5]:
                st.metric(label=f"转折点: {curve.split('/')[0].strip()}", value=f"NE = {ne_val}")
            i += 1
    
    # ... (rest of Part 3, Chart 1 remains the same)
    if NE_HIGHLIGHT in df_plot_3.index:
        highlight_data = df_plot_3.loc[NE_HIGHLIGHT]
        st.write(f"**Probabilities for NE = {NE_HIGHLIGHT} / NE = {NE_HIGHLIGHT} 时的概率:**")
        cols = st.columns(len(highlight_data))
        for idx, (col_name, prob) in enumerate(highlight_data.items()):
            if not pd.isna(prob): 
                 with cols[idx]:
                    st.metric(label=col_name.split('(')[0].strip(), value=f"{prob:.2%}") 
    st.header(f"📊 Probability Tables (X-axis = NE, from 0 to {max_NE}) / 概率表")
    for (table_name, table_data) in all_tables_3:
        with st.expander(f"**{table_name}**"): st.dataframe(table_data, use_container_width=True)

# =================================================================================
# Part 3, Chart 2
# =================================================================================
st.divider()
st.header("Part 3, Chart 2: P(Draw `>= i` Non-Engine in 5 AND >= 1 Starter in 6) / Part 3, 图2: P(抽5张含>=i张系统外 且 抽6张含>=1动点)")
st.write(f"This chart shows the cumulative probability. It uses the Fixed Starter (K) count of **{STARTER_COUNT_K}**. The X-axis is the **Non-Engine (NE) count**. / 此图表显示累积概率。使用固定的动点数 K=**{STARTER_COUNT_K}**。X轴是卡组中系统外卡牌 (NE) 的数量。")

if STARTER_COUNT_K < DECK_SIZE and max_ne_possible >= 0:
    max_NE_2 = max_ne_possible
    df_plot_3_cumulative, all_tables_3_cumulative, turning_points_3c = get_part3_cumulative_data(DECK_SIZE, STARTER_COUNT_K)
    
    # --- 修改: 使用 Altair 绘制图表 ---
    df_plot_3c_melted = df_plot_3_cumulative.reset_index().melt('NE (Non-Engine)', var_name='Curve', value_name='Probability')
    base_chart_3c = alt.Chart(df_plot_3c_melted).encode(
        x=alt.X('NE (Non-Engine):Q', title='NE (Number of Non-Engine cards in Deck)'),
        y=alt.Y('Probability:Q', axis=alt.Axis(format='%'), title='Cumulative Probability'),
        color='Curve:N',
        tooltip=['NE (Non-Engine)', 'Curve', alt.Tooltip('Probability', format='.4%')]
    )
    lines_3c = base_chart_3c.mark_line()
    tp_data_3c = [{'NE (Non-Engine)': v} for v in turning_points_3c.values()]
    if tp_data_3c:
        tp_df_3c = pd.DataFrame(tp_data_3c)
        rules_3c = alt.Chart(tp_df_3c).mark_rule(color='red', strokeDash=[5,5], size=2).encode(x='NE (Non-Engine):Q')
        st.altair_chart((lines_3c + rules_3c).interactive(), use_container_width=True)
    else:
        st.altair_chart(lines_3c.interactive(), use_container_width=True)

    # --- 新增: 边际效益分析 ---
    st.write("📈 **边际效益分析 (Marginal Utility Analysis):**")
    st.write("红色虚线标示了每条曲线收益递减的转折点。各曲线转折点如下：")
    if turning_points_3c:
        tp_cols_3c = st.columns(len(turning_points_3c))
        i = 0
        for curve, ne_val in turning_points_3c.items():
            with tp_cols_3c[i]:
                st.metric(label=f"转折点: {curve.split('(')[0].strip()}", value=f"NE = {ne_val}")
            i += 1
    
    # ... (rest of Part 3, Chart 2 remains the same)
    if NE_HIGHLIGHT in df_plot_3_cumulative.index:
        highlight_data_cumul = df_plot_3_cumulative.loc[NE_HIGHLIGHT]
        st.write(f"**Cumulative Probabilities for NE = {NE_HIGHLIGHT} / NE = {NE_HIGHLIGHT} 时的累积概率:**")
        cols_cumul = st.columns(len(highlight_data_cumul))
        for idx, (col_name, prob) in enumerate(highlight_data_cumul.items()):
            if not pd.isna(prob):
                 with cols_cumul[idx]:
                    st.metric(label=col_name.split('(')[0].strip(), value=f"{prob:.2%}") 
    st.header(f"📊 Cumulative Probability Tables (X-axis = NE, from 0 to {max_NE_2}) / 累积概率表")
    for (table_name, table_data) in all_tables_3_cumulative:
        with st.expander(f"**{table_name}**"): st.dataframe(table_data, use_container_width=True)

# =================================================================================
# Part 4
# =================================================================================
st.divider()
st.header("Part 4: P(Draw `i` Non-Engine AND `6-i` Starters in 6 cards) / Part 4: P(抽6张含i张系统外 且 6-i张动点)")
st.write(f"This chart analyzes the exact hand composition after drawing 6 cards (going second). It uses the Fixed Starter (K) count of **{STARTER_COUNT_K}**. The X-axis is the **Non-Engine (NE) count**. / 此图表分析后攻抽完6张牌后的精确手牌构成。使用固定的动点数 K=**{STARTER_COUNT_K}**。X轴是卡组中系统外卡牌 (NE) 的数量。")

if STARTER_COUNT_K < DECK_SIZE and max_ne_possible >= 0:
    max_NE_4 = max_ne_possible
    df_plot_4, all_tables_4, turning_points_4 = get_part4_data(DECK_SIZE, STARTER_COUNT_K)
    
    # --- 修改: 使用 Altair 绘制图表 ---
    df_plot_4_melted = df_plot_4.reset_index().melt('NE (Non-Engine)', var_name='Curve', value_name='Probability')
    base_chart_4 = alt.Chart(df_plot_4_melted).encode(
        x=alt.X('NE (Non-Engine):Q', title='NE (Number of Non-Engine cards in Deck)'),
        y=alt.Y('Probability:Q', axis=alt.Axis(format='%'), title='Probability'),
        color='Curve:N',
        tooltip=['NE (Non-Engine)', 'Curve', alt.Tooltip('Probability', format='.4%')]
    )
    lines_4 = base_chart_4.mark_line()
    tp_data_4 = [{'NE (Non-Engine)': v} for v in turning_points_4.values()]
    if tp_data_4:
        tp_df_4 = pd.DataFrame(tp_data_4)
        rules_4 = alt.Chart(tp_df_4).mark_rule(color='red', strokeDash=[5,5], size=2).encode(x='NE (Non-Engine):Q')
        st.altair_chart((lines_4 + rules_4).interactive(), use_container_width=True)
    else:
        st.altair_chart(lines_4.interactive(), use_container_width=True)

    # --- 新增: 边际效益分析 ---
    st.write("📈 **边际效益分析 (Marginal Utility Analysis):**")
    st.write("红色虚线标示了每条曲线收益递减的转折点。各曲线转折点如下：")
    if turning_points_4:
        tp_cols_4 = st.columns(min(len(turning_points_4), 5))
        i = 0
        for curve, ne_val in turning_points_4.items():
            with tp_cols_4[i % 5]:
                st.metric(label=f"转折点: {curve.split('(')[0].strip()}", value=f"NE = {ne_val}")
            i += 1
            
    # ... (rest of Part 4 remains the same)
    if NE_HIGHLIGHT in df_plot_4.index:
        highlight_data_4 = df_plot_4.loc[NE_HIGHLIGHT]
        st.write(f"**Exact Hand Probabilities for NE = {NE_HIGHLIGHT} / NE = {NE_HIGHLIGHT} 时的精确手牌概率:**")
        cols_4 = st.columns(len(highlight_data_4))
        for idx, (col_name, prob) in enumerate(highlight_data_4.items()):
            if not pd.isna(prob):
                with cols_4[idx]:
                    st.metric(label=col_name.split('(')[0].strip(), value=f"{prob:.2%}") 
    st.header(f"📊 Probability Tables (X-axis = NE, from 0 to {max_NE_4}) / 概率表")
    for (table_name, table_data) in all_tables_4:
        with st.expander(f"**{table_name}**"): st.dataframe(table_data, use_container_width=True)


# --- Footer ---


st.divider()
st.caption("Note: Don't dive it. Cuz the data is just for reference only. / 注：请勿过度执着计算，数据仅供参考。") 

try:
    img_meme = Image.open("meme.png") 
    target_width_meme = 300 
    
    w_percent_meme = (target_width_meme / float(img_meme.size[0]))
    target_height_meme = int((float(img_meme.size[1]) * float(w_percent_meme)))
    
    img_meme_resized = img_meme.resize((target_width_meme, target_height_meme), Image.Resampling.LANCZOS)
    
    st.image(img_meme_resized) 
except FileNotFoundError:
    st.caption("meme.png not found. (Place it in the same folder as the script)")
except Exception as e:
    st.error(f"Error loading meme image: {e}")
