# -*- coding: utf-8 -*-
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import math
from PIL import Image
import altair as alt

# --- 页面配置 (必须是第一个 Streamlit 命令) ---
st.set_page_config(
    layout="wide",
    page_title="YGO Prob Calc",
    page_icon="🎲",
    initial_sidebar_state="auto"
)

# --- Altair 图表生成函数 ---
def create_chart_with_turning_points(df_plot, turning_points, x_title, y_title, legend_title):
    """
    使用 Altair 创建带有转折点标记的交互式图表。
    """
    df_reset = df_plot.reset_index()
    x_col_name = df_reset.columns[0]
    df_melted = df_reset.melt(id_vars=[x_col_name], var_name=legend_title, value_name=y_title)

    # 基础折线图
    lines = alt.Chart(df_melted).mark_line().encode(
        x=alt.X(f'{x_col_name}:Q', title=x_title, scale=alt.Scale(zero=False)),
        y=alt.Y(f'{y_title}:Q', title=y_title, axis=alt.Axis(format='%')),
        color=f'{legend_title}:N'
    ).interactive()

    # 准备转折点数据
    points_data = []
    for curve, x_val in turning_points.items():
        if curve in df_plot.columns and x_val is not None:
            # 转折点发生在 x_val -> x_val + 1，标记在 x_val + 1
            x_highlight = x_val + 1
            if x_highlight in df_plot.index:
                y_highlight = df_plot.loc[x_highlight, curve]
                points_data.append({
                    x_col_name: x_highlight,
                    legend_title: curve,
                    y_title: y_highlight,
                })

    if not points_data:
        return lines

    points_df = pd.DataFrame(points_data)

    # 转折点标记层
    points = alt.Chart(points_df).mark_point(
        size=100, filled=True, stroke='black', strokeWidth=1, opacity=1.0,
    ).encode(
        x=alt.X(f'{x_col_name}:Q'),
        y=alt.Y(f'{y_title}:Q'),
        color=alt.Color(f'{legend_title}:N'),
        tooltip=[
            alt.Tooltip(f'{x_col_name}:Q', title=f'转折点 (X)'),
            alt.Tooltip(f'{y_title}:Q', title='概率 (Y)', format='.4%')
        ]
    )

    return (lines + points).properties(
        title=f'{y_title} vs. {x_title}'
    )

# --- 所有计算与数据处理函数 ---

@st.cache_data
def safe_comb(n, k):
    if k < 0 or n < k or n < 0:
        return 0
    try:
        return math.comb(n, k)
    except ValueError:
        return 0

@st.cache_data
def calculate_exact_prob(i, K, N, n):
    if n == 0 or K < i: return 0.0
    total_combinations = safe_comb(N, n)
    if total_combinations == 0: return 0.0
    non_starters = N - K
    draw_non_starters = n - i
    if draw_non_starters < 0: return 0.0
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

    all_tables, turning_points = [], {}
    curve_names_map = {
        "P(X >= 1)": "P(X >= 1) / 至少1张动点", "P(X >= 2)": "P(X >= 2) / 至少2张动点",
        "P(X >= 3)": "P(X >= 3) / 至少3张动点", "P(X >= 4)": "P(X >= 4) / 至少4张动点",
        "P(X = 5)": "P(X = 5) / 正好5张动点"
    }
    data_sources = [P_cumulative_full[1], P_cumulative_full[2], P_cumulative_full[3], P_cumulative_full[4], P_exact_full[5]]

    for i_curve, curve_col_name in enumerate(df_plot.columns):
        P_curve = data_sources[i_curve]
        df_table = pd.DataFrame({
            "Probability / 概率": [P_curve[k+2] for k in range(N)],
            "Marginal / 边际": [P_curve[k+2] - P_curve[k+1] for k in range(N)]
        }, index=pd.RangeIndex(1, N + 1, name="K (Starters / 动点)"))

        valid_marginals = df_table["Marginal / 边际"].dropna()
        analysis_text = "没有找到明确的边际效益转折点。/ No clear turning point found."
        if not valid_marginals.empty and valid_marginals.max() > 1e-6:
            max_marginal_gain = valid_marginals.max()
            turning_point_k = valid_marginals.idxmax()
            turning_points[curve_col_name] = turning_point_k
            analysis_text = (
                f"**边际效益分析:** 当卡组中已有 **{turning_point_k}** 张动点时, 再增加第 **{turning_point_k + 1}** 张会带来最大的概率提升 ( **{max_marginal_gain:+.4%}** )。\n\n"
                "在此点之后, 每额外增加一张动点所带来的概率提升将开始减少。"
            )

        df_display = df_table.copy()
        df_display["Probability / 概率"] = df_display["Probability / 概率"].map('{:.4%}'.format)
        df_display["Marginal / 边际"] = df_display["Marginal / 边际"].map('{:+.4%}'.format)
        all_tables.append((curve_names_map[curve_col_name], df_display, analysis_text))
    return df_plot, all_tables, turning_points

@st.cache_data
def get_combo_probability_data(D, n, K_fixed):
    max_A = D - K_fixed
    total_comb = safe_comb(D, n)
    comb_not_K = safe_comb(D - K_fixed, n)
    P_values = [1.0 - (safe_comb(D - a, n) + comb_not_K - safe_comb(D - K_fixed - a, n)) / total_comb if total_comb > 0 else 0.0 for a in range(-1, max_A + 2)]

    df_plot = pd.DataFrame({"Probability": P_values[1 : max_A + 2]}, index=pd.RangeIndex(0, max_A + 1, name="A (Insecticides)"))
    df_table = pd.DataFrame({
        "Probability / 概率": P_values[1 : max_A + 2],
        "Marginal / 边际": [P_values[i+2] - P_values[i+1] for i in range(max_A + 1)]
    }, index=pd.RangeIndex(0, max_A + 1, name="A (Insecticides / 杀虫剂)"))

    analysis_text, turning_points = "没有找到明确的边际效益转折点。", {}
    valid_marginals = df_table["Marginal / 边际"].dropna()
    if not valid_marginals.empty and valid_marginals.max() > 1e-6:
        max_marginal_gain = valid_marginals.max()
        turning_point_a = valid_marginals.idxmax()
        turning_points[df_plot.columns[0]] = turning_point_a
        analysis_text = (
            f"**边际效益分析:** 当卡组中已有 **{turning_point_a}** 张杀虫剂时, 再增加第 **{turning_point_a + 1}** 张会带来最大的概率提升 ( **{max_marginal_gain:+.4%}** )。\n\n"
            "在此点之后, 每额外增加一张杀虫剂所带来的概率提升将开始减少。"
        )
    
    df_display = df_table.copy()
    df_display["Probability / 概率"] = df_display["Probability / 概率"].map('{:.4%}'.format)
    df_display["Marginal / 边际"] = df_display["Marginal / 边际"].map('{:+.4%}'.format)

    return df_plot, df_display, analysis_text, turning_points

@st.cache_data
def calculate_part3_prob_single(NE, D, K_fixed, i):
    n = 5
    if i < 0 or i > 5 or NE < 0 or K_fixed < 0 or D < n: return 0.0
    Trash = D - NE - K_fixed
    if Trash < 0: return 0.0
    total_comb_5 = safe_comb(D, n)
    if total_comb_5 == 0: return 0.0
    
    prob_case1 = sum(safe_comb(NE, i) * safe_comb(K_fixed, k) * safe_comb(Trash, 5-i-k) / total_comb_5 for k in range(1, min(K_fixed, 5-i) + 1) if i + k <= 5)
    
    prob_case2 = 0.0
    if 5-i >= 0 and Trash >= 5-i:
        ways_5cards = safe_comb(NE, i) * safe_comb(K_fixed, 0) * safe_comb(Trash, 5-i)
        prob_5cards = ways_5cards / total_comb_5
        remaining_cards = D - 5
        if remaining_cards > 0 and K_fixed > 0:
            prob_6th_K = K_fixed / remaining_cards
            prob_case2 = prob_5cards * prob_6th_K
    
    return prob_case1 + prob_case2

@st.cache_data
def calculate_part3_prob_single_case7(NE, D, K_fixed):
    n = 5
    if NE < 5 or K_fixed == 0: return 0.0
    Trash = D - NE - K_fixed
    if Trash < 0: return 0.0
    total_comb_5 = safe_comb(D, n)
    if total_comb_5 == 0: return 0.0
    
    prob_5cards = safe_comb(NE, 5) / total_comb_5
    remaining_cards = D - 5
    prob_6th_not_K = (remaining_cards - K_fixed) / remaining_cards if remaining_cards > 0 else 1.0
    
    return prob_5cards * prob_6th_not_K

@st.cache_data
def get_part3_data(D, K_fixed):
    max_NE = D - K_fixed
    P_full = [[] for _ in range(8)]

    for ne in range(-1, max_NE + 2):
        for i in range(6): P_full[i].append(calculate_part3_prob_single(ne, D, K_fixed, i))
        P_full[7].append(calculate_part3_prob_single_case7(ne, D, K_fixed))

    df_plot = pd.DataFrame(index=pd.RangeIndex(0, max_NE + 1, name="NE (Non-Engine)"))
    df_plot["C0 (i=0 NE)"] = P_full[0][1:max_NE+2]
    df_plot["C1 (i=1 NE)"] = P_full[1][1:max_NE+2]
    df_plot["C2 (i=2 NE)"] = P_full[2][1:max_NE+2]
    df_plot["C3 (i=3 NE)"] = P_full[3][1:max_NE+2]
    df_plot["C4 (i=4 NE)"] = P_full[4][1:max_NE+2]
    df_plot["C6 (i=5 NE, >=1 K)"] = P_full[5][1:max_NE+2]
    df_plot["C7 (i=5 NE, 0 K)"] = P_full[7][1:max_NE+2]

    all_tables, turning_points = [], {}
    curve_names_map = {
        "C0 (i=0 NE)": "C0: P(0 NE in 5, >=1 K in 6)", "C1 (i=1 NE)": "C1: P(1 NE in 5, >=1 K in 6)",
        "C2 (i=2 NE)": "C2: P(2 NE in 5, >=1 K in 6)", "C3 (i=3 NE)": "C3: P(3 NE in 5, >=1 K in 6)",
        "C4 (i=4 NE)": "C4: P(4 NE in 5, >=1 K in 6)", "C6 (i=5 NE, >=1 K)": "C6: P(5 NE in 5, >=1 K in 6)",
        "C7 (i=5 NE, 0 K)": "C7: P(5 NE in 5, 0 K in 6)"
    }
    internal_indices = [0, 1, 2, 3, 4, 5, 7]

    for i, curve_col_name in enumerate(df_plot.columns):
        P_curve = P_full[internal_indices[i]]
        df_table = pd.DataFrame({
            "Probability / 概率": P_curve[1:max_NE+2],
            "Marginal / 边际": [P_curve[j+2] - P_curve[j+1] for j in range(max_NE+1)]
        }, index=pd.RangeIndex(0, max_NE+1, name="NE (Non-Engine / 系统外)"))
        
        valid_marginals = df_table["Marginal / 边际"].dropna()
        analysis_text = "没有找到明确的边际效益转折点。"
        if not valid_marginals.empty and valid_marginals.abs().max() > 1e-6:
            max_idx = valid_marginals.idxmax()
            max_marginal_gain = valid_marginals[max_idx]
            turning_point_ne = max_idx
            turning_points[curve_col_name] = turning_point_ne
            analysis_text = (
                f"**边际效益分析:** 当卡组中已有 **{turning_point_ne}** 张系统外时, 再增加第 **{turning_point_ne + 1}** 张会带来最大的概率变化 ( **{max_marginal_gain:+.4%}** )。\n\n"
                "此点是该曲线斜率最陡峭的地方。"
            )

        df_display = df_table.copy()
        df_display["Probability / 概率"] = df_display["Probability / 概率"].map('{:.4%}'.format)
        df_display["Marginal / 边际"] = df_display["Marginal / 边际"].map('{:+.4%}'.format)
        all_tables.append((curve_names_map[curve_col_name], df_display, analysis_text))

    return df_plot, all_tables, turning_points

@st.cache_data
def get_part3_cumulative_data(D, K_fixed):
    max_NE = D - K_fixed
    P_exact_full = [[calculate_part3_prob_single(ne, D, K_fixed, i) for ne in range(-1, max_NE + 2)] for i in range(6)]
    P_cumulative_full = [[0.0] * (max_NE + 3) for _ in range(5)]

    for ne_idx in range(max_NE + 3):
        p_exacts = [P_exact_full[i][ne_idx] for i in range(6)]
        P_cumulative_full[0][ne_idx] = sum(p_exacts[1:])
        P_cumulative_full[1][ne_idx] = sum(p_exacts[2:])
        P_cumulative_full[2][ne_idx] = sum(p_exacts[3:])
        P_cumulative_full[3][ne_idx] = sum(p_exacts[4:])
        P_cumulative_full[4][ne_idx] = p_exacts[5]

    df_plot = pd.DataFrame(index=pd.RangeIndex(0, max_NE + 1, name="NE (Non-Engine)"))
    df_plot["C_ge1 (>=1 NE)"] = P_cumulative_full[0][1:max_NE+2]
    df_plot["C_ge2 (>=2 NE)"] = P_cumulative_full[1][1:max_NE+2]
    df_plot["C_ge3 (>=3 NE)"] = P_cumulative_full[2][1:max_NE+2]
    df_plot["C_ge4 (>=4 NE)"] = P_cumulative_full[3][1:max_NE+2]
    df_plot["C_ge5 (>=5 NE)"] = P_cumulative_full[4][1:max_NE+2]

    all_tables, turning_points = [], {}
    curve_names_map = {
        "C_ge1 (>=1 NE)": "C_ge1: P(>=1 NE in 5, >=1 K in 6)", "C_ge2 (>=2 NE)": "C_ge2: P(>=2 NE in 5, >=1 K in 6)",
        "C_ge3 (>=3 NE)": "C_ge3: P(>=3 NE in 5, >=1 K in 6)", "C_ge4 (>=4 NE)": "C_ge4: P(>=4 NE in 5, >=1 K in 6)",
        "C_ge5 (>=5 NE)": "C_ge5: P(>=5 NE in 5, >=1 K in 6)"
    }
    
    for i_curve, curve_col_name in enumerate(df_plot.columns):
        P_curve = P_cumulative_full[i_curve]
        df_table = pd.DataFrame({
            "Probability / 概率": P_curve[1:max_NE+2],
            "Marginal / 边际": [P_curve[j+2] - P_curve[j+1] for j in range(max_NE+1)]
        }, index=pd.RangeIndex(0, max_NE+1, name="NE (Non-Engine / 系统外)"))
        
        valid_marginals = df_table["Marginal / 边际"].dropna()
        analysis_text = "没有找到明确的边际效益转折点。"
        if not valid_marginals.empty and valid_marginals.abs().max() > 1e-6:
            max_idx = valid_marginals.idxmax()
            max_marginal_gain = valid_marginals[max_idx]
            turning_point_ne = max_idx
            turning_points[curve_col_name] = turning_point_ne
            analysis_text = (
                f"**边际效益分析:** 当卡组中已有 **{turning_point_ne}** 张系统外时, 再增加第 **{turning_point_ne + 1}** 张会带来最大的概率变化 ( **{max_marginal_gain:+.4%}** )。\n\n"
                "此点是该曲线斜率最陡峭的地方。"
            )
        
        df_display = df_table.copy()
        df_display["Probability / 概率"] = df_display["Probability / 概率"].map('{:.4%}'.format)
        df_display["Marginal / 边际"] = df_display["Marginal / 边际"].map('{:+.4%}'.format)
        all_tables.append((curve_names_map[curve_col_name], df_display, analysis_text))
        
    return df_plot, all_tables, turning_points

@st.cache_data
def calculate_part4_prob_single(NE, D, K_fixed, i):
    n_draw = 6
    j = n_draw - i
    if i < 0 or j < 0 or K_fixed < j or NE < i: return 0.0
    Trash = D - K_fixed - NE
    if Trash < 0: return 0.0
    total_combinations = safe_comb(D, n_draw)
    if total_combinations == 0: return 0.0
    ways = safe_comb(NE, i) * safe_comb(K_fixed, j) * safe_comb(Trash, 0)
    return ways / total_combinations

@st.cache_data
def get_part4_data(D, K_fixed):
    max_NE = D - K_fixed
    P_full = [[calculate_part4_prob_single(ne, D, K_fixed, i) for ne in range(-1, max_NE+2)] for i in range(7)]

    df_plot = pd.DataFrame(index=pd.RangeIndex(0, max_NE+1, name="NE (Non-Engine)"))
    df_plot["C1 (1NE, 5K)"] = P_full[1][1:max_NE+2]
    df_plot["C2 (2NE, 4K)"] = P_full[2][1:max_NE+2]
    df_plot["C3 (3NE, 3K)"] = P_full[3][1:max_NE+2]
    df_plot["C4 (4NE, 2K)"] = P_full[4][1:max_NE+2]
    df_plot["C5 (5NE, 1K)"] = P_full[5][1:max_NE+2]
    df_plot["C6 (6NE, 0K)"] = P_full[6][1:max_NE+2]

    all_tables, turning_points = [], {}
    curve_names_map = {
        "C1 (1NE, 5K)": "C1: P(1 NE, 5 K in 6)", "C2 (2NE, 4K)": "C2: P(2 NE, 4 K in 6)",
        "C3 (3NE, 3K)": "C3: P(3 NE, 3 K in 6)", "C4 (4NE, 2K)": "C4: P(4 NE, 2 K in 6)",
        "C5 (5NE, 1K)": "C5: P(5 NE, 1 K in 6)", "C6 (6NE, 0K)": "C6: P(6 NE, 0 K in 6)"
    }
    
    for i_curve, curve_col_name in enumerate(df_plot.columns):
        P_curve = P_full[i_curve + 1]
        df_table = pd.DataFrame({
            "Probability / 概率": P_curve[1:max_NE+2],
            "Marginal / 边际": [P_curve[j+2] - P_curve[j+1] for j in range(max_NE+1)]
        }, index=pd.RangeIndex(0, max_NE+1, name="NE (Non-Engine / 系统外)"))
        
        valid_marginals = df_table["Marginal / 边际"].dropna()
        analysis_text = "没有找到明确的边际效益转折点。"
        if not valid_marginals.empty and valid_marginals.abs().max() > 1e-6:
            max_idx = valid_marginals.idxmax()
            max_marginal_gain = valid_marginals[max_idx]
            turning_point_ne = max_idx
            turning_points[curve_col_name] = turning_point_ne
            analysis_text = (
                f"**边际效益分析:** 当卡组中已有 **{turning_point_ne}** 张系统外时, 再增加第 **{turning_point_ne + 1}** 张会带来最大的概率变化 ( **{max_marginal_gain:+.4%}** )。\n\n"
                "此点是该曲线斜率最陡峭的地方。"
            )
        
        df_display = df_table.copy()
        df_display["Probability / 概率"] = df_display["Probability / 概率"].map('{:.4%}'.format)
        df_display["Marginal / 边际"] = df_display["Marginal / 边际"].map('{:+.4%}'.format)
        all_tables.append((curve_names_map[curve_col_name], df_display, analysis_text))
        
    return df_plot, all_tables, turning_points


# --- Analytics & Tracking Code ---
# (This can be placed after functions and before the sidebar)
GOATCOUNTER_SCRIPT = """
<script data-goatcounter="https://mikhaelise.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>
"""
GA_ID = "G-NKZ1V5K6B3"
GA_SCRIPT = f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_ID}');
</script>
"""
if 'analytics_injected' not in st.session_state:
    components.html(GOATCOUNTER_SCRIPT + GA_SCRIPT, height=0)
    st.session_state.analytics_injected = True


# --- SIDEBAR (定义用户输入变量) ---
# 这一部分必须在所有函数之后，但在主页面使用这些变量之前
try:
    img = Image.open("avatar.png")
    st.sidebar.image(img.resize((150, int(150 * img.height / img.width))), use_column_width=False)
except FileNotFoundError:
    st.sidebar.caption("avatar.png not found.")

st.sidebar.markdown("Made by mikhaElise")
st.sidebar.markdown("Bilibili: https://b23.tv/9aM3G4T")
st.sidebar.header("Parameters / 参数")

DECK_SIZE = st.sidebar.number_input("1. Total Deck Size (D) / 卡组总数", min_value=40, max_value=60, value=40)
HAND_SIZE = st.sidebar.number_input("2. Opening Hand Size (n) / 起手数", min_value=0, max_value=10, value=5)
STARTER_COUNT_K = st.sidebar.number_input("3. Starter Size (K) for Part 2-4", min_value=0, max_value=DECK_SIZE, value=min(17, DECK_SIZE))
K_HIGHLIGHT = st.sidebar.number_input("4. Highlight Starter Value (K) for Part 1", min_value=0, max_value=DECK_SIZE, value=min(17, DECK_SIZE))

max_ne_possible = max(0, DECK_SIZE - STARTER_COUNT_K)
NE_HIGHLIGHT = st.sidebar.number_input("5. Non-engine Size (NE) for Part 3-4", min_value=0, max_value=max_ne_possible, value=min(20, max_ne_possible))


# --- MAIN APP BODY (主页面渲染) ---
st.title("YGO Opening Hand Probability Calculator / YGO起手概率计算器")
st.write(f"Current Settings / 当前设置: **{DECK_SIZE}** Card Deck / 卡组总数, **{HAND_SIZE}** Card Hand / 起手卡数")
st.caption(f"Part 2, 3 & 4 Fixed Starter Count (K) / Part 2, 3 & 4 固定动点数 = **{STARTER_COUNT_K}**")


# --- Part 1 ---
st.header("Part 1: P(At least X Starter) / Part 1: 起手至少X张动点概率")
st.write("此图表显示随着动点 (K) 数量增加，抽到特定数量动点的概率。图上的圆点标记了 **边际效益转折点**，即再增加一张卡牌能带来最大概率提升的位置。")
df_plot_1, all_tables_1, turning_points_1 = get_starter_probability_data(DECK_SIZE, HAND_SIZE)
chart_1 = create_chart_with_turning_points(df_plot_1, turning_points_1, "K (Starters / 动点)", "Probability / 概率", "Curve / 曲线")
st.altair_chart(chart_1, use_container_width=True)

if K_HIGHLIGHT in df_plot_1.index:
    st.write(f"**Probabilities for K = {K_HIGHLIGHT} / K = {K_HIGHLIGHT} 时的概率:**")
    highlight_data = df_plot_1.loc[K_HIGHLIGHT]
    valid_cols = [col for col in highlight_data.index if not pd.isna(highlight_data[col])]
    cols = st.columns(len(valid_cols))
    for i, col_name in enumerate(valid_cols):
        cols[i].metric(label=col_name.split(' ')[0], value=f"{highlight_data[col_name]:.2%}")

st.header(f"📊 概率与边际效益分析表 (K=1 to {DECK_SIZE})")
for (table_name, table_data, analysis_text) in all_tables_1:
    with st.expander(f"**{table_name}**"):
        st.markdown(analysis_text)
        st.dataframe(table_data, use_container_width=True)

# --- Part 2 ---
st.divider()
st.header("Part 2: P(At least 1 Starter AND At least 1 'Insecticide') / Part 2: P(至少1动点 且 至少1杀虫剂)")
st.write(f"此图表使用固定的动点数 K=**{STARTER_COUNT_K}**。圆点标记了 **边际效益转折点**。")

if STARTER_COUNT_K >= DECK_SIZE:
    st.error(f"Error: K ({STARTER_COUNT_K}) must be less than Deck Size ({DECK_SIZE}).")
else:
    df_plot_2, df_table_2, analysis_text_2, turning_points_2 = get_combo_probability_data(DECK_SIZE, HAND_SIZE, STARTER_COUNT_K)
    chart_2 = create_chart_with_turning_points(df_plot_2, turning_points_2, "A (Insecticides / 杀虫剂)", "Probability / 概率", "Curve")
    st.altair_chart(chart_2, use_container_width=True)
    
    st.header(f"📊 概率与边际效益分析表 (A=0 to {DECK_SIZE - STARTER_COUNT_K})")
    st.markdown(analysis_text_2)
    st.dataframe(df_table_2, use_container_width=True)

# --- Part 3, Chart 1 ---
st.divider()
st.header("Part 3, Chart 1: P(Draw `i` Non-Engine in 5 AND >= 1 Starter in 6) / Part 3, 图1: P(抽5张含i张系统外 且 抽6张含>=1动点)")
st.write(f"此图表使用固定的动点数 K=**{STARTER_COUNT_K}**。圆点标记了曲线斜率最陡峭的位置。")

if STARTER_COUNT_K >= DECK_SIZE:
    st.error(f"Error: K ({STARTER_COUNT_K}) must be less than Deck Size ({DECK_SIZE}).")
else:
    df_plot_3, all_tables_3, turning_points_3 = get_part3_data(DECK_SIZE, STARTER_COUNT_K)
    chart_3 = create_chart_with_turning_points(df_plot_3, turning_points_3, "NE (Non-Engine / 系统外)", "Probability / 概率", "Curve / 曲线")
    st.altair_chart(chart_3, use_container_width=True)
    
    if NE_HIGHLIGHT in df_plot_3.index:
        st.write(f"**Probabilities for NE = {NE_HIGHLIGHT} / NE = {NE_HIGHLIGHT} 时的概率:**")
        highlight_data_3 = df_plot_3.loc[NE_HIGHLIGHT]
        valid_cols = [col for col in highlight_data_3.index if not pd.isna(highlight_data_3[col])]
        cols = st.columns(len(valid_cols))
        for i, col_name in enumerate(valid_cols):
            cols[i].metric(label=col_name.split(' ')[0], value=f"{highlight_data_3[col_name]:.2%}")

    st.header(f"📊 概率与边际效益分析表 (NE=0 to {max_ne_possible})")
    for (table_name, table_data, analysis_text) in all_tables_3:
        with st.expander(f"**{table_name}**"):
            st.markdown(analysis_text)
            st.dataframe(table_data, use_container_width=True)

# --- Part 3, Chart 2 (Cumulative) ---
st.divider()
st.header("Part 3, Chart 2: P(Draw `>= i` Non-Engine in 5 AND >= 1 Starter in 6) / Part 3, 图2: P(抽5张含>=i张系统外 且 抽6张含>=1动点)")
st.write(f"此图表显示累积概率。圆点标记了曲线斜率最陡峭的位置。")

if STARTER_COUNT_K >= DECK_SIZE:
    st.error(f"Error: K ({STARTER_COUNT_K}) must be less than Deck Size ({DECK_SIZE}).")
else:
    df_plot_3c, all_tables_3c, turning_points_3c = get_part3_cumulative_data(DECK_SIZE, STARTER_COUNT_K)
    chart_3c = create_chart_with_turning_points(df_plot_3c, turning_points_3c, "NE (Non-Engine / 系统外)", "Cumulative Probability / 累积概率", "Curve / 曲线")
    st.altair_chart(chart_3c, use_container_width=True)

    if NE_HIGHLIGHT in df_plot_3c.index:
        st.write(f"**Cumulative Probabilities for NE = {NE_HIGHLIGHT} / NE = {NE_HIGHLIGHT} 时的累积概率:**")
        highlight_data_3c = df_plot_3c.loc[NE_HIGHLIGHT]
        valid_cols = [col for col in highlight_data_3c.index if not pd.isna(highlight_data_3c[col])]
        cols = st.columns(len(valid_cols))
        for i, col_name in enumerate(valid_cols):
            cols[i].metric(label=col_name.split(' ')[0], value=f"{highlight_data_3c[col_name]:.2%}")
    
    st.header(f"📊 累积概率与边际效益分析表 (NE=0 to {max_ne_possible})")
    for (table_name, table_data, analysis_text) in all_tables_3c:
        with st.expander(f"**{table_name}**"):
            st.markdown(analysis_text)
            st.dataframe(table_data, use_container_width=True)

# --- Part 4 ---
st.divider()
st.header("Part 4: P(Draw `i` Non-Engine AND `6-i` Starters in 6 cards) / Part 4: P(抽6张含i张系统外 且 6-i张动点)")
st.write(f"此图表分析后攻抽完6张牌后的精确手牌构成。圆点标记了曲线斜率最陡峭的位置。")

if STARTER_COUNT_K >= DECK_SIZE:
    st.error(f"Error: K ({STARTER_COUNT_K}) must be less than Deck Size ({DECK_SIZE}).")
else:
    df_plot_4, all_tables_4, turning_points_4 = get_part4_data(DECK_SIZE, STARTER_COUNT_K)
    chart_4 = create_chart_with_turning_points(df_plot_4, turning_points_4, "NE (Non-Engine / 系统外)", "Exact Probability / 精确概率", "Hand Composition / 手牌构成")
    st.altair_chart(chart_4, use_container_width=True)

    if NE_HIGHLIGHT in df_plot_4.index:
        st.write(f"**Exact Hand Probabilities for NE = {NE_HIGHLIGHT} / NE = {NE_HIGHLIGHT} 时的精确手牌概率:**")
        highlight_data_4 = df_plot_4.loc[NE_HIGHLIGHT]
        valid_cols = [col for col in highlight_data_4.index if not pd.isna(highlight_data_4[col])]
        cols = st.columns(len(valid_cols))
        for i, col_name in enumerate(valid_cols):
            cols[i].metric(label=col_name.split(' ')[0], value=f"{highlight_data_4[col_name]:.2%}")
    
    st.header(f"📊 概率与边际效益分析表 (NE=0 to {max_ne_possible})")
    for (table_name, table_data, analysis_text) in all_tables_4:
        with st.expander(f"**{table_name}**"):
            st.markdown(analysis_text)
            st.dataframe(table_data, use_container_width=True)

# --- Footer ---
st.divider()
st.caption("Note: Don't dive it. Cuz the data is just for reference only. / 注：请勿过度执着计算，数据仅供参考。")
try:
    st.image("meme.png", width=300)
except FileNotFoundError:
    st.caption("meme.png not found.")
