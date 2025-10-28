# -*- coding: utf-8 -*-
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
        # P(X=0)=1 if i=0, else 0
        return 1.0 if i == 0 else 0.0
    if K < i or i < 0: # Cannot draw i if K<i or i is negative
        return 0.0
    
    total_combinations = safe_comb(N, n)
    if total_combinations == 0:
        return 0.0 # Should not happen if n>0, D>=n
    
    non_starters = N - K
    draw_non_starters = n - i
    if draw_non_starters < 0: # Cannot draw negative non-starters
        return 0.0
    
    ways_to_draw_exact = safe_comb(K, i) * safe_comb(non_starters, draw_non_starters)
    
    return ways_to_draw_exact / total_combinations

@st.cache_data
def get_starter_probability_data(N, n):
    
    plot_k_col = list(range(N + 1))
    
    # Calculate exact probabilities P(X=i) for k from -1 to N+1
    P_exact_full = [[calculate_exact_prob(i, k, N, n) for k in range(-1, N + 2)] for i in range(n + 1)] # i=0 to n
    
    # Calculate cumulative probabilities P(X>=i)
    P_cumulative_full = [[0.0] * (N + 3) for _ in range(n + 1)] # >=0 to >=n

    for k_idx in range(N + 3): # from K=-1 to N+1
        p_sum = 0.0
        # Calculate P(X >= i) by summing P(X = j) for j >= i
        for i in range(n, -1, -1): # i = n down to 0
             p_sum += P_exact_full[i][k_idx]
             P_cumulative_full[i][k_idx] = p_sum # P_cumulative_full[i] stores P(X>=i)


    # --- 1a. Prepare Plot Data for Chart 1 (Cumulative) ---
    df_plot_1 = pd.DataFrame({"K (Starters)": plot_k_col})
    df_plot_1["P(X >= 1)"] = P_cumulative_full[1][1 : N + 2] # K=0 to N
    df_plot_1["P(X >= 2)"] = P_cumulative_full[2][1 : N + 2]
    df_plot_1["P(X >= 3)"] = P_cumulative_full[3][1 : N + 2]
    df_plot_1["P(X >= 4)"] = P_cumulative_full[4][1 : N + 2]
    df_plot_1["P(X = 5)"] = P_exact_full[5][1 : N + 2] # Use P(X=5) directly
    df_plot_1 = df_plot_1.set_index("K (Starters)")

    # --- 1b. Prepare Tables Data for Chart 1 (Cumulative) ---
    all_tables_1 = []
    curve_names_1 = [
        "P(X >= 1) / 至少1张动点",
        "P(X >= 2) / 至少2张动点",
        "P(X >= 3) / 至少3张动点",
        "P(X >= 4) / 至少4张动点",
        "P(X = 5) / 正好5张动点" # P(X>=5) == P(X=5)
    ]
    data_sources_1 = [ P_cumulative_full[1], P_cumulative_full[2], P_cumulative_full[3], P_cumulative_full[4], P_exact_full[5] ]

    for i_curve in range(5): # 0 to 4
        table_K_col = list(range(1, N + 1)) # K=1 to N for tables
        P_curve = data_sources_1[i_curve] # Contains P(-1), P(0)... P(N+1)
        table_P_col = P_curve[2 : N + 2] # P(1) to P(N)
        table_D_col = [P_curve[k+2] - P_curve[k+1] for k in range(len(table_K_col))] # P(K) - P(K-1)
        table_C_col = [P_curve[k+3] - 2*P_curve[k+2] + P_curve[k+1] for k in range(len(table_K_col))] # P(K+1)-2P(K)+P(K-1)
        
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
        
        all_tables_1.append((curve_names_1[i_curve], df_display))

    # --- 2a. Prepare Plot Data for Chart 2 (Exact) ---
    df_plot_1b = pd.DataFrame({"K (Starters)": plot_k_col})
    df_plot_1b["P(X = 0)"] = P_exact_full[0][1 : N + 2] # K=0 to N
    df_plot_1b["P(X = 1)"] = P_exact_full[1][1 : N + 2]
    df_plot_1b["P(X = 2)"] = P_exact_full[2][1 : N + 2]
    df_plot_1b["P(X = 3)"] = P_exact_full[3][1 : N + 2]
    df_plot_1b["P(X = 4)"] = P_exact_full[4][1 : N + 2]
    df_plot_1b["P(X = 5)"] = P_exact_full[5][1 : N + 2]
    df_plot_1b = df_plot_1b.set_index("K (Starters)")
    
    # --- 2b. Prepare Tables Data for Chart 2 (Exact) ---
    all_tables_1b = []
    curve_names_1b = [
        "P(X = 0) / 正好0张动点",
        "P(X = 1) / 正好1张动点",
        "P(X = 2) / 正好2张动点",
        "P(X = 3) / 正好3张动点",
        "P(X = 4) / 正好4张动点",
        "P(X = 5) / 正好5张动点"
    ]
    data_sources_1b = [ P_exact_full[0], P_exact_full[1], P_exact_full[2], P_exact_full[3], P_exact_full[4], P_exact_full[5] ]

    for i_curve in range(6): # 0 to 5
        table_K_col = list(range(1, N + 1)) # K=1 to N for tables
        P_curve = data_sources_1b[i_curve] # Contains P(-1), P(0)... P(N+1)
        table_P_col = P_curve[2 : N + 2] # P(1) to P(N)
        table_D_col = [P_curve[k+2] - P_curve[k+1] for k in range(len(table_K_col))] # P(K) - P(K-1)
        table_C_col = [P_curve[k+3] - 2*P_curve[k+2] + P_curve[k+1] for k in range(len(table_K_col))] # P(K+1)-2P(K)+P(K-1)
        
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
        
        all_tables_1b.append((curve_names_1b[i_curve], df_display))

    # Return data for both charts and both sets of tables
    return df_plot_1, all_tables_1, df_plot_1b, all_tables_1b


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
        "A (Insecticides / 杀虫剂)": table_A_col, 
        "Probability / 概率": table_P_col,
        "P(A+1) - P(A) (Marginal / 边际)": table_D_col,
        "P(A+1)-2P(A)+P(A-1) (Curvature / 曲率)": table_C_col
    }).set_index("A (Insecticides / 杀虫剂)")
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
        
        final_prob = P_A_5 * P_NOT_B_given_A_5 

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
        "C0: P(0 NE in 5, >=1 K in 6) / 抽5张含0系统外, 抽6张含>=1动点",
        "C1: P(1 NE in 5, >=1 K in 6) / 抽5张含1系统外, 抽6张含>=1动点",
        "C2: P(2 NE in 5, >=1 K in 6) / 抽5张含2系统外, 抽6张含>=1动点",
        "C3: P(3 NE in 5, >=1 K in 6) / 抽5张含3系统外, 抽6张含>=1动点",
        "C4: P(4 NE in 5, >=1 K in 6) / 抽5张含4系统外, 抽6张含>=1动点",
        "C6: P(5 NE in 5, >=1 K in 6) / 抽5张含5系统外, 抽6张含>=1动点",
        "C7: P(5 NE in 5, 0 K in 6) / 抽5张含5系统外, 抽6张含0动点"
    ]
    
    for i_curve_internal in [0, 1, 2, 3, 4, 5, 7]:
        
        table_NE_col = list(range(max_NE + 1))
        P_curve = P_full[i_curve_internal]
        
        table_P_col = P_curve[1 : max_NE + 2]
        table_D_col = [P_curve[j+2] - P_curve[j+1] for j in range(len(table_NE_col))]
        table_C_col = [P_curve[j+2] - 2*P_curve[j+1] + P_curve[j] for j in range(len(table_NE_col))]
        
        df_table = pd.DataFrame({
            "NE (Non-Engine / 系统外)": table_NE_col, 
            "Probability / 概率": table_P_col,
            "Marginal / 边际": table_D_col,
            "Curvature / 曲率": table_C_col
        })
        df_table = df_table.set_index("NE (Non-Engine / 系统外)")
        
        df_display = df_table.copy()
        df_display["Probability / 概率"] = df_display["Probability / 概率"].map('{:.4%}'.format)
        df_display["Marginal / 边际"] = df_display["Marginal / 边际"].map('{:+.4%}'.format)
        df_display["Curvature / 曲率"] = df_display["Curvature / 曲率"].map('{:+.4%}'.format)
        
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
        "C_ge1: P(>=1 NE in 5, >=1 K in 6) / 抽5张含>=1系统外, 抽6张含>=1动点",
        "C_ge2: P(>=2 NE in 5, >=1 K in 6) / 抽5张含>=2系统外, 抽6张含>=1动点",
        "C_ge3: P(>=3 NE in 5, >=1 K in 6) / 抽5张含>=3系统外, 抽6张含>=1动点",
        "C_ge4: P(>=4 NE in 5, >=1 K in 6) / 抽5张含>=4系统外, 抽6张含>=1动点",
        "C_ge5: P(>=5 NE in 5, >=1 K in 6) / 抽5张含>=5系统外, 抽6张含>=1动点",
    ]

    for i_curve in range(5): 
        table_NE_col = list(range(max_NE + 1))
        P_curve = P_cumulative_full[i_curve] 
        
        table_P_col = P_curve[1 : max_NE + 2] 
        table_D_col = [P_curve[j+2] - P_curve[j+1] for j in range(len(table_NE_col))]
        table_C_col = [P_curve[j+2] - 2*P_curve[j+1] + P_curve[j] for j in range(len(table_NE_col))]
        
        df_table = pd.DataFrame({
            "NE (Non-Engine / 系统外)": table_NE_col, 
            "Probability / 概率": table_P_col,
            "Marginal / 边际": table_D_col,
            "Curvature / 曲率": table_C_col
        })
        df_table = df_table.set_index("NE (Non-Engine / 系统外)")
        
        df_display = df_table.copy()
        df_display["Probability / 概率"] = df_display["Probability / 概率"].map('{:.4%}'.format)
        df_display["Marginal / 边际"] = df_display["Marginal / 边际"].map('{:+.4%}'.format)
        df_display["Curvature / 曲率"] = df_display["Curvature / 曲率"].map('{:+.4%}'.format)
        
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
        "C1: P(1 NE, 5 K in 6) / 抽6张含1系统外, 5动点",
        "C2: P(2 NE, 4 K in 6) / 抽6张含2系统外, 4动点",
        "C3: P(3 NE, 3 K in 6) / 抽6张含3系统外, 3动点",
        "C4: P(4 NE, 2 K in 6) / 抽6张含4系统外, 2动点",
        "C5: P(5 NE, 1 K in 6) / 抽6张含5系统外, 1动点",
        "C6: P(6 NE, 0 K in 6) / 抽6张含6系统外, 0动点"
    ]
    
    for i_curve in range(1, 7): 
        
        table_NE_col = list(range(max_NE + 1))
        P_curve = P_full[i_curve] 
        
        table_P_col = P_curve[1 : max_NE + 2] 
        table_D_col = [P_curve[j+2] - P_curve[j+1] for j in range(len(table_NE_col))]
        table_C_col = [P_curve[j+2] - 2*P_curve[j+1] + P_curve[j] for j in range(len(table_NE_col))]
        
        df_table = pd.DataFrame({
            "NE (Non-Engine / 系统外)": table_NE_col, 
            "Probability / 概率": table_P_col,
            "Marginal / 边际": table_D_col,
            "Curvature / 曲率": table_C_col
        })
        df_table = df_table.set_index("NE (Non-Engine / 系统外)")
        
        df_display = df_table.copy()
        df_display["Probability / 概率"] = df_display["Probability / 概率"].map('{:.4%}'.format)
        df_display["Marginal / 边际"] = df_display["Marginal / 边际"].map('{:+.4%}'.format)
        df_display["Curvature / 曲率"] = df_display["Curvature / 曲率"].map('{:+.4%}'.format)
        
        all_tables.append((curve_names[i_curve], df_display))

    return df_plot, all_tables

st.set_page_config(layout="wide", page_title="YGO Prob Calc", page_icon="🎲🎲")


# ===== GoatCounter 代码 =====
GOATCOUNTER_SCRIPT = """
<script data-goatcounter="https://mikhaelise.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>
"""
if 'gc_injected' not in st.session_state:
    st.session_state.gc_injected = False

if not st.session_state.gc_injected:
    components.html(GOATCOUNTER_SCRIPT, height=0)
    st.session_state.gc_injected = True
# ===== GoatCounter 结束 =====


# ===== Google Analytics 代码 =====
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
      gtag('config', '{GA_ID}', {{
        'page_title': 'YGO Probability Calculator',
        'page_location': window.location.href
      }});
    </script>
    """
    components.html(GA_SCRIPT, height=0)
    st.session_state.ga_injected = True
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
st.sidebar.header("Parameters / 参数")

DECK_SIZE = st.sidebar.number_input(
    "1. Total Deck Size (D) / 卡组总数", 
    min_value=40, 
    max_value=60, 
    value=40, 
    step=1,
    help="设置卡组总数 (40-60)"
)
HAND_SIZE = st.sidebar.number_input(
    "2. Opening Hand Size (n) / 起手数", 
    min_value=0, 
    max_value=10, 
    value=5,
    step=1,
    help="设置起手抽几张牌 (0-10)。注意: Part 3 & 4 计算固定为起手5张，抽第6张。"
)
STARTER_COUNT_K = st.sidebar.number_input(
    "3. Starter Size (K) / 动点数", 
    min_value=0,
    max_value=DECK_SIZE, 
    value=min(17, DECK_SIZE), 
    step=1,
    help="为 Part 2, 3 和 4 的计算设置固定的动点 (K) 数量。" 
)

K_HIGHLIGHT = st.sidebar.number_input(
    "4. Highlight Starter Value (K) / 高亮动点数 (用于 Part 1)",
    min_value=0,
    max_value=DECK_SIZE, 
    value=min(17, DECK_SIZE), 
    step=1,
    help=f"输入一个 K 值 (0 到 {DECK_SIZE})，将在 Part 1 图表下方显示该点的精确概率。"
)


max_ne_possible = DECK_SIZE - STARTER_COUNT_K
max_ne_possible = max(0, max_ne_possible) 

NE_HIGHLIGHT = st.sidebar.number_input(
    "5. Non-engine Size（NE）/系统外数量", 
    min_value=0,
    max_value=max_ne_possible, 
    value=min(20, max_ne_possible), 
    step=1,
    help=f"输入一个 NE 值 (0 到 {max_ne_possible})，将在 Part 3 和 4 图表下方显示该点的精确概率。" 
)


st.title("YGO Opening Hand Probability Calculator / YGO起手概率计算器")
st.write(f"Current Settings / 当前设置: **{DECK_SIZE}** Card Deck / 卡组总数, **{HAND_SIZE}** Card Hand / 起手卡数")
st.caption(f"Part 2, 3 & 4 Fixed Starter Count (K) / Part 2, 3 & 4 固定动点数 = **{STARTER_COUNT_K}**")


st.header("Part 1, Chart 1: P(At least X Starter) / Part 1, 图1: 起手至少X张动点概率") # <-- Title updated
st.write("This chart shows the probability of drawing at least 'X' Starter cards (K) in your opening hand (n cards), as K (the X-axis) increases. / 此图表显示随着卡组中动点 (K) 数量 (X轴) 的增加，起手手牌 (n张) 中抽到至少X张动点的概率。") # <-- Description updated

st.subheader("Probability Formulas / 概率公式")
st.latex(r"P(X \geq i) = 1 - \sum_{j=0}^{i-1} P(X=j)")
st.latex(r"P(X = j) = \frac{\binom{K}{j} \binom{D-K}{n-j}}{\binom{D}{n}}")
st.caption("Where D = Deck Size, K = Starter Count, n = Hand Size, i = Min Starters, j = Exact Starters / 其中 D = 卡组总数, K = 动点数, n = 起手数, i = 至少动点数, j = 正好动点数")

df_plot_1, all_tables_1 = get_starter_probability_data(DECK_SIZE, HAND_SIZE) 
st.line_chart(df_plot_1)

if K_HIGHLIGHT in df_plot_1.index:
    highlight_data_1 = df_plot_1.loc[K_HIGHLIGHT]
    st.write(f"**Probabilities for K = {K_HIGHLIGHT} / K = {K_HIGHLIGHT} 时的概率:**")
    valid_cols_1 = [col for col in highlight_data_1.index if not pd.isna(highlight_data_1[col])]
    cols_1 = st.columns(len(valid_cols_1))
    col_idx_1 = 0
    for col_name, prob in highlight_data_1.items():
        if not pd.isna(prob): 
            with cols_1[col_idx_1]:
                curve_label = col_name.split('/')[0].strip() if '/' in col_name else col_name
                st.metric(label=curve_label, value=f"{prob:.2%}") 
                col_idx_1 += 1
else:
    st.caption(f"Value for K={K_HIGHLIGHT} not available in this chart (max K is {DECK_SIZE}). / K={K_HIGHLIGHT} 的值在此图表中不可用 (最大 K 为 {DECK_SIZE})。")

st.header(f"📊 Probability Tables for Chart 1 (K=1 to {DECK_SIZE}) / 图1概率表") # <-- Title updated
st.write("Tables show Probability, Marginal (P(K) - P(K-1)), and Curvature (P(K+1) - 2P(K) + P(K-1)) for each cumulative curve. / 表格显示每条累积曲线的概率，边际和曲率。") # <-- Description updated

for (table_name, table_data) in all_tables_1:
    with st.expander(f"**{table_name}**"):
        st.dataframe(table_data, use_container_width=True)

# --- (新) Part 1, Chart 2 ---
st.divider() # Add divider
st.header("Part 1, Chart 2: P(Exactly `i` Starters in Hand) / Part 1, 图2: 起手正好i张动点概率")
st.write("This chart shows the probability of drawing exactly 'i' Starter cards (K) in your opening hand (n cards), as K (the X-axis) increases. / 此图表显示随着卡组中动点 (K) 数量 (X轴) 的增加，起手手牌 (n张) 中正好抽到i张动点的概率。")

st.subheader("Probability Formula / 概率公式")
st.latex(r"P(X = i) = \frac{\binom{K}{i} \binom{D-K}{n-i}}{\binom{D}{n}}")
st.caption("Where D = Deck Size, K = Starter Count, n = Hand Size, i = Exact Starters / 其中 D = 卡组总数, K = 动点数, n = 起手数, i = 正好动点数")

# Unpack the new return values
_, _, df_plot_1b, all_tables_1b = get_starter_probability_data(DECK_SIZE, HAND_SIZE) 
st.line_chart(df_plot_1b)

# Add highlight section for Chart 2
if K_HIGHLIGHT in df_plot_1b.index:
    highlight_data_1b = df_plot_1b.loc[K_HIGHLIGHT]
    st.write(f"**Exact Probabilities for K = {K_HIGHLIGHT} / K = {K_HIGHLIGHT} 时的精确概率:**")
    valid_cols_1b = [col for col in highlight_data_1b.index if not pd.isna(highlight_data_1b[col])]
    cols_1b = st.columns(len(valid_cols_1b))
    col_idx_1b = 0
    for col_name, prob in highlight_data_1b.items():
        if not pd.isna(prob): 
            with cols_1b[col_idx_1b]:
                curve_label = col_name.split('/')[0].strip() if '/' in col_name else col_name
                st.metric(label=curve_label, value=f"{prob:.2%}") 
                col_idx_1b += 1
else:
    st.caption(f"Value for K={K_HIGHLIGHT} not available in this chart (max K is {DECK_SIZE}). / K={K_HIGHLIGHT} 的值在此图表中不可用 (最大 K 为 {DECK_SIZE})。")

st.header(f"📊 Probability Tables for Chart 2 (K=1 to {DECK_SIZE}) / 图2概率表") 
st.write("Tables show Probability, Marginal (P(K) - P(K-1)), and Curvature (P(K+1) - 2P(K) + P(K-1)) for each exact curve. / 表格显示每条精确曲线的概率，边际和曲率。") 

for (table_name, table_data) in all_tables_1b:
    with st.expander(f"**{table_name}**"):
        st.dataframe(table_data, use_container_width=True)
# --- Part 1, Chart 2 结束 ---


st.divider()
st.header("Part 2: P(At least 1 Starter AND At least 1 'Insecticide') / Part 2: P(至少1动点 且 至少1杀虫剂)")
st.write(f"This chart uses the Fixed Starter (K) count of **{STARTER_COUNT_K}** and shows how the probability changes as the 'Insecticide' (A) count (the X-axis) increases in your opening hand (n cards). / 此图表使用固定的动点数 K=**{STARTER_COUNT_K}**，显示随着卡组中‘杀虫剂’(A) 数量 (X轴) 的增加，起手手牌 (n张) 中同时抽到至少1动点和至少1杀虫剂的概率变化。")

st.subheader("Probability Formula / 概率公式")
st.latex(r"P(K \ge 1 \cap A \ge 1) = 1 - \frac{\binom{D-A}{n} + \binom{D-K}{n} - \binom{D-K-A}{n}}{\binom{D}{n}}")
st.caption("Where D = Deck Size, K = Starter Count, A = Insecticide Count, n = Hand Size / 其中 D = 卡组总数, K = 动点数, A = 杀虫剂数, n = 起手数")
        
st.caption("Assumption: This calculation assumes 'Starters' (K) and 'Insecticides' (A) are separate, non-overlapping sets of cards. / 注：此计算假设动点 (K) 和杀虫剂 (A) 是完全不重叠的两组卡。")

if STARTER_COUNT_K >= DECK_SIZE:
    st.error(f"Error: Fixed Starter Count (K={STARTER_COUNT_K}) must be less than Total Deck Size (D={DECK_SIZE}). / 错误：固定动点数必须小于卡组总数。")
else:
    max_A_part2 = DECK_SIZE - STARTER_COUNT_K
    if max_A_part2 < 0:
         st.warning("Warning: K is larger than Deck Size for Part 2 calculations. Results may be zero.")
    else:
        df_plot_2, df_table_2 = get_combo_probability_data(DECK_SIZE, HAND_SIZE, STARTER_COUNT_K)
        st.line_chart(df_plot_2)
        st.header(f"Probability Table (A=0 to {max_A_part2}) / 概率表")
        st.write("Table shows Probability, Marginal (P(A+1) - P(A)), and Curvature (P(A+1) - 2P(A) + P(A-1)). / 表格显示概率，边际和曲率。")
        df_display_2 = df_table_2.copy()
        df_display_2["Probability / 概率"] = df_display_2["Probability / 概率"].map('{:.4%}'.format)
        df_display_2["P(A+1) - P(A) (Marginal / 边际)"] = df_display_2["P(A+1) - P(A) (Marginal / 边际)"].map('{:+.4%}'.format)
        df_display_2["P(A+1)-2P(A)+P(A-1) (Curvature / 曲率)"] = df_display_2["P(A+1)-2P(A)+P(A-1) (Curvature / 曲率)"].map('{:+.4%}'.format)
        
        st.dataframe(df_display_2, use_container_width=True, height=300)


st.divider()
st.header("Part 3, Chart 1: P(Draw `i` Non-Engine in 5 AND >= 1 Starter in 6) / Part 3, 图1: P(抽5张含i张系统外 且 抽6张含>=1动点)")
st.write(f"This chart uses the Fixed Starter (K) count of **{STARTER_COUNT_K}**. The X-axis is the **Non-Engine (NE) count**. / 此图表使用固定的动点数 K=**{STARTER_COUNT_K}**。X轴是卡组中系统外卡牌 (NE) 的数量。")
st.write(f"Deck = `{STARTER_COUNT_K}` (K) + `X-axis` (NE) + `Remainder` (Trash) / 卡组 = {STARTER_COUNT_K} 动点 + X轴 (系统外) + 剩余卡 (废件)")

st.subheader("Probability Formulas / 概率公式")
st.latex(r"P(N_5 = i \cap K_6 \ge 1) = P(N_5 = i) \times P(K_6 \ge 1 | N_5 = i)")
st.latex(r"P(N_5 = i) = \frac{\binom{NE}{i} \binom{D-NE}{5-i}}{\binom{D}{5}}")
st.latex(r"P(K_6 \ge 1 | N_5 = i) = 1 - \left( \frac{\binom{T}{5-i}}{\binom{K+T}{5-i}} \times \frac{D-5-K}{D-5} \right)")
st.latex(r"P(N_5=5 \cap K_6=0) = P(N_5=5) \times P(K_6=0 | N_5=5)")
st.caption("Where D = Deck Size, K = Starter Count, NE = Non-Engine Count, T = Trash (D-K-NE) / 其中 D = 卡组总数, K = 动点数, NE = 系统外数, T = 废件数")

st.caption("Note: Calculations are for opening 5, drawing 1 (total 6 cards). / 注意：此部分计算固定为起手5张，抽第6张。")

if STARTER_COUNT_K >= DECK_SIZE:
    st.error(f"Error: Fixed Starter Count (K={STARTER_COUNT_K}) must be less than Total Deck Size (D={DECK_SIZE}). / 错误：固定动点数必须小于卡组总数。")
elif max_ne_possible < 0:
     st.warning(f"Warning: K ({STARTER_COUNT_K}) + Highlighted NE ({NE_HIGHLIGHT}) cannot exceed Deck Size ({DECK_SIZE}). Part 3 & 4 results invalid.")
else:
    max_NE = max_ne_possible
    df_plot_3, all_tables_3 = get_part3_data(DECK_SIZE, STARTER_COUNT_K)
    
    st.line_chart(df_plot_3)
    
    if NE_HIGHLIGHT in df_plot_3.index:
        highlight_data = df_plot_3.loc[NE_HIGHLIGHT]
        st.write(f"**Probabilities for NE = {NE_HIGHLIGHT} / NE = {NE_HIGHLIGHT} 时的概率:**")
        valid_cols = [col for col in highlight_data.index if not pd.isna(highlight_data[col])]
        cols = st.columns(len(valid_cols))
        col_idx = 0
        for col_name, prob in highlight_data.items():
             if not pd.isna(prob): 
                 with cols[col_idx]:
                    curve_label = col_name.split('/')[0].strip() if '/' in col_name else col_name
                    st.metric(label=curve_label, value=f"{prob:.2%}") 
                    col_idx += 1
    else:
        st.caption(f"Value for NE={NE_HIGHLIGHT} not available in this chart (max NE is {max_NE}). / NE={NE_HIGHLIGHT} 的值在此图表中不可用 (最大 NE 为 {max_NE})。")

    st.header(f"Probability Tables (X-axis = NE, from 0 to {max_NE}) / 概率表")
    st.write("Tables show Probability, Marginal (P(NE+1) - P(NE)), and Curvature (P(NE+1) - 2P(NE) + P(NE-1)). / 表格显示概率，边际和曲率。")

    for (table_name, table_data) in all_tables_3:
        with st.expander(f"**{table_name}**"):
            st.dataframe(table_data, use_container_width=True)

st.divider()
st.header("Part 3, Chart 2: P(Draw `>= i` Non-Engine in 5 AND >= 1 Starter in 6) / Part 3, 图2: P(抽5张含>=i张系统外 且 抽6张含>=1动点)")
st.write(f"This chart shows the cumulative probability. It uses the Fixed Starter (K) count of **{STARTER_COUNT_K}**. The X-axis is the **Non-Engine (NE) count**. / 此图表显示累积概率。使用固定的动点数 K=**{STARTER_COUNT_K}**。X轴是卡组中系统外卡牌 (NE) 的数量。")

st.subheader("Probability Formulas / 概率公式")
st.latex(r"P(\ge i \text{ NE in 5, } \ge 1 \text{ K in 6}) = \sum_{j=i}^{5} P(j \text{ NE in 5, } \ge 1 \text{ K in 6})")
st.caption("Cumulative probabilities are sums of the corresponding exact probabilities from Chart 1 / 累积概率是图1中相应精确概率的和")

st.caption("Note: Calculations assume opening 5, drawing 1 (total 6 cards). / 注意：计算假设起手5张，抽第6张。")

if STARTER_COUNT_K >= DECK_SIZE:
    st.error(f"Error: Fixed Starter Count (K={STARTER_COUNT_K}) must be less than Total Deck Size (D={DECK_SIZE}). / 错误：固定动点数必须小于卡组总数。")
elif max_ne_possible < 0:
     st.warning(f"Warning: K ({STARTER_COUNT_K}) + Highlighted NE ({NE_HIGHLIGHT}) cannot exceed Deck Size ({DECK_SIZE}). Part 3 results invalid.")
else:
    max_NE_2 = max_ne_possible
    df_plot_3_cumulative, all_tables_3_cumulative = get_part3_cumulative_data(DECK_SIZE, STARTER_COUNT_K)
    
    st.line_chart(df_plot_3_cumulative)

    if NE_HIGHLIGHT in df_plot_3_cumulative.index:
        highlight_data_cumul = df_plot_3_cumulative.loc[NE_HIGHLIGHT]
        st.write(f"**Cumulative Probabilities for NE = {NE_HIGHLIGHT} / NE = {NE_HIGHLIGHT} 时的累积概率:**")
        valid_cols_cumul = [col for col in highlight_data_cumul.index if not pd.isna(highlight_data_cumul[col])]
        cols_cumul = st.columns(len(valid_cols_cumul))
        col_idx_cumul = 0
        for col_name, prob in highlight_data_cumul.items():
             if not pd.isna(prob):
                 with cols_cumul[col_idx_cumul]:
                    curve_label = col_name.split('/')[0].strip() if '/' in col_name else col_name
                    st.metric(label=curve_label, value=f"{prob:.2%}") 
                    col_idx_cumul += 1
    else:
        st.caption(f"Value for NE={NE_HIGHLIGHT} not available in this chart (max NE is {max_NE_2}). / NE={NE_HIGHLIGHT} 的值在此图表中不可用 (最大 NE 为 {max_NE_2})。")
    
    st.header(f"Cumulative Probability Tables (X-axis = NE, from 0 to {max_NE_2}) / 累积概率表")
    st.write("Tables show Cumulative Probability, Marginal (P(NE+1) - P(NE)), and Curvature (P(NE+1) - 2P(NE) + P(NE-1)). / 表格显示累积概率，边际和曲率。")

    for (table_name, table_data) in all_tables_3_cumulative:
        with st.expander(f"**{table_name}**"):
            st.dataframe(table_data, use_container_width=True)

st.divider()
st.header("Part 4: P(Draw `i` Non-Engine AND `6-i` Starters in 6 cards) / Part 4: P(抽6张含i张系统外 且 6-i张动点)")
st.write(f"This chart analyzes the exact hand composition after drawing 6 cards (going second). It uses the Fixed Starter (K) count of **{STARTER_COUNT_K}**. The X-axis is the **Non-Engine (NE) count**. / 此图表分析后攻抽完6张牌后的精确手牌构成。使用固定的动点数 K=**{STARTER_COUNT_K}**。X轴是卡组中系统外卡牌 (NE) 的数量。")
st.write(f"Deck = `{STARTER_COUNT_K}` (K) + `X-axis` (NE) + `Remainder` (Trash) / 卡组 = {STARTER_COUNT_K} (动点) + X轴 (系统外) + 剩余卡 (废件)")

st.subheader("Probability Formula / 概率公式")
st.latex(r"P(i \text{ NE, } 6-i \text{ K in 6}) = \frac{\binom{NE}{i} \binom{K}{6-i} \binom{D-K-NE}{0}}{\binom{D}{6}}")
st.caption("Where D = Deck Size, K = Starter Count, NE = Non-Engine Count / 其中 D = 卡组总数, K = 动点数, NE = 系统外数")

st.caption("Note: Calculations are for going second with drawing the 6th card. / 注意：计算基于后攻抽第6张牌。")

if STARTER_COUNT_K >= DECK_SIZE:
    st.error(f"Error: Fixed Starter Count (K={STARTER_COUNT_K}) must be less than Total Deck Size (D={DECK_SIZE}). / 错误：固定动点数必须小于卡组总数。")
elif max_ne_possible < 0:
     st.warning(f"Warning: K ({STARTER_COUNT_K}) + Highlighted NE ({NE_HIGHLIGHT}) cannot exceed Deck Size ({DECK_SIZE}). Part 4 results invalid.")
else:
    max_NE_4 = max_ne_possible
    df_plot_4, all_tables_4 = get_part4_data(DECK_SIZE, STARTER_COUNT_K)
    
    st.line_chart(df_plot_4)

    if NE_HIGHLIGHT in df_plot_4.index:
        highlight_data_4 = df_plot_4.loc[NE_HIGHLIGHT]
        st.write(f"**Exact Hand Probabilities for NE = {NE_HIGHLIGHT} / NE = {NE_HIGHLIGHT} 时的精确手牌概率:**")
        valid_cols_4 = [col for col in highlight_data_4.index if not pd.isna(highlight_data_4[col])]
        cols_4 = st.columns(len(valid_cols_4))
        col_idx_4 = 0
        for col_name, prob in highlight_data_4.items():
            if not pd.isna(prob):
                with cols_4[col_idx_4]:
                    curve_label = col_name.split('/')[0].strip() if '/' in col_name else col_name
                    st.metric(label=curve_label, value=f"{prob:.2%}") 
                    col_idx_4 += 1
    else:
         st.caption(f"Value for NE={NE_HIGHLIGHT} not available in this chart (max NE is {max_NE_4}). / NE={NE_HIGHLIGHT} 的值在此图表中不可用 (最大 NE 为 {max_NE_4})。")
    
    st.header(f"Probability Tables (X-axis = NE, from 0 to {max_NE_4}) / 概率表")
    st.write("Tables show Probability, Marginal (P(NE+1) - P(NE)), and Curvature (P(NE+1) - 2P(NE) + P(NE-1)). / 表格显示概率，边际和曲率。")

    for (table_name, table_data) in all_tables_4:
        with st.expander(f"**{table_name}**"):
            st.dataframe(table_data, use_container_width=True)

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
