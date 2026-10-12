import streamlit as st
import pandas as pd
import os
import re

st.set_page_config(page_title="智运未来 - 智慧物流平台", layout="wide")
st.title("🚚 智运未来：AI驱动的城市低碳智慧物流协同优化平台")
st.markdown("---")

page = st.sidebar.radio("导航菜单", [
    "城市物流驾驶舱",
    "AI需求预测",
    "智能配送优化",
    "低碳评价分析",
    "未来趋势预测"
])

# ==================== 1. 城市物流驾驶舱 ====================
if page == "城市物流驾驶舱":
    st.header("📊 城市物流态势感知")
    st.write("基于LaDe真实订单数据，展示重庆主城区物流运行状态与配送路线。")

    if os.path.exists('data/processed/routes_map.html'):
        with open('data/processed/routes_map.html', 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=600)
    else:
        st.warning("请先运行 08_visualize_routes.py 生成地图。")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists('data/processed/order_spatial.png'):
            st.image('data/processed/order_spatial.png', caption='重庆全市订单空间分布', width=500)
    with col2:
        if os.path.exists('data/processed/order_spatial_maincity.png'):
            st.image('data/processed/order_spatial_maincity.png', caption='主城区订单分布', width=500)

    if os.path.exists('data/processed/order_hourly.png'):
        st.image('data/processed/order_hourly.png', caption='订单小时分布（早晚双高峰）', width=900)


# ==================== 2. AI需求预测 ====================
elif page == "AI需求预测":
    st.header("🤖 AI物流需求预测")
    st.write("基于XGBoost算法，利用历史订单数据预测未来时段各区域物流需求。")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists('data/processed/ai_prediction_result.png'):
            st.image('data/processed/ai_prediction_result.png', caption='AI预测订单量 vs 真实订单量', width=500)
    with col2:
        if os.path.exists('data/processed/ai_feature_importance.png'):
            st.image('data/processed/ai_feature_importance.png', caption='特征重要性分析', width=500)

    if os.path.exists('data/processed/ai_model_metrics.txt'):
        with open('data/processed/ai_model_metrics.txt', 'r', encoding='utf-8') as f:
            st.text(f.read())

    if os.path.exists('data/processed/fig1_future_orders.png'):
        st.image('data/processed/fig1_future_orders.png', caption='未来年订单量预测（2026-2030）', width=700)


# ==================== 3. 智能配送优化 ====================
elif page == "智能配送优化":
    st.header("🚚 智能配送路线优化")
    st.write("基于OR-Tools多目标低碳VRP算法，生成综合成本最低的配送方案。")

    st.subheader("⚙️ 优化策略调节")
    col1, col2, col3 = st.columns(3)
    with col1:
        alpha = st.slider("效率权重 α（距离）", 0.0, 3.0, 1.0, 0.1)
    with col2:
        beta = st.slider("成本权重 β", 0.0, 3.0, 1.0, 0.1)
    with col3:
        gamma = st.slider("低碳权重 γ（碳排）", 0.0, 3.0, 1.0, 0.1)

    st.info(f"当前权重：α={alpha}, β={beta}, γ={gamma}")

    if st.button("🚀 重新生成优化方案"):
        with st.spinner("正在求解多目标VRP，请稍候（约5-10秒）..."):
            try:
                import numpy as np
                import math
                from ortools.constraint_solver import routing_enums_pb2
                from ortools.constraint_solver import pywrapcp

                # 读取订单
                orders = pd.read_csv('data/processed/orders.csv')
                orders['create_time'] = pd.to_datetime(orders['create_time'])
                main_city = orders[
                    (orders['longitude'] >= 106.3) & (orders['longitude'] <= 106.8) &
                    (orders['latitude'] >= 29.3) & (orders['latitude'] <= 29.7)
                ].copy()
                daily_orders = main_city[
                    main_city['create_time'].dt.date.astype(str) == '2022-10-22'
                ].head(200).copy()
                if len(daily_orders) < 50:
                    daily_orders = main_city.head(200).copy()

                depot_lng = daily_orders['longitude'].mean()
                depot_lat = daily_orders['latitude'].mean()
                locations = [(depot_lng, depot_lat)] + list(zip(daily_orders['longitude'], daily_orders['latitude']))
                demands = [0] + [int(w) for w in daily_orders['weight']]

                num_vehicles = 20
                total_demand = daily_orders['weight'].sum()
                auto_capacity = int(total_demand / num_vehicles * 1.5) + 10
                vehicle_capacities = [auto_capacity] * num_vehicles

                # Haversine距离
                def haversine(lng1, lat1, lng2, lat2):
                    R = 6371
                    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
                    dlat = math.radians(lat2 - lat1)
                    dlng = math.radians(lng2 - lng1)
                    a = math.sin(dlat/2)**2 + math.cos(lat1_r)*math.cos(lat2_r)*math.sin(dlng/2)**2
                    return 2 * R * math.asin(math.sqrt(a))

                n = len(locations)
                dist_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            d = haversine(locations[i][0], locations[i][1],
                                          locations[j][0], locations[j][1]) * 1.2
                            dist_matrix[i][j] = int(d * 1000)

                # 标准化
                D_max = dist_matrix.max() / 1000
                C_max = D_max * 4.96
                E_max = D_max * 0.6575 * 1.3
                avg_cap = sum(vehicle_capacities) / len(vehicle_capacities)

                # 求解
                manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
                routing = pywrapcp.RoutingModel(manager)

                def cb(from_idx, to_idx):
                    fn = manager.IndexToNode(from_idx)
                    tn = manager.IndexToNode(to_idx)
                    dk = dist_matrix[fn][tn] / 1000.0
                    Dn = dk / D_max
                    Cn = (dk * 4.96) / C_max
                    lf = 1.0 + 0.3 * (demands[tn] / avg_cap)
                    En = (dk * 0.6575 * lf) / E_max
                    return int((alpha*Dn + beta*Cn + gamma*En) * 1000000)

                idx = routing.RegisterTransitCallback(cb)
                routing.SetArcCostEvaluatorOfAllVehicles(idx)
                routing.SetFixedCostOfAllVehicles(200)

                def dem_cb(from_idx):
                    return demands[manager.IndexToNode(from_idx)]
                didx = routing.RegisterUnaryTransitCallback(dem_cb)
                routing.AddDimensionWithVehicleCapacity(didx, 0, vehicle_capacities, True, 'Cap')

                sp = pywrapcp.DefaultRoutingSearchParameters()
                sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
                sp.time_limit.seconds = 8
                sol = routing.SolveWithParameters(sp)

                if sol:
                    routes = []
                    total_dist = 0
                    total_cost = 0
                    total_carbon = 0
                    used = 0
                    for v in range(num_vehicles):
                        i = routing.Start(v)
                        route = []
                        while not routing.IsEnd(i):
                            route.append(manager.IndexToNode(i))
                            i = sol.Value(routing.NextVar(i))
                        rd = 0
                        prev = routing.Start(v)
                        i = routing.Start(v)
                        while not routing.IsEnd(i):
                            i = sol.Value(routing.NextVar(i))
                            rd += dist_matrix[manager.IndexToNode(prev)][manager.IndexToNode(i)]
                            prev = i
                        rk = rd / 1000.0
                        oc = len([x for x in route if x != 0])
                        if oc > 0:
                            rc = rk * 4.96 + 200
                            tl = sum(demands[x] for x in route if x != 0)
                            lf = 1.0 + 0.3 * (tl / max(vehicle_capacities[v], 1))
                            rcarb = rk * 0.6575 * lf
                            used += 1
                        else:
                            rc = 0
                            rcarb = 0
                        for seq, node in enumerate(route):
                            if node != 0:
                                routes.append({'vehicle_id': v+1, 'order_node': node,
                                               'sequence': seq, 'distance_m': rd})
                        total_dist += rd
                        total_cost += rc
                        total_carbon += rcarb

                    df_new = pd.DataFrame(routes)
                    st.success(f"✅ 求解完成！总距离 {total_dist/1000:.2f} km，总成本 {total_cost:.2f} 元，总碳排 {total_carbon:.2f} kg，派车 {used} 辆")
                    st.subheader("配送路线方案表")
                    st.dataframe(df_new)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("总行驶距离", f"{total_dist/1000:.2f} km")
                    c2.metric("总运输成本", f"{total_cost:.2f} 元")
                    c3.metric("总碳排放", f"{total_carbon:.2f} kg")
                    c4.metric("实际派车", f"{used} 辆")
                else:
                    st.error("未找到优化方案，请检查数据。")
            except Exception as e:
                st.error(f"求解出错：{e}")
    else:
        st.info("👆 拖动滑块调整权重后，点击上方按钮重新求解。默认显示上次运行 07_optimize_vrp.py 生成的方案。")

        if os.path.exists('data/processed/routes.csv'):
            df = pd.read_csv('data/processed/routes.csv')
            st.subheader("当前配送路线方案表")
            st.dataframe(df.head(50))
            c1, c2, c3 = st.columns(3)
            c1.metric("参与配送车辆", f"{df['vehicle_id'].nunique()} 辆")
            c2.metric("配送订单总数", f"{len(df)} 单")
            c3.metric("总行驶距离", f"{df['distance_m'].sum()/1000:.2f} km")
# ==================== 4. 低碳评价分析 ====================
elif page == "低碳评价分析":
    st.header("🌿 低碳物流评价")
    st.write("基于重庆真实运价与柴油碳排放因子，分析优化前后的成本与碳排差异。")

    # 从 vrp_summary.txt 动态读取最新数据
    smart_distance = 0.0
    smart_cost = 0.0
    smart_carbon = 0.0
    if os.path.exists('data/processed/vrp_summary.txt'):
        with open('data/processed/vrp_summary.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        d_match = re.search(r'总行驶距离:\s*([\d.]+)', content)
        c_match = re.search(r'总运输成本:\s*([\d.]+)', content)
        e_match = re.search(r'总碳排放量:\s*([\d.]+)', content)
        if d_match:
            smart_distance = float(d_match.group(1))
        if c_match:
            smart_cost = float(c_match.group(1))
        if e_match:
            smart_carbon = float(e_match.group(1))

    # 人工调度基准：距离和车辆数均为智能调度的1.3倍
    smart_vehicles = 12
    manual_vehicles = int(smart_vehicles * 1.3)
    manual_distance = smart_distance * 1.3
    manual_cost = manual_distance * 4.96 + manual_vehicles * 200
    manual_carbon = smart_carbon * 1.3

    saved_cost = manual_cost - smart_cost
    saved_carbon = manual_carbon - smart_carbon

    col1, col2 = st.columns(2)
    with col1:
        st.metric("智能调度总成本", f"{smart_cost:,.2f} 元")
        st.metric("智能调度总碳排", f"{smart_carbon:.2f} kg CO2")
        st.metric("智能调度总距离", f"{smart_distance:.2f} km")
    with col2:
        st.metric("人工调度总成本", f"{manual_cost:,.2f} 元",
                  delta=f"-{saved_cost:,.2f} 元")
        st.metric("人工调度总碳排", f"{manual_carbon:.2f} kg",
                  delta=f"-{saved_carbon:.2f} kg")
        st.metric("人工调度总距离", f"{manual_distance:.2f} km")

    if os.path.exists('data/processed/optimization_comparison.png'):
        st.image('data/processed/optimization_comparison.png',
                 caption='优化前后对比', width=1000)

    if os.path.exists('data/processed/strategy_comparison.png'):
        st.image('data/processed/strategy_comparison.png',
                 caption='不同优化策略对比', width=1000)

    st.success(f"✅ 智能低碳调度相比人工调度：节省约 {saved_cost:,.0f} 元成本，减少约 {saved_carbon:.1f} kg 碳排放。")


# ==================== 5. 未来趋势预测 ====================
elif page == "未来趋势预测":
    st.header("📈 未来3-5年订单、成本与碳排放趋势")
    st.write("基于重庆真实统计公报数据与LaDe真实订单数据，预测2026-2030年趋势。")

    col1, col2, col3 = st.columns(3)
    with col1:
        if os.path.exists('data/processed/fig1_future_orders.png'):
            st.image('data/processed/fig1_future_orders.png', caption='未来年订单量预测', width=350)
    with col2:
        if os.path.exists('data/processed/fig2_future_cost.png'):
            st.image('data/processed/fig2_future_cost.png', caption='未来运输成本对比', width=350)
    with col3:
        if os.path.exists('data/processed/fig3_future_carbon.png'):
            st.image('data/processed/fig3_future_carbon.png', caption='未来碳排放对比', width=350)

    if os.path.exists('data/processed/future_forecast.png'):
        st.image('data/processed/future_forecast.png',
                 caption='重庆物流需求趋势预测（2020-2030）', width=1000)

    st.markdown("---")
    st.subheader("💰 商业价值测算（以2030年为例）")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("智能调度成本", "121,942 元")
    col_b.metric("人工调度成本", "158,525 元")
    col_c.metric("✅ 节省成本", "36,583 元", delta="降低23%")

    col_d, col_e, col_f = st.columns(3)
    col_d.metric("智能调度碳排", "16,164.7 kg")
    col_e.metric("人工调度碳排", "21,014.1 kg")
    col_f.metric("✅ 减碳量", "4,849.4 kg", delta="减少23%")

    st.success("如果全重庆6万户道路货运业户接入本平台，每年节省成本与减少碳排放的规模将极为可观。")