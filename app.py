# -------------------------------------------------------------
#  📊 DASHBOARD FINANCIERO AVANZADO
#      • ROIC y WACC (Kd y tasa efectiva por empresa)
#      • Resumen ordenado por Sector
#      • Gráficos separados por Sector con controles
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import time

# -------------------------------------------------------------
# ⚙️ Configuración global
# -------------------------------------------------------------
st.set_page_config(
    page_title="📊 Dashboard Financiero Avanzado",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# Parámetros CAPM por defecto (editables)
# -------------------------------------------------------------
Rf  = 0.0435   # riesgo libre
Rm  = 0.085    # retorno mercado
Tc0 = 0.21     # tasa impositiva por defecto

# Orden deseado de sectores
SECTOR_RANK = {
    "Consumer Defensive": 1,
    "Consumer Cyclical": 2,
    "Healthcare": 3,
    "Technology": 4,
    "Financial Services": 5,
    "Industrials": 6,
    "Communication Services": 7,
    "Energy": 8,
    "Real Estate": 9,
    "Utilities": 10,
    "Basic Materials": 11,
    "Unknown": 99,
}

# =============================================================
# 1. FUNCIONES AUXILIARES
# =============================================================
def safe_first(obj):
    if obj is None:
        return None
    if hasattr(obj, "dropna"):
        obj = obj.dropna()
    return obj.iloc[0] if hasattr(obj, "iloc") and not obj.empty else obj

def seek_row(df, keys):
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return pd.Series([0], index=df.columns[:1])

def calc_ke(beta):
    return Rf + beta * (Rm - Rf)

def calc_kd(interest, debt):
    return interest / debt if debt else 0

def calc_wacc(mcap, debt, ke, kd, t):
    total = mcap + debt
    return (mcap/total)*ke + (debt/total)*kd*(1-t) if total else None

def cagr4(fin, metric):
    if metric not in fin.index:
        return None
    v = fin.loc[metric].dropna().iloc[:4]
    return (v.iloc[0]/v.iloc[-1])**(1/(len(v)-1))-1 if len(v)>1 and v.iloc[-1] else None

def pct(series):
    return pd.to_numeric(series, errors="coerce") * 100

def num(series):
    return pd.to_numeric(series, errors="coerce")

def sector_sorted_df(df):
    df["Sector"] = df["Sector"].fillna("Unknown")
    df["SectorRank"] = df["Sector"].map(SECTOR_RANK).fillna(999).astype(int)
    return df.sort_values(["SectorRank", "Sector", "Ticker"]).reset_index(drop=True)

# =============================================================
# 2. OBTENER DATOS POR EMPRESA
# =============================================================
def obtener_datos_financieros(tk, Tc_def):
    tkr  = yf.Ticker(tk)
    info = tkr.info
    bs   = tkr.balance_sheet
    fin  = tkr.financials
    cf   = tkr.cashflow
    if not info or bs.empty:
        raise ValueError("Sin datos de balance/info")

    beta  = info.get("beta", 1)
    ke    = calc_ke(beta)

    debt  = safe_first(seek_row(bs, ["Total Debt", "Long Term Debt"])) or info.get("totalDebt", 0)
    cash  = safe_first(seek_row(bs, ["Cash And Cash Equivalents",
                                     "Cash And Cash Equivalents At Carrying Value",
                                     "Cash Cash Equivalents And Short Term Investments"]))
    equity= safe_first(seek_row(bs, ["Common Stock Equity","Total Stockholder Equity"]))

    interest = safe_first(seek_row(fin, ["Interest Expense"]))
    ebt      = safe_first(seek_row(fin, ["Ebt", "EBT"]))
    tax_exp  = safe_first(seek_row(fin, ["Income Tax Expense"]))
    ebit     = safe_first(seek_row(fin, ["EBIT", "Operating Income","Earnings Before Interest and Taxes"]))

    kd   = calc_kd(interest, debt)
    tax  = tax_exp / ebt if ebt else Tc_def
    mcap = info.get("marketCap", 0)
    wacc = calc_wacc(mcap, debt, ke, kd, tax)

    nopat = ebit * (1 - tax) if ebit is not None else None
    invested = equity + (debt - cash)
    roic = nopat / invested if (nopat is not None and invested) else None
    eva  = (roic - wacc) * invested if all(v is not None for v in (roic, wacc, invested)) else None

    price = info.get("currentPrice")
    fcf   = safe_first(seek_row(cf, ["Free Cash Flow"]))
    shares= info.get("sharesOutstanding")
    pfcf  = price / (fcf/shares) if (fcf and shares) else None

    return {
        "Ticker": tk,
        "Sector": info.get("sector", "Unknown"),
        "MarketCap": mcap,
        "Precio": price,
        "P/E": info.get("trailingPE"),
        "P/B": info.get("priceToBook"),
        "P/FCF": pfcf,
        "Dividend Yield %": info.get("dividendYield"),
        "Payout Ratio": info.get("payoutRatio"),
        "ROA": info.get("returnOnAssets"),
        "ROE": info.get("returnOnEquity"),
        "Current Ratio": info.get("currentRatio"),
        "Quick Ratio": info.get("quickRatio"),
        "Debt/Eq": info.get("debtToEquity"),
        "LtDebt/Eq": info.get("longTermDebtToEquity"),
        "Oper Margin": info.get("operatingMargins"),
        "Profit Margin": info.get("profitMargins"),
        "WACC": wacc,
        "ROIC": roic,
        "EVA": eva,
        "Revenue Growth": cagr4(fin, "Total Revenue"),
        "EPS Growth":     cagr4(fin, "Net Income"),
        "FCF Growth":     cagr4(cf, "Free Cash Flow") or cagr4(cf, "Operating Cash Flow"),
    }

# =============================================================
# 3. INTERFAZ PRINCIPAL
# =============================================================
def main():
    st.title("📊 Dashboard de Análisis Financiero Avanzado")

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("⚙️ Configuración")
        t_in = st.text_area("Tickers (coma)",
                            "HRL, AAPL, MSFT, ABT, O, XOM, KO, JNJ, CLX, CHD, DDOG, CB")
        max_t = st.slider("Máx tickers", 1, 100, 30)
        st.markdown("---")
        global Rf, Rm, Tc0
        Rf  = st.number_input("Risk-free (%)", 0.0, 20.0, 4.35)/100
        Rm  = st.number_input("Market return (%)", 0.0, 30.0, 8.5)/100
        Tc0 = st.number_input("Tax rate default (%)", 0.0, 50.0, 21.0)/100
        st.markdown("---")
        top_n = st.slider("Top N por sector (por market cap)", 1, 50, 25)
        min_emp = st.number_input("Mín. empresas/sector para graficar", 1, 50, 1)
        ylim_abs = st.slider("Límite eje Y para % (valor absoluto)", 10, 200, 100)

    tickers = [t.strip().upper() for t in t_in.split(",") if t.strip()][:max_t]

    # ---------- Botón Analizar ----------
    if st.button("🔍 Analizar", type="primary"):
        if not tickers:
            st.warning("Ingresa al menos un ticker")
            return

        datos, errs, bar = [], [], st.progress(0)
        for i, tk in enumerate(tickers, 1):
            try:
                datos.append(obtener_datos_financieros(tk, Tc0))
            except Exception as e:
                errs.append({"Ticker": tk, "Error": str(e)})
            bar.progress(i / len(tickers))
            time.sleep(1)
        bar.empty()

        if not datos:
            st.error("Sin datos válidos.")
            if errs: st.table(pd.DataFrame(errs))
            return

        df = sector_sorted_df(pd.DataFrame(datos))

        # Copias: una numérica para graficar (df_plot) y otra formateada (df_disp)
        df_plot = df.copy()
        df_disp = df.copy()

        # Formateo % para mostrar
        pct_cols = ["Dividend Yield %", "Payout Ratio", "ROA", "ROE", "Oper Margin", "Profit Margin", "WACC", "ROIC"]
        for col in pct_cols:
            df_disp[col] = df_disp[col].apply(lambda x: f"{x*100:,.2f}%" if pd.notnull(x) else "N/D")

        # =====================================================
        # Sección 1: Resumen General
        # =====================================================
        st.header("📋 Resumen General (agrupado por Sector)")
        resumen_cols = ["Ticker", "Sector", "Precio", "P/E", "P/B", "P/FCF",
                        "Dividend Yield %", "Payout Ratio", "ROA", "ROE",
                        "Current Ratio", "Debt/Eq", "Oper Margin", "Profit Margin",
                        "WACC", "ROIC", "EVA"]
        st.dataframe(
            df_disp[resumen_cols].dropna(how='all', axis=1),
            use_container_width=True,
            height=420
        )

        if errs:
            st.subheader("🚫 Tickers con error")
            st.table(pd.DataFrame(errs))

        # Lista de sectores presentes (ya ordenados)
        sectors = df_plot["Sector"].dropna().unique().tolist()

        # Helper para limitar por sector (Top N por MarketCap y mínimo de empresas)
        def sector_slice(sec_df):
            sec_df = sec_df.sort_values("MarketCap", ascending=False).head(top_n)
            return sec_df if len(sec_df) >= min_emp else pd.DataFrame(columns=sec_df.columns)

        # =====================================================
        # Sección 2: Análisis de Valoración (por sector)
        # =====================================================
        st.header("💰 Análisis de Valoración (por Sector)")
        for s in sectors:
            sec = sector_slice(df_plot[df_plot["Sector"] == s])
            if sec.empty:
                continue
            with st.expander(f"Sector: {s}  •  {len(sec)} empresas", expanded=False):
                c1, c2 = st.columns(2)

                with c1:
                    st.caption("Ratios de Valoración (P/E, P/B, P/FCF)")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sec_val = sec[["Ticker", "P/E", "P/B", "P/FCF"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                    sec_val.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("Ratio")
                    st.pyplot(fig); plt.close()

                with c2:
                    st.caption("Dividend Yield (%)")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    dy = pct(sec["Dividend Yield %"])
                    pd.DataFrame({"Dividend Yield %": dy.values}, index=sec["Ticker"]).plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%"); ax.set_ylim(-ylim_abs, ylim_abs)
                    st.pyplot(fig); plt.close()

        # =====================================================
        # Sección 3: Rentabilidad y Eficiencia (por sector)
        # =====================================================
        st.header("📈 Rentabilidad y Eficiencia (por Sector)")
        for s in sectors:
            sec = sector_slice(df_plot[df_plot["Sector"] == s])
            if sec.empty: continue
            with st.expander(f"Sector: {s}  •  {len(sec)} empresas", expanded=False):
                t1, t2, t3 = st.columns(3)

                with t1:
                    st.caption("ROE vs ROA (%)")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    rr = pd.DataFrame({
                        "ROE": pct(sec["ROE"]).values,
                        "ROA": pct(sec["ROA"]).values
                    }, index=sec["Ticker"])
                    rr.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%"); ax.set_ylim(-ylim_abs, ylim_abs)
                    st.pyplot(fig); plt.close()

                with t2:
                    st.caption("Márgenes (%)")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    mm = pd.DataFrame({
                        "Oper Margin": pct(sec["Oper Margin"]).values,
                        "Profit Margin": pct(sec["Profit Margin"]).values
                    }, index=sec["Ticker"])
                    mm.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%"); ax.set_ylim(-ylim_abs, ylim_abs)
                    st.pyplot(fig); plt.close()

                with t3:
                    st.caption("ROIC vs WACC (%)")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    rw = pd.DataFrame({
                        "ROIC": pct(sec["ROIC"]).values,
                        "WACC": pct(sec["WACC"]).values
                    }, index=sec["Ticker"])
                    rw.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%"); ax.set_ylim(-ylim_abs, ylim_abs)
                    st.pyplot(fig); plt.close()

        # =====================================================
        # Sección 4: Estructura de Capital y Liquidez (por sector)
        # =====================================================
        st.header("🏦 Estructura de Capital y Liquidez (por Sector)")
        for s in sectors:
            sec = sector_slice(df_plot[df_plot["Sector"] == s])
            if sec.empty: continue
            with st.expander(f"Sector: {s}  •  {len(sec)} empresas", expanded=False):
                c3, c4 = st.columns(2)

                with c3:
                    st.caption("Apalancamiento")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    lev = sec[["Ticker", "Debt/Eq", "LtDebt/Eq"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                    lev.plot(kind="bar", stacked=True, ax=ax, rot=45)
                    ax.axhline(1, color="red", linestyle="--")
                    ax.set_ylabel("Ratio")
                    st.pyplot(fig); plt.close()

                with c4:
                    st.caption("Liquidez")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    liq = sec[["Ticker", "Current Ratio", "Quick Ratio"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                    liq.plot(kind="bar", ax=ax, rot=45)
                    ax.axhline(1, color="green", linestyle="--")
                    ax.set_ylabel("Ratio")
                    st.pyplot(fig); plt.close()

        # =====================================================
        # Sección 5: Crecimiento (por sector)
        # =====================================================
        st.header("🚀 Crecimiento (CAGR 3-4 años, por Sector)")
        for s in sectors:
            sec = sector_slice(df_plot[df_plot["Sector"] == s])
            if sec.empty: continue
            with st.expander(f"Sector: {s}  •  {len(sec)} empresas", expanded=False):
                fig, ax = plt.subplots(figsize=(12, 6))
                gdf = pd.DataFrame({
                    "Revenue Growth": pct(sec["Revenue Growth"]).values,
                    "EPS Growth": pct(sec["EPS Growth"]).values,
                    "FCF Growth": pct(sec["FCF Growth"]).values
                }, index=sec["Ticker"])
                gdf.plot(kind="bar", ax=ax, rot=45)
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_ylabel("%"); ax.set_ylim(-ylim_abs, ylim_abs)
                st.pyplot(fig); plt.close()

        # =====================================================
        # Sección 6: Análisis individual
        # =====================================================
        st.header("🔍 Análisis por Empresa")
        pick = st.selectbox("Selecciona empresa", df_disp["Ticker"].unique())
        det = df_disp[df_disp["Ticker"] == pick].iloc[0]

        cA, cB, cC = st.columns(3)
        with cA:
            st.metric("Precio", f"${det['Precio']:,.2f}" if det['Precio'] else "N/D")
            st.metric("P/E", det["P/E"])
            st.metric("P/B", det["P/B"])
        with cB:
            st.metric("ROIC", det["ROIC"])
            st.metric("WACC", det["WACC"])
            st.metric("EVA", f"{det['EVA']:,.0f}" if pd.notnull(det["EVA"]) else "N/D")
        with cC:
            st.metric("ROE", det["ROE"])
            st.metric("Dividend Yield", det["Dividend Yield %"])
            st.metric("Debt/Eq", det["Debt/Eq"])

        st.subheader("ROIC vs WACC")
        if det["ROIC"] != "N/D" and det["WACC"] != "N/D":
            r_val = float(str(det["ROIC"]).rstrip("%"))
            w_val = float(str(det["WACC"]).rstrip("%"))
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(["ROIC", "WACC"], [r_val, w_val],
                   color=["green" if r_val > w_val else "red", "gray"])
            ax.set_ylabel("%")
            st.pyplot(fig); plt.close()
            st.success("✅ Crea valor (ROIC > WACC)" if r_val > w_val else "❌ Destruye valor (ROIC < WACC)")
        else:
            st.info("Datos insuficientes para comparar ROIC/WACC")

# -------------------------------------------------------------
if __name__ == "__main__":
    main()
