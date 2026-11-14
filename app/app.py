from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
st.set_page_config(page_title='Prognos: Turbofan RUL Prediction', layout='wide')


@st.cache_data
def load_data():
    demo = pd.read_csv(HERE / 'demo_data.csv.gz')
    final = pd.read_csv(HERE / 'final_results.csv')
    base = pd.read_csv(HERE / 'baseline_results.csv')
    return demo, final, base


demo, final, base = load_data()

st.title('Prognos: Remaining Useful Life Prediction for Turbofan Engines')
st.caption('NASA C-MAPSS | GRU ensemble of 5 networks per subset | predictions use only past cycles')

tab1, tab2, tab3 = st.tabs(['Engine explorer', 'Results', 'About'])

# ------------------------------------------------------------------ Engine explorer
with tab1:
    c1, c2, c3 = st.columns(3)
    fd = c1.selectbox('Dataset subset', sorted(demo['subset'].unique()))
    d = demo[demo['subset'] == fd]
    unit = c2.selectbox('Test engine', sorted(d['unit'].unique()))
    e = d[d['unit'] == unit].sort_values('cycle')
    last = int(e['cycle'].max())
    cyc = c3.slider('Engine age: cycles observed so far', int(e['cycle'].min()), last, last)
    thr = st.slider('Maintenance alert when predicted RUL falls below (cycles)', 10, 60, 30)

    seen = e[e['cycle'] <= cyc]
    now = seen.iloc[-1]

    m1, m2, m3 = st.columns(3)
    m1.metric('Predicted RUL', f'{now.pred_rul:.0f} cycles', f'model spread +/- {now.pred_std:.1f}', delta_color='off')
    m2.metric('True RUL (for evaluation)', f'{now.true_rul:.0f} cycles')
    m3.metric('Error (predicted - true)', f'{now.pred_rul - now.true_rul:+.0f} cycles')

    if now.pred_rul < thr:
        st.error(f'Maintenance alert: predicted RUL ({now.pred_rul:.0f}) is below the threshold of {thr} cycles.')
    else:
        st.success(f'No alert: predicted RUL ({now.pred_rul:.0f}) is above the threshold of {thr} cycles.')

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(e['cycle'], e['true_rul'], 'k--', lw=1, label='true RUL (known only for evaluation)')
    ax.fill_between(seen['cycle'], seen['pred_rul'] - seen['pred_std'], seen['pred_rul'] + seen['pred_std'],
                    alpha=0.25, label='spread of the 5 networks')
    ax.plot(seen['cycle'], seen['pred_rul'], lw=2, label='predicted RUL')
    ax.axhline(thr, color='r', ls=':', lw=1, label='alert threshold')
    ax.axvline(cyc, color='grey', lw=0.8)
    ax.set_xlabel('cycle'); ax.set_ylabel('RUL (cycles)')
    ax.set_title(f'{fd}, test engine {unit}')
    ax.legend(loc='upper right')
    st.pyplot(fig)

    with st.expander('Sensor signals (normalised per operating condition)'):
        st.line_chart(seen.set_index('cycle')[['s4', 's7', 's11', 's12']])

    st.caption('For test engines the true RUL is known only at the last cycle (NASA RUL file); '
               'earlier values are derived as RUL_last + (last cycle - cycle). '
               'The models were trained with the RUL capped at 125, so predictions flatten near 125 early in life.')

# ------------------------------------------------------------------ Results
with tab2:
    st.subheader('GRU (chosen configuration, mean of 5 seeds) vs. best untuned baseline')
    ch = final[final['chosen']].set_index('subset')
    best_base = base[base['model'] != 'Dummy (mean)'].sort_values('test_rmse').groupby('subset').head(1).set_index('subset')

    table = pd.DataFrame({
        'GRU test RMSE (5 seeds)': ch.apply(lambda r: f'{r.rmse_mean:.2f} ± {r.rmse_std:.2f}', axis=1),
        'GRU NASA score': ch.apply(lambda r: f'{r.nasa_mean:,.0f} ± {r.nasa_std:,.0f}', axis=1),
        'Best baseline': best_base['model'],
        'Baseline test RMSE': best_base['test_rmse'].round(2),
        'RMSE reduction': ((1 - ch['rmse_mean'] / best_base['test_rmse']) * 100).round(0).astype(int).astype(str) + '%',
        'GRU RMSE vs capped truth': ch['rmse_capped_mean'].round(2),
    })
    st.dataframe(table, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(ch))
    ax2.bar(x - 0.2, best_base.loc[ch.index, 'test_rmse'], 0.4, label='best baseline')
    ax2.bar(x + 0.2, ch['rmse_mean'], 0.4, yerr=ch['rmse_std'], label='GRU (5 seeds)')
    ax2.set_xticks(x); ax2.set_xticklabels(ch.index)
    ax2.set_ylabel('test RMSE (cycles)'); ax2.legend()
    st.pyplot(fig2)

    st.info('Evaluation: one prediction per test engine at its last recorded cycle, compared with the true RUL '
            'from NASA\'s RUL file (lower is better). FD002 and FD004 have a much larger raw RMSE mainly because '
            'the models are trained with the RUL capped at 125 while some test engines have a true RUL of about 195; '
            'against the capped truth the RMSE is about 14-15.')

# ------------------------------------------------------------------ About
with tab3:
    st.markdown("""
**What is this?** A model that reads the sensor history of a jet engine and estimates how many cycles it can still run (Remaining Useful Life, RUL), so that maintenance can be planned before a failure.

**Data:** NASA C-MAPSS turbofan degradation simulation (four subsets, FD001-FD004, with 1 or 6 operating conditions and 1 or 2 fault modes).

**Pipeline:** data verification -> EDA -> per-condition normalisation (fitted on training data only) -> 30-cycle sliding windows -> baselines (Ridge, Random Forest, XGBoost) -> LSTM / GRU / 1D-CNN -> Optuna tuning on validation engines -> 5-seed evaluation -> error analysis.

**Main findings:** the GRU is accurate close to failure (RMSE of roughly 2.5 to 6.4 cycles when fewer than 30 cycles remain); its main weakness is late warnings at medium RUL (60-100 cycles).

**Limitations:** results are comparable to classic LSTM / CNN-LSTM papers but behind recent attention-based models; published results use different protocols, so comparisons are approximate.

**Data source:** A. Saxena and K. Goebel (2008), Turbofan Engine Degradation Simulation Data Set, NASA Ames Prognostics Data Repository.
""")