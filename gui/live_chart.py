# gui/live_chart.py
from PyQt6 import uic
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
import pandas as pd
import json
import numpy as np

from utils.utils import safe_float


class LiveChart(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.lastTime = None
        self.lastSigc = 0
        self.origSigc = 0
        self.webview = QWebEngineView()
        layout = QVBoxLayout(self)
        layout.addWidget(self.webview)
        self.setLayout(layout)

        # --- HTML + JS ---
        self.html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
            <style>
                body, html { margin:0; padding:0; height:100%; }
                #chart { width:100%; height:100%; }
            </style>
        </head>
        <body>
            <div id="chart"></div>
            <script>
                var layout = {
                    autosize: true,
                    margin: { l: 50, r: 50, t: 30, b: 50 },
                    xaxis: { title: "Temps", type: 'date' },
                    yaxis: { title: "Prix" },
                    legend: { x: 0, y: 1, bgcolor: 'rgba(255,255,255,0.8)', bordercolor: 'black', borderwidth: 1 },
                    hovermode: 'closest'
                };
                Plotly.newPlot('chart', [], layout, {responsive: true});
                function updateChart(data) {
                    Plotly.react('chart', data.traces, data.layout);
                }
            </script>
        </body>
        </html>
        """
        self.webview.setHtml(self.html_template)

    def _to_serializable(self, val):
        if isinstance(val, (pd.Timestamp, np.datetime64)):
            return val.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(val, (np.integer, np.int64)):
            return int(val)
        if isinstance(val, (np.floating, np.float64)):
            return float(val)
        if isinstance(val, pd.Timedelta):
            return val.total_seconds()
        return val

    def update_display(self, data):
        df = data.get('df')
        current_bid = data.get('current_bid', 0)
        if df is None or df.empty:
            return

        sigcChanged = False
        if self.lastTime is None or self.lastTime != df.index[-1]:
            self.lastTime = df.index[-1]
            self.origSigc = df['sigc'].iloc[-1] if 'sigc' in df.columns and df['sigc'].iloc[-1] == 4 else 0
        else:
            newSigc = df['sigc'].iloc[-1] if 'sigc' in df.columns and df['sigc'].iloc[-1] == 4 else 0
            if newSigc != self.origSigc:
                sigcChanged = True

        cl_value = 'close_renko' if 'close_renko' in df.columns else 'close'
        op_value = 'open_renko' if 'open_renko' in df.columns else 'open'

        traces = []
        legend_up_shown = False
        legend_down_shown = False

        brick_size = abs(df[cl_value].iloc[0] - df[op_value].iloc[0]) if len(df) > 1 else 0.0001
        min_interval = df.index.to_series().diff().dt.total_seconds().dropna().min() if len(df) > 1 else 1800
        half_width = pd.Timedelta(seconds=min_interval * 0.4)

        # === BRIQUES + HOVER ===
        for _, row in df.iterrows():
            t = row.name
            open_val = row[op_value]
            close_val = row[cl_value]
            high = row['high']
            low = row['low']
            base = min(open_val, close_val)
            is_up = close_val >= open_val
            color = 'rgba(0, 255, 0, 0.8)' if is_up else 'rgba(255, 0, 0, 0.8)'
            name = 'Renko Up' if is_up else 'Renko Down'

            show_legend = False
            if is_up and not legend_up_shown:
                show_legend = True
                legend_up_shown = True
            elif not is_up and not legend_down_shown:
                show_legend = True
                legend_down_shown = True

            # === HOVER SÉCURISÉ ===
            try:
                hover_text = (
                    f"<b>Renko Brick</b><br>"
                    f"Time: {t.strftime('%H:%M:%S')}<br>"
                    f"Open: {safe_float(open_val, fmt='.2f')}<br>"
                    f"Close: {safe_float(close_val, fmt='.2f')}<br>"
                    f"High: {safe_float(high, fmt='.2f')}<br>"
                    f"Low: {safe_float(low, fmt='.2f')}<br>"
                    f"RSI: {safe_float(row.get('RSI'), fmt='.1f')}<br>"
                    f"MACD: {safe_float(row.get('MACD_line'), fmt='.3f')}"
                )
            except Exception as e:
                print(f"Hover error: {e}")
                hover_text = "Erreur hover"

            traces.append({
                'type': 'scatter', 'x': [self._to_serializable(t)], 'y': [base + brick_size / 2],
                'mode': 'markers', 'marker': {'size': 1, 'opacity': 0}, 'name': name,
                'showlegend': False, 'hoverinfo': 'text', 'text': [hover_text],
                'hoverlabel': {'bgcolor': 'rgba(0,0,0,0.8)', 'font': {'color': 'yellow', 'size': 11}}
            })

            # Rectangle
            x_rect = [self._to_serializable(t - half_width), self._to_serializable(t - half_width),
                      self._to_serializable(t + half_width), self._to_serializable(t + half_width),
                      self._to_serializable(t - half_width)]
            y_rect = [base, base + brick_size, base + brick_size, base, base]

            traces.append({
                'type': 'scatter', 'x': x_rect, 'y': y_rect, 'fill': 'toself',
                'fillcolor': color, 'line': {'color': color, 'width': 1},
                'mode': 'lines', 'name': name, 'showlegend': show_legend, 'hoverinfo': 'skip'
            })

        # === Ligne BID ===
        traces.append({
            'type': 'scatter',
            'x': [self._to_serializable(df.index[0]), self._to_serializable(df.index[-1])],
            'y': [current_bid, current_bid],
            'mode': 'lines',
            'line': {'color': 'blue', 'dash': 'dash', 'width': 1.5},
            'name': f'Bid: {current_bid:.2f}',
            'hoverinfo': 'skip'
        })

        # === EMA ===
        if 'EMA' in df.columns:
            ema = df['EMA'].dropna()
            if not ema.empty:
                traces.append({
                    'type': 'scatter',
                    'x': [self._to_serializable(t) for t in ema.index],
                    'y': ema.values.tolist(),
                    'mode': 'lines',
                    'line': {'color': 'orange', 'width': 2},
                    'name': 'EMA9'
                })

        # === BOLLINGER BANDS ===
        if all(col in df.columns for col in ['bb_mavg', 'bb_hband', 'bb_lband']):
            bb_mavg = df['bb_mavg'].dropna()
            bb_hband = df['bb_hband'].dropna()
            bb_lband = df['bb_lband'].dropna()
            if not bb_mavg.empty:
                traces.append({
                    'type': 'scatter', 'x': [self._to_serializable(t) for t in bb_mavg.index],
                    'y': bb_mavg.values.tolist(), 'mode': 'lines',
                    'line': {'color': 'blue', 'width': 1}, 'name': 'BB Moyenne'
                })
                traces.append({
                    'type': 'scatter', 'x': [self._to_serializable(t) for t in bb_hband.index],
                    'y': bb_hband.values.tolist(), 'mode': 'lines',
                    'line': {'color': 'gray', 'dash': 'dash', 'width': 1}, 'name': 'BB Haut', 'showlegend': False
                })
                traces.append({
                    'type': 'scatter', 'x': [self._to_serializable(t) for t in bb_lband.index],
                    'y': bb_lband.values.tolist(), 'mode': 'lines',
                    'line': {'color': 'gray', 'dash': 'dash', 'width': 1}, 'name': 'BB Bas', 'showlegend': False
                })

        # === SIGNAUX (Rules) ===
        if 'sigo' in df.columns:
            buy = df[df['sigo'] == 1]
            sell = df[df['sigo'] == -1]
            if not buy.empty:
                traces.append({
                    'type': 'scatter', 'x': [self._to_serializable(t) for t in buy.index],
                    'y': buy[cl_value].tolist(), 'mode': 'markers',
                    'marker': {'symbol': 'triangle-up', 'color': 'lime', 'size': 10},
                    'name': 'Achat (Rule)'
                })
            if not sell.empty:
                traces.append({
                    'type': 'scatter', 'x': [self._to_serializable(t) for t in sell.index],
                    'y': sell[cl_value].tolist(), 'mode': 'markers',
                    'marker': {'symbol': 'triangle-down', 'color': 'red', 'size': 10},
                    'name': 'Vente (Rule)'
                })

        if 'sigc' in df.columns:
            clos_5 = df[df['sigc'] == 5]
            clos_4 = df[df['sigc'] == 4]
            colorc = 'purple' if not sigcChanged and t==self.lastTime else 'pink'
            """
            clos_6 = df[df['sigc'] == 6]
            if not clos_6.empty:
                traces.append({
                    'type': 'scatter', 'x': [self._to_serializable(t) for t in clos_6.index],
                    'y': clos_6[cl_value].tolist(), 'mode': 'markers',
                    'marker': {'symbol': 'circle', 'color': 'blue', 'size': 5},
                    'name': 'Clôture (6)'
                })
            if not clos_5.empty:
                traces.append({
                    'type': 'scatter', 'x': [self._to_serializable(t) for t in clos_5.index],
                    'y': clos_5[cl_value].tolist(), 'mode': 'markers',
                    'marker': {'symbol': 'circle', 'color': 'orange', 'size': 5},
                    'name': 'Clôture (5)'
                })
            """
            if not clos_4.empty:
                traces.append({
                    'type': 'scatter', 'x': [self._to_serializable(t) for t in clos_4.index],
                    'y': clos_4[cl_value].tolist(), 'mode': 'markers',
                    'marker': {'symbol': 'circle', 'color': colorc, 'size': 5},
                    'name': 'Clôture (4)'
                })

        # === SIGNAUX (IA) ===
        if 'pred_signal' in df.columns:
            scale = (df[cl_value].max() - df[cl_value].min()) * 0.02
            buy_ia = df[df['pred_signal'] > 0.5]
            sell_ia = df[df['pred_signal'] < -0.5]
            if not buy_ia.empty:
                traces.append({
                    'type': 'scatter', 'x': [self._to_serializable(t) for t in buy_ia.index],
                    'y': (buy_ia[cl_value] + scale).tolist(), 'mode': 'markers',
                    'marker': {'symbol': 'star', 'color': 'cyan', 'size': 9},
                    'name': 'Achat (IA)'
                })
            if not sell_ia.empty:
                traces.append({
                    'type': 'scatter', 'x': [self._to_serializable(t) for t in sell_ia.index],
                    'y': (sell_ia[cl_value] - scale).tolist(), 'mode': 'markers',
                    'marker': {'symbol': 'star', 'color': 'magenta', 'size': 9},
                    'name': 'Vente (IA)'
                })

        # === ENVOI ===
        js_code = f"""
        updateChart({{
            traces: {json.dumps(traces)},
            layout: {{}}
        }});
        """
        self.webview.page().runJavaScript(js_code)
