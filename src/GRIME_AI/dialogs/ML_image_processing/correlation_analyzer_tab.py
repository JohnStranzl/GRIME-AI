#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# correlation_analyzer_tab.py
# Correlates GRIME AI ROI metrics with user-supplied sensor CSV files.
# Author: John Edward Stranzl, Jr.
# License: Apache License, Version 2.0

import os
import re
import glob
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from fpdf import FPDF

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QFileDialog, QMessageBox, QListWidgetItem,
                              QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout,
                              QLabel, QComboBox, QCheckBox, QGroupBox, QFormLayout,
                              QSizePolicy, QAbstractItemView, QHeaderView, QPushButton,
                              QScrollArea, QGridLayout)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor

from GRIME_AI import PROJECT_ROOT
from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor
from GRIME_AI.GRIME_AI_CSS_Styles import BUTTON_CSS_STEEL_BLUE


# ──────────────────────────────────────────────────────────────────────────────
# Datetime detection helpers
# ──────────────────────────────────────────────────────────────────────────────

DATETIME_KEYWORDS = re.compile(
    r'(datetime|timestamp|date_time|date time|dt)', re.IGNORECASE
)
DATE_KEYWORDS = re.compile(
    r'\bdate\b', re.IGNORECASE
)
TIME_KEYWORDS = re.compile(
    r'\btime\b', re.IGNORECASE
)

DATETIME_FORMATS = [
    '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d',
    '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M', '%m/%d/%Y',
    '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M', '%d/%m/%Y',
    '%Y%m%d %H:%M:%S', '%Y%m%d',
    '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S',
]


def try_parse_datetime(series: pd.Series, n_sample: int = 20) -> bool:
    """Return True if the series looks like datetimes."""
    sample = series.dropna().head(n_sample).astype(str)
    parsed = 0
    for val in sample:
        try:
            pd.to_datetime(val)
            parsed += 1
        except Exception:
            pass
    return parsed / max(len(sample), 1) >= 0.7


def detect_datetime_columns(df: pd.DataFrame) -> dict:
    """
    Returns a dict with keys 'datetime', 'date', 'time' mapping to
    column names (or None).  Always scans for all three regardless of
    whether a combined datetime column is found.
    """
    result = {'datetime': None, 'date': None, 'time': None}

    for col in df.columns:
        if DATETIME_KEYWORDS.search(col) and try_parse_datetime(df[col]):
            result['datetime'] = col
            break  # found combined — but keep scanning for date/time below

    for col in df.columns:
        if DATE_KEYWORDS.search(col) and try_parse_datetime(df[col]):
            result['date'] = col
            break

    for col in df.columns:
        if TIME_KEYWORDS.search(col):
            result['time'] = col
            break

    return result


def build_datetime_series(df: pd.DataFrame, detection: dict,
                           date_col: str, time_col: str,
                           datetime_col: str) -> pd.Series:
    """
    Construct a UTC-naive datetime Series from user selections.
    Uses pandas default flexible datetime parsing.
    """
    if datetime_col and datetime_col in df.columns:
        return pd.to_datetime(df[datetime_col],
                              errors='coerce')
    if date_col and time_col and date_col in df.columns and time_col in df.columns:
        combined = df[date_col].astype(str).str.strip() + ' ' + \
                   df[time_col].astype(str).str.strip()
        return pd.to_datetime(combined, errors='coerce')
    if date_col and date_col in df.columns:
        return pd.to_datetime(df[date_col], errors='coerce')
    return pd.Series(pd.NaT, index=df.index)


# ──────────────────────────────────────────────────────────────────────────────
# Sensor file tab widget
# ──────────────────────────────────────────────────────────────────────────────

class SensorFileTab(QWidget):
    """
    One tab per sensor CSV.  Shows:
      - Auto-detected / user-overridable datetime column assignments
      - 5-row preview table
      - Checkable list of data columns to include
    """

    def __init__(self, filepath: str, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.df = None
        self.full_df = None
        self.detection = {}
        self._build_ui()
        self._load_file()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # ── datetime assignment ──────────────────────────────────────────────
        grp_dt = QGroupBox("Date / Time Columns")
        form = QFormLayout(grp_dt)

        self.combo_datetime = QComboBox()
        self.combo_datetime.addItem("— none —")
        form.addRow("Combined datetime:", self.combo_datetime)

        self.combo_date = QComboBox()
        self.combo_date.addItem("— none —")
        form.addRow("Date column:", self.combo_date)

        self.combo_time = QComboBox()
        self.combo_time.addItem("— none —")
        form.addRow("Time column:", self.combo_time)

        layout.addWidget(grp_dt)

        # ── preview ──────────────────────────────────────────────────────────
        grp_preview = QGroupBox("Preview (first 5 rows)")
        v = QVBoxLayout(grp_preview)
        self.table_preview = QTableWidget()
        self.table_preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_preview.setMinimumHeight(110)
        self.table_preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        v.addWidget(self.table_preview)
        layout.addWidget(grp_preview)

        # ── data columns ─────────────────────────────────────────────────────
        grp_cols = QGroupBox("Data Columns to Correlate")
        v2 = QVBoxLayout(grp_cols)

        btn_row = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_none = QPushButton("None")
        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addStretch()
        v2.addLayout(btn_row)

        from PyQt5.QtWidgets import QListWidget
        self.list_columns = QListWidget()
        self.list_columns.setMinimumHeight(100)
        v2.addWidget(self.list_columns)
        layout.addWidget(grp_cols)

    # ── file loading ──────────────────────────────────────────────────────────
    def _load_file(self):
        try:
            # Full file for correlation — drop all-NaN rows from any bloated CSVs
            self.full_df = pd.read_csv(self.filepath, low_memory=False)
            self.full_df = self.full_df.dropna(how='all')
            # Preview only — first 5 valid rows
            self.df = self.full_df.head(2000)
        except Exception as e:
            QMessageBox.warning(self, "Load Error",
                                f"Could not read {self.filepath}:\n{e}")
            return

        cols = list(self.df.columns)

        # Populate combos
        for combo in (self.combo_datetime, self.combo_date, self.combo_time):
            combo.addItems(cols)

        # Auto-detect
        self.detection = detect_datetime_columns(self.df)
        self._apply_detection()

        # Preview
        self._fill_preview()

        # Data columns — exclude any column that looks like a datetime/date/time
        # column, whether auto-detected or not, and exclude quality-code columns
        dt_cols = {v for v in self.detection.values() if v}
        for col in cols:
            if col in dt_cols:
                continue
            if (DATETIME_KEYWORDS.search(col) or
                    DATE_KEYWORDS.search(col) or
                    TIME_KEYWORDS.search(col)):
                continue
            item = QListWidgetItem(col)
            item.setCheckState(Qt.Unchecked)
            self.list_columns.addItem(item)

    def _apply_detection(self):
        def set_combo(combo, value):
            if value:
                idx = combo.findText(value)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        set_combo(self.combo_datetime, self.detection.get('datetime'))
        set_combo(self.combo_date,     self.detection.get('date'))
        set_combo(self.combo_time,     self.detection.get('time'))

    def _fill_preview(self):
        preview = self.df.head(5)
        self.table_preview.setColumnCount(len(preview.columns))
        self.table_preview.setRowCount(len(preview))
        self.table_preview.setHorizontalHeaderLabels(list(preview.columns))
        for r, (_, row) in enumerate(preview.iterrows()):
            for c, val in enumerate(row):
                self.table_preview.setItem(r, c, QTableWidgetItem(str(val)))

    # ── helpers ───────────────────────────────────────────────────────────────
    def _select_all(self):
        for i in range(self.list_columns.count()):
            self.list_columns.item(i).setCheckState(Qt.Checked)

    def _select_none(self):
        for i in range(self.list_columns.count()):
            self.list_columns.item(i).setCheckState(Qt.Unchecked)

    def get_selected_data_columns(self) -> list:
        result = []
        for i in range(self.list_columns.count()):
            item = self.list_columns.item(i)
            if item.checkState() == Qt.Checked:
                result.append(item.text())
        return result

    def get_datetime_selection(self) -> dict:
        def val(combo):
            t = combo.currentText()
            return None if t == "— none —" else t
        return {
            'datetime': val(self.combo_datetime),
            'date':     val(self.combo_date),
            'time':     val(self.combo_time),
        }

    def build_aligned_df(self) -> pd.DataFrame:
        """Return a DataFrame with a '_datetime' index + selected data columns."""
        if self.full_df is None:
            return pd.DataFrame()
        sel = self.get_datetime_selection()
        dt_series = build_datetime_series(self.full_df, self.detection,
                                          sel['date'], sel['time'], sel['datetime'])
        data_cols = self.get_selected_data_columns()
        if not data_cols:
            return pd.DataFrame()
        out = self.full_df[data_cols].copy()
        out.insert(0, '_datetime', dt_series)
        out = out.dropna(subset=['_datetime'])
        out = out.sort_values('_datetime')
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Worker thread
# ──────────────────────────────────────────────────────────────────────────────

class CorrelationWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)   # path to PDF
    error    = pyqtSignal(str)

    def __init__(self, roi_df, sensor_dfs, roi_cols, method, tolerance_h,
                 output_dir, parent=None):
        super().__init__(parent)
        self.roi_df      = roi_df          # ROI metrics with '_datetime' col
        self.sensor_dfs  = sensor_dfs      # list of (label, df) with '_datetime'
        self.roi_cols    = roi_cols
        self.method      = method          # 'pearson' | 'spearman' | 'kendall'
        self.tolerance_h = tolerance_h
        self.output_dir  = output_dir

    def run(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        tol = pd.Timedelta(hours=self.tolerance_h)

        # ── 1. Align all sensor files against ROI timestamps ─────────────────
        self.progress.emit("Aligning sensor data to ROI timestamps…")
        merged = self.roi_df[['_datetime'] + self.roi_cols].copy()
        merged = merged.sort_values('_datetime')

        for label, sdf in self.sensor_dfs:
            sdf = sdf.sort_values('_datetime')
            sensor_cols = [c for c in sdf.columns if c != '_datetime']
            renamed = {c: f"{label}::{c}" for c in sensor_cols}
            sdf = sdf.rename(columns=renamed)
            merged = pd.merge_asof(
                merged, sdf,
                on='_datetime',
                direction='nearest',
                tolerance=tol
            )

        n_aligned = merged.dropna(how='all',
                                   subset=[c for c in merged.columns
                                           if c != '_datetime']).shape[0]
        self.progress.emit(
            f"Aligned {n_aligned} / {len(merged)} ROI observations.")

        # Collect all sensor column names
        sensor_cols_all = [c for c in merged.columns
                           if c not in ['_datetime'] + self.roi_cols]

        if not sensor_cols_all:
            self.error.emit("No sensor columns found after alignment.")
            return

        # ── 2. Correlation table ──────────────────────────────────────────────
        self.progress.emit("Computing correlations…")

        # Coerce all data columns to numeric — skips quality-code string columns
        for col in self.roi_cols + sensor_cols_all:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors='coerce')

        corr_rows = []
        for rc in self.roi_cols:
            for sc in sensor_cols_all:
                pair = merged[[rc, sc]].dropna()
                if len(pair) < 5:
                    continue
                # Skip if either column is still non-numeric after coercion
                if not (pd.api.types.is_numeric_dtype(pair[rc]) and
                        pd.api.types.is_numeric_dtype(pair[sc])):
                    continue
                x, y = pair[rc].values.astype(float), pair[sc].values.astype(float)
                try:
                    if self.method == 'pearson':
                        r, p = stats.pearsonr(x, y)
                    elif self.method == 'spearman':
                        r, p = stats.spearmanr(x, y)
                    else:
                        r, p = stats.kendalltau(x, y)
                except Exception:
                    continue
                sig = ('***' if p < 0.001 else '**' if p < 0.01
                        else '*' if p < 0.05 else 'ns')
                corr_rows.append({
                    'ROI Column': rc,
                    'Sensor Column': sc,
                    'r': round(r, 4),
                    'p': round(p, 6),
                    'n': len(pair),
                    'Significance': sig,
                })

        corr_df = pd.DataFrame(corr_rows)
        corr_df.to_csv(
            os.path.join(self.output_dir, 'correlation_table.csv'), index=False)

        # ── 3. Build PDF ──────────────────────────────────────────────────────
        self.progress.emit("Generating PDF report…")
        pdf_path = os.path.join(self.output_dir, 'correlation_report.pdf')

        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=True, margin=10)
        pdf.set_margins(10, 10, 10)

        # ── Title page ────────────────────────────────────────────────────────
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, 'GRIME AI - Correlation Analysis Report', ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 7,
                 f"Method: {self.method.capitalize()}    "
                 f"Tolerance: {self.tolerance_h} h    "
                 f"ROI observations: {len(merged)}", ln=True)
        pdf.ln(4)

        # ── Correlation table ─────────────────────────────────────────────────
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, 'Correlation Summary', ln=True)
        pdf.ln(2)

        if not corr_df.empty:
            col_widths = [55, 65, 18, 22, 14, 28]
            headers = ['ROI Column', 'Sensor Column', 'r', 'p-value', 'n', 'Sig.']
            pdf.set_font('Helvetica', 'B', 8)
            pdf.set_fill_color(46, 134, 171)
            pdf.set_text_color(255, 255, 255)
            for w, h in zip(col_widths, headers):
                pdf.cell(w, 6, h, border=1, fill=True)
            pdf.ln()
            pdf.set_text_color(0, 0, 0)
            fill = False
            for _, row in corr_df.iterrows():
                pdf.set_fill_color(240, 244, 248) if fill else pdf.set_fill_color(255, 255, 255)
                pdf.set_font('Helvetica', '', 7)
                vals = [str(row['ROI Column']), str(row['Sensor Column']),
                        f"{row['r']:.4f}", f"{row['p']:.6f}",
                        str(row['n']), str(row['Significance'])]
                for w, v in zip(col_widths, vals):
                    pdf.cell(w, 5, v, border=1, fill=True)
                pdf.ln()
                fill = not fill
        else:
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(0, 7, 'No valid correlation pairs found.', ln=True)

        # ── Heatmap ───────────────────────────────────────────────────────────
        self.progress.emit("Generating heatmap…")
        heatmap_path = self._make_heatmap(corr_df, merged)
        if heatmap_path:
            pdf.add_page()
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, 'Correlation Heatmap', ln=True)
            pdf.image(heatmap_path, x=10, y=None, w=267)

        # ── Scatter plots ─────────────────────────────────────────────────────
        self.progress.emit("Generating scatter plots…")
        if not corr_df.empty:
            top_pairs = (corr_df.assign(abs_r=corr_df['r'].abs())
                         .sort_values('abs_r', ascending=False)
                         .head(20))
            scatter_paths = []
            for _, row in top_pairs.iterrows():
                path = self._make_scatter(
                    merged, row['ROI Column'], row['Sensor Column'],
                    row['r'], row['p'], row['n'])
                if path:
                    scatter_paths.append(path)

            if scatter_paths:
                pdf.add_page()
                pdf.set_font('Helvetica', 'B', 12)
                pdf.cell(0, 8, 'Scatter Plots (top pairs by |r|)', ln=True)
                # 2 per row
                for i in range(0, len(scatter_paths), 2):
                    row_paths = scatter_paths[i:i+2]
                    y_before = pdf.get_y()
                    for j, p in enumerate(row_paths):
                        pdf.image(p, x=10 + j*135, y=y_before, w=130)
                    pdf.set_y(y_before + 90)
                    if pdf.get_y() > 170 and i + 2 < len(scatter_paths):
                        pdf.add_page()
                        pdf.set_font('Helvetica', 'B', 12)
                        pdf.cell(0, 8, 'Scatter Plots (continued)', ln=True)

        # ── Time series ───────────────────────────────────────────────────────
        self.progress.emit("Generating time series plots…")
        ts_paths = self._make_timeseries(merged, sensor_cols_all)
        if ts_paths:
            pdf.add_page()
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, 'Time Series Overlays', ln=True)
            for p in ts_paths:
                if pdf.get_y() > 150:
                    pdf.add_page()
                pdf.image(p, x=10, y=None, w=267)
                pdf.ln(3)

        pdf.output(pdf_path)
        self.finished.emit(pdf_path)

    # ── plot helpers ──────────────────────────────────────────────────────────
    def _make_heatmap(self, corr_df, merged) -> str:
        if corr_df.empty:
            return None
        pivot = corr_df.pivot_table(
            index='ROI Column', columns='Sensor Column', values='r')
        if pivot.empty:
            return None
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*0.9),
                                        max(4, len(pivot.index)*0.6)))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=ax,
                    linewidths=0.3, cbar_kws={'shrink': 0.8})
        ax.set_title(f'{self.method.capitalize()} Correlation Heatmap', fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'heatmap.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _make_scatter(self, merged, roi_col, sensor_col, r, p, n) -> str:
        pair = merged[[roi_col, sensor_col, '_datetime']].dropna()
        if len(pair) < 3:
            return None
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(pair[sensor_col], pair[roi_col], alpha=0.6, s=30)
        # regression line
        m, b = np.polyfit(pair[sensor_col].values, pair[roi_col].values, 1)
        xs = np.linspace(pair[sensor_col].min(), pair[sensor_col].max(), 100)
        ax.plot(xs, m*xs + b, 'r--', linewidth=1.5)
        sig = ('***' if p < 0.001 else '**' if p < 0.01
               else '*' if p < 0.05 else 'ns')
        ax.set_xlabel(sensor_col.split('::')[-1], fontsize=9)
        ax.set_ylabel(roi_col, fontsize=9)
        ax.set_title(f'r={r:.3f} {sig}  n={n}', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        safe = re.sub(r'[^\w]', '_', f"{roi_col}_vs_{sensor_col}")[:80]
        path = os.path.join(self.output_dir, f'scatter_{safe}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _make_timeseries(self, merged, sensor_cols) -> list:
        """
        One figure per ROI column.
        ROI data plotted as scatter at its natural timestamps.
        Sensor data plotted at FULL resolution from the original sensor DataFrames
        so the complete time series is visible, not just the aligned subset.
        """
        paths = []
        for roi_col in self.roi_cols:
            valid_sensors = [c for c in sensor_cols
                             if merged[[roi_col, c]].dropna().shape[0] >= 3]
            if not valid_sensors:
                continue
            n_rows = 1 + len(valid_sensors)
            fig, axes = plt.subplots(n_rows, 1,
                                     figsize=(11, 1.8 * n_rows),
                                     sharex=True)
            if n_rows == 1:
                axes = [axes]

            # ROI line+markers — sparse, at matched timestamps
            roi_data = merged[['_datetime', roi_col]].dropna()
            axes[0].plot(roi_data['_datetime'], roi_data[roi_col],
                         color='steelblue', linewidth=0.8, alpha=0.7)
            axes[0].scatter(roi_data['_datetime'], roi_data[roi_col],
                            s=8, color='steelblue', alpha=0.9, zorder=3)
            axes[0].set_ylabel(roi_col, fontsize=8)
            axes[0].grid(True, alpha=0.3)

            for i, sc in enumerate(valid_sensors):
                ax = axes[i + 1]
                # Extract label and sensor file label from "FileLabel::ColName"
                parts = sc.split('::', 1)
                file_label = parts[0] if len(parts) == 2 else None
                col_name   = parts[1] if len(parts) == 2 else sc

                # Use full-resolution sensor data if available
                full_df = None
                if file_label is not None:
                    for lbl, sdf in self.sensor_dfs:
                        if lbl == file_label and '_datetime' in sdf.columns and col_name in sdf.columns:
                            full_df = sdf[['_datetime', col_name]].dropna()
                            break

                if full_df is not None and len(full_df) > 0:
                    ax.plot(full_df['_datetime'], full_df[col_name],
                            linewidth=0.8, alpha=0.8)
                else:
                    # fallback to merged (aligned) data
                    pair = merged[['_datetime', sc]].dropna()
                    ax.plot(pair['_datetime'], pair[sc],
                            linewidth=1.2, alpha=0.8)

                ax.set_ylabel(col_name, fontsize=8)
                ax.grid(True, alpha=0.3)

            axes[-1].xaxis.set_major_formatter(
                mdates.AutoDateFormatter(mdates.AutoDateLocator()))
            plt.setp(axes[-1].xaxis.get_majorticklabels(),
                     rotation=30, ha='right')
            fig.suptitle(f'Time Series: {roi_col}', fontsize=11)
            plt.tight_layout()

            safe = re.sub(r'[^\w]', '_', roi_col)[:60]
            path = os.path.join(self.output_dir, f'timeseries_{safe}.png')
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
        return paths


# ──────────────────────────────────────────────────────────────────────────────
# Main tab
# ──────────────────────────────────────────────────────────────────────────────

class CorrelationAnalyzerTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # EXPECTED UI WIDGETS (loaded via loadUi before wire_connections):
        # - self.lineEdit_roiFolder
        # - self.pushButton_browseROI
        # - self.label_roiStatus
        # - self.listWidget_roiColumns
        # - self.pushButton_roiSelectAll
        # - self.pushButton_roiSelectNone
        # - self.pushButton_addSensorFile
        # - self.tabWidget_sensorFiles
        # - self.comboBox_method
        # - self.doubleSpinBox_tolerance
        # - self.pushButton_correlate
        # - self.label_status
        # - self.tabWidget_results   (right panel — populated after correlation)

        self._roi_df = None
        self._worker = None

    # ── wiring ────────────────────────────────────────────────────────────────
    def wire_connections(self):
        self.pushButton_browseROI.clicked.connect(self._browse_roi_folder)
        self.pushButton_browseROI.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.lineEdit_roiFolder.editingFinished.connect(self._load_roi_folder)

        self.pushButton_roiSelectAll.clicked.connect(self._roi_select_all)
        self.pushButton_roiSelectNone.clicked.connect(self._roi_select_none)

        self.pushButton_addSensorFile.clicked.connect(self._add_sensor_file)
        self.pushButton_addSensorFile.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.tabWidget_sensorFiles.tabCloseRequested.connect(
            self.tabWidget_sensorFiles.removeTab)

        self.pushButton_correlate.clicked.connect(self._run_correlation)
        self.pushButton_correlate.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        # Placeholder in results tab widget
        self._show_placeholder("Run a correlation to see results here.")

        # Restore saved folder
        saved = JsonEditor().getValue("CorrelationAnalyzer_ROIFolder")
        if saved:
            self.lineEdit_roiFolder.setText(saved)
            self._load_roi_folder()

    # ── ROI folder ────────────────────────────────────────────────────────────
    def _browse_roi_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select ROI Images Folder", str(PROJECT_ROOT))
        if not folder:
            return
        self.lineEdit_roiFolder.setText(folder)
        self._load_roi_folder()

    def _load_roi_folder(self):
        folder = self.lineEdit_roiFolder.text().strip()
        if not folder or not os.path.isdir(folder):
            return
        JsonEditor().update_json_entry("CorrelationAnalyzer_ROIFolder", folder)

        # Find ROI metrics CSV — look for *_roi_metrics.csv
        matches = glob.glob(os.path.join(folder, '*_roi_metrics.csv'))
        if not matches:
            self.label_roiStatus.setText(
                "⚠ No ROI metrics CSV found. Run 'Extract ROI Features' first.")
            self.listWidget_roiColumns.clear()
            self._roi_df = None
            return

        # Use most recent
        csv_path = sorted(matches)[-1]
        try:
            self._roi_df = pd.read_csv(csv_path)
        except Exception as e:
            self.label_roiStatus.setText(f"Error reading {csv_path}:\n{e}")
            return

        # Parse timestamps from ROI df
        self._roi_df = self._parse_roi_timestamps(self._roi_df)

        self.label_roiStatus.setText(
            f"✓ {os.path.basename(csv_path)}  ({len(self._roi_df)} rows)")

        # Populate column checklist — skip path/date/time/method cols
        skip = {'Image Path', 'Mask Path', 'Capture Date', 'Capture Time',
                '_datetime', 'Clustering Method', 'Clustering Params'}
        self.listWidget_roiColumns.clear()
        for col in self._roi_df.columns:
            if col in skip:
                continue
            # Skip non-numeric
            if not pd.api.types.is_numeric_dtype(self._roi_df[col]):
                continue
            item = QListWidgetItem(col)
            item.setCheckState(Qt.Unchecked)
            self.listWidget_roiColumns.addItem(item)

    def _parse_roi_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a '_datetime' column from Capture Date + Capture Time."""
        df = df.copy()
        if 'Capture Date' in df.columns and 'Capture Time' in df.columns:
            combined = (df['Capture Date'].astype(str).str.strip() + ' ' +
                        df['Capture Time'].astype(str).str.strip())
            df['_datetime'] = pd.to_datetime(combined,

                                              errors='coerce')
        elif 'Capture Date' in df.columns:
            df['_datetime'] = pd.to_datetime(
                df['Capture Date'], errors='coerce')
        else:
            df['_datetime'] = pd.NaT
        df = df.dropna(subset=['_datetime']).sort_values('_datetime')
        return df

    # ── ROI column helpers ────────────────────────────────────────────────────
    def _roi_select_all(self):
        for i in range(self.listWidget_roiColumns.count()):
            self.listWidget_roiColumns.item(i).setCheckState(Qt.Checked)

    def _roi_select_none(self):
        for i in range(self.listWidget_roiColumns.count()):
            self.listWidget_roiColumns.item(i).setCheckState(Qt.Unchecked)

    def _get_selected_roi_cols(self) -> list:
        result = []
        for i in range(self.listWidget_roiColumns.count()):
            item = self.listWidget_roiColumns.item(i)
            if item.checkState() == Qt.Checked:
                result.append(item.text())
        return result

    # ── Sensor file tabs ──────────────────────────────────────────────────────
    def _add_sensor_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Sensor CSV File(s)", str(PROJECT_ROOT),
            "CSV Files (*.csv)")
        for path in paths:
            label = os.path.splitext(os.path.basename(path))[0]
            tab = SensorFileTab(path)
            self.tabWidget_sensorFiles.addTab(tab, label[:20])

    # ── Correlation run ───────────────────────────────────────────────────────
    def _run_correlation(self):
        if self._roi_df is None:
            QMessageBox.warning(self, "No ROI Data",
                                "Load a ROI metrics folder first.")
            return

        roi_cols = self._get_selected_roi_cols()
        if not roi_cols:
            QMessageBox.warning(self, "No ROI Columns",
                                "Select at least one ROI column to correlate.")
            return

        if self.tabWidget_sensorFiles.count() == 0:
            QMessageBox.warning(self, "No Sensor Files",
                                "Add at least one sensor CSV file.")
            return

        # Collect sensor data
        sensor_dfs = []
        for i in range(self.tabWidget_sensorFiles.count()):
            tab = self.tabWidget_sensorFiles.widget(i)
            label = self.tabWidget_sensorFiles.tabText(i)
            sdf = tab.build_aligned_df()
            if sdf.empty:
                QMessageBox.warning(
                    self, "No Sensor Columns",
                    f"Sensor file '{label}' has no selected data columns "
                    f"or no valid datetime column.")
                return
            sensor_dfs.append((label, sdf))

        roi_folder = self.lineEdit_roiFolder.text().strip()
        output_dir = os.path.join(roi_folder, 'correlation')

        method = self.comboBox_method.currentText().lower()
        tolerance_h = self.doubleSpinBox_tolerance.value()

        self.label_status.setText("Running…")
        self.pushButton_correlate.setEnabled(False)

        self._worker = CorrelationWorker(
            roi_df      = self._roi_df,
            sensor_dfs  = sensor_dfs,
            roi_cols    = roi_cols,
            method      = method,
            tolerance_h = tolerance_h,
            output_dir  = output_dir,
        )
        self._worker.progress.connect(self.label_status.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, pdf_path: str):
        self.pushButton_correlate.setEnabled(True)
        self.label_status.setText(f"✓ Done. PDF saved to:\n{pdf_path}")

        # Load results data and populate the results tabs
        output_dir = os.path.dirname(pdf_path)
        corr_csv = os.path.join(output_dir, 'correlation_table.csv')
        try:
            corr_df = pd.read_csv(corr_csv)
        except Exception:
            corr_df = pd.DataFrame()

        self._populate_results_tabs(corr_df, output_dir)

        QMessageBox.information(
            self, "Correlation Complete",
            f"PDF report saved to:\n{pdf_path}")

    def _on_error(self, msg: str):
        self.pushButton_correlate.setEnabled(True)
        self.label_status.setText("Error — see details.")
        QMessageBox.critical(self, "Correlation Error", msg)

    # ── Results tab population ─────────────────────────────────────────────────
    def _show_placeholder(self, text: str):
        """Clear results tabs and show a placeholder message."""
        self.tabWidget_results.clear()
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFont(QFont("Arial", 11))
        self.tabWidget_results.addTab(lbl, "Results")

    def _populate_results_tabs(self, corr_df: pd.DataFrame, output_dir: str):
        """Build all four results tabs from the correlation output."""
        self.tabWidget_results.clear()

        # ── Tab 1: Correlation Table ──────────────────────────────────────────
        self._build_table_tab(corr_df)

        # ── Tab 2: Heatmap ────────────────────────────────────────────────────
        heatmap_path = os.path.join(output_dir, 'heatmap.png')
        self._build_image_tab("Heatmap", heatmap_path)

        # ── Tab 3: Scatter Plots ──────────────────────────────────────────────
        scatter_paths = sorted(
            glob.glob(os.path.join(output_dir, 'scatter_*.png')))
        self._build_scrollable_grid_tab("Scatter Plots", scatter_paths, cols=2)

        # ── Tab 4: Time Series ────────────────────────────────────────────────
        ts_paths = sorted(
            glob.glob(os.path.join(output_dir, 'timeseries_*.png')))
        self._build_scrollable_grid_tab("Time Series", ts_paths, cols=1)

    def _build_table_tab(self, corr_df: pd.DataFrame):
        """Tab 1 — sortable QTableWidget of correlation results."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        if corr_df.empty:
            layout.addWidget(QLabel("No correlation results."))
        else:
            tbl = QTableWidget(len(corr_df), len(corr_df.columns))
            tbl.setHorizontalHeaderLabels(list(corr_df.columns))
            tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
            tbl.setAlternatingRowColors(True)
            tbl.setSortingEnabled(True)
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            tbl.horizontalHeader().setStretchLastSection(True)

            for r, (_, row) in enumerate(corr_df.iterrows()):
                for c, val in enumerate(row):
                    item = QTableWidgetItem(str(val))
                    item.setTextAlignment(Qt.AlignCenter)
                    # Color-code by r value
                    if corr_df.columns[c] == 'r':
                        try:
                            rv = float(val)
                            if rv >= 0.5:
                                item.setBackground(QColor(200, 230, 200))
                            elif rv <= -0.5:
                                item.setBackground(QColor(230, 200, 200))
                        except Exception:
                            pass
                    tbl.setItem(r, c, item)

            layout.addWidget(tbl)

        self.tabWidget_results.addTab(tab, "Correlation Table")

    def _build_image_tab(self, title: str, image_path: str):
        """A tab showing a single image scaled to fit."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        if os.path.exists(image_path):
            pix = QPixmap(image_path)
            lbl.setPixmap(pix)
            lbl.setScaledContents(False)
            # Scale on resize via custom resizeEvent on the label
            lbl._pixmap = pix
            orig_resize = lbl.resizeEvent

            def _resize(event, _lbl=lbl, _orig=orig_resize):
                if hasattr(_lbl, '_pixmap') and not _lbl._pixmap.isNull():
                    scaled = _lbl._pixmap.scaled(
                        _lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    _lbl.setPixmap(scaled)
                _orig(event)

            lbl.resizeEvent = _resize
        else:
            lbl.setText(f"Image not found:\n{image_path}")

        layout.addWidget(lbl)
        self.tabWidget_results.addTab(tab, title)

    def _build_scrollable_grid_tab(self, title: str, image_paths: list, cols: int = 2):
        """A scrollable tab with images arranged in a grid."""
        tab = QWidget()
        outer = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(6)
        scroll.setWidget(container)

        if not image_paths:
            grid.addWidget(QLabel("No images generated."), 0, 0)
        else:
            for idx, path in enumerate(image_paths):
                r, c = divmod(idx, cols)
                lbl = QLabel()
                lbl.setAlignment(Qt.AlignCenter)
                if os.path.exists(path):
                    pix = QPixmap(path)
                    # Fixed display size; maintain aspect ratio
                    w = 540 if cols == 1 else 400
                    scaled = pix.scaledToWidth(w, Qt.SmoothTransformation)
                    lbl.setPixmap(scaled)
                    lbl.setToolTip(os.path.basename(path))
                else:
                    lbl.setText(os.path.basename(path))
                grid.addWidget(lbl, r, c)

        self.tabWidget_results.addTab(tab, title)
