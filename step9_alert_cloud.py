#!/usr/bin/env python3
"""
step9_alert_cloud.py  (v3 — 6-token, tiered confidence)
=========================================================
Crypto trading signal system — runs on GitHub Actions every 4 hours.
Sends a Telegram message every run: signals found, or "no strong signal".

Tokens:    BTC, ETH, BNB, SOL, XRP
Timeframe: 4h candles  |  Training window: ~7 years per token

Confidence tiers:
  🟡  55–60%  Weak     — slight lean, treat as FYI
  🟠  60–70%  Moderate — meaningful signal, watch closely
  🟢/🔴 ≥70% Strong   — historically consistent, highest conviction

Only tokens with confidence ≥ 55% appear in the message.
A message is always sent (signals found, or quiet check confirmed).

Optimisations applied (steps 10–18):
  • Tuned RF hyperparameters per token (steps 11, 14, 18)
  • Per-token move thresholds from threshold sweep (steps 13, 18)
  • 7 years training data — critical for BTC, XRP edge (step 17)
  • Purged walk-forward CV confirmed edge is real (step 15)
  • Funding/OI, MA crossovers, feature selection — all tested, all dropped

Test mode: python step9_alert_cloud.py --test
Real mode: python step9_alert_cloud.py
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import urllib.request
import json
import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# SETTINGS  (secrets stored in GitHub — never in code)
# ============================================================

TG_BOT_TOKEN = os.environ.get('TG_BOT_TOKEN', '')
TG_CHAT_ID   = os.environ.get('TG_CHAT_ID',   '')

TIMEFRAME       = '4h'
PREDICT_AHEAD   = 1         # predict 1 × 4h candle ahead
N_CANDLES       = 15_300    # ~7 years of 4h candles
MIN_CONF        = 0.55      # minimum confidence to include in message

# Confidence tier thresholds
TIER_WEAK     = 0.55
TIER_MODERATE = 0.60
TIER_STRONG   = 0.70

# Per-token config: tuned move threshold + RF hyperparameters
TOKEN_CONFIGS = {
    'BTC/USDT': {
        'label':          'BTC',
        'move_threshold': 0.02,      # 2.0% — step 13
        'clf_params': {
            'n_estimators':      100,
            'max_depth':         None,
            'min_samples_leaf':  10,
            'max_features':      0.3,
            'min_samples_split': 5,
            'class_weight':      None,
            'random_state':      42,
            'n_jobs':            -1,
        },
    },
    'ETH/USDT': {
        'label':          'ETH',
        'move_threshold': 0.025,     # 2.5% — step 14
        'clf_params': {
            'n_estimators':      200,
            'max_depth':         None,
            'min_samples_leaf':  18,
            'max_features':      'log2',
            'min_samples_split': 3,
            'class_weight':      None,
            'random_state':      42,
            'n_jobs':            -1,
        },
    },
    'BNB/USDT': {
        'label':          'BNB',
        'move_threshold': 0.02,      # 2.0% — step 13
        'clf_params': {
            'n_estimators':      100,
            'max_depth':         10,
            'min_samples_leaf':  2,
            'max_features':      'sqrt',
            'min_samples_split': 3,
            'class_weight':      None,
            'random_state':      42,
            'n_jobs':            -1,
        },
    },
    'SOL/USDT': {
        'label':          'SOL',
        'move_threshold': 0.025,     # 2.5% — step 13
        'clf_params': {
            'n_estimators':      75,
            'max_depth':         10,
            'min_samples_leaf':  3,
            'max_features':      0.3,
            'min_samples_split': 2,
            'class_weight':      None,
            'random_state':      42,
            'n_jobs':            -1,
        },
    },
    'XRP/USDT': {
        'label':          'XRP',
        'move_threshold': 0.025,     # 2.5% — step 18
        'clf_params': {
            'n_estimators':      200,
            'max_depth':         None,
            'min_samples_leaf':  2,
            'max_features':      'log2',
            'min_samples_split': 8,
            'class_weight':      None,
            'random_state':      42,
            'n_jobs':            -1,
        },
    },
}

FEATURE_COLS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_width', 'bb_pos',
    'ma7_21_ratio', 'ma21_50_ratio', 'ma50_200_ratio',
    'price_vs_ma7', 'price_vs_ma21', 'price_vs_ma50', 'price_vs_ma200',
    'mom3', 'mom7', 'mom14', 'mom21',
    'atr', 'vol_ratio', 'hl_ratio', 'oc_ratio',
    'near_resistance', 'near_support', 'touch_count
