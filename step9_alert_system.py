#!/usr/bin/env python3
"""
step9_alert_system.py
=====================
Automated crypto trading signal system for BTC, ETH, BNB, SOL

Uses the best configs discovered in step8:
  - 4h candles, 2% move threshold
  - Random Forest or Extra Trees
  - 60-65% confidence cutoff depending on token

For each token, produces:
  1. Direction (UP / DOWN)
  2. Confidence % (how sure the model is)
  3. Expected % move (from a regression model)
  4. Estimated target price

Alerts fire as Mac desktop notifications + are saved to a CSV log.

How to run:
  python3 ~/Documents/Claude/step9_alert_system.py

How to automate (every 4 hours via cron):
  1. Open Terminal
  2. Type: crontab -e
  3. Press i to enter insert mode
  4. Add this line (checks at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00):
       0 0,4,8,12,16,20 * * * python3 ~/Documents/Claude/step9_alert_system.py >> ~/Documents/Claude/step9_cron.log 2>&1
  5. Press Escape, then type :wq and Enter to save
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
import subprocess
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SETTINGS - Best configs per token from step8 results
# ============================================================

LOG_FILE = os.path.expanduser("~/Documents/Claude/step9_alerts_log.csv")

TIMEFRAME   = '4h'
PREDICT_AHEAD = 1          # 1 x 4h candle = 4 hours ahead
SR_THRESHOLD_PCT = 1.0     # % proximity to count as "near" S/R level
MOVE_THRESHOLD   = 0.02    # 2% minimum move to count as a signal

TOKEN_CONFIGS = {
    'BTC/USDT': {
        'label':            'BTC',
        'model':            'extra_trees',
        'regime':           'all_conditions',
        'confidence_cutoff': 0.60,
        'n_candles':        13000,
    },
    'ETH/USDT': {
        'label':            'ETH',
        'model':            'random_forest',
        'regime':           'all_conditions',
        'confidence_cutoff': 0.65,
        'n_candles':        13000,
    },
    'BNB/USDT': {
        'label':            'BNB',
        'model':            'random_forest',
        'regime':           'uptrend_only',
        'confidence_cutoff': 0.65,
        'n_candles':        13000,
    },
    'SOL/USDT': {
        'label':            'SOL',
        'model':            'extra_trees',
        'regime':           'uptrend_only',
        'confidence_cutoff': 0.65,
        'n_candles':         8000,
    },
}

FEATURE_COLS = [
    'returns', 'log_returns', 'candle_body', 'candle_range', 'rel_body_size',
    'ma_5_ratio', 'ma_10_ratio', 'ma_200_ratio', 'ma200_slope',
    'momentum_3', 'momentum_5', 'momentum_10',
    'volume_ratio', 'vol_expansion',
    'rsi', 'bb_position', 'bb_width', 'macd', 'macd_signal',
    'hour_of_day', 'day_of_week', 'is_weekend',
    'higher_high', 'lower_low', 'consec_green', 'consec_red',
    'dist_to_resistance_20', 'dist_to_resistance_50',
    'dist_to_support_20',    'dist_to_support_50',
    'near_resistance_20',    'near_resistance_50',
    'near_support_20',       'near_support_50',
    'resistance_touches_20', 'resistance_touches_50',
    'support_touches_20',    'support_touches_50',
    'sfp_bearish', 'sfp_bullish',
]


# ============================================================
# ALERTS
# ============================================================

def send_mac_notification(title, message):
    """Fire a Mac desktop notification with sound."""
    try:
        script = f'display notification "{message}" with title "{title}" sound name "Glass"'
        subprocess.run(['osascript', '-e', script], check=True)
    except Exception as e:
        print(f"  (Notification failed: {e})")


# ============================================================
# DATA FETCHING
# ============================================================

def fetch_candles(symbol, timeframe, n_candles):
    print(f"  Fetching {n_candles} {timeframe} candles for {symbol}...", end='', flush=True)
    exchange = ccxt.binance({'enableRateLimit': True})
    all_candles = []
    tf_ms = {'1h': 3_600_000, '4h': 14_400_000}[timeframe]
    since = exchange.milliseconds() - (n_candles + 200) * tf_ms
    batch_size = 1000

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_size)
        if not batch:
            break
        all_candles.extend(batch)
        if len(batch) < batch_size:
            break
        since = batch[-1][0] + 1
        if len(all_candles) >= n_candles + 200:
            break

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    df = df.tail(n_candles + 100).reset_index(drop=True)
    print(f" {len(df)} rows loaded.")
    return df


# ============================================================
# FEATURE ENGINEERING  (same pipeline as step8)
# ============================================================

def add_features(df):
    df = df.copy()

    # --- Basic candle features ---
    df['returns']      = df['close'].pct_change()
    df['log_returns']  = np.log(df['close'] / df['close'].shift(1))
    df['candle_body']  = (df['close'] - df['open']).abs() / df['open']
    df['candle_range'] = (df['high'] - df['low']) / df['open']
    df['rel_body_size']= (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-10)

    # --- Moving averages ---
    for w in [5, 10, 20, 50, 200]:
        df[f'ma_{w}'] = df['close'].rolling(w).mean()

    df['ma_5_ratio']   = df['close'] / df['ma_5']
    df['ma_10_ratio']  = df['close'] / df['ma_10']
    df['ma_200_ratio'] = df['close'] / df['ma_200']
    df['is_uptrend']   = (df['close'] > df['ma_200']).astype(int)
    df['ma200_slope']  = df['ma_200'].pct_change(5)

    # --- Momentum ---
    for p in [3, 5, 10]:
        df[f'momentum_{p}'] = df['close'].pct_change(p)

    # --- Volume ---
    df['volume_ma']    = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
    df['vol_expansion']= (df['volume_ratio'] > 1.5).astype(int)

    # --- RSI ---
    delta = df['close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # --- Bollinger Bands ---
    df['bb_mid']      = df['close'].rolling(20).mean()
    df['bb_std']      = df['close'].rolling(20).std()
    df['bb_upper']    = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower']    = df['bb_mid'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width']    = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-10)

    # --- MACD ---
    ema12          = df['close'].ewm(span=12, adjust=False).mean()
    ema26          = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']     = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # --- Time features ---
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

    # --- Price structure (vectorized consecutive candle count) ---
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low']   = (df['low']  < df['low'].shift(1)).astype(int)

    is_green = df['close'] > df['open']
    is_red   = df['close'] < df['open']

    grp_green = (is_green != is_green.shift()).cumsum()
    counts_green = is_green.groupby(grp_green).cumcount() + 1
    df['consec_green'] = np.where(is_green, counts_green, 0)

    grp_red = (is_red != is_red.shift()).cumsum()
    counts_red = is_red.groupby(grp_red).cumcount() + 1
    df['consec_red'] = np.where(is_red, counts_red, 0)

    # --- Support / Resistance ---
    prox = SR_THRESHOLD_PCT / 100
    for w in [20, 50]:
        df[f'resistance_{w}'] = df['high'].rolling(w).max()
        df[f'support_{w}']    = df['low'].rolling(w).min()

        df[f'dist_to_resistance_{w}'] = (df[f'resistance_{w}'] - df['close']) / df['close']
        df[f'dist_to_support_{w}']    = (df['close'] - df[f'support_{w}']  ) / df['close']
        df[f'near_resistance_{w}']    = (df[f'dist_to_resistance_{w}'] < prox).astype(int)
        df[f'near_support_{w}']       = (df[f'dist_to_support_{w}']    < prox).astype(int)

        # Touch count (vectorized)
        res = df[f'resistance_{w}'].shift(1)
        sup = df[f'support_{w}'].shift(1)
        near_res = (df['high'] >= res * (1 - prox)) & (df['low'] <= res * (1 + prox))
        near_sup  = (df['low'] <= sup * (1 + prox)) & (df['high'] >= sup * (1 - prox))
        df[f'resistance_touches_{w}'] = near_res.rolling(w).sum()
        df[f'support_touches_{w}']    = near_sup.rolling(w).sum()

    # --- SFP (Swing Failure Pattern) ---
    df['sfp_bearish'] = ((df['high'] > df['resistance_20'].shift(1)) &
                         (df['close'] < df['resistance_20'].shift(1))).astype(int)
    df['sfp_bullish'] = ((df['low']  < df['support_20'].shift(1)) &
                         (df['close'] > df['support_20'].shift(1))).astype(int)

    return df


# ============================================================
# MAIN PREDICTION LOGIC
# ============================================================

def run_prediction(symbol, config):
    label = config['label']
    cutoff = config['confidence_cutoff']
    print(f"\n{'─'*52}")
    print(f"  {label} ({config['regime']}, {config['model']}, ≥{cutoff:.0%} confidence)")

    # 1. Fetch and engineer features
    df = fetch_candles(symbol, TIMEFRAME, config['n_candles'])
    df = add_features(df)

    # 2. Build classification target (UP=1 / DOWN=0 / NaN=no big move)
    future_return = df['close'].shift(-PREDICT_AHEAD).sub(df['close']).div(df['close'])
    df['target']  = np.where(future_return >  MOVE_THRESHOLD, 1,
                    np.where(future_return < -MOVE_THRESHOLD, 0, np.nan))
    df['future_return'] = future_return          # regression target

    # 3. Apply regime filter
    if config['regime'] == 'uptrend_only':
        df_model = df[df['is_uptrend'] == 1].copy()
    else:
        df_model = df.copy()

    # 4. Latest candle = what we want to predict for; everything before = training
    available = df_model.dropna(subset=FEATURE_COLS)
    train_df  = available.iloc[:-1]              # exclude last row (the live candle)
    live_row  = available.iloc[-1]

    if len(train_df) < 200:
        print(f"  ⚠️  Not enough training data ({len(train_df)} rows). Skipping.")
        return None

    X_train  = train_df[FEATURE_COLS].values
    X_live   = live_row[FEATURE_COLS].values.reshape(1, -1)

    # For classification, only use rows that had a clear big move
    mask_clf = train_df['target'].notna()
    X_clf    = train_df.loc[mask_clf, FEATURE_COLS].values
    y_clf    = train_df.loc[mask_clf, 'target'].values

    # For regression, use all rows (including small moves)
    mask_reg = train_df['future_return'].notna()
    X_reg    = train_df.loc[mask_reg, FEATURE_COLS].values
    y_reg    = train_df.loc[mask_reg, 'future_return'].values

    if len(X_clf) < 50:
        print(f"  ⚠️  Not enough big-move examples ({len(X_clf)}). Skipping.")
        return None

    current_price = live_row['close']
    signal_time   = live_row['timestamp']
    is_uptrend    = bool(live_row['is_uptrend'])

    # 5. Check regime requirement at live candle
    if config['regime'] == 'uptrend_only' and not is_uptrend:
        print(f"  ⏸️  {label}: Currently NOT in uptrend (below 200 MA). No signal.")
        return {'symbol': label, 'status': 'no_signal', 'reason': 'not_in_uptrend',
                'price': current_price, 'time': signal_time}

    # 6. Train classifier — direction (UP / DOWN)
    if config['model'] == 'random_forest':
        clf = RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                     random_state=42, n_jobs=-1)
    else:
        clf = ExtraTreesClassifier(n_estimators=300, max_features='sqrt',
                                   random_state=42, n_jobs=-1)
    clf.fit(X_clf, y_clf)

    # 7. Train regressor — magnitude (expected % move)
    reg = RandomForestRegressor(n_estimators=200, max_features='sqrt',
                                random_state=42, n_jobs=-1)
    reg.fit(X_reg, y_reg)

    # 8. Predict on live candle
    proba = clf.predict_proba(X_live)[0]
    # clf.classes_ might be [0,1] or just [0] or [1] — handle safely
    classes = list(clf.classes_)
    prob_up   = proba[classes.index(1)] if 1 in classes else 0.0
    prob_down = proba[classes.index(0)] if 0 in classes else 0.0

    expected_move = float(reg.predict(X_live)[0])
    confidence    = max(prob_up, prob_down)
    direction     = 'UP' if prob_up >= prob_down else 'DOWN'
    arrow         = '▲' if direction == 'UP' else '▼'
    trend_label   = 'uptrend ✅' if is_uptrend else 'downtrend'

    print(f"  Trend:          {trend_label}")
    print(f"  Signal:         {arrow} {direction}")
    print(f"  Prob UP:        {prob_up:.1%}   |   Prob DOWN: {prob_down:.1%}")
    print(f"  Confidence:     {confidence:.1%}  (cutoff: {cutoff:.0%})")
    print(f"  Expected move:  {expected_move*100:+.2f}%")
    print(f"  Current price:  ${current_price:,.4f}")

    # 9. Does it meet the confidence threshold?
    if confidence < cutoff:
        print(f"  ⏸️  No signal — confidence {confidence:.1%} < cutoff {cutoff:.0%}")
        return {'symbol': label, 'status': 'no_signal', 'reason': 'low_confidence',
                'price': current_price, 'time': signal_time, 'confidence': confidence}

    target_price = current_price * (1 + expected_move)
    print(f"  🚨 SIGNAL FIRES!  Target price:  ${target_price:,.4f}")

    return {
        'symbol':        label,
        'status':        'signal',
        'direction':     direction,
        'confidence':    confidence,
        'prob_up':       prob_up,
        'prob_down':     prob_down,
        'current_price': current_price,
        'expected_move': expected_move,
        'target_price':  target_price,
        'signal_time':   signal_time,
        'regime':        config['regime'],
        'model':         config['model'],
        'is_uptrend':    is_uptrend,
    }


# ============================================================
# LOGGING
# ============================================================

def log_signal(s):
    row = {
        'run_time':          datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'signal_candle_time': str(s.get('signal_time', '')),
        'symbol':            s['symbol'],
        'direction':         s.get('direction', ''),
        'confidence_pct':    f"{s.get('confidence', 0)*100:.1f}",
        'prob_up_pct':       f"{s.get('prob_up', 0)*100:.1f}",
        'prob_down_pct':     f"{s.get('prob_down', 0)*100:.1f}",
        'current_price':     f"{s.get('current_price', 0):.4f}",
        'expected_move_pct': f"{s.get('expected_move', 0)*100:+.2f}",
        'target_price':      f"{s.get('target_price', 0):.4f}",
        'regime':            s.get('regime', ''),
        'model':             s.get('model', ''),
        'outcome':           'PENDING',   # fill in manually later
        'actual_price_4h':   '',          # fill in manually later
    }
    log_df = pd.DataFrame([row])
    if os.path.exists(LOG_FILE):
        existing = pd.read_csv(LOG_FILE)
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(LOG_FILE, index=False)
    print(f"  📝 Logged → {LOG_FILE}")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"\n{'='*52}")
    print(f"  🚀  Trading Signal System  —  {now_str}")
    print(f"{'='*52}")

    fired_signals = []

    for symbol, config in TOKEN_CONFIGS.items():
        try:
            result = run_prediction(symbol, config)
            if result and result.get('status') == 'signal':
                fired_signals.append(result)
        except Exception as e:
            print(f"\n  ❌  Error processing {symbol}: {e}")

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"  SUMMARY  —  {now_str}")
    print(f"{'='*52}")

    if not fired_signals:
        print("  ⏸️  No high-confidence signals right now.")
        print("       Check again at the next 4h candle close.")
        send_mac_notification("Crypto Signal Check", "No signals firing. Check again in 4h.")
    else:
        for s in fired_signals:
            emoji = '🟢' if s['direction'] == 'UP' else '🔴'
            arrow = '▲' if s['direction'] == 'UP' else '▼'
            print(f"\n  {emoji} {arrow}  {s['symbol']} — {s['direction']}")
            print(f"     Confidence:     {s['confidence']:.1%}  "
                  f"(UP {s['prob_up']:.1%} | DOWN {s['prob_down']:.1%})")
            print(f"     Current price:  ${s['current_price']:,.4f}")
            print(f"     Expected move:  {s['expected_move']*100:+.2f}%")
            print(f"     Target price:   ${s['target_price']:,.4f}")
            print(f"     Regime:         {s['regime']}  |  Model: {s['model']}")

            notif_msg = (
                f"{s['direction']} | Conf: {s['confidence']:.0%} | "
                f"Move: {s['expected_move']*100:+.1f}% | "
                f"Price: ${s['current_price']:,.2f} → ${s['target_price']:,.2f}"
            )
            send_mac_notification(f"🚨 {s['symbol']} Trade Signal", notif_msg)
            log_signal(s)

    print(f"\n{'='*52}\n")
    return fired_signals


if __name__ == '__main__':
    main()
