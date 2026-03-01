#!/usr/bin/env python3
"""
step9_alert_cloud.py
====================
Crypto trading signal system — runs on GitHub Actions every 4 hours.
Sends signals to Telegram when a high-confidence trade is detected.

Tokens:    BTC, ETH, BNB, SOL
Timeframe: 4h candles
Signal:    Direction (UP/DOWN) + Confidence % + Expected % move + Target price

Test mode: python step9_alert_cloud.py --test
Real mode: python step9_alert_cloud.py
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
import urllib.request
import urllib.parse
import json
import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# SETTINGS  (tokens stored as GitHub Secrets — never in code)
# ============================================================

TG_BOT_TOKEN = os.environ.get('TG_BOT_TOKEN', '')   # set in GitHub Secrets
TG_CHAT_ID   = os.environ.get('TG_CHAT_ID',   '')   # set in GitHub Secrets

TIMEFRAME        = '4h'
PREDICT_AHEAD    = 1        # 1 x 4h candle = 4 hours ahead
SR_THRESHOLD_PCT = 1.0
MOVE_THRESHOLD   = 0.02     # 2% minimum move to qualify as a signal

TOKEN_CONFIGS = {
    'BTC/USDT': {
        'label':             'BTC',
        'model':             'extra_trees',
        'regime':            'all_conditions',
        'confidence_cutoff':  0.60,
        'n_candles':         8000,
    },
    'ETH/USDT': {
        'label':             'ETH',
        'model':             'random_forest',
        'regime':            'all_conditions',
        'confidence_cutoff':  0.65,
        'n_candles':         8000,
    },
    'BNB/USDT': {
        'label':             'BNB',
        'model':             'random_forest',
        'regime':            'uptrend_only',
        'confidence_cutoff':  0.65,
        'n_candles':         8000,
    },
    'SOL/USDT': {
        'label':             'SOL',
        'model':             'extra_trees',
        'regime':            'uptrend_only',
        'confidence_cutoff':  0.65,
        'n_candles':         6000,
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
# TELEGRAM
# ============================================================

def send_telegram(message: str):
    """Send a Telegram message to your chat."""
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("  ⚠️  TG_BOT_TOKEN or TG_CHAT_ID not set. Skipping notification.")
        return
    try:
        url  = f'https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage'
        data = json.dumps({
            'chat_id':    TG_CHAT_ID,
            'text':       message,
            'parse_mode': 'HTML',
        }).encode('utf-8')
        req = urllib.request.Request(
            url, data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        urllib.request.urlopen(req, timeout=15)
        print("  📱 Telegram message sent.")
    except Exception as e:
        print(f"  ❌ Telegram error: {e}")


# ============================================================
# TEST MODE  —  runs instantly, no ML, just verifies Telegram
# ============================================================

def run_test():
    now_str = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    print("\n🧪 TEST MODE — sending a fake signal to Telegram...")

    msg = (
        "🧪 <b>TEST SIGNAL — Setup Verified ✅</b>\n\n"
        "This is a fake signal to confirm your alerts are working.\n\n"
        "🟢 <b>▲ ETH — UP</b>\n"
        "Confidence:     68%\n"
        "Expected move:  +2.8%\n"
        "Price now:      $3,412.00\n"
        "Target price:   $3,507.54\n\n"
        "🟢 <b>▲ BTC — UP</b>\n"
        "Confidence:     63%\n"
        "Expected move:  +3.1%\n"
        "Price now:      $84,200.00\n"
        "Target price:   $86,810.20\n\n"
        f"⏰ {now_str}\n"
        "✅ If you can read this, your setup is working!"
    )

    send_telegram(msg)
    print("✅ Test complete. Check your Telegram!")


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
    print(f" {len(df)} rows.")
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(df):
    df = df.copy()

    df['returns']       = df['close'].pct_change()
    df['log_returns']   = np.log(df['close'] / df['close'].shift(1))
    df['candle_body']   = (df['close'] - df['open']).abs() / df['open']
    df['candle_range']  = (df['high'] - df['low']) / df['open']
    df['rel_body_size'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-10)

    for w in [5, 10, 20, 50, 200]:
        df[f'ma_{w}'] = df['close'].rolling(w).mean()

    df['ma_5_ratio']   = df['close'] / df['ma_5']
    df['ma_10_ratio']  = df['close'] / df['ma_10']
    df['ma_200_ratio'] = df['close'] / df['ma_200']
    df['is_uptrend']   = (df['close'] > df['ma_200']).astype(int)
    df['ma200_slope']  = df['ma_200'].pct_change(5)

    for p in [3, 5, 10]:
        df[f'momentum_{p}'] = df['close'].pct_change(p)

    df['volume_ma']     = df['volume'].rolling(20).mean()
    df['volume_ratio']  = df['volume'] / (df['volume_ma'] + 1e-10)
    df['vol_expansion'] = (df['volume_ratio'] > 1.5).astype(int)

    delta = df['close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    df['bb_mid']      = df['close'].rolling(20).mean()
    df['bb_std']      = df['close'].rolling(20).std()
    df['bb_upper']    = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower']    = df['bb_mid'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width']    = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-10)

    ema12             = df['close'].ewm(span=12, adjust=False).mean()
    ema26             = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low']   = (df['low']  < df['low'].shift(1)).astype(int)

    is_green = df['close'] > df['open']
    is_red   = df['close'] < df['open']
    grp_g    = (is_green != is_green.shift()).cumsum()
    grp_r    = (is_red   != is_red.shift()).cumsum()
    df['consec_green'] = np.where(is_green, is_green.groupby(grp_g).cumcount() + 1, 0)
    df['consec_red']   = np.where(is_red,   is_red.groupby(grp_r).cumcount()   + 1, 0)

    prox = SR_THRESHOLD_PCT / 100
    for w in [20, 50]:
        df[f'resistance_{w}'] = df['high'].rolling(w).max()
        df[f'support_{w}']    = df['low'].rolling(w).min()
        df[f'dist_to_resistance_{w}'] = (df[f'resistance_{w}'] - df['close']) / df['close']
        df[f'dist_to_support_{w}']    = (df['close'] - df[f'support_{w}'])    / df['close']
        df[f'near_resistance_{w}']    = (df[f'dist_to_resistance_{w}'] < prox).astype(int)
        df[f'near_support_{w}']       = (df[f'dist_to_support_{w}']    < prox).astype(int)
        res = df[f'resistance_{w}'].shift(1)
        sup = df[f'support_{w}'].shift(1)
        df[f'resistance_touches_{w}'] = ((df['high'] >= res*(1-prox)) & (df['low'] <= res*(1+prox))).rolling(w).sum()
        df[f'support_touches_{w}']    = ((df['low']  <= sup*(1+prox)) & (df['high']>= sup*(1-prox))).rolling(w).sum()

    df['sfp_bearish'] = ((df['high'] > df['resistance_20'].shift(1)) &
                         (df['close'] < df['resistance_20'].shift(1))).astype(int)
    df['sfp_bullish'] = ((df['low']  < df['support_20'].shift(1)) &
                         (df['close'] > df['support_20'].shift(1))).astype(int)
    return df


# ============================================================
# PREDICTION
# ============================================================

def run_prediction(symbol, config):
    label  = config['label']
    cutoff = config['confidence_cutoff']
    print(f"\n{'─'*52}")
    print(f"  {label}  ({config['regime']}, {config['model']}, ≥{cutoff:.0%})")

    df = fetch_candles(symbol, TIMEFRAME, config['n_candles'])
    df = add_features(df)

    future_return       = df['close'].shift(-PREDICT_AHEAD).sub(df['close']).div(df['close'])
    df['target']        = np.where(future_return >  MOVE_THRESHOLD, 1,
                          np.where(future_return < -MOVE_THRESHOLD, 0, np.nan))
    df['future_return'] = future_return

    df_model  = df[df['is_uptrend'] == 1].copy() if config['regime'] == 'uptrend_only' else df.copy()
    available = df_model.dropna(subset=FEATURE_COLS)
    train_df  = available.iloc[:-1]
    live_row  = available.iloc[-1]

    if len(train_df) < 200:
        print(f"  ⚠️  Not enough training data. Skipping.")
        return None

    mask_clf  = train_df['target'].notna()
    X_clf     = train_df.loc[mask_clf, FEATURE_COLS].values
    y_clf     = train_df.loc[mask_clf, 'target'].values
    mask_reg  = train_df['future_return'].notna()
    X_reg     = train_df.loc[mask_reg, FEATURE_COLS].values
    y_reg     = train_df.loc[mask_reg, 'future_return'].values
    X_live    = live_row[FEATURE_COLS].values.reshape(1, -1)

    current_price = live_row['close']
    is_uptrend    = bool(live_row['is_uptrend'])

    if config['regime'] == 'uptrend_only' and not is_uptrend:
        print(f"  ⏸️  Not in uptrend. No signal.")
        return None

    if len(X_clf) < 50:
        print(f"  ⚠️  Not enough signal examples. Skipping.")
        return None

    clf = (ExtraTreesClassifier if config['model'] == 'extra_trees' else RandomForestClassifier)(
        n_estimators=150, max_features='sqrt', random_state=42, n_jobs=-1
    )
    clf.fit(X_clf, y_clf)

    reg = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=42, n_jobs=-1)
    reg.fit(X_reg, y_reg)

    proba     = clf.predict_proba(X_live)[0]
    classes   = list(clf.classes_)
    prob_up   = proba[classes.index(1)] if 1 in classes else 0.0
    prob_down = proba[classes.index(0)] if 0 in classes else 0.0

    expected_move = float(reg.predict(X_live)[0])
    confidence    = max(prob_up, prob_down)
    direction     = 'UP' if prob_up >= prob_down else 'DOWN'
    arrow         = '▲' if direction == 'UP' else '▼'

    print(f"  {arrow} {direction}  |  Confidence: {confidence:.1%}  |  Expected: {expected_move*100:+.2f}%")
    print(f"  Current: ${current_price:,.4f}")

    if confidence < cutoff:
        print(f"  ⏸️  Below cutoff ({cutoff:.0%}). No signal.")
        return None

    target_price = current_price * (1 + expected_move)
    print(f"  🚨 SIGNAL!  Target: ${target_price:,.4f}")

    return {
        'symbol':        label,
        'direction':     direction,
        'confidence':    confidence,
        'prob_up':       prob_up,
        'prob_down':     prob_down,
        'current_price': current_price,
        'expected_move': expected_move,
        'target_price':  target_price,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    now_str = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    print(f"\n{'='*52}")
    print(f"  🚀  Crypto Signal Check  —  {now_str}")
    print(f"{'='*52}")

    signals = []
    for symbol, config in TOKEN_CONFIGS.items():
        try:
            result = run_prediction(symbol, config)
            if result:
                signals.append(result)
        except Exception as e:
            print(f"\n  ❌  Error on {symbol}: {e}")

    print(f"\n{'='*52}")
    print(f"  RESULT: {len(signals)} signal(s) fired")
    print(f"{'='*52}")

    if not signals:
        print("  No signals this check.")
        send_telegram(f"✅ <b>Crypto check complete</b> — no signals this run.\n⏰ {now_str}")
    else:
        lines = [f"🚨 <b>{len(signals)} Signal(s) Found!</b>\n⏰ {now_str}\n"]
        for s in signals:
            emoji = '🟢' if s['direction'] == 'UP' else '🔴'
            arrow = '▲' if s['direction'] == 'UP' else '▼'
            print(f"\n  {emoji} {arrow} {s['symbol']} — {s['direction']}")
            print(f"     Confidence:    {s['confidence']:.1%}")
            print(f"     Current price: ${s['current_price']:,.4f}")
            print(f"     Expected move: {s['expected_move']*100:+.2f}%")
            print(f"     Target price:  ${s['target_price']:,.4f}")

            lines.append(
                f"{emoji} <b>{arrow} {s['symbol']} — {s['direction']}</b>\n"
                f"Confidence:     {s['confidence']:.0%}\n"
                f"Expected move:  {s['expected_move']*100:+.1f}%\n"
                f"Price now:      ${s['current_price']:,.2f}\n"
                f"Target price:   ${s['target_price']:,.2f}\n"
            )
        send_telegram('\n'.join(lines))

    print(f"\nDone — {now_str}\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    if '--test' in sys.argv:
        run_test()
    else:
        main()
