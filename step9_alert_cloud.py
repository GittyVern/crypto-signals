#!/usr/bin/env python3
"""
step9_alert_cloud.py  (v4 — incremental candle caching)
=========================================================
Crypto trading signal system — runs on GitHub Actions every 4 hours.
Sends a Telegram message every run: signals found, or "no strong signal".

Tokens:    BTC, ETH, BNB, SOL, XRP
Timeframe: 4h candles  |  Training window: ~7 years per token

Confidence tiers:
  🟡  55-60%  Weak     — slight lean, treat as FYI
  🟠  60-70%  Moderate — meaningful signal, watch closely
  🟢/🔴 >=70% Strong  — historically consistent, highest conviction

Only tokens with confidence >= 55% appear in the message.
A message is always sent (signals found, or quiet check confirmed).

Candle caching:
  - First run: fetches full 7yr history, saves to data/*.csv in repo
  - Every run after: loads saved CSV, fetches only new candles since last run
  - CSV is committed back to repo after each run (persistent, no data loss)
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
# SETTINGS
# ============================================================

TG_BOT_TOKEN = os.environ.get('TG_BOT_TOKEN', '')
TG_CHAT_ID   = os.environ.get('TG_CHAT_ID',   '')

TIMEFRAME     = '4h'
PREDICT_AHEAD = 1
N_CANDLES     = 15_300   # ~7 years of 4h candles
MIN_CONF      = 0.55

TIER_WEAK     = 0.55
TIER_MODERATE = 0.60
TIER_STRONG   = 0.70

CACHE_DIR = 'data'   # saved to repo — persistent across runs

TOKEN_CONFIGS = {
    'BTC/USDT': {
        'label':          'BTC',
        'move_threshold': 0.02,
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
        'move_threshold': 0.025,
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
        'move_threshold': 0.02,
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
        'move_threshold': 0.025,
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
        'move_threshold': 0.025,
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
    'near_resistance', 'near_support', 'touch_count_hi', 'touch_count_lo',
    'sfp_bear', 'sfp_bull',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
    'regime', 'higher_high', 'lower_low', 'trend_str',
]


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("  TG_BOT_TOKEN or TG_CHAT_ID not set. Skipping.")
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
            method='POST',
        )
        urllib.request.urlopen(req, timeout=15)
        print("  Telegram message sent.")
    except Exception as e:
        print(f"  Telegram error: {e}")


# ============================================================
# TEST MODE
# ============================================================

def run_test():
    now_str = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    print("\nTEST MODE — sending a fake signal to Telegram...")
    msg = (
        "TEST SIGNAL — Setup Verified\n\n"
        "This is a fake signal to confirm alerts are working.\n\n"
        "ETH UP | Strong >=70%\n"
        "Confidence:     74%\n"
        "Expected move:  +2.8%\n"
        "Price now:      $3,412.00\n"
        "Target price:   $3,507.54\n\n"
        f"Time: {now_str}\n"
        "If you can read this, your setup is working!"
    )
    send_telegram(msg)
    print("Test complete. Check your Telegram!")


# ============================================================
# DATA FETCHING — incremental, saves to data/ folder in repo
# ============================================================

def fetch_candles(symbol: str, timeframe: str, n_candles: int) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, symbol.replace('/', '_') + '_' + timeframe + '.csv')

    exchange = ccxt.okx({'enableRateLimit': True})
    tf_ms = {'1h': 3600000, '4h': 14400000}[timeframe]

    if os.path.exists(cache_file):
        cached = pd.read_csv(cache_file, parse_dates=['ts'])
        last_ts_ms = int(cached['ts'].iloc[-1].timestamp() * 1000)
        since = last_ts_ms + tf_ms
        print(f"  {symbol}: loading saved history, fetching new candles only...", end='', flush=True)
    else:
        since = exchange.milliseconds() - (n_candles + 200) * tf_ms
        cached = None
        print(f"  {symbol}: no saved history, fetching full 7yr history...", end='', flush=True)

    all_bars = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        all_bars.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000:
            break

    if all_bars:
        new_df = pd.DataFrame(all_bars, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        new_df['ts'] = pd.to_datetime(new_df['ts'], unit='ms')
        df = pd.concat([cached, new_df], ignore_index=True) if cached is not None else new_df
    else:
        df = cached

    df = df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    df = df.tail(n_candles).reset_index(drop=True)
    df.to_csv(cache_file, index=False)

    yrs = round(len(df) * 4 / 8760, 1)
    new_count = len(all_bars) if all_bars else 0
    print(f" {len(df)} rows ({yrs}yr), {new_count} new candles fetched.")
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c  = df['close']

    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    rs    = gain.ewm(com=13, adjust=False).mean() / loss.ewm(com=13, adjust=False).mean()
    df['rsi'] = 100 - 100 / (1 + rs)

    ema12             = c.ewm(span=12, adjust=False).mean()
    ema26             = c.ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    sma20          = c.rolling(20).mean()
    std20          = c.rolling(20).std()
    bb_upper       = sma20 + 2 * std20
    bb_lower       = sma20 - 2 * std20
    df['bb_width'] = (bb_upper - bb_lower) / sma20
    df['bb_pos']   = (c - bb_lower) / (bb_upper - bb_lower + 1e-9)

    for p in [7, 21, 50, 100, 200]:
        df[f'ma{p}'] = c.rolling(p).mean()
    df['ma7_21_ratio']   = df['ma7']  / (df['ma21']  + 1e-9)
    df['ma21_50_ratio']  = df['ma21'] / (df['ma50']  + 1e-9)
    df['ma50_200_ratio'] = df['ma50'] / (df['ma200'] + 1e-9)
    for p in [7, 21, 50, 200]:
        df[f'price_vs_ma{p}'] = c / (df[f'ma{p}'] + 1e-9) - 1

    for p in [3, 7, 14, 21]:
        df[f'mom{p}'] = c.pct_change(p)

    df['atr']       = (df['high'] - df['low']).rolling(14).mean() / (c + 1e-9)
    df['vol_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)
    df['hl_ratio']  = (df['high'] - df['low']) / (c + 1e-9)
    df['oc_ratio']  = (df['close'] - df['open']) / (df['open'] + 1e-9)

    roll_high = df['high'].rolling(20).max()
    roll_low  = df['low'].rolling(20).min()
    tol = 0.005
    df['near_resistance'] = ((roll_high - c) / (c + 1e-9) < tol).astype(int)
    df['near_support']    = ((c - roll_low)  / (c + 1e-9) < tol).astype(int)
    df['touch_count_hi']  = (abs(df['high'] - roll_high) / (c + 1e-9) < tol).astype(int)
    df['touch_count_lo']  = (abs(df['low']  - roll_low)  / (c + 1e-9) < tol).astype(int)

    prev_high = df['high'].shift(1).rolling(5).max()
    prev_low  = df['low'].shift(1).rolling(5).min()
    df['sfp_bear'] = ((df['high'] > prev_high) & (df['close'] < prev_high)).astype(int)
    df['sfp_bull'] = ((df['low']  < prev_low)  & (df['close'] > prev_low)).astype(int)

    df['hour_sin']   = np.sin(2 * np.pi * df['ts'].dt.hour / 24)
    df['hour_cos']   = np.cos(2 * np.pi * df['ts'].dt.hour / 24)
    df['dow_sin']    = np.sin(2 * np.pi * df['ts'].dt.dayofweek / 7)
    df['dow_cos']    = np.cos(2 * np.pi * df['ts'].dt.dayofweek / 7)
    df['is_weekend'] = (df['ts'].dt.dayofweek >= 5).astype(int)

    df['regime']      = (c > df['ma200']).astype(int)
    df['higher_high'] = ((df['high'] > df['high'].shift(1)) &
                          (df['high'].shift(1) > df['high'].shift(2))).astype(int)
    df['lower_low']   = ((df['low'] < df['low'].shift(1)) &
                          (df['low'].shift(1) < df['low'].shift(2))).astype(int)
    df['trend_str']   = df['higher_high'].rolling(5).sum() - df['lower_low'].rolling(5).sum()

    return df


# ============================================================
# CONFIDENCE TIER
# ============================================================

def get_tier(confidence: float) -> tuple:
    if confidence >= TIER_STRONG:
        return ('🟢', 'Strong >=70%')
    elif confidence >= TIER_MODERATE:
        return ('🟠', 'Moderate 60-70%')
    else:
        return ('🟡', 'Weak 55-60%')


# ============================================================
# PREDICTION
# ============================================================

def run_prediction(symbol: str, config: dict):
    label     = config['label']
    threshold = config['move_threshold']
    print(f"\n--- {label}  |  threshold={threshold*100:.1f}% ---")

    df = fetch_candles(symbol, TIMEFRAME, N_CANDLES)
    df = add_features(df)

    future_ret       = df['close'].shift(-PREDICT_AHEAD) / df['close'] - 1
    df['target']     = np.nan
    df.loc[future_ret >  threshold, 'target'] = 1
    df.loc[future_ret < -threshold, 'target'] = 0
    df['future_ret'] = future_ret

    available = df.dropna(subset=FEATURE_COLS)
    if len(available) < 250:
        print(f"  Not enough data. Skipping.")
        return None

    train_df = available.iloc[:-1]
    live_row = available.iloc[-1]
    X_live   = live_row[FEATURE_COLS].values.reshape(1, -1)

    clf_mask = train_df['target'].notna()
    X_clf    = train_df.loc[clf_mask, FEATURE_COLS].values
    y_clf    = train_df.loc[clf_mask, 'target'].values

    if len(X_clf) < 50 or len(np.unique(y_clf)) < 2:
        print(f"  Not enough labelled examples. Skipping.")
        return None

    clf = RandomForestClassifier(**config['clf_params'])
    clf.fit(X_clf, y_clf)

    reg_mask = train_df['future_ret'].notna()
    reg      = RandomForestRegressor(n_estimators=100, max_features='sqrt',
                                      random_state=42, n_jobs=-1)
    reg.fit(train_df.loc[reg_mask, FEATURE_COLS].values,
            train_df.loc[reg_mask, 'future_ret'].values)

    proba         = clf.predict_proba(X_live)[0]
    classes       = list(clf.classes_)
    prob_up       = proba[classes.index(1)] if 1 in classes else 0.0
    prob_down     = proba[classes.index(0)] if 0 in classes else 0.0
    confidence    = max(prob_up, prob_down)
    direction     = 'UP' if prob_up >= prob_down else 'DOWN'
    arrow         = 'UP' if direction == 'UP' else 'DOWN'
    expected_move = float(reg.predict(X_live)[0])
    current_price = float(live_row['close'])
    target_price  = current_price * (1 + expected_move)

    print(f"  {direction}  |  Confidence: {confidence:.1%}  |  Expected: {expected_move*100:+.2f}%")

    if confidence < MIN_CONF:
        print(f"  Below {MIN_CONF:.0%} minimum. No signal.")
        return None

    tier_emoji, tier_label = get_tier(confidence)
    if confidence >= TIER_STRONG and direction == 'DOWN':
        tier_emoji = '🔴'

    print(f"  SIGNAL!  Tier: {tier_label}  Target: ${target_price:,.4f}")

    return {
        'symbol':        label,
        'direction':     direction,
        'arrow':         arrow,
        'confidence':    confidence,
        'expected_move': expected_move,
        'current_price': current_price,
        'target_price':  target_price,
        'tier_emoji':    tier_emoji,
        'tier_label':    tier_label,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    now_str = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    print(f"\n=== Crypto Signal Check  {now_str} ===")

    signals = []
    for symbol, config in TOKEN_CONFIGS.items():
        try:
            result = run_prediction(symbol, config)
            if result:
                signals.append(result)
        except Exception as e:
            print(f"\n  Error on {symbol}: {e}")

    print(f"\n=== RESULT: {len(signals)} signal(s) above {MIN_CONF:.0%} ===")

    if not signals:
        send_telegram(
            f"✅ <b>Crypto check complete</b>\n"
            f"No token above 55% confidence this run.\n"
            f"⏰ {now_str}"
        )
    else:
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        lines = [f"📊 <b>Signal Update</b>  |  ⏰ {now_str}\n"]
        for s in signals:
            lines.append(
                f"{s['tier_emoji']} <b>{s['arrow']} {s['symbol']} - "
                f"{s['direction']}</b>  |  <b>{s['tier_label']}</b>\n"
                f"Confidence:     {s['confidence']:.0%}\n"
                f"Expected move:  {s['expected_move']*100:+.1f}%\n"
                f"Price now:      ${s['current_price']:,.4f}\n"
                f"Target price:   ${s['target_price']:,.4f}\n"
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
