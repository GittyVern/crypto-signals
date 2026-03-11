#!/usr/bin/env python3
"""
step9_alert_cloud.py  (v6 — paper trading mode)
=========================================================
Crypto trading signal system — runs on GitHub Actions every 4 hours.

Changes in v6:
  - Paper trading mode: logs hypothetical trades to data/paper_trades.csv
  - Per-token stop-loss strategy (from backtest results):
      BTC / ETH / BNB  →  fixed 2% stop-loss (hold 1 candle, 4h)
      SOL / XRP        →  trailing 2% stop-loss (ride winners up to 80h)
  - Telegram message always includes paper trade updates and scoreboard

Candle caching:
  - First run: fetches full 7yr history, saves to data/*.csv in repo
  - Every run after: loads saved CSV, fetches only new candles since last run
  - CSV is committed back to repo after each run (persistent, no data loss)

Paper trades:
  - Stored in data/paper_trades.csv (also committed to repo)
  - One open trade max per token at any time
  - Closed trades show final P&L; open trades show live unrealised P&L
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


# ============================================================
# PAPER TRADING CONFIG
# ============================================================

PAPER_TRADES_FILE = os.path.join(CACHE_DIR, 'paper_trades.csv')

FIXED_SL_PCT = 0.02   # 2% stop-loss for fixed strategy (BTC, ETH, BNB)
TRAIL_PCT    = 0.02   # 2% trailing stop-loss           (SOL, XRP)
MAX_CANDLES  = 20     # max 4h candles to hold a trailing-stop trade (= 80h)

# Strategy per token — based on walk-forward backtest results
PAPER_STRATEGY = {
    'BTC': 'fixed',
    'ETH': 'fixed',
    'BNB': 'fixed',
    'SOL': 'trailing',
    'XRP': 'trailing',
}


# ============================================================
# TOKEN CONFIGS
# ============================================================

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
            'n_jobs':            1,
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
            'n_jobs':            1,
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
            'n_jobs':            1,
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
            'n_jobs':            1,
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
            'n_jobs':            1,
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
        return ('🟢', 'Strong ≥70%')
    elif confidence >= TIER_MODERATE:
        return ('🟠', 'Moderate 60-70%')
    else:
        return ('🟡', 'Weak 55-60%')


# ============================================================
# PREDICTION  (returns signal dict AND candle DataFrame)
# ============================================================

def run_prediction(symbol: str, config: dict):
    """
    Run model prediction for one token.
    Always returns (signal_dict or None, feature_df or None).
    The df is returned even when there is no signal — it's needed
    to update any open paper trades for that token.
    """
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
        return None, df

    train_df = available.iloc[:-1]
    live_row = available.iloc[-1]
    X_live   = live_row[FEATURE_COLS].values.reshape(1, -1)

    clf_mask = train_df['target'].notna()
    X_clf    = train_df.loc[clf_mask, FEATURE_COLS].values
    y_clf    = train_df.loc[clf_mask, 'target'].values

    if len(X_clf) < 50 or len(np.unique(y_clf)) < 2:
        print(f"  Not enough labelled examples. Skipping.")
        return None, df

    clf = RandomForestClassifier(**config['clf_params'])
    clf.fit(X_clf, y_clf)

    reg_mask = train_df['future_ret'].notna()
    reg      = RandomForestRegressor(n_estimators=100, max_features='sqrt',
                                      random_state=42, n_jobs=1)
    reg.fit(train_df.loc[reg_mask, FEATURE_COLS].values,
            train_df.loc[reg_mask, 'future_ret'].values)

    proba         = clf.predict_proba(X_live)[0]
    classes       = list(clf.classes_)
    prob_up       = proba[classes.index(1)] if 1 in classes else 0.0
    prob_down     = proba[classes.index(0)] if 0 in classes else 0.0
    confidence    = max(prob_up, prob_down)
    direction     = 'UP' if prob_up >= prob_down else 'DOWN'
    expected_move = float(reg.predict(X_live)[0])
    current_price = float(live_row['close'])
    target_price  = current_price * (1 + expected_move)
    candle_ts     = live_row['ts']

    print(f"  {direction}  |  Confidence: {confidence:.1%}  |  Expected: {expected_move*100:+.2f}%")

    if confidence < MIN_CONF:
        print(f"  Below {MIN_CONF:.0%} minimum. No signal.")
        return None, df

    if abs(expected_move) < 0.005:
        print(f"  Expected move too small ({expected_move*100:+.2f}%). No signal.")
        return None, df

    tier_emoji, tier_label = get_tier(confidence)
    if confidence >= TIER_STRONG and direction == 'DOWN':
        tier_emoji = '🔴'

    print(f"  SIGNAL!  Tier: {tier_label}  Target: ${target_price:,.4f}")

    return {
        'symbol':        label,        # e.g. 'BTC'
        'full_symbol':   symbol,       # e.g. 'BTC/USDT'
        'direction':     direction,
        'confidence':    confidence,
        'expected_move': expected_move,
        'current_price': current_price,
        'target_price':  target_price,
        'tier_emoji':    tier_emoji,
        'tier_label':    tier_label,
        'candle_ts':     candle_ts,
    }, df


# ============================================================
# PAPER TRADING — load / save / update / open
# ============================================================

_PAPER_COLS = [
    'trade_id', 'token', 'full_symbol', 'direction', 'strategy',
    'entry_ts', 'entry_price', 'confidence', 'exp_move',
    'status', 'exit_ts', 'exit_price', 'pnl_pct', 'exit_reason', 'candles_held',
]


def load_paper_trades() -> pd.DataFrame:
    """Load paper trades CSV. Returns an empty DataFrame if the file doesn't exist yet."""
    if os.path.exists(PAPER_TRADES_FILE):
        df = pd.read_csv(PAPER_TRADES_FILE, parse_dates=['entry_ts', 'exit_ts'])
        return df
    return pd.DataFrame(columns=_PAPER_COLS)


def save_paper_trades(trades_df: pd.DataFrame):
    """Save paper trades back to CSV in the data/ folder."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    trades_df.to_csv(PAPER_TRADES_FILE, index=False)
    n_open   = (trades_df['status'] == 'open').sum()   if not trades_df.empty else 0
    n_closed = (trades_df['status'] == 'closed').sum() if not trades_df.empty else 0
    print(f"  Paper trades saved: {n_open} open, {n_closed} closed.")


def update_open_trades(trades_df: pd.DataFrame, candle_data: dict):
    """
    Walk through every open paper trade and check whether it should close,
    using the latest candle data fetched this run.

    Fixed trades  → close after 1 candle (4h), with 2% hard stop
    Trailing trades → ride up to MAX_CANDLES candles, exit on 2% reversal

    Returns (updated_df, list_of_newly_closed_summaries).
    """
    closed_this_run = []

    if trades_df.empty:
        return trades_df, closed_this_run

    for idx in trades_df.index:
        row = trades_df.loc[idx]
        if row['status'] != 'open':
            continue

        full_sym = row['full_symbol']
        df = candle_data.get(full_sym)
        if df is None:
            print(f"  {row['token']}: no candle data this run — skipping trade update.")
            continue

        entry_ts   = pd.to_datetime(row['entry_ts'])
        post_entry = df[df['ts'] > entry_ts].reset_index(drop=True)

        if len(post_entry) == 0:
            continue   # trade opened this very run; nothing to evaluate yet

        direction   = row['direction']
        entry_price = float(row['entry_price'])
        strategy    = row['strategy']

        pnl = exit_price = exit_ts = exit_reason = candles_held = None

        # ── Fixed stop-loss: one 4h candle, 2% hard stop ──────────────────────
        if strategy == 'fixed':
            c  = post_entry.iloc[0]
            hi = float(c['high'])
            lo = float(c['low'])
            cl = float(c['close'])

            if direction == 'UP':
                if lo / entry_price - 1 <= -FIXED_SL_PCT:
                    pnl         = -FIXED_SL_PCT
                    exit_price  = entry_price * (1 - FIXED_SL_PCT)
                    exit_reason = 'sl_hit'
                else:
                    exit_price  = cl
                    pnl         = cl / entry_price - 1
                    exit_reason = 'closed_4h'
            else:   # SHORT
                if hi / entry_price - 1 >= FIXED_SL_PCT:
                    pnl         = -FIXED_SL_PCT
                    exit_price  = entry_price * (1 + FIXED_SL_PCT)
                    exit_reason = 'sl_hit'
                else:
                    exit_price  = cl
                    pnl         = -(cl / entry_price - 1)
                    exit_reason = 'closed_4h'

            exit_ts      = c['ts']
            candles_held = 1

        # ── Trailing stop: ride the move, exit on 2% reversal ─────────────────
        else:
            candles = post_entry.head(MAX_CANDLES)

            if direction == 'UP':
                peak        = entry_price
                trail_level = peak * (1 - TRAIL_PCT)
                for i, c in candles.iterrows():
                    # 1. Check stop FIRST against the current trail
                    if float(c['low']) <= trail_level:
                        pnl          = trail_level / entry_price - 1
                        exit_price   = trail_level
                        exit_ts      = c['ts']
                        exit_reason  = 'trailing_stop'
                        candles_held = i + 1
                        break
                    # 2. Only if stop wasn't hit: update peak from new high
                    if float(c['high']) > peak:
                        peak        = float(c['high'])
                        trail_level = peak * (1 - TRAIL_PCT)
            else:   # SHORT
                trough      = entry_price
                trail_level = trough * (1 + TRAIL_PCT)
                for i, c in candles.iterrows():
                    # 1. Check stop FIRST against the current trail
                    if float(c['high']) >= trail_level:
                        pnl          = -(trail_level / entry_price - 1)
                        exit_price   = trail_level
                        exit_ts      = c['ts']
                        exit_reason  = 'trailing_stop'
                        candles_held = i + 1
                        break
                    # 2. Only if stop wasn't hit: update trough from new low
                    if float(c['low']) < trough:
                        trough      = float(c['low'])
                        trail_level = trough * (1 + TRAIL_PCT)

            # Not stopped out — check if we've hit max candles
            if pnl is None and len(candles) >= MAX_CANDLES:
                last         = candles.iloc[-1]
                exit_price   = float(last['close'])
                exit_ts      = last['ts']
                candles_held = MAX_CANDLES
                pnl          = (exit_price / entry_price - 1) if direction == 'UP' \
                               else -(exit_price / entry_price - 1)
                exit_reason  = 'timed_out'

        # ── Write result back ──────────────────────────────────────────────────
        if pnl is not None:
            trades_df.at[idx, 'status']       = 'closed'
            trades_df.at[idx, 'exit_ts']      = exit_ts
            trades_df.at[idx, 'exit_price']   = exit_price
            trades_df.at[idx, 'pnl_pct']      = pnl
            trades_df.at[idx, 'exit_reason']  = exit_reason
            trades_df.at[idx, 'candles_held'] = candles_held

            reason_pretty = {
                'sl_hit': 'SL hit', 'trailing_stop': 'trail SL hit',
                'closed_4h': '4h close', 'timed_out': 'max hold',
            }.get(exit_reason, exit_reason)

            print(f"  Paper trade closed: {row['token']} {direction}  "
                  f"PnL={pnl*100:+.2f}%  ({reason_pretty})")

            closed_this_run.append({
                'token':        row['token'],
                'direction':    direction,
                'entry_price':  entry_price,
                'exit_price':   exit_price,
                'pnl_pct':      pnl,
                'exit_reason':  exit_reason,
                'candles_held': int(candles_held),
            })

    return trades_df, closed_this_run


def open_paper_trade(trades_df: pd.DataFrame, signal: dict) -> tuple:
    """
    Open a new paper trade for the given signal.
    If a trade is already open for this token, skip (no duplicates).
    Returns (updated_df, was_opened_bool).
    """
    token    = signal['symbol']
    full_sym = signal['full_symbol']

    # Don't open a second trade if one is already running for this token
    if not trades_df.empty:
        existing = trades_df[(trades_df['token'] == token) & (trades_df['status'] == 'open')]
        if not existing.empty:
            print(f"  {token}: paper trade already open — skipping duplicate.")
            return trades_df, False

    strategy = PAPER_STRATEGY.get(token, 'fixed')
    trade_id = f"{token}_{pd.Timestamp(signal['candle_ts']).strftime('%Y%m%d_%H%M')}"

    new_row = pd.DataFrame([{
        'trade_id':    trade_id,
        'token':       token,
        'full_symbol': full_sym,
        'direction':   signal['direction'],
        'strategy':    strategy,
        'entry_ts':    signal['candle_ts'],
        'entry_price': signal['current_price'],
        'confidence':  round(signal['confidence'], 4),
        'exp_move':    round(signal['expected_move'], 4),
        'status':      'open',
        'exit_ts':     pd.NaT,
        'exit_price':  np.nan,
        'pnl_pct':     np.nan,
        'exit_reason': '',
        'candles_held': np.nan,
    }])

    trades_df = pd.concat([trades_df, new_row], ignore_index=True)
    print(f"  {token}: paper {signal['direction']} opened @ ${signal['current_price']:,.4f}"
          f"  (strategy: {strategy} 2% SL)")
    return trades_df, True


# ============================================================
# TELEGRAM MESSAGE BUILDER
# ============================================================

def build_message(signals: list, closed_this_run: list, new_paper_tokens: list,
                  trades_df: pd.DataFrame, candle_data: dict, now_str: str) -> str:
    """
    Build the full Telegram message for this 4h run.
    Sections:
      1. Signals (or "no signal" if quiet)
      2. Paper Trading — closed this run, open trades, scoreboard
    """
    SEP  = "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    lines = [f"📊 <b>Signal Check</b>  ⏰ {now_str}"]

    # ── 1. Signals ────────────────────────────────────────────────────────────
    lines.append(SEP)
    if signals:
        for s in sorted(signals, key=lambda x: x['confidence'], reverse=True):
            tier_e = s['tier_emoji']
            strat  = PAPER_STRATEGY.get(s['symbol'], 'fixed')
            paper_note = (
                f"\n→ 📋 Paper {s['direction']} opened ({strat} 2% SL)"
                if s['symbol'] in new_paper_tokens else ""
            )
            lines.append(
                f"{tier_e} <b>{s['symbol']} {s['direction']} — {s['tier_label']}</b>\n"
                f"Confidence:  {s['confidence']:.0%}\n"
                f"Expected:    {s['expected_move']*100:+.1f}%\n"
                f"Price:       ${s['current_price']:,.4f}\n"
                f"Target:      ${s['target_price']:,.4f}"
                f"{paper_note}"
            )
    else:
        lines.append("✅ No signal met both filters (≥55% conf + ≥0.5% move).")

    # ── 2. Paper Trading ──────────────────────────────────────────────────────
    has_closed = len(closed_this_run) > 0
    has_open   = (not trades_df.empty) and (trades_df['status'] == 'open').any()
    has_hist   = (not trades_df.empty) and (trades_df['status'] == 'closed').any()

    if has_closed or has_open or has_hist:
        lines.append(SEP)
        lines.append("📋 <b>Paper Trading</b>")

        # Closed this run
        if has_closed:
            lines.append("")
            reason_map = {
                'sl_hit':        'SL hit',
                'trailing_stop': 'trail hit',
                'closed_4h':     '4h close',
                'timed_out':     'max hold',
            }
            for c in closed_this_run:
                pct = c['pnl_pct'] * 100
                if pct > 0:
                    emoji = "✅"
                elif c['exit_reason'] in ('sl_hit', 'trailing_stop'):
                    emoji = "🛑"
                else:
                    emoji = "⛔"
                reason    = reason_map.get(c['exit_reason'], c['exit_reason'])
                held_str  = f"{c['candles_held']} candle{'s' if c['candles_held'] != 1 else ''}"
                lines.append(
                    f"  {emoji} {c['token']} {c['direction']}: "
                    f"<b>{pct:+.2f}%</b>  ({reason}, {held_str})"
                )

        # Currently open trades
        if has_open:
            open_trades = trades_df[trades_df['status'] == 'open']
            lines.append("")
            for _, t in open_trades.iterrows():
                ep        = float(t['entry_price'])
                token     = t['token']
                direction = t['direction']
                full_sym  = t['full_symbol']
                df_c      = candle_data.get(full_sym)
                since_str = pd.to_datetime(t['entry_ts']).strftime('%m/%d %H:%M')

                if df_c is not None:
                    cp  = float(df_c.iloc[-1]['close'])
                    ur  = (cp / ep - 1) if direction == 'UP' else -(cp / ep - 1)
                    price_str = f"${ep:,.4f} → ${cp:,.4f} ({ur*100:+.2f}%)"
                else:
                    price_str = f"${ep:,.4f}"

                lines.append(f"  🔓 {token} {direction} @ {price_str}  [since {since_str} UTC]")

        # Scoreboard
        if has_hist:
            closed_df  = trades_df[trades_df['status'] == 'closed']
            wins       = int((closed_df['pnl_pct'] > 0).sum())
            losses     = int((closed_df['pnl_pct'] <= 0).sum())
            total      = len(closed_df)
            avg_pnl    = closed_df['pnl_pct'].mean() * 100
            total_pnl  = closed_df['pnl_pct'].sum() * 100
            lines.append(
                f"\n  🏆 {total} closed | {wins}W {losses}L | "
                f"Avg {avg_pnl:+.2f}% | Total {total_pnl:+.1f}%"
            )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    now_str = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    print(f"\n=== Crypto Signal Check  {now_str} ===")

    # ── Step 1: Load paper trades ──────────────────────────────────────────────
    trades_df = load_paper_trades()
    n_open = int((trades_df['status'] == 'open').sum()) if not trades_df.empty else 0
    print(f"\n  Paper trades: {len(trades_df)} total, {n_open} open")

    # ── Step 2: Fetch candles + run predictions for all tokens ─────────────────
    signals     = []
    candle_data = {}   # full_symbol → feature DataFrame

    for symbol, config in TOKEN_CONFIGS.items():
        try:
            result, df = run_prediction(symbol, config)
            if df is not None:
                candle_data[symbol] = df
            if result:
                signals.append(result)
        except Exception as e:
            print(f"\n  Error on {symbol}: {e}")

    print(f"\n=== RESULT: {len(signals)} signal(s) above {MIN_CONF:.0%} ===")

    # ── Step 3: Update open paper trades against latest candle data ────────────
    trades_df, closed_this_run = update_open_trades(trades_df, candle_data)

    # ── Step 4: Open new paper trades for any signals detected ─────────────────
    new_paper_tokens = []
    for signal in signals:
        trades_df, opened = open_paper_trade(trades_df, signal)
        if opened:
            new_paper_tokens.append(signal['symbol'])

    # ── Step 5: Save paper trades CSV (will be committed back to repo) ──────────
    save_paper_trades(trades_df)

    # ── Step 6: Build and send Telegram message ────────────────────────────────
    msg = build_message(
        signals, closed_this_run, new_paper_tokens,
        trades_df, candle_data, now_str
    )
    send_telegram(msg)

    print(f"\nDone — {now_str}\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    if '--test' in sys.argv:
        run_test()
    else:
        main()
