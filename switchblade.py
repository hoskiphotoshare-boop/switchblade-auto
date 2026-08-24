# ==========================================================
# SWITCHBLADE v47.80 - GITHUB ACTIONS EDITION (NO CACHE)
# ==========================================================
import sys
import matplotlib
matplotlib.use('Agg') # MUST BE BEFORE pyplot import for headless rendering
import matplotlib.pyplot as plt

import backtrader as bt
import yfinance as yf
import datetime
import pandas as pd
import requests
import io
import numpy as np
import os
import json
import time
import shutil
import warnings
import gc
import base64
from functools import partial
import multiprocessing

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# ==========================================
# DIRECTORY SETUP FOR GITHUB ACTIONS
# ==========================================
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# POINT-IN-TIME (PIT) UNIVERSE ENGINE
# ==========================================
class PITUniverseManager:
    def __init__(self):
        self.snapshots = {}
        self.all_tickers = set()
        print("   [PIT Engine] Loading index constituent changes strictly from CSV...")
        self.csv_data = self._load_csv_data()
        self._build_sp500()
        self._build_ndx()
        self._build_sp1000_proxy_universe()

    def _load_csv_data(self):
        csv_path = os.path.join(DATA_DIR, "full_production_holdings.csv")
        if not os.path.exists(csv_path):
            sys.exit(f"      [CRITICAL ERROR] PIT Universe CSV not found at {csv_path}. Execution aborted.")
        try:
            df = pd.read_csv(csv_path)
            df['Filing_Date'] = pd.to_datetime(df['Filing_Date']).dt.date
            df['Ticker'] = df['Ticker'].apply(self._clean_ticker)
            print(f"      [PIT Engine] Loaded {len(df)} records from {csv_path}")
            return df
        except Exception as e:
            sys.exit(f"      [CRITICAL ERROR] Error loading CSV: {e}")

    def _clean_ticker(self, symbol):
        if not symbol or pd.isna(symbol): return None
        s = str(symbol).split('|')[0].strip().upper().replace('.', '-').replace('/', '-')
        if not s or any(char.isdigit() for char in s) or len(s) > 8: return None
        return s

    def _build_sp500(self):
        timeline = []
        master_tickers = set()
        sp500_data = self.csv_data[self.csv_data['Fund'].str.contains('SP500|IVV|SPY', case=False, na=False)]
        if sp500_data.empty: sys.exit("      [CRITICAL ERROR] No S&P 500 data found.")
        dates = sorted(sp500_data['Filing_Date'].unique(), reverse=True)
        for dt in dates:
            members = set(sp500_data[sp500_data['Filing_Date'] == dt]['Ticker'].dropna())
            if members:
                timeline.append((dt, members))
                master_tickers.update(members)
        timeline.sort(key=lambda x: x[0], reverse=True)
        self.snapshots['SP500'] = timeline
        self.all_tickers.update(master_tickers)

    def _build_ndx(self):
        timeline = []
        unique_tickers = set()
        ndx_data = self.csv_data[self.csv_data['Fund'].str.contains('NASDAQ100|QQQ', case=False, na=False)]
        if ndx_data.empty: sys.exit("      [CRITICAL ERROR] No Nasdaq-100 data found.")
        dates = sorted(ndx_data['Filing_Date'].unique(), reverse=True)
        for dt in dates:
            members = set(ndx_data[ndx_data['Filing_Date'] == dt]['Ticker'].dropna())
            if members:
                timeline.append((dt, members))
                unique_tickers.update(members)
        timeline.sort(key=lambda x: x[0], reverse=True)
        self.snapshots['NDX'] = timeline
        self.all_tickers.update(unique_tickers)

    def _build_sp1000_proxy_universe(self):
        mid_timeline, small_timeline = [], []
        unique_mid, unique_small = set(), set()
        ijh_data = self.csv_data[self.csv_data['Fund'].str.contains('SP400|IJH|MDY', case=False, na=False)]
        ijr_data = self.csv_data[self.csv_data['Fund'].str.contains('SP600|IJR|VIOO', case=False, na=False)]
        
        for dt in sorted(ijh_data['Filing_Date'].unique(), reverse=True):
            members = set(ijh_data[ijh_data['Filing_Date'] == dt]['Ticker'].dropna())
            if members: mid_timeline.append((dt, members)); unique_mid.update(members)
            
        for dt in sorted(ijr_data['Filing_Date'].unique(), reverse=True):
            members = set(ijr_data[ijr_data['Filing_Date'] == dt]['Ticker'].dropna())
            if members: small_timeline.append((dt, members)); unique_small.update(members)

        mid_timeline.sort(key=lambda x: x[0], reverse=True)
        small_timeline.sort(key=lambda x: x[0], reverse=True)
        self.snapshots['SP1000_MID'] = mid_timeline
        self.snapshots['SP1000_SMALL'] = small_timeline
        self.all_tickers.update(unique_mid)
        self.all_tickers.update(unique_small)

    def is_constituent(self, symbol, universe, query_date, historical_mcap=None, current_price=None):
        if symbol not in self.all_tickers: return False
        keys = ['SP500', 'NDX', 'SP1000_MID', 'SP1000_SMALL'] if universe == "STANDARD" else [universe]
        for k in keys:
            if k in self.snapshots:
                for snap_date, constituents in self.snapshots[k]:
                    if query_date >= snap_date:
                        if symbol in constituents:
                            if k in ['SP1000_MID', 'SP1000_SMALL']:
                                if current_price is not None and current_price < 5.0: return False
                                return True
                            return True
                        break
        return False
    def get_all_tickers(self): return list(self.all_tickers)

pit_manager = PITUniverseManager()

# ==========================================
# PRE-FLIGHT INTEGRITY CHECK
# ==========================================
def verify_yfinance_integrity(tickers=["QQQ", "SPY"], threshold=0.01):
    print("    -> [DIAGNOSTIC] Running Pre-Flight yfinance Integrity Check...")
    prices = {t: {} for t in tickers}
    try:
        bulk = yf.download(tickers, period="5d", group_by='ticker', progress=False, threads=False)
        for t in tickers: prices[t]['Bulk'] = bulk[t]['Close'].dropna().iloc[-1]
        for t in tickers: prices[t]['Single'] = yf.Ticker(t).history(period="5d")['Close'].dropna().iloc[-1]
        intraday = yf.download(tickers, period="1d", interval="1m", group_by='ticker', progress=False, threads=False)
        for t in tickers: prices[t]['1-Min'] = intraday[t]['Close'].dropna().iloc[-1]

        abort = False
        print("-" * 75)
        for t in tickers:
            p_list = list(prices[t].values())
            p_max, p_min = max(p_list), min(p_list)
            diff_pct = (p_max - p_min) / p_min
            print(f"      {t}: Bulk=${prices[t]['Bulk']:<7.2f} | Single=${prices[t]['Single']:<7.2f} | 1m=${prices[t]['1-Min']:<7.2f} | Max Diff={diff_pct*100:.2f}%")
            if diff_pct > threshold: abort = True
        print("-" * 75)

        if abort: sys.exit("\n    [CRITICAL ERROR] yfinance endpoints are out of sync by >1%!")
        else: print("    -> [PASS] Integrity check clear. Prices are perfectly synchronized.\n")
    except Exception as e:
        print(f"    -> [WARNING] Integrity check encountered a network error: {e}\n    -> Continuing with caution...\n")

verify_yfinance_integrity()

# ==========================================
# STRATEGY CONFIGURATION (JSON DRIVEN)
# ==========================================
backtest_start_date = "2012-01-18"
max_stocks_per_univ = 4000
use_multiprocessing = True

GLOBAL_NITRO_ETFS = "SPXL, SPXS, TQQQ, SQQQ, UDOW, SDOW, TNA, TZA, MIDU, EDC, EDZ, YINN, YANG, EURL, INDL, TECL, TECS, SOXL, SOXS, FNGU, FNGD, WEBL, WEBS, FAS, FAZ, ERX, ERY, CURE, LABU, LABD, DRN, DRV, UTSL, DUSL, RETL, UGL, GLL, AGQ, ZSL, UCO, SCO, BOIL, KOLD, TMF, TMV, UST, PST, BITU, ETHU, UVXY"

# Load parameters from config.json
config_path = os.path.join(DATA_DIR, "config.json")
try:
    with open(config_path, "r") as f:
        cfg = json.load(f)
        mode = cfg.get("mode", "Backtest Mode") # Fallback to Backtest Mode
except Exception as e:
    sys.exit(f"[CRITICAL ERROR] Could not load config.json: {e}")

print(f">>> MODE: {mode}")

strategies = []
def parse_list(s_input): return [x.strip().upper() for x in s_input.split(',') if x.strip()] if s_input else []
def pack_strat(s_cfg):
    return {
        'name': s_cfg['name'], 'universe': s_cfg['universe'], 'custom_list': parse_list(s_cfg['custom_list']),
        'sma_period': s_cfg['sma_period'], 'reentry_sma_period': s_cfg['reentry_period'], 
        'rebalance_days': 21, 'top_n_stocks': 10, 'top_n_3x': 5, 
        'confirmation_days': s_cfg['confirm_days'], 'guard_mode': 'GRADUATED', 
        'allow_3x_in_stock_picks': False, 'momentum_long': s_cfg['mom_long'], 'momentum_short': s_cfg['mom_short']
    }

if "S1" in cfg: strategies.append(pack_strat(cfg["S1"]))
if "S2" in cfg: strategies.append(pack_strat(cfg["S2"]))

print(f">>> CONFIGURATION LOADED: {len(strategies)} Strategies Ready.\n")

# ==========================================
# DOWNLOAD HELPERS (NO PARQUET CACHING)
# ==========================================
def batch_download(tickers, start_date, end_date=None, chunk_size=150):
    all_data_list = []
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    print(f"    -> Downloading {len(tickers)} tickers in batches...")
    for i, chunk in enumerate(chunks):
        try:
            if end_date:
                safe_end_date = end_date + datetime.timedelta(days=1) if isinstance(end_date, datetime.date) else end_date
                batch = yf.download(chunk, start=start_date, end=safe_end_date, group_by='ticker', progress=False, auto_adjust=True, threads=False)
            else:
                batch = yf.download(chunk, start=start_date, group_by='ticker', progress=False, auto_adjust=True, threads=False)

            if not batch.empty:
                if len(chunk) == 1 and not isinstance(batch.columns, pd.MultiIndex):
                    batch.columns = pd.MultiIndex.from_product([chunk, batch.columns])
                
                batch.index = pd.to_datetime(batch.index, utc=True).tz_localize(None).normalize()
                batch = batch.groupby(batch.index).last()
                all_data_list.append(batch)
            time.sleep(0.3)
        except Exception as e:
            pass
    return pd.concat(all_data_list, axis=1) if all_data_list else None

class SwitchbladeStrategy(bt.Strategy):
    params = (
        ('name', 'Strategy'), ('universe', 'STANDARD'), ('start_date', None), ('custom_list', []),
        ('momentum_long', 126), ('momentum_short', 21), ('rebalance_days', 21), ('top_n_stocks', 10),
        ('top_n_3x', 5), ('cash_buffer', 0.05), ('sma_period', 200), ('reentry_sma_period', 100),
        ('confirmation_days', 8), ('guard_mode', 'GRADUATED'), ('allow_3x_in_stock_picks', False),
        ('tickers_sp500', []), ('tickers_xlg', []),
    )
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        if self.params.start_date and dt < self.params.start_date: return
        print(f"[{self.params.name}] {dt.isoformat()}: {txt}")

    def __init__(self):
        self.iwm = self.getdatabyname("IWM"); self.qqq = self.getdatabyname("QQQ")
        self.spy = self.getdatabyname("SPY"); self.xlg = self.getdatabyname("XLG")
        self.gld = self.getdatabyname("GLD"); self.tlt = self.getdatabyname("TLT")
        self.ief = self.getdatabyname("IEF"); self.bil = self.getdatabyname("BIL")
        self.tqqq = self.getdatabyname("TQQQ"); self.spxl = self.getdatabyname("SPXL")
        self.smas = {}
        self.guards = {'IWM': self.iwm, 'QQQ': self.qqq, 'SPY': self.spy, 'XLG': self.xlg, 'GLD': self.gld, 'TLT': self.tlt, 'IEF': self.ief, 'BIL': self.bil}
        for name, data in self.guards.items():
            self.smas[name] = {'exit': bt.indicators.SimpleMovingAverage(data.close, period=self.params.sma_period), 'entry': bt.indicators.SimpleMovingAverage(data.close, period=self.params.reentry_sma_period)}

        self.univ_map = {'ALL_STOCKS': [], 'XLG_TOP5': [], 'GOLD': [self.gld], 'LONG_BOND': [self.tlt], 'MED_BOND': [self.ief], 'CASH': [self.bil], 'TQQQ': [self.tqqq], 'SPXL': [self.spxl]}
        self.base_exclude = [self.iwm, self.qqq, self.spy, self.xlg, self.gld, self.tlt, self.ief, self.bil]
        self.nitro_list = parse_list(GLOBAL_NITRO_ETFS)

        target_universe = []
        if self.params.universe == 'TQQQ_ONLY': target_universe = [self.tqqq]
        elif self.params.universe == 'CUSTOM':
            for d in self.datas:
                if d._name in self.params.custom_list: target_universe.append(d)
        else:
            for d in self.datas:
                if any(d is asset for asset in self.base_exclude): continue
                if d._name in self.nitro_list:
                    if self.params.allow_3x_in_stock_picks: target_universe.append(d)
                else: target_universe.append(d)
        self.univ_map['ALL_STOCKS'] = target_universe
        for d in self.datas:
             if d._name in self.params.tickers_sp500[:60]: self.univ_map['XLG_TOP5'].append(d)

        self.timer = 0; self.current_mode = "INIT"; self.graduated_state = "FIRMLY_BEARISH"
        self.pending_mode = None; self.pending_state = None; self.confirm_counter = 0
        self.total_switches = 0; self.total_orders = 0; self.last_year = None; self.val_history = []

    def get_sma(self, asset_name):
        is_defensive = self.current_mode in ["GOLD", "LONG_BOND", "MED_BOND", "CASH", "INIT"]
        return self.smas[asset_name]['entry'][0] if is_defensive else self.smas[asset_name]['exit'][0]

    def get_rankings(self, universe, top_n):
        ranks = []
        curr_dt = self.datetime.date(0)
        for d in universe:
            if self.params.universe in ['STANDARD', 'SP500', 'NDX', 'SP1000']:
                curr_price = d.close[0] if len(d) > 0 else 0
                if not pit_manager.is_constituent(d._name, self.params.universe, curr_dt, current_price=curr_price): continue
            if len(d) > self.params.momentum_long:
                try:
                    if d.datetime.date(0) == curr_dt and d.close[0] > 0:
                        try:
                            vol_array = d.volume.get(size=20)
                            if len(vol_array) < 20 or np.mean(vol_array) < 100000: continue
                        except Exception: continue
                        start_long = d.close[-self.params.momentum_long]
                        start_short = d.close[-self.params.momentum_short]
                        end = d.close[0]
                        if start_long > 0 and start_short > 0:
                            blended_score = (((end - start_long) / start_long) * 0.70) + (((end - start_short) / start_short) * 0.30)
                            ranks.append((d._name, blended_score))
                except IndexError: continue
        ranks.sort(key=lambda x: x[1], reverse=True)
        return ranks[:top_n]

    def print_holdings(self, mode, context="Rebalance"):
        action = "SWITCH" if context == "Switch" else "REBALANCE"
        assets = [x[0] for x in self.get_rankings(self.univ_map[mode], self.params.top_n_stocks)] if mode == "ALL_STOCKS" else [x[0] for x in self.get_rankings(self.univ_map[mode], 5)] if mode == "XLG_TOP5" else [mode] if mode in ["TQQQ", "SPXL", "GOLD", "LONG_BOND", "MED_BOND"] else ['BIL']
        if mode == "GOLD": assets = ['GLD']
        elif mode == "LONG_BOND": assets = ['TLT']
        elif mode == "MED_BOND": assets = ['IEF']
        self.log(f"{self.graduated_state}, {mode} -> {action}: {assets}")

    def prenext(self): self.next()

    def next(self):
        dt = self.datetime.date(0)
        if self.params.start_date and dt < self.params.start_date: return
        self.val_history.append({'Date': dt, 'Value': self.broker.getvalue(), 'Mode': self.current_mode})
        if self.last_year != dt.year: self.last_year = dt.year; print(f"    -> [{self.params.name}] Processing Year: {dt.year}...")
        
        bull_iwm = self.iwm.close[0] > self.get_sma('IWM')
        bull_qqq = self.qqq.close[0] > self.get_sma('QQQ')
        bull_spy = self.spy.close[0] > self.get_sma('SPY')
        bull_xlg = self.xlg.close[0] > self.get_sma('XLG')
        any_bull = bull_iwm or bull_qqq or bull_spy or bull_xlg
        all_bull = bull_iwm and bull_qqq and bull_spy and bull_xlg
        raw_mode = "CASH"; potential_state = self.graduated_state

        if self.params.guard_mode == "NONE": raw_mode = "ALL_STOCKS"; potential_state = "FIRMLY_BULLISH"
        elif self.params.guard_mode == "GRADUATED":
            if self.graduated_state == "FIRMLY_BEARISH" and any_bull: potential_state = "CAUTIOUSLY_OPTIMISTIC"
            elif self.graduated_state == "CAUTIOUSLY_OPTIMISTIC": potential_state = "FIRMLY_BULLISH" if all_bull else ("FIRMLY_BEARISH" if not any_bull else "CAUTIOUSLY_OPTIMISTIC")
            elif self.graduated_state == "FIRMLY_BULLISH" and not all_bull: potential_state = "SLIGHTLY_BEARISH"
            elif self.graduated_state == "SLIGHTLY_BEARISH": potential_state = "FIRMLY_BULLISH" if all_bull else ("FIRMLY_BEARISH" if not any_bull else "SLIGHTLY_BEARISH")
            
            if potential_state in ["FIRMLY_BEARISH", "SLIGHTLY_BEARISH"]:
                tg, tt, te = (self.smas['GLD']['entry'][0], self.smas['TLT']['entry'][0], self.smas['IEF']['entry'][0]) if self.current_mode in ["GOLD", "LONG_BOND", "MED_BOND", "CASH", "INIT"] else (self.smas['GLD']['exit'][0], self.smas['TLT']['exit'][0], self.smas['IEF']['exit'][0])
                if self.gld.close[0] > tg: raw_mode = "GOLD"
                elif self.tlt.close[0] > tt: raw_mode = "LONG_BOND"
                elif self.ief.close[0] > te: raw_mode = "MED_BOND"
            else:
                # FIX: Explicit universe check prevents TQQQ from misrouting to ALL_STOCKS
                if self.params.universe == 'TQQQ_ONLY': raw_mode = "TQQQ"
                elif bull_iwm: raw_mode = "ALL_STOCKS"
                elif bull_qqq: raw_mode = "TQQQ"
                elif bull_spy: raw_mode = "SPXL"
                elif bull_xlg: raw_mode = "XLG_TOP5"
        else:
            risk_on = bull_iwm if self.params.guard_mode == "IWM_ONLY" else all_bull
            if risk_on: 
                if self.params.universe == 'TQQQ_ONLY': raw_mode = "TQQQ"
                else: raw_mode = "ALL_STOCKS"
            elif self.gld.close[0] > self.get_sma('GLD'): raw_mode = "GOLD"
            elif self.tlt.close[0] > self.get_sma('TLT'): raw_mode = "LONG_BOND"
            elif self.ief.close[0] > self.get_sma('IEF'): raw_mode = "MED_BOND"

        force_rebalance = False
        if raw_mode != self.current_mode:
            if raw_mode == self.pending_mode: self.confirm_counter += 1
            else: self.pending_mode = raw_mode; self.pending_state = potential_state; self.confirm_counter = 1
            if self.confirm_counter >= self.params.confirmation_days:
                self.graduated_state = self.pending_state; self.total_switches += 1
                self.current_mode = raw_mode; self.timer = self.params.rebalance_days
                self.confirm_counter = 0; self.pending_mode = None; self.pending_state = None
                force_rebalance = True
                self.print_holdings(raw_mode, context="Switch")
        else:
            if self.params.guard_mode == "GRADUATED" and potential_state != self.graduated_state: self.graduated_state = potential_state
            self.confirm_counter = 0; self.pending_mode = None; self.pending_state = None

        self.timer += 1
        if (self.timer >= self.params.rebalance_days) or force_rebalance:
            if not force_rebalance: self.print_holdings(self.current_mode, context="Monthly Rebalance")
            self.timer = 0
            target_names = [x[0] for x in self.get_rankings(self.univ_map['ALL_STOCKS'], self.params.top_n_stocks)] if self.current_mode == "ALL_STOCKS" else [x[0] for x in self.get_rankings(self.univ_map['XLG_TOP5'], 5)] if self.current_mode == "XLG_TOP5" else ['TQQQ'] if self.current_mode == "TQQQ" else ['SPXL'] if self.current_mode == "SPXL" else ['GLD'] if self.current_mode == "GOLD" else ['TLT'] if self.current_mode == "LONG_BOND" else ['IEF'] if self.current_mode == "MED_BOND" else ['BIL']
            target_assets = [self.getdatabyname(x) for x in target_names]
            for d, pos in self.getpositions().items():
                if pos.size != 0 and d._name not in target_names: self.order_target_percent(d, target=0.0)
            if target_assets:
                weight = (1.0 - self.params.cash_buffer) / len(target_assets)
                for d in target_assets: self.order_target_percent(d, target=weight)


    def stop(self):
        self.log("\n" + "="*50)
        self.log(f"FINAL STATE DUMP: {self.params.name}")
        self.log("="*50)
        curr_dd_str = "N/A"
        try:
            if self.val_history:
                curr_val = self.broker.getvalue()
                peak_val = max(v['Value'] for v in self.val_history)
                curr_dd = ((peak_val - curr_val) / peak_val) * 100 if peak_val > 0 else 0
                curr_dd_str = f"{curr_dd:.2f}%"
        except Exception: pass
        self.log(f"Current Mode    : {self.current_mode}")
        self.log(f"Graduated State : {self.graduated_state}")
        self.log(f"Pending Mode    : {self.pending_mode}")
        self.log(f"Confirm Counter : {self.confirm_counter} / {self.params.confirmation_days}")
        self.log(f"Rebalance Timer : {self.timer} / {self.params.rebalance_days}")
        self.log(f"Current Drawdown: {curr_dd_str}")
        self.log("-" * 50)
        self.log("GUARD DOG STATUS (Final Day):")
        for name, data in self.guards.items():
            try:
                px = data.close[0]; sma_ex = self.smas[name]['exit'][0]; sma_en = self.smas[name]['entry'][0]
                dist_ex = ((px - sma_ex) / sma_ex) * 100 if sma_ex else 0
                dist_en = ((px - sma_en) / sma_en) * 100 if sma_en else 0
                self.log(f"  {name.ljust(4)}: Px: {px:6.2f} | Exit Dist: {dist_ex:>+6.2f}% | Entry Dist: {dist_en:>+6.2f}%")
            except: pass
        self.log("-" * 50)
        active_positions = [d._name for d, pos in self.getpositions().items() if pos.size != 0]
        self.log(f"Final Holdings  : {active_positions}")
        self.log("="*50 + "\n")

def load_and_prep_data_superset():
    guards = ["IWM", "QQQ", "SPY", "XLG", "GLD", "TLT", "IEF", "BIL"]
    all_nitro_etfs = parse_list(GLOBAL_NITRO_ETFS)
    t_sp500 = pit_manager.get_all_tickers()
    all_tickers = list(set(t_sp500 + guards + all_nitro_etfs))[:max_stocks_per_univ]
    
    print(f"   [System] Master Universe Size: {len(all_tickers)} Tickers")
    print("   -> Starting FRESH BATCH DOWNLOAD...")

    if mode == "Backtest Mode":
        target_start = datetime.datetime.strptime(backtest_start_date, "%Y-%m-%d").date()
        required_start = target_start - datetime.timedelta(days=365)
    else:
        # Execution mode only needs enough bars for the longest SMA + MOM
        required_start = datetime.date.today() - datetime.timedelta(days=400)

    data = batch_download(all_tickers, required_start, end_date=None)

    if data is not None:
        data.index = pd.to_datetime(data.index, utc=True).tz_localize(None).normalize()
        data = data.groupby(data.index).last().loc[:, ~data.columns.duplicated()].sort_index()
        
        # FIX: Fill missing data gaps and drop completely dead tickers to prevent NaN propagation
        data = data.ffill().bfill()
        if 'SPY' in data.columns.levels[0]:
            data = data.dropna(subset=[('SPY', 'Close')])

    t_xlg = t_sp500[:60] if t_sp500 else []
    return data, t_sp500, [], [], t_xlg


def worker_run_strategy(config, data_master, t_sp500, t_sp1000, t_ndx, t_xlg, backtest_start_date):
    try:
        print(f"   [Runner] Starting: {config['name']}")
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        target_start_pd = pd.Timestamp(backtest_start_date).normalize()
        data_start_pd = target_start_pd - pd.Timedelta(days=365)
        added_tickers = set()
        guards = ["IWM", "QQQ", "SPY", "XLG", "GLD", "TLT", "IEF", "BIL"]
        required_targets = ["TQQQ", "SPXL"]
        available_tickers = data_master.columns.get_level_values(0).unique()

        for g in guards + required_targets:
            if g in available_tickers:
                try:
                    df = data_master[g].dropna(subset=['Close'])
                    if not df.empty and (df['Close'] > 0).all():
                        cerebro.adddata(bt.feeds.PandasData(dataname=df, name=g, fromdate=data_start_pd.to_pydatetime()))
                        added_tickers.add(g)
                except KeyError: pass

        candidates = set()
        if config['universe'] == 'TQQQ_ONLY': candidates.add("TQQQ")
        elif config['universe'] == 'CUSTOM': candidates.update(config['custom_list'])
        else:
            valid_lev = parse_list(GLOBAL_NITRO_ETFS) if config['allow_3x_in_stock_picks'] else []
            candidates = set(pit_manager.get_all_tickers() + valid_lev)

        min_required_bars = config['momentum_long'] + 5
        added_count = 0
        for t in candidates:
            if t in added_tickers or t in guards: continue
            if t in available_tickers:
                try:
                    df = data_master[t]
                    if 'Close' in df.columns:
                        df = df.dropna(subset=['Close'])
                        if not df.empty and len(df) >= min_required_bars:
                            cerebro.adddata(bt.feeds.PandasData(dataname=df, name=t, fromdate=data_start_pd.to_pydatetime()))
                            added_count += 1
                except KeyError: pass
        
        if config['universe'] == 'TQQQ_ONLY': print(f"   [Runner] {config['name']}: Target ETF (TQQQ) pre-loaded successfully.")
        else: print(f"   [Runner] {config['name']}: {added_count} candidate stocks loaded.")

        cerebro.addstrategy(SwitchbladeStrategy, **config, tickers_sp500=t_sp500, tickers_xlg=t_xlg, start_date=target_start_pd.date())
        results = cerebro.run()
        strat = results[0]
        if not hasattr(strat, 'val_history') or not strat.val_history: return None
        history_df = pd.DataFrame(strat.val_history)
        history_df['Date'] = pd.to_datetime(history_df['Date'])
        history_df.set_index('Date', inplace=True)
        start_val = 100000; end_val = history_df['Value'].iloc[-1]
        ret = ((end_val - start_val) / start_val) * 100
        dd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
        curr_dd = strat.analyzers.drawdown.get_analysis().get('drawdown', 0.0)
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
        
        years = (history_df.index[-1] - history_df.index[0]).days / 365.25
        cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 else 0
        daily_returns = history_df['Value'].pct_change().dropna()
        downside_returns = daily_returns[daily_returns < 0]
        downside_dev = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (cagr / downside_dev) if downside_dev > 0 else 0
        calmar = cagr / (dd / 100) if dd > 0 else 0

        print(f"   [Runner] Finished: {config['name']} (Total Ret: {ret:,.2f}%)")
        return {
            'Strategy': config['name'], 'Return %': ret, 'MaxDD %': dd, 'CurrDD %': curr_dd,
            'Sharpe': sharpe, 'Sortino': sortino, 'Calmar': calmar,
            'Trades': strat.total_orders, 'Switches': strat.total_switches, 'History': history_df
        }
    except Exception as e: return None

def run_batch_backtest():
    global use_multiprocessing
    data_master, t_sp500, t_sp1000, t_ndx, t_xlg = load_and_prep_data_superset()
    if data_master is None: return
    summary_stats = []
    start_date_pd = pd.Timestamp(datetime.datetime.strptime(backtest_start_date, "%Y-%m-%d")).normalize()

    worker = partial(worker_run_strategy, data_master=data_master, t_sp500=t_sp500, t_sp1000=t_sp1000, t_ndx=t_ndx, t_xlg=t_xlg, backtest_start_date=backtest_start_date)
    if use_multiprocessing and len(strategies) > 1:
        print(f"\n   --- PARALLEL EXECUTION STARTED ({multiprocessing.cpu_count()} CORES) ---")
        try:
            ctx = multiprocessing.get_context('fork')
            with ctx.Pool(processes=len(strategies)) as pool: results = pool.map(worker, strategies)
            summary_stats = [r for r in results if r]
        except Exception:
            for config in strategies:
                res = worker(config)
                if res: summary_stats.append(res)
    else:
        for config in strategies:
            res = worker(config)
            if res: summary_stats.append(res)

    print("\n   -> Calculating Benchmarks...")
    available_tickers = data_master.columns.get_level_values(0).unique()
    for b in ['SPY', 'QQQ', 'TQQQ']:
        if b in available_tickers:
            try:
                df = data_master[b]
                if 'Close' in df.columns:
                    df = df['Close'].dropna()
                    df = df[df.index >= start_date_pd]
                    if df.empty or df.iloc[0] == 0: continue
                    s_norm = (df / df.iloc[0]) * 100000
                    ret = ((df.iloc[-1] - df.iloc[0]) / df.iloc[0]) * 100
                    dd_series = ((s_norm - s_norm.cummax()) / s_norm.cummax()) * 100
                    dd = abs(dd_series.min())
                    curr_dd = abs(dd_series.iloc[-1])
                    years = (df.index[-1] - df.index[0]).days / 365.25
                    cagr = (df.iloc[-1] / df.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
                    daily_returns = df.pct_change().dropna()
                    downside_returns = daily_returns[daily_returns < 0]
                    downside_dev = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252) if len(downside_returns) > 0 else 0
                    sortino = (cagr / downside_dev) if downside_dev > 0 else 0
                    calmar = cagr / (dd / 100) if dd > 0 else 0
                    summary_stats.append({
                        'Strategy': f"BENCHMARK: {b}", 'Return %': ret, 'MaxDD %': dd, 'CurrDD %': curr_dd,
                        'Sharpe': 0.0, 'Sortino': sortino, 'Calmar': calmar,
                        'Trades': 0, 'Switches': 0, 'History': pd.DataFrame({'Value': s_norm})
                    })
            except Exception as e: pass

    print("\n" + "="*50 + "\n   FINAL LEAGUE TABLE\n" + "="*50)
    if summary_stats:
        df = pd.DataFrame(summary_stats).drop(columns=['History']).sort_values(by='Return %', ascending=False)
        pd.options.display.float_format = '{:,.2f}'.format
        print(df.to_string(index=False))
        
        # Headless Plotting - Advanced 2 Panel Chart
        fig, (ax, ax_dd) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 1.5]}, sharex=True)
        palette = ['#648FFF', '#DC267F', '#FE6100', '#785EF0', '#FFB000']
        bg_colors = {'GOLD': '#FFFACD', 'LONG_BOND': '#E0F7FA', 'MED_BOND': '#E0F7FA', 'CASH': '#F5F5F5', 'ALL_STOCKS': '#FFFFFF', 'TQQQ': '#FFFFFF', 'SPXL': '#FFFFFF', 'XLG_TOP5': '#FFFFFF', 'INIT': '#FFFFFF'}

        best_strat = next((s for s in summary_stats if "BENCHMARK" not in s['Strategy']), None)
        all_starts = [s['History'].index[0] for s in summary_stats if not s['History'].empty]
        anchor_date = max(all_starts) if all_starts else start_date_pd

        if best_strat and 'Mode' in best_strat['History'].columns:
            hist = best_strat['History']
            hist = hist[hist.index >= anchor_date]
            hist['group'] = (hist['Mode'] != hist['Mode'].shift()).cumsum()
            for g, data in hist.groupby('group'):
                ax.axvspan(data.index[0], data.index[-1], color=bg_colors.get(data['Mode'].iloc[0], '#FFFFFF'), alpha=0.5, lw=0)

        color_idx = 0
        for res in summary_stats:
            df = res['History']
            if df.empty: continue
            df_slice = df[df.index >= anchor_date]
            if df_slice.empty or df_slice['Value'].iloc[0] == 0: continue
            norm = (df_slice['Value'] / df_slice['Value'].iloc[0]) * 100
            lbl = res['Strategy']; is_bench = "BENCHMARK" in lbl
            clr = '#000000' if "SPY" in lbl else ('#555555' if "QQQ" in lbl else '#333333') if is_bench else palette[color_idx % len(palette)]
            lw = 1.5 if is_bench else 2.5; ls = ':' if is_bench else '-'; alpha = 0.6 if is_bench else 1.0
            if not is_bench: color_idx += 1

            ax.plot(norm.index, norm.values, label=lbl, color=clr, linewidth=lw, alpha=alpha, linestyle=ls)
            dd_series = (df_slice['Value'] / df_slice['Value'].cummax() - 1) * 100
            ax_dd.plot(dd_series.index, dd_series.values, color=clr, linewidth=lw*0.8, alpha=alpha, linestyle=ls)
            ax_dd.fill_between(dd_series.index, dd_series.values, 0, color=clr, alpha=0.1 if not is_bench else 0.05)

        ax.set_title(f"Performance (Rebased to 100 at {anchor_date.date()})"); ax.set_yscale('log'); ax.grid(True, which="both", ls="-", alpha=0.15); ax.legend(loc='upper left')
        ax_dd.set_title("Drawdown Profile", fontsize=10); ax_dd.set_ylabel("Drawdown %"); ax_dd.grid(True, which="both", ls="-", alpha=0.15)
        plt.tight_layout()
        
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', facecolor='white', bbox_inches='tight')
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
        
        if hasattr(sys.stdout, 'inject_html'):
            sys.stdout.inject_html(f'<br><img src="data:image/png;base64,{img_base64}" style="max-width:100%; border: 2px solid #333;"><br>')

def run_execution():
    def json_serial(obj):
        if isinstance(obj, (datetime.date, datetime.datetime)): return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    data_master, _, _, _, _ = load_and_prep_data_superset()
    if data_master is None: return

    available_tickers = data_master.columns.get_level_values(0).unique()
    all_dates = data_master.index.unique().sort_values()

    def get_sma_series(ticker, period):
        if ticker in available_tickers:
            try:
                if 'Close' in data_master[ticker].columns: return data_master[ticker]['Close'].rolling(window=period).mean()
            except KeyError: pass
        return None

    execution_reports = []
    for i, config in enumerate(strategies):
        strat_id = i + 1; s_name = config['name']
        univ_setting = str(config.get('universe', 'STANDARD')).strip().upper()
        custom_tickers = config.get('custom_list', [])
        mom_long = config.get('momentum_long', 126); mom_short = config.get('momentum_short', 21)
        
        state_file = os.path.join(DATA_DIR, f"portfolio_state_S{strat_id}.json")
        state = {"current_mode": "INIT", "graduated_state": "FIRMLY_BEARISH", "pending_mode": None, "pending_state": None, "confirm_counter": 0, "timer": 0, "holdings": {}}
        
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f: state = json.load(f)
            except: pass

        last_run_pd = pd.to_datetime(state.get('last_run_date', "1900-01-01")).normalize()
        missed_days = all_dates[all_dates > last_run_pd]
        
        sma_entry = config['reentry_sma_period']; sma_exit = config['sma_period']
        guards = ['IWM', 'QQQ', 'SPY', 'XLG', 'GLD', 'TLT', 'IEF']
        sma_series, price_series = {}, {}
        for g in guards:
            sma_series[f"{g}_exit"] = get_sma_series(g, sma_exit)
            sma_series[f"{g}_entry"] = get_sma_series(g, sma_entry)
            if g in available_tickers: price_series[g] = data_master[g]['Close']

        dates_to_process = missed_days if len(missed_days) > 0 else [all_dates[-1]]
        processing_today_only = (len(missed_days) == 0)
        force_rebalance_today = False

        for current_date in dates_to_process:
            date_str = current_date.strftime('%Y-%m-%d')
            is_defensive = state['current_mode'] in ["GOLD", "LONG_BOND", "MED_BOND", "CASH", "INIT"]
            sma_type = "entry" if is_defensive else "exit"

            def check_bull(ticker):
                try: return price_series[ticker].loc[current_date] > sma_series[f"{ticker}_{sma_type}"].loc[current_date]
                except: return False

            any_bull = any([check_bull(t) for t in ['IWM','QQQ','SPY','XLG']])
            all_bull = all([check_bull(t) for t in ['IWM','QQQ','SPY','XLG']])
            
            pot_state = state.get('graduated_state', "FIRMLY_BEARISH")
            if config['guard_mode'] == "GRADUATED":
                if pot_state == "FIRMLY_BEARISH" and any_bull: pot_state = "CAUTIOUSLY_OPTIMISTIC"
                elif pot_state == "CAUTIOUSLY_OPTIMISTIC": pot_state = "FIRMLY_BULLISH" if all_bull else ("FIRMLY_BEARISH" if not any_bull else "CAUTIOUSLY_OPTIMISTIC")
                elif pot_state == "FIRMLY_BULLISH" and not all_bull: pot_state = "SLIGHTLY_BEARISH"
                elif pot_state == "SLIGHTLY_BEARISH": pot_state = "FIRMLY_BULLISH" if all_bull else ("FIRMLY_BEARISH" if not any_bull else "SLIGHTLY_BEARISH")

            raw_mode = "CASH"
            if pot_state in ["FIRMLY_BEARISH", "SLIGHTLY_BEARISH"]:
                if check_bull('GLD'): raw_mode = "GOLD"
                elif check_bull('TLT'): raw_mode = "LONG_BOND"
                elif check_bull('IEF'): raw_mode = "MED_BOND"
            else:
                if config['universe'] == 'TQQQ_ONLY': raw_mode = "TQQQ"
                elif check_bull('IWM'): raw_mode = "ALL_STOCKS"
                elif check_bull('QQQ'): raw_mode = "TQQQ"
                elif check_bull('SPY'): raw_mode = "SPXL"
                elif check_bull('XLG'): raw_mode = "XLG_TOP5"

            if not processing_today_only:
                if raw_mode != state['current_mode']:
                    if raw_mode == state.get('pending_mode'): state['confirm_counter'] += 1
                    else: state['pending_mode'] = raw_mode; state['pending_state'] = pot_state; state['confirm_counter'] = 1
                    if state['confirm_counter'] >= config['confirmation_days']:
                        state['graduated_state'] = state['pending_state']; state['current_mode'] = raw_mode
                        state['timer'] = config['rebalance_days']; state['confirm_counter'] = 0
                else:
                    state['confirm_counter'] = 0
                    if config['guard_mode'] == "GRADUATED" and pot_state != state['graduated_state']:
                        state['graduated_state'] = pot_state
                
                state['timer'] += 1
                if state['timer'] >= config['rebalance_days']:
                    state['timer'] = 0
                    if current_date == dates_to_process[-1]: force_rebalance_today = True

            state['last_run_date'] = date_str

        # --- REBALANCE ENGINE ---
        force_rebalance = force_rebalance_today
        curr_holdings = list(state.get('holdings', {}).keys())
        curr_mode = state['current_mode']

        if univ_setting == 'TQQQ_ONLY':
            if curr_mode in ['ALL_STOCKS', 'XLG_TOP5']: state['current_mode'] = 'TQQQ'; force_rebalance = True
            if curr_mode == 'TQQQ' and (not curr_holdings or any(x != 'TQQQ' for x in curr_holdings)): force_rebalance = True
        elif univ_setting == 'STANDARD' and curr_mode == 'ALL_STOCKS':
            if not curr_holdings or curr_holdings == ['TQQQ']: force_rebalance = True

        if force_rebalance:
            target_assets = []
            if state['current_mode'] == "ALL_STOCKS":
                univ = pit_manager.get_all_tickers()
                ranked = []; latest_dt = all_dates[-1]
                for t in univ:
                    if t in available_tickers:
                        try:
                            df_t = data_master[t]
                            if 'Close' in df_t.columns:
                                closes = df_t['Close'].dropna()
                                curr_price = closes.loc[latest_dt] if latest_dt in closes.index else 0
                                
                                # Volume check
                                if 'Volume' in df_t.columns and latest_dt in df_t['Volume'].index:
                                    idx_vol = df_t['Volume'].index.get_loc(latest_dt)
                                    if idx_vol >= 20:
                                        if df_t['Volume'].iloc[idx_vol-20:idx_vol].mean() < 100000:
                                            continue 
                                            
                                if univ_setting == 'STANDARD' and not pit_manager.is_constituent(t, 'STANDARD', latest_dt.date(), current_price=curr_price): continue
                                
                                if len(closes) > mom_long and latest_dt in closes.index:
                                    idx = closes.index.get_loc(latest_dt)
                                    if idx >= mom_long:
                                        start_long = closes.iloc[idx - mom_long]; start_short = closes.iloc[idx - mom_short] if idx >= mom_short else start_long; end = closes.iloc[idx]
                                        if start_long > 0 and start_short > 0:
                                            blended_roc = (((end - start_long) / start_long) * 0.70) + (((end - start_short) / start_short) * 0.30)
                                            ranked.append((t, blended_roc))
                        except Exception: pass
                ranked.sort(key=lambda x: x[1], reverse=True)
                target_assets = [x[0] for x in ranked[:config['top_n_stocks']]]
            elif state['current_mode'] == "TQQQ": target_assets = ['TQQQ']
            elif state['current_mode'] == "SPXL": target_assets = ['SPXL']
            elif state['current_mode'] == "GOLD": target_assets = ['GLD']
            elif state['current_mode'] == "LONG_BOND": target_assets = ['TLT']
            elif state['current_mode'] == "MED_BOND": target_assets = ['IEF']
            else: target_assets = ['BIL']

            state['timer'] = 0
            state['holdings'] = {t: f"{100/len(target_assets):.2f}%" for t in target_assets} if target_assets else {}

        with open(state_file, 'w') as f: json.dump(state, f, indent=4, default=json_serial)
        execution_reports.append({'name': s_name, 'state': state})

    print("\n================ FINAL SYSTEM STATE ================")
    for rep in execution_reports:
        print(f"[{rep['name']}] Mode: {rep['state']['current_mode']} | State: {rep['state']['graduated_state']}")
        print(f"   -> Holdings: {list(rep['state'].get('holdings', {}).keys())}")

# ==========================================
# CUSTOM HTML LOGGER
# ==========================================
class HTMLConsoleLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.filepath = filepath
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("<html><head><style>body { background-color: #121212; color: #00FF00; font-family: 'Courier New', monospace; padding: 15px; white-space: pre-wrap; font-size: 14px; }</style></head><body>\n")
            f.write(f"<h3>SWITCHBLADE RUN: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>\n")

    def write(self, message):
        self.terminal.write(message)
        safe_msg = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        with open(self.filepath, "a", encoding="utf-8") as f: f.write(safe_msg)
        self.flush()

    def inject_html(self, html_content):
        with open(self.filepath, "a", encoding="utf-8") as f: f.write(f"\n{html_content}\n")

    def flush(self): self.terminal.flush()
    def close(self):
        with open(self.filepath, "a", encoding="utf-8") as f: f.write("\n</body></html>")

if __name__ == "__main__":
    os.makedirs("./docs", exist_ok=True)
    log_file = "./docs/latest_run.html"
    
    logger = HTMLConsoleLogger(log_file)
    sys.stdout = logger
    sys.stderr = logger  # Captures all yfinance delisted/404 errors so they appear in HTML

    try:
        if mode == "Backtest Mode": run_batch_backtest()
        else: run_execution()
    finally:
        logger.close()
        sys.stdout = logger.terminal
        sys.stderr = sys.__stderr__
