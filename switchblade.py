# =========================================================================
# SWITCHBLADE v47.78 - HEADLESS GITHUB EDITION (PIT + PARQUET)
# =========================================================================

import matplotlib
matplotlib.use('Agg')  # Headless backend for Linux servers

import backtrader as bt
import yfinance as yf
import datetime
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import time
import warnings
import gc
import sys
import base64
from functools import partial
import multiprocessing

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# ==========================================
# CONFIGURATION
# ==========================================
mode = "Backtest Mode"  # Set to Backtest or Execution depending on your CI/CD workflow
state_dir = "./state"
data_source = "Force Fresh Download"
cache_filename = "switchblade_data.parquet"
backtest_start_date = "2011-08-12" 
max_stocks_per_univ = 3000
use_multiprocessing = True

GLOBAL_NITRO_ETFS = "SPXL, SPXS, TQQQ, SQQQ, UDOW, SDOW, TNA, TZA, MIDU, EDC, EDZ, YINN, YANG, EURL, INDL, TECL, TECS, SOXL, SOXS, FNGU, FNGD, WEBL, WEBS, FAS, FAZ, ERX, ERY, CURE, LABU, LABD, DRN, DRV, UTSL, DUSL, RETL, UGL, GLL, AGQ, ZSL, UCO, SCO, BOIL, KOLD, TMF, TMV, UST, PST, BITU, ETHU, UVXY"

S1_ENABLE = True
S1_NAME = "Standard (205/20, 5, 102/22)"
S1_UNIVERSE = "STANDARD"
S1_CUSTOM_LIST = ""
S1_SMA = 254
S1_REENTRY = 19
S1_REBAL_DAYS = 21
S1_TOP_STOCKS = 10
S1_MOM_LONG = 179
S1_MOM_SHORT = 86
S1_CONFIRM_DAYS = 6
S1_GUARD_MODE = "GRADUATED"
S1_ALLOW_NITRO = False

S2_ENABLE = True
S2_NAME = "TQQQ_Only (205/20, 5, 102/22)"
S2_UNIVERSE = "TQQQ_ONLY"
S2_CUSTOM_LIST = GLOBAL_NITRO_ETFS
S2_SMA = 254
S2_REENTRY = 19
S2_REBAL_DAYS = 21
S2_TOP_STOCKS = 1
S2_MOM_LONG = 179
S2_MOM_SHORT = 86
S2_CONFIRM_DAYS = 6
S2_GUARD_MODE = "GRADUATED"
S2_ALLOW_NITRO = False

strategies = []
def parse_list(s_input):
    if not s_input: return []
    return [x.strip().upper() for x in s_input.split(',') if x.strip()]

def pack_strat(enable, name, univ, cust, sma, ren, reb, top, conf, grd, nitro, mom_long, mom_short):
    if enable:
        return {
            'name': name, 'universe': univ, 'custom_list': parse_list(cust),
            'sma_period': sma, 'reentry_sma_period': ren, 'rebalance_days': reb,
            'top_n_stocks': top, 'top_n_3x': 5, 'confirmation_days': conf,
            'guard_mode': grd, 'allow_3x_in_stock_picks': nitro,
            'momentum_long': mom_long, 'momentum_short': mom_short
        }
    return None

s1 = pack_strat(S1_ENABLE, S1_NAME, S1_UNIVERSE, S1_CUSTOM_LIST, S1_SMA, S1_REENTRY, S1_REBAL_DAYS, S1_TOP_STOCKS, S1_CONFIRM_DAYS, S1_GUARD_MODE, S1_ALLOW_NITRO, S1_MOM_LONG, S1_MOM_SHORT)
if s1: strategies.append(s1)
s2 = pack_strat(S2_ENABLE, S2_NAME, S2_UNIVERSE, S2_CUSTOM_LIST, S2_SMA, S2_REENTRY, S2_REBAL_DAYS, S2_TOP_STOCKS, S2_CONFIRM_DAYS, S2_GUARD_MODE, S2_ALLOW_NITRO, S2_MOM_LONG, S2_MOM_SHORT)
if s2: strategies.append(s2)

print(f">>> CONFIGURATION LOADED: {len(strategies)} Strategies Ready.")

# ==========================================
# POINT-IN-TIME (PIT) UNIVERSE ENGINE
# ==========================================
class PITUniverseManager:
    """
    Scrapes Wikipedia historical additions and removals to build point-in-time constituent lists
    for SP500, NDX, and SP1000 (SP400 + SP600), eliminating survivorship bias.
    """
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        self.snapshots = {}
        self.all_tickers = set()
        print("   [PIT Engine] Scraped historical index constituent changes from Wikipedia...")
        self._build_sp500()
        self._build_ndx()
        self._build_sp1000()

    def _clean_ticker(self, symbol):
        if not symbol or pd.isna(symbol): return None
        s = str(symbol).strip().upper().replace('.', '-')
        return s if s and s not in ['CWEN-A', 'NAN', 'NONE'] else None

    def _parse_wiki_changes(self, url, curr_table_idx=0, change_table_idx=1, symbol_cols=['Symbol', 'Ticker']):
        try:
            r = requests.get(url, headers=self.headers, timeout=10)
            tables = pd.read_html(io.StringIO(r.text))
            
            curr_df = tables[curr_table_idx]
            curr_tickers = set()
            for col in symbol_cols:
                if col in curr_df.columns:
                    curr_tickers = set(filter(None, [self._clean_ticker(t) for t in curr_df[col].tolist()]))
                    break

            changes = []
            if len(tables) > change_table_idx:
                change_df = tables[change_table_idx]
                if isinstance(change_df.columns, pd.MultiIndex):
                    change_df.columns = ['_'.join(col).strip() for col in change_df.columns.values]

                date_col = next((c for c in change_df.columns if 'date' in c.lower()), None)
                add_col = next((c for c in change_df.columns if 'added' in c.lower() and ('ticker' in c.lower() or 'symbol' in c.lower())), None)
                rem_col = next((c for c in change_df.columns if 'removed' in c.lower() and ('ticker' in c.lower() or 'symbol' in c.lower())), None)

                if not add_col: add_col = next((c for c in change_df.columns if 'added' in c.lower()), None)
                if not rem_col: rem_col = next((c for c in change_df.columns if 'removed' in c.lower()), None)

                if date_col:
                    for _, row in change_df.iterrows():
                        try:
                            dt = pd.to_datetime(row[date_col]).date()
                            added = self._clean_ticker(row[add_col]) if add_col else None
                            removed = self._clean_ticker(row[rem_col]) if rem_col else None
                            if added or removed:
                                changes.append({'date': dt, 'added': added, 'removed': removed})
                        except Exception: continue

            changes.sort(key=lambda x: x['date'], reverse=True)
            timeline = [(datetime.date.today(), set(curr_tickers))]
            master_tickers = set(curr_tickers)
            curr_state = set(curr_tickers)

            for change in changes:
                dt = change['date']
                added = change['added']
                removed = change['removed']
                if added: master_tickers.add(added)
                if removed: master_tickers.add(removed)
                if added and added in curr_state: curr_state.remove(added)
                if removed: curr_state.add(removed)
                timeline.append((dt, set(curr_state)))

            return timeline, master_tickers
        except Exception as e:
            print(f"      [PIT Warning] Failed parsing {url}: {e}")
            return [(datetime.date.today(), set())], set()

    def _build_sp500(self):
        timeline, tickers = self._parse_wiki_changes('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 0, 1)
        self.snapshots['SP500'] = timeline; self.all_tickers.update(tickers)

    def _build_ndx(self):
        timeline, tickers = self._parse_wiki_changes('https://en.wikipedia.org/wiki/Nasdaq-100', 4, 5)
        self.snapshots['NDX'] = timeline; self.all_tickers.update(tickers)

    def _build_sp1000(self):
        t_mid, tickers_mid = self._parse_wiki_changes('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies', 0, 1)
        t_small, tickers_small = self._parse_wiki_changes('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies', 0, 1)
        self.snapshots['SP1000_MID'] = t_mid; self.snapshots['SP1000_SMALL'] = t_small
        self.all_tickers.update(tickers_mid); self.all_tickers.update(tickers_small)

    def is_constituent(self, symbol, universe, query_date):
        if universe not in self.snapshots and universe != "STANDARD": return True
        keys = ['SP500', 'NDX', 'SP1000_MID', 'SP1000_SMALL'] if universe == "STANDARD" else [universe]
        for k in keys:
            if k not in self.snapshots: continue
            for snap_date, constituents in self.snapshots[k]:
                if query_date >= snap_date:
                    if symbol in constituents: return True
                    break
        return False

    def get_all_tickers(self): return list(self.all_tickers)

pit_manager = PITUniverseManager()

# ==========================================
# PARQUET & DOWNLOAD HELPERS
# ==========================================
def save_parquet(df, path):
    save_df = df.copy()
    if isinstance(save_df.columns, pd.MultiIndex):
        save_df.columns = [f"{col[0]}_{col[1]}" for col in save_df.columns]
    save_df.to_parquet(path, engine='pyarrow')

def load_parquet(path):
    df = pd.read_parquet(path, engine='pyarrow')
    if not isinstance(df.columns, pd.MultiIndex):
        tuples = [(c.split('_', 1)[0], c.split('_', 1)[1]) if '_' in c else (c, '') for c in df.columns]
        df.columns = pd.MultiIndex.from_tuples(tuples)
    return df

def batch_download(tickers, start_date, end_date=None, chunk_size=150):
    all_data_list = []
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    print(f"   -> Downloading {len(tickers)} tickers in batches...")
    for i, chunk in enumerate(chunks):
        try:
            if end_date:
                safe_end_date = end_date + datetime.timedelta(days=1) if isinstance(end_date, datetime.date) else end_date
                batch = yf.download(chunk, start=start_date, end=safe_end_date, group_by='ticker', progress=False, auto_adjust=True, threads=False)
            else:
                batch = yf.download(chunk, start=start_date, group_by='ticker', progress=False, auto_adjust=True, threads=False)

            if not batch.empty:
                # yfinance drops MultiIndex if chunk len == 1. We force it back for Parquet mapping.
                if len(chunk) == 1 and not isinstance(batch.columns, pd.MultiIndex):
                    batch.columns = pd.MultiIndex.from_product([chunk, batch.columns])
                    
                batch.index = pd.to_datetime(batch.index, utc=True).tz_localize(None).normalize()
                batch = batch.groupby(batch.index).last()
                all_data_list.append(batch)
            time.sleep(0.3)
        except Exception as e:
            print(f"      Batch {i+1} Failed: {e}")
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
            self.smas[name] = {
                'exit': bt.indicators.SimpleMovingAverage(data.close, period=self.params.sma_period),
                'entry': bt.indicators.SimpleMovingAverage(data.close, period=self.params.reentry_sma_period)
            }

        self.univ_map = {
            'ALL_STOCKS': [], 'XLG_TOP5': [], 'GOLD': [self.gld], 'LONG_BOND': [self.tlt],
            'MED_BOND': [self.ief], 'CASH': [self.bil], 'TQQQ': [self.tqqq], 'SPXL': [self.spxl]
        }

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
                    else: continue
                else: target_universe.append(d)

        self.univ_map['ALL_STOCKS'] = target_universe

        for d in self.datas:
             if d._name in self.params.tickers_sp500[:60]:
                 self.univ_map['XLG_TOP5'].append(d)

        self.timer = 0; self.current_mode = "INIT"; self.graduated_state = "FIRMLY_BEARISH"
        self.pending_mode = None; self.pending_state = None; self.confirm_counter = 0
        self.total_switches = 0; self.total_orders = 0; self.last_year = None; self.val_history = []

    def start(self):
        print(f"[{self.params.name}] Indicators Calculated. Warming up...")

    def notify_order(self, order):
        if self.params.start_date and self.datetime.date(0) < self.params.start_date: return
        if order.status in [order.Completed]: self.total_orders += 1

    def get_sma(self, asset_name):
        is_defensive = self.current_mode in ["GOLD", "LONG_BOND", "MED_BOND", "CASH", "INIT"]
        if is_defensive: return self.smas[asset_name]['entry'][0]
        else: return self.smas[asset_name]['exit'][0]

    def get_rankings(self, universe, top_n):
        ranks = []
        curr_dt = self.datetime.date(0)
        for d in universe:
            if self.params.universe in ['STANDARD', 'SP500', 'NDX', 'SP1000']:
                if not pit_manager.is_constituent(d._name, self.params.universe, curr_dt):
                    continue

            if len(d) > self.params.momentum_long:
                try:
                    if d.datetime.date(0) == curr_dt and d.close[0] > 0:
                        start_long = d.close[-self.params.momentum_long]
                        start_short = d.close[-self.params.momentum_short]
                        end = d.close[0]
                        if start_long > 0 and start_short > 0:
                            roc_long = (end - start_long) / start_long
                            roc_short = (end - start_short) / start_short
                            blended_score = (roc_long * 0.70) + (roc_short * 0.30)
                            ranks.append((d._name, blended_score))
                except IndexError: continue
        ranks.sort(key=lambda x: x[1], reverse=True)
        return ranks[:top_n]

    def print_holdings(self, mode, context="Rebalance"):
        action = "HOLD"
        if context == "Switch": action = "SWITCH"
        elif context == "Monthly Rebalance": action = "REBALANCE"
        assets = []
        if mode == "ALL_STOCKS": assets = [x[0] for x in self.get_rankings(self.univ_map[mode], self.params.top_n_stocks)]
        elif mode == "XLG_TOP5": assets = [x[0] for x in self.get_rankings(self.univ_map[mode], 5)]
        elif mode == "TQQQ": assets = ['TQQQ']
        elif mode == "SPXL": assets = ['SPXL']
        elif mode == "GOLD": assets = ['GLD']
        elif mode == "LONG_BOND": assets = ['TLT']
        elif mode == "MED_BOND": assets = ['IEF']
        else: assets = ['BIL']
        self.log(f"{self.graduated_state}, {mode} -> {action}: {assets}")

    def prenext(self):
        self.next()

    def next(self):
        dt = self.datetime.date(0)
        if self.params.start_date and dt < self.params.start_date: return

        self.val_history.append({'Date': dt, 'Value': self.broker.getvalue(), 'Mode': self.current_mode})

        if self.last_year != dt.year:
            self.last_year = dt.year
            print(f"   -> [{self.params.name}] Processing Year: {dt.year}...")

        bull_iwm = self.iwm.close[0] > self.get_sma('IWM')
        bull_qqq = self.qqq.close[0] > self.get_sma('QQQ')
        bull_spy = self.spy.close[0] > self.get_sma('SPY')
        bull_xlg = self.xlg.close[0] > self.get_sma('XLG')

        any_bull = bull_iwm or bull_qqq or bull_spy or bull_xlg
        all_bull = bull_iwm and bull_qqq and bull_spy and bull_xlg

        raw_mode = "CASH"; potential_state = self.graduated_state

        if self.params.guard_mode == "NONE":
            raw_mode = "ALL_STOCKS"; potential_state = "FIRMLY_BULLISH"
        elif self.params.guard_mode == "GRADUATED":
            if self.graduated_state == "FIRMLY_BEARISH":
                if any_bull: potential_state = "CAUTIOUSLY_OPTIMISTIC"
            elif self.graduated_state == "CAUTIOUSLY_OPTIMISTIC":
                if all_bull: potential_state = "FIRMLY_BULLISH"
                elif not any_bull: potential_state = "FIRMLY_BEARISH"
            elif self.graduated_state == "FIRMLY_BULLISH":
                if not all_bull: potential_state = "SLIGHTLY_BEARISH"
            elif self.graduated_state == "SLIGHTLY_BEARISH":
                if all_bull: potential_state = "FIRMLY_BULLISH"
                elif not any_bull: potential_state = "FIRMLY_BEARISH"

            if potential_state in ["FIRMLY_BEARISH", "SLIGHTLY_BEARISH"]:
                tg, tt, te = (self.smas['GLD']['entry'][0], self.smas['TLT']['entry'][0], self.smas['IEF']['entry'][0]) if self.current_mode in ["GOLD", "LONG_BOND", "MED_BOND", "CASH", "INIT"] else (self.smas['GLD']['exit'][0], self.smas['TLT']['exit'][0], self.smas['IEF']['exit'][0])
                if self.gld.close[0] > tg: raw_mode = "GOLD"
                elif self.tlt.close[0] > tt: raw_mode = "LONG_BOND"
                elif self.ief.close[0] > te: raw_mode = "MED_BOND"
                else: raw_mode = "CASH"
            else:
                if bull_iwm: raw_mode = "ALL_STOCKS"
                elif bull_qqq: raw_mode = "TQQQ"
                elif bull_spy: raw_mode = "SPXL"
                elif bull_xlg: raw_mode = "XLG_TOP5"
                else: raw_mode = "CASH"
        else:
            risk_on = bull_iwm if self.params.guard_mode == "IWM_ONLY" else (all_bull if self.current_mode != "ALL_STOCKS" else all_bull)
            if risk_on: raw_mode = "ALL_STOCKS"
            elif self.gld.close[0] > self.get_sma('GLD'): raw_mode = "GOLD"
            elif self.tlt.close[0] > self.get_sma('TLT'): raw_mode = "LONG_BOND"
            elif self.ief.close[0] > self.get_sma('IEF'): raw_mode = "MED_BOND"
            else: raw_mode = "CASH"

        force_rebalance = False
        if raw_mode != self.current_mode:
            if raw_mode == self.pending_mode: self.confirm_counter += 1
            else:
                self.pending_mode = raw_mode; self.pending_state = potential_state; self.confirm_counter = 1
            if self.confirm_counter >= self.params.confirmation_days:
                self.graduated_state = self.pending_state; self.total_switches += 1
                self.current_mode = raw_mode; self.timer = self.params.rebalance_days
                self.confirm_counter = 0; self.pending_mode = None; self.pending_state = None
                force_rebalance = True
                self.print_holdings(raw_mode, context="Switch")
        else:
            if self.params.guard_mode == "GRADUATED" and potential_state != self.graduated_state:
                 self.log(f"[STATE SHIFT] {self.graduated_state} -> {potential_state} (Silent Update)")
                 self.graduated_state = potential_state
            self.confirm_counter = 0; self.pending_mode = None; self.pending_state = None

        self.timer += 1
        if (self.timer >= self.params.rebalance_days) or force_rebalance:
            if not force_rebalance: self.print_holdings(self.current_mode, context="Monthly Rebalance")
            self.timer = 0

            target_names = []
            if self.current_mode == "ALL_STOCKS": target_names = [x[0] for x in self.get_rankings(self.univ_map['ALL_STOCKS'], self.params.top_n_stocks)]
            elif self.current_mode == "XLG_TOP5": target_names = [x[0] for x in self.get_rankings(self.univ_map['XLG_TOP5'], 5)]
            elif self.current_mode == "TQQQ": target_names = ['TQQQ']
            elif self.current_mode == "SPXL": target_names = ['SPXL']
            elif self.current_mode == "GOLD": target_names = ['GLD']
            elif self.current_mode == "LONG_BOND": target_names = ['TLT']
            elif self.current_mode == "MED_BOND": target_names = ['IEF']
            else: target_names = ['BIL']

            target_assets = [self.getdatabyname(x) for x in target_names]

            for d, pos in self.getpositions().items():
                if pos.size != 0 and d._name not in target_names: self.order_target_percent(d, target=0.0)

            if not target_assets: return
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
    print("   [Data] Preparing point-in-time universe Parquet data environment...")
    
    active_path = cache_filename
    guards = ["IWM", "QQQ", "SPY", "XLG", "GLD", "TLT", "IEF", "BIL"]
    all_nitro_etfs = parse_list(GLOBAL_NITRO_ETFS)
    
    t_sp500 = pit_manager.get_all_tickers()
    all_tickers = list(set(t_sp500 + guards + all_nitro_etfs))[:max_stocks_per_univ]
    print(f"   [System] Master Universe Size: {len(all_tickers)} Tickers")

    data = None
    force_refresh = (data_source == "Force Fresh Download")

    if not force_refresh and os.path.exists(active_path):
        print(f"2. PARQUET CACHE FOUND: Executing High-Speed Update...")
        try:
            data = load_parquet(active_path)
            data = data.loc[:, ~data.columns.duplicated()]

            if not data.empty:
                data.index = pd.to_datetime(data.index, utc=True).tz_localize(None).normalize()
                anchor_date = data.index[0].date()
                last_date = data.index[-1].date()
                today = pd.Timestamp.now(tz='UTC').date()

                existing_cols = list(data.columns.levels[0])
                valid_existing = []; missing_tickers = []

                for t in all_tickers:
                    if t in existing_cols:
                        valid_days = data[t]['Close'].dropna().shape[0]
                        if valid_days > 5: valid_existing.append(t)
                        else: missing_tickers.append(t)
                    else: missing_tickers.append(t)

                existing_to_update = valid_existing

                if last_date < today:
                    if len(data) > 1: data = data.iloc[:-1]
                    last_safe_date = data.index[-1].date()
                    actual_start = last_safe_date + datetime.timedelta(days=1)

                    print(f"   [Delta Update] Fetching {actual_start} to LATEST for {len(existing_to_update)} existing tickers...")
                    delta_data = batch_download(existing_to_update, actual_start, end_date=None)

                    if delta_data is not None and not delta_data.empty:
                        data = pd.concat([data, delta_data])
                        data = data.groupby(data.index).last()
                        data = data.loc[:, ~data.columns.duplicated()].sort_index()

                if missing_tickers:
                    print(f"\n   [Smart Update] DETECTED {len(missing_tickers)} NEW OR BROKEN TICKERS.")
                    missing_data = batch_download(missing_tickers, anchor_date, end_date=None)
                    if missing_data is not None and not missing_data.empty:
                        data = data.drop(columns=[t for t in missing_tickers if t in data.columns.levels[0]], level=0, errors='ignore')
                        data = pd.concat([data, missing_data], axis=1)
                        data = data.groupby(data.index).last()
                        data = data.loc[:, ~data.columns.duplicated()].sort_index()

                save_parquet(data, active_path)

        except Exception as e:
            print(f"   [Cache Error] {e}. Falling back to fresh download.")
            data = None

    if data is None or force_refresh:
        print("   -> Starting FRESH BATCH DOWNLOAD...")
        target_start = datetime.datetime.strptime(backtest_start_date, "%Y-%m-%d").date()
        required_start = target_start - datetime.timedelta(days=365)
        data = batch_download(all_tickers, required_start, end_date=None)
        if data is not None: save_parquet(data, active_path)

    if data is not None:
        data.index = pd.to_datetime(data.index, utc=True).tz_localize(None).normalize()
        data = data.groupby(data.index).last()
        data = data.loc[:, ~data.columns.duplicated()]
        data = data.sort_index()
        data.ffill(inplace=True)

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

        for g in guards + required_targets:
            if g in data_master.columns.levels[0]:
                df = data_master[g].dropna(subset=['Close'])
                if not df.empty:
                    cerebro.adddata(bt.feeds.PandasData(dataname=df, name=g, fromdate=data_start_pd.to_pydatetime()))
                    added_tickers.add(g)

        candidates = set()
        if config['universe'] == "TQQQ_ONLY": candidates.add("TQQQ")
        elif config['universe'] == "CUSTOM": candidates.update(config['custom_list'])
        else:
            valid_lev = parse_list(GLOBAL_NITRO_ETFS) if config['allow_3x_in_stock_picks'] else []
            candidates = set(pit_manager.get_all_tickers() + valid_lev)

        min_required_bars = config['momentum_long'] + 5
        added_count = 0

        for t in candidates:
            if t in added_tickers or t in guards: continue
            if t in data_master.columns.levels[0]:
                df = data_master[t].dropna(subset=['Close'])
                if not df.empty:
                    if len(df) >= min_required_bars:
                        cerebro.adddata(bt.feeds.PandasData(dataname=df, name=t, fromdate=data_start_pd.to_pydatetime()))
                        added_tickers.add(t); added_count += 1

        if config['universe'] == "TQQQ_ONLY":
            print(f"   [Runner] {config['name']}: Target ETF (TQQQ) pre-loaded successfully.")
        else:
            print(f"   [Runner] {config['name']}: {added_count} candidate stocks loaded.")

        if added_count == 0 and config['universe'] not in ["TQQQ_ONLY"]:
            print(f"   [Runner] {config['name']}: No candidate stocks loaded. Aborting strategy.")
            return None

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
    except Exception as e:
        print(f"   [Error] {config['name']} Failed: {e}")
        return None

def run_batch_backtest():
    global use_multiprocessing
    if 'use_multiprocessing' not in globals(): use_multiprocessing = False

    data_master, t_sp500, t_sp1000, t_ndx, t_xlg = load_and_prep_data_superset()
    if data_master is None: return

    summary_stats = []
    start_date_pd = pd.Timestamp(datetime.datetime.strptime(backtest_start_date, "%Y-%m-%d")).normalize()

    if use_multiprocessing and len(strategies) > 1:
        print(f"\n   --- PARALLEL EXECUTION STARTED ({multiprocessing.cpu_count()} CORES) ---")
        gc.collect()
        worker = partial(worker_run_strategy, data_master=data_master, t_sp500=t_sp500, t_sp1000=t_sp1000, t_ndx=t_ndx, t_xlg=t_xlg, backtest_start_date=backtest_start_date)
        try:
            ctx = multiprocessing.get_context('fork')
            with ctx.Pool(processes=len(strategies)) as pool:
                results = pool.map(worker, strategies)
            summary_stats = [r for r in results if r]
        except Exception as e:
            print(f"[Error] Parallel Failed: {e}. Switching to Serial.")
            use_multiprocessing = False

    if not use_multiprocessing or not summary_stats:
        print(f"\n   --- SERIAL EXECUTION STARTED ---")
        for config in strategies:
            res = worker_run_strategy(config, data_master, t_sp500, t_sp1000, t_ndx, t_xlg, backtest_start_date)
            if res: summary_stats.append(res)

    print("\n   -> Calculating Benchmarks...")
    for b in ['SPY', 'QQQ', 'TQQQ']:
        if b in data_master.columns.levels[0]:
            try:
                df = data_master[b]['Close'].dropna(); df = df[df.index >= start_date_pd]
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
            except Exception as e:
                print(f"   [Warning] Benchmark {b} failed: {e}")

    print("\n" + "="*50 + "\n   FINAL LEAGUE TABLE\n" + "="*50)
    if summary_stats:
        df = pd.DataFrame(summary_stats).drop(columns=['History']).sort_values(by='Return %', ascending=False)
        pd.options.display.float_format = '{:,.2f}'.format
        print(df.to_string(index=False))

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
            sys.stdout.inject_html(f'<br><br><img src="data:image/png;base64,{img_base64}" style="max-width:100%; border: 2px solid #333;"><br>')
        else:
            print("   -> (Graphs omitted in non-HTML console environments)")
    else:
        print("[CRITICAL] No results were generated from any strategy.")

def run_execution():
    print(f"\n" + "="*50 + "\n   >>> EXECUTION MODE: V47.78 (PIT SYNCHRONIZED) <<<\n" + "="*50)

    def json_serial(obj):
        if isinstance(obj, (datetime.date, datetime.datetime)): return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    print("   [Data] Loading market data...")
    data_master, t_sp500, t_sp1000, t_ndx, t_xlg = load_and_prep_data_superset()

    if data_master is None:
        print("   [Error] Data load failed. Aborting.")
        return

    full_live_path = state_dir
    execution_reports = []

    if not os.path.exists(full_live_path):
        os.makedirs(full_live_path, exist_ok=True)

    def get_sma_series(ticker, period):
        return data_master[ticker]['Close'].rolling(window=period).mean() if ticker in data_master.columns.levels[0] else None

    for i, config in enumerate(strategies):
        strat_id = i + 1; s_name = config['name']
        univ_setting = str(config.get('universe', 'STANDARD')).strip().upper()
        custom_tickers = [x.strip().upper() for x in config.get('custom_list', [])]
        mom_long = config.get('momentum_long', 126); mom_short = config.get('momentum_short', 21)
        state_file = f"{full_live_path}/portfolio_state_S{strat_id}.json"

        print(f"\n   >>> STRATEGY {strat_id}: {s_name}")
        state = {"current_mode": "INIT", "graduated_state": "FIRMLY_BEARISH", "pending_mode": None, "pending_state": None, "confirm_counter": 0, "timer": 0, "last_rebal_date": "1900-01-01", "last_run_date": "1900-01-01", "holdings": {}}

        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f: state = json.load(f)
            except: print("       [Warning] State file corrupt. Resetting.")

        last_run_pd = pd.to_datetime(state.get('last_run_date', "1900-01-01")).normalize()
        all_dates = data_master.index.unique().sort_values()
        missed_days = all_dates[all_dates > last_run_pd]

        sma_entry = config['reentry_sma_period']; sma_exit = config['sma_period']
        guards = ['IWM', 'QQQ', 'SPY', 'XLG', 'GLD', 'TLT', 'IEF']
        sma_series, price_series = {}, {}

        for g in guards:
            sma_series[f"{g}_exit"] = get_sma_series(g, sma_exit); sma_series[f"{g}_entry"] = get_sma_series(g, sma_entry)
            if g in data_master.columns.levels[0]: price_series[g] = data_master[g]['Close']

        dates_to_process = missed_days if len(missed_days) > 0 else [all_dates[-1]]
        processing_today_only = (len(missed_days) == 0); force_rebalance_today = False

        for current_date in dates_to_process:
            date_str = current_date.strftime('%Y-%m-%d')
            is_defensive = state['current_mode'] in ["GOLD", "LONG_BOND", "MED_BOND", "CASH", "INIT"]
            sma_type = "entry" if is_defensive else "exit"

            def check_bull(ticker):
                try: return price_series[ticker].loc[current_date] > sma_series[f"{ticker}_{sma_type}"].loc[current_date]
                except: return False

            bull_iwm = check_bull('IWM'); bull_qqq = check_bull('QQQ'); bull_spy = check_bull('SPY'); bull_xlg = check_bull('XLG')
            any_bull = bull_iwm or bull_qqq or bull_spy or bull_xlg; all_bull = bull_iwm and bull_qqq and bull_spy and bull_xlg
            current_grad_state = state.get('graduated_state', "FIRMLY_BEARISH"); potential_state = current_grad_state

            if config['guard_mode'] == "NONE": potential_state = "FIRMLY_BULLISH"
            elif config['guard_mode'] == "GRADUATED":
                if current_grad_state == "FIRMLY_BEARISH" and any_bull: potential_state = "CAUTIOUSLY_OPTIMISTIC"
                elif current_grad_state == "CAUTIOUSLY_OPTIMISTIC": potential_state = "FIRMLY_BULLISH" if all_bull else ("FIRMLY_BEARISH" if not any_bull else "CAUTIOUSLY_OPTIMISTIC")
                elif current_grad_state == "FIRMLY_BULLISH" and not all_bull: potential_state = "SLIGHTLY_BEARISH"
                elif current_grad_state == "SLIGHTLY_BEARISH": potential_state = "FIRMLY_BULLISH" if all_bull else ("FIRMLY_BEARISH" if not any_bull else "SLIGHTLY_BEARISH")
            else: potential_state = "FIRMLY_BULLISH" if (bull_iwm if config['guard_mode']=='IWM_ONLY' else any_bull) else "FIRMLY_BEARISH"

            raw_mode = "CASH"
            if config['guard_mode'] == "NONE": raw_mode = "ALL_STOCKS"
            elif potential_state in ["FIRMLY_BEARISH", "SLIGHTLY_BEARISH"]:
                def check_def(ticker):
                    try: return price_series[ticker].loc[current_date] > sma_series[f"{ticker}_{sma_type}"].loc[current_date]
                    except: return False
                if check_def('GLD'): raw_mode = "GOLD"
                elif check_def('TLT'): raw_mode = "LONG_BOND"
                elif check_def('IEF'): raw_mode = "MED_BOND"
            else:
                if univ_setting == 'TQQQ_ONLY': raw_mode = "TQQQ"
                elif bull_iwm: raw_mode = "ALL_STOCKS"
                elif bull_qqq: raw_mode = "TQQQ"
                elif bull_spy: raw_mode = "SPXL"
                elif bull_xlg: raw_mode = "XLG_TOP5"

            if not processing_today_only:
                if raw_mode != state['current_mode']:
                    if raw_mode == state.get('pending_mode'): state['confirm_counter'] += 1
                    else: state['pending_mode'] = raw_mode; state['pending_state'] = potential_state; state['confirm_counter'] = 1
                    if state['confirm_counter'] >= config['confirmation_days']:
                        print(f"       [SIGNAL] State Confirmed: {state['current_mode']} -> {raw_mode}")
                        state['graduated_state'] = state['pending_state']; state['current_mode'] = raw_mode; state['timer'] = config['rebalance_days']; state['confirm_counter'] = 0; state['pending_mode'] = None; state['pending_state'] = None
                else:
                    if config['guard_mode'] == "GRADUATED" and potential_state != current_grad_state: state['graduated_state'] = potential_state
                    state['confirm_counter'] = 0; state['pending_mode'] = None; state['pending_state'] = None

                state['timer'] += 1
                if state['timer'] >= config['rebalance_days']:
                    state['timer'] = 0
                    if current_date == dates_to_process[-1]: force_rebalance_today = True

            state['last_run_date'] = date_str

        force_rebalance = force_rebalance_today; curr_holdings = list(state.get('holdings', {}).keys()); curr_mode = state['current_mode']

        if univ_setting == 'TQQQ_ONLY':
            if curr_mode in ['ALL_STOCKS', 'XLG_TOP5']: state['current_mode'] = 'TQQQ'; force_rebalance = True
            if curr_mode == 'TQQQ' and (not curr_holdings or any(x != 'TQQQ' for x in curr_holdings)): force_rebalance = True
        elif univ_setting == 'CUSTOM' and curr_mode == 'ALL_STOCKS':
            if any(x not in custom_tickers for x in curr_holdings) or not curr_holdings: force_rebalance = True
        elif univ_setting == 'STANDARD' and curr_mode == 'ALL_STOCKS':
            if not curr_holdings or curr_holdings == ['TQQQ']: force_rebalance = True

        if force_rebalance:
            print(f"       >>> EXECUTING REBALANCE ({state['current_mode']})...")
            target_assets = []
            if state['current_mode'] == "ALL_STOCKS":
                 univ = custom_tickers if univ_setting == "CUSTOM" else pit_manager.get_all_tickers() if not config['allow_3x_in_stock_picks'] else list(set(pit_manager.get_all_tickers() + parse_list(GLOBAL_NITRO_ETFS)))
                 ranked = []; latest_dt = all_dates[-1]
                 for t in univ:
                     if t in data_master.columns.levels[0]:
                         if univ_setting == 'STANDARD' and not pit_manager.is_constituent(t, 'STANDARD', latest_dt.date()): continue
                         closes = data_master[t]['Close'].dropna()
                         if len(closes) > mom_long and latest_dt in closes.index:
                             idx = closes.index.get_loc(latest_dt)
                             if idx >= mom_long:
                                 start_long = closes.iloc[idx - mom_long]; start_short = closes.iloc[idx - mom_short] if idx >= mom_short else start_long; end = closes.iloc[idx]
                                 if start_long > 0 and start_short > 0:
                                     blended_roc = (((end - start_long) / start_long) * 0.70) + (((end - start_short) / start_short) * 0.30)
                                     ranked.append((t, blended_roc))
                 ranked.sort(key=lambda x: x[1], reverse=True); target_assets = [x[0] for x in ranked[:config['top_n_stocks']]]
            elif state['current_mode'] == "XLG_TOP5":
                 ranked = []; latest_dt = all_dates[-1]
                 for t in t_xlg[:50]:
                     if t in data_master.columns.levels[0]:
                         closes = data_master[t]['Close'].dropna()
                         if len(closes) > mom_long and latest_dt in closes.index:
                             idx = closes.index.get_loc(latest_dt); start = closes.iloc[idx-mom_long]; end = closes.iloc[idx]
                             if start > 0: ranked.append((t, (end-start)/start))
                 ranked.sort(key=lambda x: x[1], reverse=True); target_assets = [x[0] for x in ranked[:5]]
            elif state['current_mode'] == "TQQQ": target_assets = ['TQQQ']
            elif state['current_mode'] == "SPXL": target_assets = ['SPXL']
            elif state['current_mode'] == "GOLD": target_assets = ['GLD']
            elif state['current_mode'] == "LONG_BOND": target_assets = ['TLT']
            elif state['current_mode'] == "MED_BOND": target_assets = ['IEF']
            else: target_assets = ['BIL']

            print(f"       -> TARGET ALLOCATION: {target_assets}")
            state['timer'] = 0; state['last_rebal_date'] = datetime.date.today().isoformat()
            state['holdings'] = {t: f"{100/len(target_assets):.2f}%" for t in target_assets} if target_assets else {}
        else: print(f"       [HOLD] No action required today.")

        try:
            with open(state_file, 'w') as f: json.dump(state, f, indent=4, default=json_serial)
            print(f"       [System] State Saved.")
        except: pass

        guard_report = {}
        for g in guards:
            try:
                latest_dt = dates_to_process[-1] if len(dates_to_process) > 0 else all_dates[-1]
                px = price_series[g].loc[latest_dt]; sma_ex = sma_series[f"{g}_exit"].loc[latest_dt]; sma_en = sma_series[f"{g}_entry"].loc[latest_dt]
                guard_report[g] = {'px': px, 'dist_ex': ((px - sma_ex) / sma_ex) * 100 if pd.notna(sma_ex) and sma_ex != 0 else 0, 'dist_en': ((px - sma_en) / sma_en) * 100 if pd.notna(sma_en) and sma_en != 0 else 0}
            except: pass
        execution_reports.append({'name': s_name, 'state': state, 'guards': guard_report})

    print("\n" + "="*60 + "\n   >>> END-OF-RUN STATE SUMMARY <<<\n" + "="*60)
    for rep in execution_reports:
        st = rep['state']
        print(f"\n   [STRATEGY]: {rep['name']}")
        print(f"   -> State: {st['current_mode']} ({st['graduated_state']})")
        print(f"   -> Pending: Mode={st['pending_mode']}, State={st['pending_state']}")
        print(f"   -> Counters: Confirm Day = {st['confirm_counter']} / {config['confirmation_days']}, Rebalance Timer = {st['timer']} / {config['rebalance_days']}")
        print(f"   -> Current Holdings: {list(st.get('holdings', {}).keys())}")
        print(f"   -> Guard Status (vs Exit SMA / vs Re-entry SMA):")
        for g, vals in rep['guards'].items(): print(f"      {g.ljust(4)} | Px: {vals['px']:>7.2f} | Exit Dist: {vals['dist_ex']:>+7.2f}% | Entry Dist: {vals['dist_en']:>+7.2f}%")
    print("\n" + "="*50 + "\n   EXECUTION COMPLETE\n" + "="*50)

# ==========================================
# MAIN ENTRY & HTML LOGGER
# ==========================================
class HTMLConsoleLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("<html><head><style>body { background-color: #121212; color: #00FF00; font-family: 'Courier New', monospace; padding: 15px; white-space: pre-wrap; font-size: 14px; }</style></head><body>\n")
            f.write(f"<h3>SWITCHBLADE RUN LOG: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>\n")

    def write(self, message):
        self.terminal.write(message)
        safe_msg = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        with open(self.filepath, "a", encoding="utf-8") as f: f.write(safe_msg)

    def inject_html(self, html_content):
        with open(self.filepath, "a", encoding="utf-8") as f: f.write(f"\n{html_content}\n")

    def flush(self): self.terminal.flush()
    def close(self):
        with open(self.filepath, "a", encoding="utf-8") as f: f.write("\n</body></html>")

if __name__ == "__main__":
    if 'strategies' in globals() and strategies:
        os.makedirs("output", exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        local_log_path = f"output/Switchblade_Log_{timestamp}.html"
        
        logger = HTMLConsoleLogger(local_log_path)
        sys.stdout = logger

        try:
            print(f"\n>>> MODE: {mode}")
            if mode == "Backtest Mode": run_batch_backtest()
            else: run_execution()
        finally:
            logger.close()
            sys.stdout = logger.terminal
    else: print("\n>>> PLEASE RUN CONFIGURATION CELL (STEP 2) FIRST.")
