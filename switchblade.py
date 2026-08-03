# ====================================================
# SWITCHBLADE v47.55 - HEADLESS GITHUB ACTIONS EDITION
# ====================================================

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

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# ==========================================
# CONFIGURATION
# ==========================================
mode = "Execution Mode"  # Default for GitHub Actions
state_dir = "./state"
GLOBAL_NITRO_ETFS = "SPXL, SPXS, TQQQ, SQQQ, UDOW, SDOW, TNA, TZA, MIDU, EDC, EDZ, YINN, YANG, EURL, INDL, TECL, TECS, SOXL, SOXS, FNGU, FNGD, WEBL, WEBS, FAS, FAZ, ERX, ERY, CURE, LABU, LABD, DRN, DRV, UTSL, DUSL, RETL, UGL, GLL, AGQ, ZSL, UCO, SCO, BOIL, KOLD, TMF, TMV, UST, PST, BITU, ETHU, UVXY"

S1_ENABLE = True
S1_NAME = "Standard (205/20, 5, 102/22)"
S1_UNIVERSE = "STANDARD"
S1_CUSTOM_LIST = ""
S1_SMA = 205
S1_REENTRY = 20
S1_REBAL_DAYS = 21
S1_TOP_STOCKS = 10
S1_MOM_LONG = 106
S1_MOM_SHORT = 25
S1_CONFIRM_DAYS = 5
S1_GUARD_MODE = "GRADUATED"
S1_ALLOW_NITRO = False

S2_ENABLE = True
S2_NAME = "TQQQ_Only (205/20, 5, 102/22)"
S2_UNIVERSE = "TQQQ_ONLY"
S2_CUSTOM_LIST = GLOBAL_NITRO_ETFS
S2_SMA = 205
S2_REENTRY = 20
S2_REBAL_DAYS = 21
S2_TOP_STOCKS = 1
S2_MOM_LONG = 106
S2_MOM_SHORT = 25
S2_CONFIRM_DAYS = 5
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

# ==========================================
# DATA FETCHING UTILITIES
# ==========================================
def get_tickers(universe):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        def extract(url, cols=['Symbol', 'Ticker']):
            r = requests.get(url, headers=headers)
            tables = pd.read_html(io.StringIO(r.text))
            for df in tables:
                for c in cols:
                    if c in df.columns: 
                        raw_list = [str(t).replace('.', '-') for t in df[c].tolist()]
                        return [t for t in raw_list if t not in ['CWEN-A']] 
            return []
        if universe == "SP500": return extract('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        if universe == "NDX": return extract('https://en.wikipedia.org/wiki/Nasdaq-100', ['Ticker', 'Symbol'])
        if universe == "SP1000":
            mid = extract('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies', ['Symbol', 'Ticker Symbol'])
            small = extract('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies', ['Symbol', 'Ticker Symbol'])
            return mid + small
    except Exception as e:
        print(f"Error fetching ticker list: {e}")
        return []

def batch_download(tickers, start_date, end_date, chunk_size=100):
    all_data_list = []
    safe_end_date = end_date + datetime.timedelta(days=1) if isinstance(end_date, datetime.date) else end_date
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    print(f" Downloading {len(tickers)} tickers in chunks...")
    for i, chunk in enumerate(chunks):
        try:
            batch = yf.download(chunk, start=start_date, end=safe_end_date, group_by='ticker', progress=False, auto_adjust=True, threads=True)
            if not batch.empty: all_data_list.append(batch)
            time.sleep(0.3)
        except Exception as e:
            print(f" Batch {i+1} Failed: {e}")
    return pd.concat(all_data_list, axis=1) if all_data_list else None

def load_and_prep_data():
    t_sp500 = get_tickers("SP500"); t_ndx = get_tickers("NDX"); t_sp1000 = get_tickers("SP1000")
    t_xlg = t_sp500[:60] if t_sp500 else []
    guards = ["IWM", "QQQ", "SPY", "XLG", "GLD", "TLT", "IEF", "BIL"]
    all_nitro_etfs = parse_list(GLOBAL_NITRO_ETFS)
    all_tickers = list(set(t_sp500 + t_ndx + t_sp1000 + t_xlg + guards + all_nitro_etfs))
    print(f" Master Universe Size: {len(all_tickers)} Tickers")
    
    start_date = datetime.date.today() - datetime.timedelta(days=450)
    data = batch_download(all_tickers, start_date, datetime.date.today())
    if data is not None:
        data.index = pd.to_datetime(data.index, utc=True).normalize().tz_localize(None)
        data = data[~data.index.duplicated(keep='last')]
        data = data.loc[:, ~data.columns.duplicated()]
        data = data.sort_index()
    return data, t_sp500, t_sp1000, t_ndx, t_xlg

# ==========================================
# EXECUTION ENGINE
# ==========================================
def run_execution():
    print("\n" + "="*50 + "\n >>> SWITCHBLADE HEADLESS EXECUTION <<<\n" + "="*50)
    
    os.makedirs(state_dir, exist_ok=True)
    
    def json_serial(obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    data_master, t_sp500, t_sp1000, t_ndx, t_xlg = load_and_prep_data()
    if data_master is None:
        print(" [Error] Data download failed. Aborting.")
        return

    def get_sma_series(ticker, period):
        if ticker in data_master.columns.levels[0]:
            return data_master[ticker]['Close'].rolling(window=period).mean()
        return None

    execution_reports = []

    for i, config in enumerate(strategies):
        strat_id = i + 1
        s_name = config['name']
        univ_setting = str(config.get('universe', 'STANDARD')).strip().upper()
        custom_tickers = [x.strip().upper() for x in config.get('custom_list', [])]
        mom_long = config.get('momentum_long', 126)
        mom_short = config.get('momentum_short', 21)

        state_file = f"{state_dir}/portfolio_state_S{strat_id}.json"

        print(f"\n >>> STRATEGY {strat_id}: {s_name}")

        state = {
            "current_mode": "INIT",
            "graduated_state": "FIRMLY_BEARISH",
            "pending_mode": None,
            "pending_state": None,
            "confirm_counter": 0,
            "timer": 0,
            "last_rebal_date": "1900-01-01",
            "last_run_date": "1900-01-01",
            "holdings": {}
        }

        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                print(f" Loaded existing state from {state_file}")
            except Exception as e:
                print(f" Could not read state file: {e}. Resetting.")

        try:
            last_run_pd = pd.to_datetime(state.get('last_run_date', "1900-01-01")).normalize()
        except:
            last_run_pd = pd.Timestamp("1900-01-01")

        all_dates = data_master.index.unique().sort_values()
        missed_days = all_dates[all_dates > last_run_pd]

        sma_entry = config['reentry_sma_period']
        sma_exit = config['sma_period']
        guards = ['IWM', 'QQQ', 'SPY', 'XLG', 'GLD', 'TLT', 'IEF']
        sma_series, price_series = {}, {}

        for g in guards:
            sma_series[f"{g}_exit"] = get_sma_series(g, sma_exit)
            sma_series[f"{g}_entry"] = get_sma_series(g, sma_entry)
            if g in data_master.columns.levels[0]:
                price_series[g] = data_master[g]['Close']

        dates_to_process = missed_days if len(missed_days) > 0 else [all_dates[-1]]
        processing_today_only = (len(missed_days) == 0)
        force_rebalance_today = False

        for current_date in dates_to_process:
            date_str = current_date.strftime('%Y-%m-%d')
            is_defensive = state['current_mode'] in ["GOLD", "LONG_BOND", "MED_BOND", "CASH", "INIT"]
            sma_type = "entry" if is_defensive else "exit"

            def check_bull(ticker):
                try:
                    px = price_series[ticker].loc[current_date]
                    ma = sma_series[f"{ticker}_{sma_type}"].loc[current_date]
                    return px > ma
                except: return False

            bull_iwm = check_bull('IWM')
            bull_qqq = check_bull('QQQ')
            bull_spy = check_bull('SPY')
            bull_xlg = check_bull('XLG')

            any_bull = bull_iwm or bull_qqq or bull_spy or bull_xlg
            all_bull = bull_iwm and bull_qqq and bull_spy and bull_xlg

            current_grad_state = state.get('graduated_state', "FIRMLY_BEARISH")
            potential_state = current_grad_state

            if config['guard_mode'] == "NONE":
                potential_state = "FIRMLY_BULLISH"
            elif config['guard_mode'] == "GRADUATED":
                if current_grad_state == "FIRMLY_BEARISH":
                    if any_bull: potential_state = "CAUTIOUSLY_OPTIMISTIC"
                elif current_grad_state == "CAUTIOUSLY_OPTIMISTIC":
                    if all_bull: potential_state = "FIRMLY_BULLISH"
                    elif not any_bull: potential_state = "FIRMLY_BEARISH"
                    else: potential_state = "CAUTIOUSLY_OPTIMISTIC"
                elif current_grad_state == "FIRMLY_BULLISH":
                    if not all_bull: potential_state = "SLIGHTLY_BEARISH"
                elif current_grad_state == "SLIGHTLY_BEARISH":
                    if all_bull: potential_state = "FIRMLY_BULLISH"
                    elif not any_bull: potential_state = "FIRMLY_BEARISH"
                    else: potential_state = "SLIGHTLY_BEARISH"
            else:
                potential_state = "FIRMLY_BULLISH" if (bull_iwm if config['guard_mode']=='IWM_ONLY' else any_bull) else "FIRMLY_BEARISH"

            raw_mode = "CASH"
            if config['guard_mode'] == "NONE":
                raw_mode = "ALL_STOCKS"
            elif potential_state in ["FIRMLY_BEARISH", "SLIGHTLY_BEARISH"]:
                def check_def(ticker):
                    try: return price_series[ticker].loc[current_date] > sma_series[f"{ticker}_{sma_type}"].loc[current_date]
                    except: return False

                if check_def('GLD'): raw_mode = "GOLD"
                elif check_def('TLT'): raw_mode = "LONG_BOND"
                elif check_def('IEF'): raw_mode = "MED_BOND"
                else: raw_mode = "CASH"
            else:
                if univ_setting == 'TQQQ_ONLY':
                    raw_mode = "TQQQ"
                else:
                    if bull_iwm: raw_mode = "ALL_STOCKS"
                    elif bull_qqq: raw_mode = "TQQQ"
                    elif bull_spy: raw_mode = "SPXL"
                    elif bull_xlg: raw_mode = "XLG_TOP5"
                    else: raw_mode = "CASH"

            if not processing_today_only:
                if raw_mode != state['current_mode']:
                    if raw_mode == state.get('pending_mode'):
                        state['confirm_counter'] += 1
                    else:
                        state['pending_mode'] = raw_mode
                        state['pending_state'] = potential_state
                        state['confirm_counter'] = 1

                    if state['confirm_counter'] >= config['confirmation_days']:
                        print(f" [SIGNAL] State Confirmed: {state['current_mode']} -> {raw_mode}")
                        state['graduated_state'] = state['pending_state']
                        state['current_mode'] = raw_mode
                        state['timer'] = config['rebalance_days']
                        state['confirm_counter'] = 0
                        state['pending_mode'] = None
                        state['pending_state'] = None
                else:
                    if config['guard_mode'] == "GRADUATED" and potential_state != current_grad_state:
                        print(f" [STATE SHIFT] {current_grad_state} -> {potential_state}")
                        state['graduated_state'] = potential_state

                    state['confirm_counter'] = 0
                    state['pending_mode'] = None
                    state['pending_state'] = None

                state['timer'] += 1
                if state['timer'] >= config['rebalance_days']:
                    state['timer'] = 0
                    if current_date == dates_to_process[-1]:
                        force_rebalance_today = True

            state['last_run_date'] = date_str

        force_rebalance = force_rebalance_today
        curr_holdings = list(state.get('holdings', {}).keys())
        curr_mode = state['current_mode']

        if force_rebalance or not curr_holdings:
            print(f" >>> EXECUTING REBALANCE ({state['current_mode']})...")
            target_assets = []

            if state['current_mode'] == "ALL_STOCKS":
                 univ = custom_tickers if univ_setting == "CUSTOM" else list(set(t_sp500 + t_ndx + t_sp1000 + t_xlg))
                 ranked = []
                 latest_dt = all_dates[-1]
                 for t in univ:
                     if t in data_master.columns.levels[0]:
                         closes = data_master[t]['Close'].dropna()
                         if len(closes) > mom_long and latest_dt in closes.index:
                             idx = closes.index.get_loc(latest_dt)
                             if idx >= mom_long:
                                 start_long = closes.iloc[idx - mom_long]
                                 start_short = closes.iloc[idx - mom_short] if idx >= mom_short else start_long
                                 end = closes.iloc[idx]

                                 if start_long > 0 and start_short > 0:
                                     roc_long = (end - start_long) / start_long
                                     roc_short = (end - start_short) / start_short
                                     blended_roc = (roc_long * 0.70) + (roc_short * 0.30)
                                     ranked.append((t, blended_roc))

                 ranked.sort(key=lambda x: x[1], reverse=True)
                 target_assets = [x[0] for x in ranked[:config['top_n_stocks']]]

            elif state['current_mode'] == "TQQQ": target_assets = ['TQQQ']
            elif state['current_mode'] == "SPXL": target_assets = ['SPXL']
            elif state['current_mode'] == "GOLD": target_assets = ['GLD']
            elif state['current_mode'] == "LONG_BOND": target_assets = ['TLT']
            elif state['current_mode'] == "MED_BOND": target_assets = ['IEF']
            else: target_assets = ['BIL']

            print(f" -> TARGET ALLOCATION: {target_assets}")
            state['timer'] = 0
            state['last_rebal_date'] = datetime.date.today().isoformat()
            state['holdings'] = {t: f"{100/len(target_assets):.2f}%" for t in target_assets} if target_assets else {}

        with open(state_file, 'w') as f:
            json.dump(state, f, indent=4, default=json_serial)
        print(f" State successfully saved to {state_file}")

        execution_reports.append({'name': s_name, 'state': state})

    print("\n" + "="*50 + "\n >>> END-OF-RUN SUMMARY <<<\n" + "="*50)
    for rep in execution_reports:
        st = rep['state']
        print(f" Strategy: {rep['name']}")
        print(f"   Mode: {st['current_mode']} ({st['graduated_state']})")
        print(f"   Holdings: {list(st.get('holdings', {}).keys())}\n")

class HTMLConsoleLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w") as f:
            f.write("<html><head><style>body { background-color: #121212; color: #00FF00; font-family: 'Courier New', monospace; padding: 15px; white-space: pre-wrap; font-size: 14px; }</style></head><body>\n")
            f.write(f"<h3>SWITCHBLADE RUN LOG: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>\n")

    def write(self, message):
        self.terminal.write(message)
        safe_msg = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(safe_msg)
            
    def flush(self):
        self.terminal.flush()

    def close(self):
        with open(self.filepath, "a") as f:
            f.write("\n</body></html>")

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    html_log_path = f"output/Switchblade_Log_{timestamp}.html"
    
    logger = HTMLConsoleLogger(html_log_path)
    sys.stdout = logger

    try:
        run_execution()
    finally:
        logger.close()
        sys.stdout = logger.terminal
