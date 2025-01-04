import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet
from scipy import stats
import os
from dotenv import load_dotenv
import oracledb
from sqlalchemy import create_engine
import urllib.parse
from sqlalchemy import text
import time
import oracledb.exceptions
from datetime import datetime
import pytz

# Function to toggle debugging mode
def toggle_debug_mode():
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    st.session_state.debug_mode = not st.session_state.debug_mode
    return st.session_state.debug_mode

def safe_get_data(data_dict, key):
    """Safely access data dictionary with proper error handling and column name normalization"""
    if data_dict is None or key not in data_dict or data_dict[key] is None:
        return pd.DataFrame()
        
    # Get the dataframe
    df = data_dict[key]
    
    # Convert all column names to lowercase
    df.columns = df.columns.str.lower()
    
    if toggle_debug_mode():
        print(f"Data for '{key}':")
        print(df.head())
        print("Columns:", df.columns)
    
    return df  
def filter_by_date(df, date_col, start_date, end_date):
    """Filter DataFrame by date range"""
    try:
        if df.empty:
            return df
            
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Filter by date range
        mask = (
            (df[date_col].dt.date >= pd.to_datetime(start_date).date()) &
            (df[date_col].dt.date <= pd.to_datetime(end_date).date())
        )
        
        return df[mask]
    except Exception as e:
        st.error(f"Error filtering data by date: {str(e)}")
        return df    
def get_date_column(df):
    """Find the appropriate date column in a DataFrame"""
    possible_date_cols = ['transactiondate', 'opendate', 'date', 'exclusiontime', 'login_time']
    for col in possible_date_cols:
        if col in df.columns:
            return col
    return None

def safe_get_trend_data(df, value_col='sumdeposit'):
    """Safely get trend data with proper date handling"""
    if df is None or df.empty:
        return pd.DataFrame()
        
    date_col = get_date_column(df)
    if date_col is None:
        st.warning(f"No date column found in data")
        return pd.DataFrame()
        
    if value_col not in df.columns:
        st.warning(f"Value column {value_col} not found in data")
        return pd.DataFrame()
        
    try:
        trend_data = df.groupby(pd.to_datetime(df[date_col]).dt.date)[value_col].sum().reset_index()
        trend_data.columns = ['date', 'amount']
        return trend_data
    except Exception as e:
        st.error(f"Error creating trend data: {str(e)}")
        return pd.DataFrame()


def validate_dataframe(df, required_columns, tab_name):
    """Validate DataFrame has required columns and data"""
    if df is None or df.empty:
        st.warning(f"No data available for {tab_name}")
        return False
        
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing required columns for {tab_name}: {', '.join(missing_cols)}")
        return False
        
    return True

def safe_filter_date(df, date_col, start_date, end_date, tab_name):
    """Safely filter DataFrame by date with error handling"""
    try:
        if df.empty:
            return df
            
        # Debug info
        if st.checkbox(f"Debug {tab_name} Date Filtering"):
            st.write(f"Date column type: {df[date_col].dtype}")
            st.write("Sample dates:", df[date_col].head())
        
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        df = df.dropna(subset=[date_col])
        
        filtered_df = df[
            (df[date_col].dt.date >= pd.to_datetime(start_date).date()) &
            (df[date_col].dt.date <= pd.to_datetime(end_date).date())
        ]
        
        if filtered_df.empty:
            st.warning(f"No data available for selected date range in {tab_name}")
        
        return filtered_df
        
    except Exception as e:
        st.error(f"Error filtering data in {tab_name}: {str(e)}")
        return df
def get_last_update_time():
    try:
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            return datetime.now(pytz.timezone('Asia/Jerusalem'))
            
        # Get the most recent modification time of any cache file
        cache_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.csv')]
        if not cache_files:
            return datetime.now(pytz.timezone('Asia/Jerusalem'))
            
        latest_mtime = max(os.path.getmtime(f) for f in cache_files)
        latest_time = datetime.fromtimestamp(latest_mtime)
        
        # Convert to Israel timezone
        israel_tz = pytz.timezone('Asia/Jerusalem')
        return latest_time.astimezone(israel_tz)
    except Exception as e:
        st.error(f"Error getting last update time: {str(e)}")
        return datetime.now(pytz.timezone('Asia/Jerusalem'))

def safe_metric(df, column, operation='sum', default="N/A"):
    """Safely compute metric with error handling"""
    try:
        if df is None or df.empty:
            return default
        if operation == 'count':  # Add count operation
            return f"{len(df):,.0f}"
        elif operation == 'sum':
            return f"{df[column].sum():,.0f}"
        elif operation == 'mean':
            return f"{df[column].mean():.1f}%"
        return default
    except Exception as e:
        st.warning(f"Error computing metric for {column}: {str(e)}")
        return default

def safe_value(df, column, default=0):
    """Safely extract value from DataFrame with error handling"""
    try:
        if df is None or df.empty or column not in df.columns:
            return default
        value = df[column].iloc[0]
        return value if pd.notnull(value) else default
    except:
        return default

# Create columns first
col1, col2, col3 = st.columns(3)

QUERIES = {
    # ... existing queries remain the same ...
    'login_duration': """
        select round(median(l.logoutdate-l.logindate)*24*60,2) as median_login_duration_minutes
          from gamer.logins l
            join gamer.ir_sys_useraccounts ua on ua.userid = l.userid
            join casino.users u on u.userid = ua.userid
            join gamer.skins sk on u.skinid = sk.skinid 
         where l.logindate >= date'2025-01-01'
           and sk.skinorigin = 15
           and sk.skin not in ('TestCasino.bet.br', 'OjoTest.bet.br')
           and ua.internalaccount != 1
    """,       
    'registrations': """
        Select rt.description,
               u.userid,
               u2.username,
               ud.user_parent_id,
               u.opendate,
               ua.account_block_reasons,
               ei.description,
               sk.skin,
               t.accounttypename,
               u.realbalance,
               ua.verifiedplayer,
               ua.verified_bank_details,
               ua.mobile_verification
          from gamer.ir_sys_useraccounts ua
          join gamer.userdetails2 ud on ua.userid = ud.userid
          join casino.users u on u.userid = ud.userid
          join gamer.users2 u2 on u2.userid = ud.userid
          join gamer.skins sk on u.skinid = sk.skinid
          join gamer.registration_type rt on ud.registration_type_id = rt.id
          join gamer.eid_status ei on ua.eid_status = ei.id
          join gamer.accounttypes t on t.accounttypeid = ud.casinoaccounttypeid
         where u.opendate >= to_date('01/01/2025 00:00:00', 'dd/mm/yyyy hh24:mi:ss')
          and sk.skinorigin = 15
          and sk.skin not in ('TestCasino.bet.br', 'OjoTest.bet.br')
          and ua.internalaccount != 1
    """,
    'breakdown': """
        Select rt.description,
               u.userid,
               u2.username,
               ud.user_parent_id,
               u.opendate,
               ua.account_block_reasons,
               ei.description,
               sk.skin,
               t.accounttypename,
               u.realbalance,
               ua.verifiedplayer,
               ua.verified_bank_details,
               ua.mobile_verification,       
               abr.id                   as br_id,
               abr.name                 as br_name
          from gamer.ir_sys_useraccounts ua
          join gamer.userdetails2 ud on ua.userid = ud.userid
          join casino.users u on u.userid = ud.userid
          join gamer.users2 u2 on u2.userid = ud.userid
          join gamer.skins sk on u.skinid = sk.skinid
          join gamer.registration_type rt on ud.registration_type_id = rt.id
          join gamer.eid_status ei on ua.eid_status = ei.id
          join gamer.accounttypes t on t.accounttypeid = ud.casinoaccounttypeid
          join table(reporting.del2tab(nvl(ua.account_block_reasons, 0))) br on 1 = 1
          join gamer.account_block_reason abr on abr.id = br.column_value
         where u.opendate < to_date('01/01/2025 00:00:00', 'dd/mm/yyyy hh24:mi:ss')
          and sk.skinorigin = 15
          and sk.skin not in ('TestCasino.bet.br', 'OjoTest.bet.br')
          and ua.internalaccount != 1
    """,
    'exclusions': """
    SELECT 
        se.exclusiontime,
        se2.name,
        COUNT(*) as count,
        COUNT(DISTINCT se.userid) as unique_count
    FROM gamer.selfexclusions se
    JOIN gamer.self_exclusion_types se2 ON se2.id = se.self_exclusion_type_id
    JOIN casino.users cu ON cu.userid = se.userid
    JOIN gamer.skingroupskins sgs ON cu.skinid = sgs.skinid
    WHERE se.createddate >= date'2025-01-01'
    AND sgs.groupid = 25
    GROUP BY se.exclusiontime, se2.name
    """,
    'quick_deposits': """
        select 
               count(distinct et.userid) as CountUsers,
               count(*) as CountDeposits,
               round(avg(et.amount*ccr.conversionrate),2) as AVGDeposit,
               round(sum(et.amount*ccr.conversionrate)) as SUMDeposit
          from gamer.userdetails2 gu
          join casino.users cu on cu.userid = gu.userid
          join gamer.ir_sys_useraccounts ua on ua.userid = gu.userid
          join gamer.skingroupskins sgs on cu.skinid = sgs.skinid
          join casino.skins cs on cs.id = cu.skinid
          join casino.brands b on b.id = cs.brand_id
          join gamer.ir_sys_exttrans et on gu.userid = et.userid
          join gamer.ir_sys_exttranstypes tt on tt.externaltransactiontypeid = et.externaltransactiontypeid
          join gamer.ir_sys_exttransstatuses s on s.statusid = et.externaltransactionstatusid
          join gamer.currency_conv_rates_current ccr
            on ccr.basecurrencyid = et.currencyid
           and ccr.currencyid = 30
         where et.transactiondate >= date'2025-01-01'
           and tt.typename in ('Sale', 'Manual deposit')
           and s.status = 'Approved'
           and et.isquickdeposit = 1
           and sgs.groupid = 25
           and ua.internalaccount != 1
    """,
    
    'regular_deposits': """
        select 
               count(distinct et.userid) as CountUsers,
               count(*) as CountDeposits,
               round(sum(et.amount*ccr.conversionrate)) as SUMDeposit,
               round(avg(et.amount*ccr.conversionrate),2) as AVGfirstDeposit,
               round(avg(gu.casinofirstdepositamount*ccr.conversionrate),2) as AVGDeposit,
               round(median(et.amount*ccr.conversionrate),2) as MDNDfirsteposit,
               round(median(gu.casinofirstdepositamount*ccr.conversionrate),2) as MDNDeposit
          from gamer.userdetails2 gu
          join casino.users cu on cu.userid = gu.userid
          join gamer.ir_sys_useraccounts ua on ua.userid = gu.userid
          join gamer.skingroupskins sgs on cu.skinid = sgs.skinid
          join casino.skins cs on cs.id = cu.skinid
          join casino.brands b on b.id = cs.brand_id
          join gamer.ir_sys_exttrans et on gu.userid = et.userid
          join gamer.ir_sys_exttranstypes tt on tt.externaltransactiontypeid = et.externaltransactiontypeid
          join gamer.ir_sys_exttransstatuses s on s.statusid = et.externaltransactionstatusid
          join gamer.currency_conv_rates_current ccr
            on ccr.basecurrencyid = et.currencyid
           and ccr.currencyid = 30
         where et.transactiondate >= date'2025-01-01'
           and tt.typename in ('Sale', 'Manual deposit')
           and s.status = 'Approved'
           and et.isquickdeposit != 1
           and sgs.groupid = 25
           and ua.internalaccount != 1
    """,
    
    'withdrawals': """
        SELECT 
            COUNT(DISTINCT et.userid) AS CountUsers,
            COUNT(*) AS CountDeposits,
            ROUND(AVG(NVL(et.amount*ccr.conversionrate, 0)), 2) AS AVGDeposit,
            ROUND(SUM(NVL(et.amount*ccr.conversionrate, 0))) AS SUMDeposit
        FROM gamer.ir_sys_exttrans et
        JOIN gamer.ir_sys_exttranstypes tt ON tt.externaltransactiontypeid = et.externaltransactiontypeid
        JOIN gamer.ir_sys_exttransstatuses s ON s.statusid = et.externaltransactionstatusid
        JOIN gamer.ir_sys_useraccounts ua ON ua.userid = et.userid
        JOIN casino.users cu ON cu.userid = et.userid
        JOIN gamer.skingroupskins sgs ON cu.skinid = sgs.skinid
        JOIN gamer.currency_conv_rates_current ccr ON ccr.basecurrencyid = et.currencyid AND ccr.currencyid = 30
        WHERE et.transactiondate >= DATE '2025-01-01'
        AND tt.typename IN ('Winning', 'Manual withdraw')
        AND s.status = 'Approved'
        AND sgs.groupid = 25
        AND ua.internalaccount != 1
    """,
    
    'withdrawal_processors': """
        SELECT 
            p.name AS processor,
            COUNT(*) AS deposit_count,
            ROUND(SUM(NVL(e.amount/cr.conversionrate, 0))) AS deposit_amount,
            ROUND(AVG(NVL(e.amount/cr.conversionrate, 0))) AS average_deposit,
            ROUND(RATIO_TO_REPORT(COUNT(*)) OVER() * 100, 1) AS pct,
            ROUND(MAX(NVL(e.amount/cr.conversionrate, 0))) AS biggest_deposit
        FROM gamer.ir_sys_exttrans e
        JOIN gamer.ir_sys_exttranstypes tt ON tt.externaltransactiontypeid = e.externaltransactiontypeid
        JOIN gamer.ir_sys_useraccounts ua ON ua.userid = e.userid
        JOIN gamer.ir_sys_exttransstatuses s ON s.statusid = e.externaltransactionstatusid
        JOIN gamer.ir_sys_processors p ON p.processorid = e.processorid
        JOIN gamer.currency_conv_rates_current cr ON cr.currencyid = e.currencyid
        WHERE tt.typename IN ('Winning', 'Manual withdraw')
        AND s.status = 'Approved'
        AND e.transactiondate >= DATE '2025-01-01'
        GROUP BY p.name
    """,
    
    'game_providers': """
        select at.applicationtype_name,
               count(*) as sessions_count, 
               sum(s.gamecount) as rounds_count, 
               sum(s.totalbet) as total_bet, 
               avg(s.totalbet/s.gamecount) as avg_bet, 
               median(s.totalbet/s.gamecount) as median_bet,
               median(s.gamecount) as median_rounds_in_session
          from casino.game_sessions s
          join casino.menu_items mi on mi.itemid = s.menu_itemid
          join gamer.ir_sys_useraccounts ua on ua.userid = s.userid
          join casino.application_types at on at.applicationtypeid = mi.categoryid
          join casino.application_types at2 on at2.applicationtypeid = at.pa_typeid
          join casino.users cu on cu.userid = s.userid
          join gamer.skingroupskins sgs on cu.skinid = sgs.skinid
          join gamer.currency_conv_rates_current ccr
            on ccr.basecurrencyid = s.currencyid
           and ccr.currencyid = 30
         where s.opendate >= to_date('01/01/2025', 'dd/mm/yyyy')
           and sgs.groupid = 25
           and s.realmoney = 1
           and s.gamecount > 0
           and ua.internalaccount != 1
         group by at.applicationtype_name
    """,
    'popular_games': """
        select m.item_title, 
               sum(s.gamecount) as total_rounds, 
               median(s.totalbet/s.gamecount) as median_bet
          from casino.game_sessions s
          join casino.menu_items m on m.itemid = s.menu_itemid
          join casino.users cu on cu.userid = s.userid
          join gamer.skingroupskins sgs on cu.skinid = sgs.skinid
          join gamer.ir_sys_useraccounts ua on ua.userid = s.userid
         where s.opendate >= to_date('01/01/2025', 'dd/mm/yyyy')
           and sgs.groupid = 25
           and s.realmoney = 1
           and s.gamecount > 0
           and ua.internalaccount != 1
         group by m.item_title
         order by sum(s.gamecount) desc
    """,
    
    'lead_processors_deposits': """
        select p.name as processor, 
               count(*) as deposit_count,
               round(ratio_to_report(count(*)) over()*100, 1) as pct,
               round(sum(e.amount/cr.conversionrate)) as deposit_amount, 
               round(avg(e.amount/cr.conversionrate)) as average_deposit, 
               round(max(e.amount/cr.conversionrate)) as biggest_deposit
          from gamer.ir_sys_exttrans e
          join gamer.ir_sys_exttranstypes tt on tt.externaltransactiontypeid = e.externaltransactiontypeid
          join gamer.ir_sys_exttransstatuses s on s.statusid = e.externaltransactionstatusid
          join gamer.ir_sys_processors p on p.processorid = e.processorid
          join gamer.currency_conv_rates_current cr on cr.currencyid = e.currencyid
          join casino.users cu on cu.userid = e.userid
          join gamer.skingroupskins sgs on sgs.skinid = cu.skinid
          join gamer.ir_sys_useraccounts ua on ua.userid = e.userid
         where tt.typename in ('Sale', 'Manual deposit')
           and s.status = 'Approved'
           and e.transactiondate >= to_date('01/01/2025', 'dd/mm/yyyy')
           and cr.basecurrencyid = 30
           and sgs.groupid = 25
           and ua.internalaccount != 1
        group by p.name
        order by sum(e.amount/cr.conversionrate) desc
    """
}
load_dotenv()

# Database configuration
def get_db_config():
    try:
        host = os.getenv('DB_HOST', 'rac-cluster-7-8-scan')
        port = os.getenv('DB_PORT', '1521')
        service_name = os.getenv('DB_SERVICE', 'sgames_bi')
        
        # Try to initialize thick mode
        try:
            oracledb.init_oracle_client()
            thick_mode = True
        except Exception as e:
            st.warning(f"Could not initialize Oracle thick client: {str(e)}")
            st.warning("Falling back to thin mode with modified queries")
            thick_mode = False
        
        dsn = f"""(DESCRIPTION=
                    (ADDRESS=(PROTOCOL=TCP)(HOST={host})(PORT={port}))
                    (CONNECT_DATA=(SERVICE_NAME={service_name}))
                )"""
                    
        return {
            'user': os.getenv('DB_USER', 'Product'),
            'password': os.getenv('DB_PASSWORD', 'Product!2022'),
            'dsn': dsn,
            'thick_mode': thick_mode
        }
    except Exception as e:
        st.error(f"Error loading database configuration: {str(e)}")
        return None

def fetch_and_cache_data():
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    all_data = {}
    
    db_config = get_db_config()
    if not db_config:
        st.warning("Database configuration not available. Using sample data.")
        return generate_sample_data()
        
    try:
        # Modify queries based on mode
        modified_queries = QUERIES.copy()
        if not db_config.get('thick_mode'):
            # Modify problematic queries for thin mode
            modified_queries['withdrawal_processors'] = """
                select 
                    cast(p.name as varchar2(100)) as processor,
                    count(*) as deposit_count,
                    round(sum(e.amount/cr.conversionrate)) as deposit_amount,
                    round(avg(e.amount/cr.conversionrate)) as average_deposit,
                    round(ratio_to_report(count(*)) over()*100, 1) as pct,
                    round(max(e.amount/cr.conversionrate)) as biggest_deposit
                from gamer.ir_sys_exttrans e
                join gamer.ir_sys_exttranstypes tt on tt.externaltransactiontypeid = e.externaltransactiontypeid
                join gamer.ir_sys_useraccounts ua on ua.userid = e.userid
                join gamer.ir_sys_exttransstatuses s on s.statusid = e.externaltransactionstatusid
                join gamer.ir_sys_processors p on p.processorid = e.processorid
                join gamer.currency_conv_rates_current cr on cr.currencyid = e.currencyid
                join casino.users cu on cu.userid = e.userid
                join gamer.skingroupskins sgs on sgs.skinid = cu.skinid
                where tt.typename in ('Winning', 'Manual withdraw')
                and s.status = 'Approved'
                and e.transactiondate >= to_date('01/01/2025', 'dd/mm/yyyy')
                and cr.basecurrencyid = 30
                and sgs.groupid = 25
                and ua.internalaccount != 1
                group by p.name
            """
            modified_queries['lead_processors_deposits'] = """
                select 
                    cast(p.name as varchar2(100)) as processor,
                    count(*) as deposit_count,
                    round(ratio_to_report(count(*)) over()*100, 1) as pct,
                    round(sum(e.amount/cr.conversionrate)) as deposit_amount,
                    round(avg(e.amount/cr.conversionrate)) as average_deposit,
                    round(max(e.amount/cr.conversionrate)) as biggest_deposit
                from gamer.ir_sys_exttrans e
                join gamer.ir_sys_exttranstypes tt 
                    on tt.externaltransactiontypeid = e.externaltransactiontypeid
                join gamer.ir_sys_exttransstatuses s 
                    on s.statusid = e.externaltransactionstatusid
                join gamer.ir_sys_processors p 
                    on p.processorid = e.processorid
                join gamer.currency_conv_rates_current cr 
                    on cr.currencyid = e.currencyid
                join casino.users cu 
                    on cu.userid = e.userid
                join gamer.skingroupskins sgs 
                    on sgs.skinid = cu.skinid
                join gamer.ir_sys_useraccounts ua 
                    on ua.userid = e.userid
                where tt.typename in ('Sale', 'Manual deposit')
                and s.status = 'Approved'
                and e.transactiondate >= to_date('01/01/2025', 'dd/mm/yyyy')
                and cr.basecurrencyid = 30
                and sgs.groupid = 25
                and ua.internalaccount != 1
                group by cast(p.name as varchar2(100))
                order by sum(e.amount/cr.conversionrate) desc
            """
            modified_queries['login_duration'] = """
                SELECT 
                    trunc(us.login_time) as login_date,
                    COUNT(*) as session_count,
                    ROUND(AVG(
                        EXTRACT(HOUR FROM (us.logout_time - us.login_time))*60 + 
                        EXTRACT(MINUTE FROM (us.logout_time - us.login_time))
                    ), 1) as avg_duration_minutes
                FROM gamer.user_sessions us
                JOIN casino.users cu ON cu.userid = us.userid
                JOIN gamer.skingroupskins sgs ON cu.skinid = sgs.skinid
                WHERE us.login_time >= DATE '2025-01-01'
                AND sgs.groupid = 25
                GROUP BY trunc(us.login_time)
                ORDER BY login_date
            """
            
        with oracledb.connect(**{k: v for k, v in db_config.items() if k != 'thick_mode'}) as connection:
            for query_name, query in modified_queries.items():
                try:
                    csv_path = os.path.join(cache_dir, f"{query_name}.csv")
                    
                    if os.path.exists(csv_path) and (datetime.now() - datetime.fromtimestamp(os.path.getmtime(csv_path))).seconds < 3600:
                        try:
                            all_data[query_name] = pd.read_csv(csv_path)
                            st.success(f"Loaded {query_name} from cache")
                        except Exception as e:
                            st.error(f"Error reading cache for {query_name}: {str(e)}")
                            all_data[query_name] = None
                        continue
                    
                    st.info(f"Fetching {query_name} data from database...")
                    start_time = time.time()
                    
                    # Execute query as string
                    df = pd.read_sql(query, connection)
                    
                    elapsed = time.time() - start_time
                    st.success(f"Fetched {query_name} in {elapsed:.2f} seconds")
                    
                    df.to_csv(csv_path, index=False)
                    all_data[query_name] = df
                    
                except Exception as e:
                    st.error(f"Error fetching {query_name}: {str(e)}")
                    all_data[query_name] = None
                    
        # If all queries failed, use sample data
        if all(v is None for v in all_data.values()):
            st.warning("No data could be fetched. Using sample data.")
            return generate_sample_data()
                    
        return all_data
                    
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        st.warning("Using sample data due to database connection error.")
        return generate_sample_data()
@st.cache_data(ttl=3600)
def load_data():
    """Load and cache data with a 1-hour timeout"""
    try:
        data = fetch_and_cache_data()
        if data is None:
            st.warning("Using sample data due to fetch failure")
            return generate_sample_data()
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return generate_sample_data()

# Load data with proper error handling
with st.spinner('Fetching data...'):
    data = load_data()
# Generate stub data when DB is not available
def generate_sample_data():
    """Generate realistic sample data matching Oracle queries"""
    sample_data = {
        'registrations': pd.DataFrame({
            'description': ['Standard', 'Quick', 'Standard'] * 10,
            'userid': range(1, 31),
            'username': [f'user_{i}' for i in range(1, 31)],
            'user_parent_id': [None] * 30,
            'opendate': pd.date_range(start='2025-01-01', periods=30),
            'account_block_reasons': [None] * 30,
            'description_2': ['Verified'] * 30,
            'skin': ['Casino1.bet.br', 'Casino2.bet.br'] * 15,
            'accounttypename': ['Standard'] * 30,
            'realbalance': np.random.uniform(0, 1000, 30),
            'verifiedplayer': np.random.choice([0, 1], 30),
            'verified_bank_details': np.random.choice([0, 1], 30),
            'mobile_verification': np.random.choice([0, 1], 30)
        }),
        
        'breakdown': pd.DataFrame({
            'description': ['Standard'] * 30,
            'userid': range(1, 31),
            'username': [f'user_{i}' for i in range(1, 31)],
            'opendate': pd.date_range(start='2025-01-01', periods=30),
            'br_id': np.random.randint(1, 5, 30),
            'br_name': ['Reason_' + str(i) for i in np.random.randint(1, 5, 30)],
            'skin': ['Casino1.bet.br', 'Casino2.bet.br'] * 15,
            'verifiedplayer': np.random.choice([0, 1], 30),
            'verified_bank_details': np.random.choice([0, 1], 30),
            'mobile_verification': np.random.choice([0, 1], 30)
        }),
        
        'exclusions': pd.DataFrame({
            'exclusiontime': pd.date_range(start='2025-01-01', periods=30),
            'name': ['Self-Exclusion', 'Temporary Ban'] * 15,
            'count(*)': np.random.randint(1, 10, 30),
            'count(distinctult.base_userid)': np.random.randint(1, 8, 30)
        }),
        
        'login_duration': pd.DataFrame({
            'median_login_duration_minutes': [45.5]
        }),
        
        'quick_deposits': pd.DataFrame({
            'CountUsers': [500],
            'CountDeposits': [750],
            'AVGDeposit': [250.75],
            'SUMDeposit': [188062.50]
        }),
        
        'regular_deposits': pd.DataFrame({
            'CountUsers': [1200],
            'CountDeposits': [2000],
            'SUMDeposit': [500000.00],
            'AVGfirstDeposit': [300.25],
            'AVGDeposit': [250.00],
            'MDNDfirsteposit': [200.00],
            'MDNDeposit': [175.00]
        }),
        
        'withdrawals': pd.DataFrame({
            'CountUsers': [800],
            'CountDeposits': [1000],
            'AVGDeposit': [350.25],
            'SUMDeposit': [350250.00]
        }),
        
        'withdrawal_processors': pd.DataFrame({
            'processor': ['PIX', 'Bank Transfer', 'Credit Card'] * 2,
            'deposit_count': np.random.randint(50, 200, 6),
            'deposit_amount': np.random.uniform(10000, 100000, 6),
            'average_deposit': np.random.uniform(100, 1000, 6),
            'pct': np.random.uniform(1, 100, 6),
            'biggest_deposit': np.random.uniform(1000, 5000, 6)
        }),
        
        'game_providers': pd.DataFrame({
            'applicationtype_name': ['Provider_' + str(i) for i in range(1, 6)],
            'sessions_count': np.random.randint(1000, 5000, 5),
            'rounds_count': np.random.randint(5000, 20000, 5),
            'total_bet': np.random.uniform(50000, 200000, 5),
            'avg_bet': np.random.uniform(10, 100, 5),
            'median_bet': np.random.uniform(5, 50, 5),
            'median_rounds_in_session': np.random.randint(20, 100, 5)
        }),
        
        'popular_games': pd.DataFrame({
            'item_title': [f'Game_{i}' for i in range(1, 31)],
            'total_rounds': np.random.randint(10000, 100000, 30),
            'median_bet': np.random.uniform(5, 200, 30)
        }),
        
        'lead_processors_deposits': pd.DataFrame({
            'processor': ['PIX', 'Bank Transfer', 'Credit Card', 'E-wallet', 'Mobile Payment'] * 2,
            'deposit_count': np.random.randint(1000, 5000, 10),
            'pct': [20, 18, 15, 12, 10, 8, 7, 5, 3, 2],
            'deposit_amount': np.random.uniform(100000, 1000000, 10),
            'average_deposit': np.random.uniform(200, 1000, 10),
            'biggest_deposit': np.random.uniform(5000, 20000, 10)
        })
    }
    return sample_data
# Add forecasting capability
def create_forecast(data, date_col, value_col, periods=30):
    """Create forecast using Prophet"""
    # Prepare data for Prophet
    df = pd.DataFrame({'ds': data[date_col], 'y': data[value_col]})
    
    # Initialize and fit Prophet model
    model = Prophet(
        changepoint_prior_scale=0.15,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative'
    )
    
    model.fit(df)
    
    # Make future dataframe for prediction
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return forecast

def create_detailed_statistics(df, columns_to_display, title, formatters=None):
    st.subheader(title)
    if formatters is None:
        formatters = {}
    if not df.empty:
        # Verify columns exist before trying to display them
        existing_columns = [col for col in columns_to_display if col in df.columns]
        if not existing_columns:
            st.warning("No valid columns found for detailed statistics")
            return
            
        styled_df = df[existing_columns].style
        for col, formatter in formatters.items():
            if col in existing_columns:
                styled_df = styled_df.format({col: formatter})
        st.dataframe(styled_df, use_container_width=True)

def safe_plot(fig, title="", use_container_width=True):
    """Safely display plotly figure with error handling"""
    try:
        container = st.container()
        with container:
            st.plotly_chart(fig, use_container_width=use_container_width)
    except Exception as e:
        st.error(f"Error displaying {title}: {str(e)}")

# Usage example:
    fig = px.line(
        daily_regs,
        x='opendate',
        y='count',
        title="Daily Registration Trend"
    )
    safe_plot(fig, "Registration Trend")

# First initialize session state
def init_session_state():
    if 'first_run' not in st.session_state:
        st.session_state.first_run = True
    if 'selected_skins' not in st.session_state:
        st.session_state.selected_skins = []
    if 'date_range' not in st.session_state:
        st.session_state.date_range = (datetime.now() - timedelta(days=30), datetime.now())
    if 'download_option' not in st.session_state:
        st.session_state.download_option = None

init_session_state()

with st.spinner('Fetching data...'):
    data = load_data()
#debug data
if st.sidebar.checkbox("Debug Data Loading"):
    st.sidebar.write("Data keys:", list(data.keys()))
    st.sidebar.write("Cache status:", os.path.exists("cache"))
    st.sidebar.write("Cache files:", [f for f in os.listdir("cache")] if os.path.exists("cache") else "No cache directory")

st.markdown("""
    <style>
    /* Base container styling */
    .block-container {
        padding: 3rem 2rem 1rem;
        max-width: 95%;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin: 0.5rem 0;
    }

    div[data-testid="stMetric"] > div:first-child {
        font-size: 0.875rem;
        font-weight: 500;
        color: #6B7280;
    }

    div[data-testid="stMetric"] > div:last-child {
        font-size: 1.875rem;
        font-weight: 600;
        color: #111827;
    }

    /* Chart containers */
    div[data-testid="stHorizontalBlock"] {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin: 1rem 0;
    }

    /* Tables */
    .dataframe {
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-radius: 8px;
    }

    .dataframe thead tr th {
        background-color: #F9FAFB !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
    }

    .dataframe tbody tr td {
        padding: 0.75rem 1.5rem !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        padding: 0 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 1rem 0.5rem;
        font-weight: 500;
    }

    /* Headers */
    h1, h2, h3 {
        color: #111827;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


# Get last update time using the utility function
last_update = get_last_update_time()
timestamp = last_update.strftime("%Y-%m-%d %H:%M:%S")

# Display timestamp in a more prominent way
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Last Update:** {timestamp} (Israel Time)")

st.sidebar.markdown("---")
with st.sidebar.expander("Data Loading Status", expanded=False):
    st.write("Last Update:", get_last_update_time().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Status indicators for each data source
    for key in data.keys():
        if data[key] is not None and not data[key].empty:
            st.success(f"✓ {key}: Loaded successfully ({len(data[key])} rows)")
        else:
            st.error(f"✗ {key}: Failed to load")
    
    if st.checkbox("Show Debug Information"):
        st.write("Data keys:", list(data.keys()))
        st.write("Cache status:", os.path.exists("cache"))
        st.write("Cache files:", [f for f in os.listdir("cache")] if os.path.exists("cache") else "No cache directory")

# Create main tabs
tabs = st.tabs([
    "Registrations",
    "Registration Breakdown",
    "Exclusions",
    "Games Analytics",
    "Deposits & Withdrawals",
    "Game Providers",
    "Session Analytics",
    "Forecasting"
])

# Date range selector with debug info
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(datetime.now() - timedelta(days=30), datetime.now()),
    max_value=datetime.now(),
    key='date_input'
)

if st.sidebar.checkbox("Debug Date Range"):
    st.sidebar.write("Selected start date:", date_range[0])
    st.sidebar.write("Selected end date:", date_range[1])
    st.sidebar.write("Date range type:", type(date_range))
# Skin selector
if 'registrations' in data and data['registrations'] is not None:
    
    # Try different possible column names for skin
    possible_skin_columns = ['skin', 'SKIN', 'Skin']
    skin_column = None
    for col in possible_skin_columns:
        if col in data['registrations'].columns:
            skin_column = col
            break
    
    if skin_column:
        all_skins = data['registrations'][skin_column].unique()
        if st.session_state.first_run:
            st.session_state.selected_skins = all_skins.tolist()
            st.session_state.first_run = False

        selected_skins = st.sidebar.multiselect(
            "Select Skins",
            options=all_skins,
            default=st.session_state.selected_skins,
            key='selected_skins'
        )
    else:
        st.warning("Skin column not found in the data. Available columns: " + 
                  ", ".join(data['registrations'].columns.tolist()))
        selected_skins = []
else:
    st.warning("Registration data not available, skin filtering disabled")
    selected_skins = []

# Registrations Tab
with tabs[0]:
    registrations_data = safe_get_data(data, 'registrations')
    deposits_data = safe_get_data(data, 'regular_deposits')
    quick_deposits_data = safe_get_data(data, 'quick_deposits')

    # Key metrics
    if not registrations_data.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_regs = len(registrations_data['userid'].unique())
            st.metric("Total Registrations", f"{total_regs:,}")
        
        # Calculate real users (users who have deposited)
        with col2:
            depositing_users = 0
            if not deposits_data.empty and 'countusers' in deposits_data.columns:
                depositing_users += deposits_data['countusers'].iloc[0]
            if not quick_deposits_data.empty and 'countusers' in quick_deposits_data.columns:
                depositing_users += quick_deposits_data['countusers'].iloc[0]
            real_users_pct = (depositing_users / len(registrations_data) * 100) if len(registrations_data) > 0 else 0
            st.metric("Real Users", f"{depositing_users:,} ({real_users_pct:.1f}%)")
            
        with col3:
            verified_pct = (registrations_data['verifiedplayer'].mean() * 100)
            st.metric("Verified Players", f"{verified_pct:.1f}%")
        with col4:
            total_balance = registrations_data['realbalance'].sum()
            st.metric("Total Balance", f"R$ {total_balance:,.2f}")
    
    if st.checkbox("Debug Registration Data"):
        st.write("Registration Data Columns:", registrations_data.columns.tolist())
        st.write("First few rows:", registrations_data.head())
        
    if validate_dataframe(registrations_data, ['opendate', 'verifiedplayer', 'realbalance', 'skin'], "Registrations"):
        filtered_registrations = safe_filter_date(
            registrations_data, 
            'opendate', 
            date_range[0], 
            date_range[1], 
            "Registrations"
        )
        
        if not filtered_registrations.empty:
            # Daily registrations trend
            daily_regs = filtered_registrations.groupby(
                pd.to_datetime(filtered_registrations['opendate']).dt.date
            ).size().reset_index(name='count')
            
            fig_trend = px.line(
                daily_regs,
                x='opendate',
                y='count',
                title="Daily Registration Trend"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Registration by skin distribution
            if 'skin' in filtered_registrations.columns:
                skin_dist = filtered_registrations.groupby('skin').size().reset_index(name='count')
                fig_skin = px.pie(
                    skin_dist,
                    values='count',
                    names='skin',
                    title="Registrations by Skin"
                )
                st.plotly_chart(fig_skin, use_container_width=True)
                
            # Add detailed statistics table
            st.subheader("Detailed Registration Statistics")
            detailed_stats = filtered_registrations.groupby([pd.to_datetime(filtered_registrations['opendate']).dt.date])\
                .agg({
                    'userid': 'count',
                    'verifiedplayer': 'mean',
                    'realbalance': 'sum'
                }).reset_index()
            detailed_stats.columns = ['Date', 'Daily Registrations', 'Verification Rate', 'Total Balance']
            
            st.dataframe(
                detailed_stats.style.format({
                    'Daily Registrations': '{:,.0f}',
                    'Verification Rate': '{:.1%}',
                    'Total Balance': 'R$ {:,.2f}'
                }),
                use_container_width=True
            )

                
# Registration Breakdown Tab
with tabs[1]:
    breakdown_data = safe_get_data(data, 'breakdown')
    
    # Key metrics
    if not breakdown_data.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            verified_bank = (breakdown_data['verified_bank_details'].mean() * 100)
            st.metric("Bank Verified", f"{verified_bank:.1f}%")
        with col2:
            mobile_verified = (breakdown_data['mobile_verification'].mean() * 100)
            st.metric("Mobile Verified", f"{mobile_verified:.1f}%")
        with col3:
            blocked_users = breakdown_data['br_name'].notna().sum()
            st.metric("Blocked Users", blocked_users)
    if validate_dataframe(breakdown_data, ['br_name', 'verifiedplayer', 'verified_bank_details', 'mobile_verification'], "Breakdown"):
        filtered_breakdown = safe_filter_date(
            breakdown_data, 
            'opendate', 
            date_range[0], 
            date_range[1], 
            "Registration Breakdown"
        )

        if st.checkbox("Debug Breakdown Data"):
            st.write("Breakdown Data Columns:", breakdown_data.columns.tolist())
            st.write("First few rows:", breakdown_data.head())

        breakdown_data.columns = breakdown_data.columns.str.lower()

        if not filtered_breakdown.empty:
            # Block reasons chart
            block_dist = filtered_breakdown.groupby('br_name').size().reset_index(name='count')
            fig_block = px.bar(
                block_dist, 
                x='br_name', 
                y='count',
                title="Distribution of Block Reasons"
            )
            st.plotly_chart(fig_block, use_container_width=True)
            
            # Verification rates chart
            verif_data = pd.DataFrame({
                'Status': ['Player', 'Bank', 'Mobile'],
                'Verified': [
                    filtered_breakdown['verifiedplayer'].mean() * 100,
                    filtered_breakdown['verified_bank_details'].mean() * 100,
                    filtered_breakdown['mobile_verification'].mean() * 100
                ]
            })
            fig_verif = px.bar(
                verif_data, 
                x='Status', 
                y='Verified',
                title="Verification Rates (%)"
            )
            st.plotly_chart(fig_verif, use_container_width=True)
# Exclusions Tab
with tabs[2]:
    exclusions_data = safe_get_data(data, 'exclusions')
    if validate_dataframe(exclusions_data, ['exclusiontime', 'name', 'count', 'unique_count'], "Exclusions"):
        filtered_exclusions = safe_filter_date(
            exclusions_data,
            'exclusiontime',
            date_range[0],
            date_range[1],
            "Exclusions"
        )
        
        if not filtered_exclusions.empty:
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Exclusions", len(filtered_exclusions))
            with col2:
                unique_users = filtered_exclusions['unique_count'].sum()
                st.metric("Unique Users Excluded", unique_users)
            with col3:
                avg_daily = filtered_exclusions['count'].mean()
                st.metric("Average Daily Exclusions", f"{avg_daily:.1f}")
            
            # Exclusions trend
            fig_trend = px.line(
                filtered_exclusions,
                x='exclusiontime',
                y='count',
                color='name',
                title="Exclusions Trend by Type"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Distribution by type
            type_dist = filtered_exclusions.groupby('name')['count'].sum().reset_index()
            fig_dist = px.pie(
                type_dist,
                values='count',
                names='name',
                title="Distribution of Exclusion Types"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
# Games Analytics Tab
with tabs[3]:
    games_data = safe_get_data(data, 'popular_games')
    
    # Key metrics
    if not games_data.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_rounds = games_data['total_rounds'].sum()
            st.metric("Total Game Rounds", f"{total_rounds:,.0f}")
        with col2:
            avg_bet = games_data['median_bet'].mean()
            st.metric("Average Bet", f"R$ {avg_bet:.2f}")
        with col3:
            median_bet = games_data['median_bet'].median()
            st.metric("Median Bet", f"R$ {median_bet:.2f}")
        with col4:
            unique_games = len(games_data['item_title'].unique())
            st.metric("Active Games", unique_games)

    if validate_dataframe(games_data, ['item_title', 'total_rounds', 'median_bet'], "Games"):
        # Top games visualization
        top_games = games_data.nlargest(10, 'total_rounds')
        fig_top = px.bar(
            top_games, 
            x='item_title', 
            y='total_rounds',
            title="Top 10 Games by Total Rounds"
        )
        st.plotly_chart(fig_top, use_container_width=True)
        
        # Games distribution scatter plot
        fig_scatter = px.scatter(
            games_data, 
            x='total_rounds', 
            y='median_bet',
            hover_data=['item_title'],
            title="Games Distribution by Rounds and Median Bet"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Detailed games table
        st.subheader("Detailed Games Statistics")
        try:
            format_dict = {
                'total_rounds': '{:,.0f}',
                'median_bet': 'R$ {:,.2f}'
            }
            st.dataframe(
                games_data.style.format(format_dict),
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"Could not format games table: {str(e)}")
with tabs[4]:
    deposit_subtabs = st.tabs([
        "Overview",
        "Quick Deposits", 
        "Regular Deposits", 
        "Withdrawals",
        "Lead Processors"
    ])
    
with deposit_subtabs[0]:  # Overview tab
    # Load all required data at the start
    quick_deposits_data = safe_get_data(data, 'quick_deposits')
    regular_deposits_data = safe_get_data(data, 'regular_deposits')
    withdrawals_data = safe_get_data(data, 'withdrawals')
    
    # Top metrics section
    col1, col2, col3 = st.columns(3)
    
    # Quick Deposits with proper error handling
    if validate_dataframe(quick_deposits_data, ['countusers', 'sumdeposit'], "Quick Deposits"):
        with col1:
            quick_deposit_amount = safe_value(quick_deposits_data, 'sumdeposit')
            quick_deposit_users = safe_value(quick_deposits_data, 'countusers')
            st.metric(
                "Quick Deposits",
                f"R$ {quick_deposit_amount:,.2f}",
                f"{quick_deposit_users} users"
            )

    # Regular Deposits
    if validate_dataframe(regular_deposits_data, ['countusers', 'sumdeposit'], "Regular Deposits"):
        with col2:
            st.metric(
                "Regular Deposits",
                f"R$ {regular_deposits_data['sumdeposit'].iloc[0]:,.2f}",
                f"{regular_deposits_data['countusers'].iloc[0]} users"
            )
    
    # Withdrawals
    if validate_dataframe(withdrawals_data, ['countusers', 'sumdeposit'], "Withdrawals"):
        with col3:
            st.metric(
                "Total Withdrawals",
                f"R$ {withdrawals_data['sumdeposit'].iloc[0]:,.2f}",
                f"{withdrawals_data['countusers'].iloc[0]} users"
            )

    # New visualizations section
    st.markdown("### Deposit Trends and Distribution")
    
    # Create deposits trend chart
    if not quick_deposits_data.empty and not regular_deposits_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily trend comparison chart
            quick_trend = safe_get_trend_data(quick_deposits_data)
            regular_trend = safe_get_trend_data(regular_deposits_data)
            
            if not quick_trend.empty and not regular_trend.empty:
                combined_df = pd.DataFrame({
                    'date': quick_trend['date'],
                    'Quick Deposits': quick_trend['amount'],
                    'Regular Deposits': regular_trend['amount']
                })
                
                fig_trend = px.line(
                    combined_df,
                    x='date',
                    y=['Quick Deposits', 'Regular Deposits'],
                    title="Daily Deposits Trend",
                    labels={'value': 'Amount (R$)', 'date': 'Date'}
                )
                fig_trend.update_layout(
                    yaxis_title="Amount (R$)",
                    legend_title="Deposit Type",
                    height=400
                )
                st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # Distribution pie chart
            total_quick = quick_deposits_data['sumdeposit'].sum()
            total_regular = regular_deposits_data['sumdeposit'].sum()
            total_withdrawals = withdrawals_data['sumdeposit'].sum() if not withdrawals_data.empty else 0
            
            pie_data = pd.DataFrame({
                'Type': ['Quick Deposits', 'Regular Deposits', 'Withdrawals'],
                'Amount': [total_quick, total_regular, total_withdrawals]
            })
            
            fig_pie = px.pie(
                pie_data,
                values='Amount',
                names='Type',
                title="Transaction Distribution"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

    # Original Transaction Overview table
    overview_stats = pd.DataFrame({
        'Type': ['Quick Deposits', 'Regular Deposits', 'Withdrawals'],
        'Users': [
            quick_deposits_data['countusers'].iloc[0] if not quick_deposits_data.empty else 0,
            regular_deposits_data['countusers'].iloc[0] if not regular_deposits_data.empty else 0,
            withdrawals_data['countusers'].iloc[0] if not withdrawals_data.empty else 0
        ],
        'Total Amount': [
            quick_deposits_data['sumdeposit'].sum() if not quick_deposits_data.empty else 0,
            regular_deposits_data['sumdeposit'].sum() if not regular_deposits_data.empty else 0,
            withdrawals_data['sumdeposit'].sum() if not withdrawals_data.empty else 0
        ]
    })

    st.markdown("### Transaction Overview")
    create_detailed_statistics(
        overview_stats,
        ['Type', 'Users', 'Total Amount'],
        "",  # Empty string because we already have the header above
        {
            'Total Amount': 'R$ {:,.2f}',
            'Users': '{:,}'
        }
    )

    # Quick Deposits Details
    with deposit_subtabs[2]:  # Now index 2 for Quick Deposits
        if validate_dataframe(quick_deposits_data, ['countusers', 'countdeposits', 'avgdeposit', 'sumdeposit'], "Quick Deposits Detail"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Quick Deposits", f"{quick_deposits_data['countdeposits'].iloc[0]:,}")
            with col2:
                avg_deposit = quick_deposits_data['sumdeposit'].mean() / quick_deposits_data['countdeposits'].mean()
                st.metric("Average Quick Deposit", f"R$ {avg_deposit:,.2f}")
            with col3:
                st.metric("Unique Users", f"{quick_deposits_data['countusers'].iloc[0]:,}")
            
            # Add detailed quick deposits statistics
            create_detailed_statistics(
                quick_deposits_data,
                ['transactiondate', 'countdeposits', 'sumdeposit', 'avgdeposit'],
                "Quick Deposits Detailed Statistics",
                {
                    'sumdeposit': 'R$ {:,.2f}',
                    'avgdeposit': 'R$ {:,.2f}',
                    'countdeposits': '{:,}'
                }
            )

    # Regular Deposits Details
    with deposit_subtabs[2]:
        if validate_dataframe(regular_deposits_data, ['countusers', 'countdeposits', 'avgdeposit', 'sumdeposit'], "Regular Deposits Detail"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Regular Deposits", regular_deposits_data['countdeposits'].iloc[0])
            with col2:
                st.metric("Average Regular Deposit", f"R$ {regular_deposits_data['avgdeposit'].iloc[0]:,.2f}")
            with col3:
                st.metric("Unique Users", regular_deposits_data['countusers'].iloc[0])
            
            # Add detailed regular deposits statistics
            create_detailed_statistics(
                regular_deposits_data,
                ['transactiondate', 'countdeposits', 'sumdeposit', 'avgdeposit'],
                "Regular Deposits Detailed Statistics",
                {
                    'sumdeposit': 'R$ {:,.2f}',
                    'avgdeposit': 'R$ {:,.2f}',
                    'countdeposits': '{:,}'
                }
            )
    with deposit_subtabs[3]:  # Withdrawals tab
        if validate_dataframe(withdrawals_data, ['countusers', 'countdeposits', 'avgdeposit', 'sumdeposit'], "Withdrawals Detail"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Withdrawals", withdrawals_data['countdeposits'].iloc[0])
            with col2:
                st.metric("Average Withdrawal", f"R$ {withdrawals_data['avgdeposit'].iloc[0]:,.2f}")
            with col3:
                st.metric("Unique Users", withdrawals_data['countusers'].iloc[0])
            
            # Add detailed withdrawals statistics
            create_detailed_statistics(
                withdrawals_data,
                ['countdeposits', 'sumdeposit', 'avgdeposit'],
                "Withdrawals Detailed Statistics",
                {
                    'sumdeposit': 'R$ {:,.2f}',
                    'avgdeposit': 'R$ {:,.2f}',
                    'countdeposits': '{:,}'
                }
            )
    with deposit_subtabs[4]:  # Lead Processors tab
        lead_processors_data = safe_get_data(data, 'lead_processors_deposits')
        if validate_dataframe(lead_processors_data, ['processor', 'deposit_count', 'deposit_amount', 'average_deposit', 'pct'], "Lead Processors"):
            # Top metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Processors", 
                    len(lead_processors_data['processor'].unique())
                )
            with col2:
                st.metric(
                    "Total Deposits", 
                    f"R$ {lead_processors_data['deposit_amount'].sum():,.2f}"
                )
            with col3:
                avg_deposit = lead_processors_data['average_deposit'].mean()
                st.metric(
                    "Average Deposit",
                    f"R$ {avg_deposit:,.2f}"
                )
                
            # Processors chart
            fig_processors = px.bar(
                lead_processors_data,
                x='processor',
                y='deposit_amount',
                title="Deposits by Processor",
                color='pct',
                labels={'deposit_amount': 'Total Deposits (R$)', 'pct': 'Percentage (%)'}
            )
            st.plotly_chart(fig_processors, use_container_width=True)
            
            # Distribution pie chart
            fig_dist = px.pie(
                lead_processors_data,
                values='deposit_amount',
                names='processor',
                title="Distribution of Deposits by Processor"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Detailed statistics table
            create_detailed_statistics(
                lead_processors_data,
                ['processor', 'deposit_count', 'deposit_amount', 'average_deposit', 'pct', 'biggest_deposit'],
                "Detailed Processor Statistics",
                {
                    'deposit_amount': 'R$ {:,.2f}',
                    'average_deposit': 'R$ {:,.2f}',
                    'biggest_deposit': 'R$ {:,.2f}',
                    'pct': '{:.1f}%',
                    'deposit_count': '{:,}'
                }
            )
# Game provider         
with tabs[5]:
    providers_data = safe_get_data(data, 'game_providers')
    if validate_dataframe(providers_data, ['applicationtype_name', 'sessions_count', 'rounds_count', 'total_bet', 'avg_bet', 'median_bet'], "Game Providers"):
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_sessions = safe_value(providers_data, 'sessions_count', 0)
            st.metric("Total Sessions", f"{total_sessions:,}")
        with col2:
            st.metric("Total Rounds", f"{providers_data['rounds_count'].sum():,}")
        with col3:
            st.metric("Total Bet Amount", f"R$ {providers_data['total_bet'].sum():,.2f}")
        
        # Provider Performance Charts
        st.subheader("Provider Performance")
        
        # Sessions by Provider
        fig_provider_sessions = px.bar(
            providers_data,
            x='applicationtype_name',
            y=['sessions_count', 'rounds_count'],
            title="Sessions and Rounds by Provider",
            barmode='group',
            labels={
                'applicationtype_name': 'Provider',
                'sessions_count': 'Sessions',
                'rounds_count': 'Rounds'
            }
        )
        st.plotly_chart(fig_provider_sessions, use_container_width=True)
        
        # Average Bet by Provider
        fig_provider_bets = px.bar(
            providers_data,
            x='applicationtype_name',
            y=['avg_bet', 'median_bet'],
            title="Average and Median Bet by Provider",
            barmode='group',
            labels={
                'applicationtype_name': 'Provider',
                'avg_bet': 'Average Bet (R$)',
                'median_bet': 'Median Bet (R$)'
            }
        )
        st.plotly_chart(fig_provider_bets, use_container_width=True)
        
        # Rename columns for detailed statistics
        providers_data = providers_data.rename(columns={
            'applicationtype_name': 'Provider',
            'sessions_count': 'Sessions',
            'rounds_count': 'Rounds',
            'total_bet': 'Total Bet',
            'avg_bet': 'Average Bet',
            'median_bet': 'Median Bet'
        })
        
        # Detailed statistics table
        create_detailed_statistics(
            providers_data,
            ['Provider', 'Sessions', 'Rounds', 'Total Bet', 'Average Bet', 'Median Bet'],
            "Game Provider Statistics",
            {
                'Total Bet': 'R$ {:,.2f}',
                'Average Bet': 'R$ {:,.2f}',
                'Median Bet': 'R$ {:,.2f}',
                'Sessions': '{:,}',
                'Rounds': '{:,}'
            }
        )
# Session Analytics Tab
with tabs[6]:
    try:
        login_data = pd.read_csv('login_duration.csv')
        login_data['login_time'] = pd.to_datetime(login_data['login_time'])
        
        # Calculate daily metrics
        daily_sessions = login_data.groupby(login_data['login_time'].dt.date).agg({
            'duration_minutes': ['count', 'mean', 'median']
        }).reset_index()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sessions", f"{len(login_data):,}")
        with col2:
            st.metric("Average Duration", f"{login_data['duration_minutes'].mean():.1f} min")
        with col3:
            st.metric("Median Duration", f"{login_data['duration_minutes'].median():.1f} min")
            
        # Display trend chart
        fig_sessions = px.line(
            daily_sessions,
            x='login_time',
            y=('duration_minutes', 'mean'),
            title="Daily Average Session Duration"
        )
        st.plotly_chart(fig_sessions, use_container_width=True)
            # Distribution visualization
        fig_dist = px.histogram(
                login_data, 
                x='duration_minutes',
                nbins=30,
                title="Distribution of Session Durations",
                labels={'duration_minutes': 'Duration (minutes)'}
            )
        fig_dist.update_layout(
                showlegend=False,
                xaxis_title="Session Duration (minutes)",
                yaxis_title="Frequency"
            )
        st.plotly_chart(fig_dist, use_container_width=True)
            
            # Daily session trend
        daily_sessions = login_data.groupby(login_data['login_time'].dt.date).agg({
                'duration_minutes': ['count', 'mean', 'median']
            }).reset_index()
        daily_sessions.columns = ['date', 'session_count', 'avg_duration', 'median_duration']
            
        fig_trend = px.line(
                daily_sessions,
                x='date',
                y=['avg_duration', 'median_duration'],
                title="Session Duration Trend",
                labels={
                    'value': 'Duration (minutes)',
                    'date': 'Date',
                    'variable': 'Metric'
                }
            )
        st.plotly_chart(fig_trend, use_container_width=True)

            # Detailed statistics table
        st.subheader("Session Duration Statistics")
        st.dataframe(
                daily_sessions.style.format({
                    'session_count': '{:,}',
                    'avg_duration': '{:.1f} min',
                    'median_duration': '{:.1f} min'
                }),
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Error loading session data: {str(e)}")
        st.info("Please check if the login_duration.csv file is present in the correct location.")
def create_forecast_data(data_dict, metric_type):
    try:
        if not data_dict or not metric_type:
            return pd.DataFrame(), None, None
            
        if metric_type == "Deposits":
            quick_df = safe_get_data(data_dict, 'quick_deposits')
            regular_df = safe_get_data(data_dict, 'regular_deposits')
            
            if not quick_df.empty and not regular_df.empty:
                df = pd.concat([
                    quick_df.assign(type='Quick'),
                    regular_df.assign(type='Regular')
                ])
                df = df.groupby('transactiondate')['sumdeposit'].sum().reset_index()
                date_col = 'transactiondate'
                value_col = 'sumdeposit'
            else:
                return pd.DataFrame(), None, None
                
        elif metric_type == "Game Rounds":
            df = safe_get_data(data_dict, 'popular_games')
            if not df.empty:
                df = df.groupby('opendate')['total_rounds'].sum().reset_index()
                date_col = 'opendate'
                value_col = 'total_rounds'
            else:
                return pd.DataFrame(), None, None
        else:
            df = safe_get_data(data_dict, 'registrations')
            if not df.empty:
                df = df.groupby(pd.to_datetime(df['opendate']).dt.date).size().reset_index()
                df.columns = ['opendate', 'count']
                date_col = 'opendate'
                value_col = 'count'
            else:
                return pd.DataFrame(), None, None
            
        df[date_col] = pd.to_datetime(df[date_col])
        return df, date_col, value_col
        
    except Exception as e:
        st.error(f"Error creating forecast data: {str(e)}")
        return pd.DataFrame(), None, None

def create_forecast(data, date_col, value_col, periods=30):
    try:
        if data.empty or not date_col or not value_col:
            return None
            
        df = pd.DataFrame({'ds': data[date_col], 'y': data[value_col]})
        
        model = Prophet(
            changepoint_prior_scale=0.15,
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative'
        )
        
        model.fit(df)
        future = model.make_future_dataframe(periods=periods)
        return model.predict(future)
        
    except Exception as e:
        st.error(f"Error creating forecast: {str(e)}")
        return None
# Forecasting Tab
with tabs[7]:
    st.markdown("### Forecast Analysis")
    forecast_periods = st.slider("Forecast Periods (Days)", 7, 90, 30)
    
    # Generate forecasts for all metrics
    for metric in ["Registrations", "Deposits", "Game Rounds"]:
        df, date_col, value_col = create_forecast_data(data, metric)
        
        if not df.empty and date_col and value_col:
            forecast = create_forecast(df, date_col, value_col, periods=forecast_periods)
            
            if forecast is not None:
                st.subheader(f"{metric} Forecast")
                
                fig_forecast = go.Figure()
                
                # Add actual values
                fig_forecast.add_trace(go.Scatter(
                    x=df[date_col],
                    y=df[value_col],
                    name="Actual",
                    mode="lines+markers"
                ))
                
                # Add forecast values (ensure non-negative)
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=np.maximum(forecast['yhat'], 0),  # Ensure non-negative
                    name="Forecast",
                    mode="lines",
                    line=dict(dash='dash')
                ))
                
                # Add confidence intervals (ensure non-negative)
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=np.maximum(forecast['yhat_upper'], 0),
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,100,255,0.2)',
                    name='Upper Bound'
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=np.maximum(forecast['yhat_lower'], 0),
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,100,255,0.2)',
                    name='Lower Bound'
                ))
                
                fig_forecast.update_layout(
                    title=f"{metric} Forecast",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
# Sidebar download options
st.sidebar.markdown("---")
download_option = st.sidebar.selectbox(
    "Select Data to Download",
    list(data.keys())
)
if st.sidebar.button("Download Data"):
    csv = data[download_option].to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{download_option}.csv",
        mime="text/csv"
    )
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()
    
# Debug mode toggle
st.sidebar.title("Debug Mode")
if st.sidebar.button("Toggle Debug Mode"):
    toggle_debug_mode()