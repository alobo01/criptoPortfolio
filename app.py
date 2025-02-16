import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# -----------------------------
# Utility functions
# -----------------------------

def load_csv_files(folder_path):
    """
    Load and combine CSV files from a given folder path.
    """
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = [pd.read_csv(file) for file in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def get_quote_currency(pair):
    """
    Attempt to extract the quote currency from the pair.
    Extend this list as needed for your trading pairs.
    """
    quote_currencies = ['EUR', 'USDT', 'FDUSD']
    for quote in quote_currencies:
        if pair.endswith(quote):
            return quote
    return 'UNKNOWN'  # fallback if not found

def get_base_currency(pair):
    """
    Extract the base currency by removing the known quote currency from the end of the pair.
    """
    quote = get_quote_currency(pair)
    if quote != 'UNKNOWN':
        return pair.replace(quote, '')
    else:
        # If we cannot find a known quote, fallback to the beginning of the pair.
        # This is a simplistic fallback, adapt to your specific usage if needed.
        return pair

def process_orders(df):
    """
    Process orders using a FIFO approach for each base currency.
    Returns:
        closed_positions_df: DataFrame of closed positions
        open_positions_dict: Dictionary of currently open positions
        orphan_sell_orders: List of SELL orders that occurred without a matching BUY
    """
    # Convert necessary columns
    df['Price'] = df['Price'].astype(float)
    df['Executed'] = df['Executed'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
    df['Amount'] = df['Amount'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
    df['Date(UTC)'] = pd.to_datetime(df['Date(UTC)'])
    df['Quote Currency'] = df['Pair'].apply(get_quote_currency)
    df['Base Currency']  = df['Pair'].apply(get_base_currency)

    # Sort by date ascending
    df = df.sort_values(by='Date(UTC)').reset_index(drop=True)

    open_positions = {}   # { base_currency: [ {price, amount, date}, ... ], ... }
    closed_positions = []
    orphan_sell_orders = []  # To store SELL orders with no prior BUY

    for _, row in df.iterrows():
        base_currency = row['Base Currency']
        side = row['Side']
        price = row['Price']
        amount = row['Executed']
        date = row['Date(UTC)']

        if side == 'BUY':
            # Open a new position (or add to open positions)
            if base_currency not in open_positions:
                open_positions[base_currency] = []
            open_positions[base_currency].append({
                'price': price,
                'amount': amount,
                'date': date
            })
        elif side == 'SELL':
            # If there is no open position, add the order to orphan list instead of crashing
            if base_currency not in open_positions or not open_positions[base_currency]:
                orphan_sell_orders.append(row)
                continue

            # Close existing positions using FIFO
            remaining_to_sell = amount
            while remaining_to_sell > 0 and open_positions[base_currency]:
                open_order = open_positions[base_currency][0]
                if open_order['amount'] <= remaining_to_sell:
                    closed_amount = open_order['amount']
                    profit_loss_usdt = closed_amount * (price - open_order['price'])
                    closed_positions.append({
                        'Base Currency': base_currency,
                        'Open Price': open_order['price'],
                        'Close Price': price,
                        'Amount': closed_amount,
                        'Open Date': open_order['date'],
                        'Close Date': date,
                        'Profit/Loss USDT': profit_loss_usdt
                    })
                    remaining_to_sell -= open_order['amount']
                    open_positions[base_currency].pop(0)
                else:
                    closed_amount = remaining_to_sell
                    profit_loss_usdt = closed_amount * (price - open_order['price'])
                    closed_positions.append({
                        'Base Currency': base_currency,
                        'Open Price': open_order['price'],
                        'Close Price': price,
                        'Amount': closed_amount,
                        'Open Date': open_order['date'],
                        'Close Date': date,
                        'Profit/Loss USDT': profit_loss_usdt
                    })
                    open_order['amount'] -= remaining_to_sell
                    remaining_to_sell = 0

    closed_positions_df = pd.DataFrame(closed_positions)
    return closed_positions_df, open_positions, orphan_sell_orders


def open_positions_to_df(open_positions_dict):
    """
    Convert the open_positions dictionary into a DataFrame for display.
    """
    records = []
    for base_cur, positions_list in open_positions_dict.items():
        for pos in positions_list:
            records.append({
                'Base Currency': base_cur,
                'Open Price': pos['price'],
                'Amount': pos['amount'],
                'Open Date': pos['date']
            })
    return pd.DataFrame(records)


def calculate_statistics(closed_positions_df):
    """
    Calculate various interesting statistics for the closed positions.
    Returns a dictionary of stats.
    """
    stats = {}
    if closed_positions_df.empty:
        return stats

    # Basic stats
    stats['Total Closed Trades'] = len(closed_positions_df)
    total_pnl = closed_positions_df['Profit/Loss USDT'].sum()
    stats['Total PnL (USDT)'] = total_pnl

    # Compute per-trade returns in percentage:
    # Profit/Loss % = (Profit/Loss USDT) / (Open Price * Amount) * 100
    # Handle division-by-zero carefully:
    cost_basis = closed_positions_df['Open Price'] * closed_positions_df['Amount']
    closed_positions_df['Profit/Loss %'] = closed_positions_df['Profit/Loss USDT'] / cost_basis.replace(0, float('nan')) * 100

    # Win / Loss stats
    winning_trades = closed_positions_df[closed_positions_df['Profit/Loss USDT'] > 0]
    losing_trades = closed_positions_df[closed_positions_df['Profit/Loss USDT'] < 0]
    stats['Winning Trades'] = len(winning_trades)
    stats['Losing Trades'] = len(losing_trades)
    stats['Win Rate'] = (len(winning_trades) / len(closed_positions_df) * 100) if len(closed_positions_df) > 0 else 0

    # Average Profit/Loss in USDT
    stats['Average PnL (USDT)'] = closed_positions_df['Profit/Loss USDT'].mean()

    # Average Profit/Loss in %
    stats['Average PnL (%)'] = closed_positions_df['Profit/Loss %'].mean()

    # By month - number of operations
    closed_positions_df['Month'] = closed_positions_df['Close Date'].dt.to_period('M')
    monthly_counts = closed_positions_df.groupby('Month')['Base Currency'].count()
    stats['Operations per Month'] = monthly_counts.to_dict()

    return stats, closed_positions_df


def plot_monthly_percentage_bar(closed_positions_df):
    """
    Create a barplot of the sum (or average) percentage of profit per month.
    """
    if closed_positions_df.empty:
        st.write("No closed positions to plot.")
        return

    # Group by Month
    monthly_pnl_df = closed_positions_df.copy()
    monthly_pnl_df['Month'] = monthly_pnl_df['Close Date'].dt.to_period('M')

    # Sum of PnL(%) can be misleading if not weighted. 
    # Usually we want Weighted Average of returns or an overall ratio:
    # Weighted approach: sum of PnL USDT / sum of cost basis * 100 for each month.
    monthly_pnl_df['Cost Basis'] = monthly_pnl_df['Open Price'] * monthly_pnl_df['Amount']
    monthly_grouped = monthly_pnl_df.groupby('Month').agg({
        'Profit/Loss USDT': 'sum',
        'Cost Basis': 'sum'
    })
    monthly_grouped['Monthly Profit %'] = monthly_grouped.apply(
        lambda row: (row['Profit/Loss USDT'] / row['Cost Basis'] * 100) if row['Cost Basis'] != 0 else 0,
        axis=1
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    monthly_grouped['Monthly Profit %'].plot(kind='bar', ax=ax)
    ax.set_title('Monthly Percentage Profit')
    ax.set_xlabel('Month')
    ax.set_ylabel('Profit %')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.title("Dynamic Portfolio Viewer")
    st.sidebar.header("Upload CSV File")

    # Upload CSV file
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        st.write(f"**Number of rows in the uploaded CSV:** {df.shape[0]}")
        st.subheader("Data Preview")
        st.write(df.head())

        # Process orders and capture orphan SELL orders
        closed_positions_df, open_positions_dict, orphan_sell_orders = process_orders(df)
        
        # Show Open Positions
        open_positions_df = open_positions_to_df(open_positions_dict)
        st.subheader("Open Positions")
        if open_positions_df.empty:
            st.write("No open positions.")
        else:
            st.write(open_positions_df)

        # Show Closed Positions
        st.subheader("Closed Positions")
        if closed_positions_df.empty:
            st.write("No closed positions yet.")
        else:
            st.write(closed_positions_df.head(50))

        # ----- Missing BUY Data for SELL orders -----
        if orphan_sell_orders:
            st.subheader("Missing BUY Data for SELL Orders")
            st.info("The following SELL order(s) have no matching BUY order. Please provide the missing BUY data:")

            # Loop through each orphan SELL order and display a form
            for i, sell_order in enumerate(orphan_sell_orders):
                st.write(f"**SELL Order #{i+1}:**")
                st.write(f"Base Currency: {sell_order['Base Currency']}")
                st.write(f"Sell Price: {sell_order['Price']}")
                st.write(f"Amount: {sell_order['Executed']}")
                st.write(f"Sell Date: {sell_order['Date(UTC)']}")
                
                with st.form(key=f"missing_buy_form_{i}"):
                    missing_buy_price = st.number_input("Enter the missing BUY price", min_value=0.0, step=0.0001, format="%.4f")
                    missing_buy_date = st.date_input("Enter the missing BUY date")
                    submitted = st.form_submit_button("Submit Missing BUY Data")
                    
                    if submitted:
                        profit_loss_usdt = sell_order['Executed'] * (sell_order['Price'] - missing_buy_price)
                        manual_closed_op = {
                            'Base Currency': sell_order['Base Currency'],
                            'Open Price': missing_buy_price,
                            'Close Price': sell_order['Price'],
                            'Amount': sell_order['Executed'],
                            'Open Date': pd.Timestamp(missing_buy_date),
                            'Close Date': pd.Timestamp(sell_order['Date(UTC)']),
                            'Profit/Loss USDT': profit_loss_usdt
                        }
                        if "manual_missing_operations" not in st.session_state:
                            st.session_state.manual_missing_operations = []
                        st.session_state.manual_missing_operations.append(manual_closed_op)
                        st.success("Missing BUY data submitted for this SELL order!")

        # Display any manually submitted missing BUY data
        if "manual_missing_operations" in st.session_state and st.session_state.manual_missing_operations:
            st.subheader("Manually Added Closed Operations")
            manual_closed_df = pd.DataFrame(st.session_state.manual_missing_operations)
            st.write(manual_closed_df)

        # Continue with displaying statistics, plots, etc.
        stats, closed_positions_df = calculate_statistics(closed_positions_df)
        st.subheader("Interesting Statistics")
        if stats:
            for key, value in stats.items():
                if key != 'Operations per Month':
                    st.write(f"**{key}:** {value}")
                else:
                    st.write("**Operations per Month:**")
                    st.write(value)
        else:
            st.write("No statistics available.")

        # Profit per month, trimester, and year (Absolute and Percentage)
        # First ensure we have the 'Profit/Loss %' in closed_positions_df
        if 'Profit/Loss %' not in closed_positions_df.columns:
            # In case empty or something unexpected
            closed_positions_df['Profit/Loss %'] = 0.0

        closed_positions_df['Month'] = closed_positions_df['Close Date'].dt.to_period('M')
        closed_positions_df['Trimester'] = closed_positions_df['Close Date'].dt.to_period('Q')
        closed_positions_df['Year'] = closed_positions_df['Close Date'].dt.year

        time_period = st.selectbox("Select Time Period", ["Month", "Trimester", "Year"])
        profit_type = st.selectbox("Select Profit Type", ["Absolute", "Percentage"])

        # Map time period to the actual column
        if time_period == "Month":
            time_column = "Month"
        elif time_period == "Trimester":
            time_column = "Trimester"
        else:
            time_column = "Year"

        # Map profit type
        if profit_type == "Absolute":
            profit_column = "Profit/Loss USDT"
        else:
            profit_column = "Profit/Loss %"

        # Group by time period
        profit_summary = (closed_positions_df
                          .groupby(time_column)[profit_column]
                          .agg(['sum', 'mean', 'std'])
                          .rename(columns={'sum': 'Sum', 'mean': 'Mean', 'std': 'Std Dev'}))

        st.subheader(f"{profit_type} Profit per {time_period}")
        st.write(profit_summary)

        # Histogram of chosen profit type
        fig, ax = plt.subplots()
        closed_positions_df[profit_column].hist(bins=20, ax=ax)
        ax.set_title(f"{profit_type} Profit Distribution")
        ax.set_xlabel(f"{profit_type} Profit (USDT or %)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Line Plot of Cumulative Profit
        closed_positions_df['Cumulative Profit'] = closed_positions_df[profit_column].cumsum()
        fig, ax = plt.subplots()
        closed_positions_df.plot(x='Close Date', y='Cumulative Profit', ax=ax)
        ax.set_title(f"Cumulative {profit_type} Profit")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Cumulative {profit_type} Profit (USDT or %)")
        st.pyplot(fig)

        # Barplot: Percentage of profit per month
        st.subheader("Monthly Percentage Profit")
        plot_monthly_percentage_bar(closed_positions_df)

if __name__ == "__main__":
    main()
