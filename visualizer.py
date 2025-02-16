import os
import pandas as pd

pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def load_csv_files(folder_path):
    """
    Load all CSV files in a folder into a single DataFrame.
    """
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = [pd.read_csv(file) for file in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def get_base_currency(pair):
    """
    Extract the base currency from the trading pair by considering
    EUR, USDT, and FDUSD as quote currencies.
    """
    quote_currencies = ['EUR', 'USDT', 'FDUSD']
    for quote in quote_currencies:
        if pair.endswith(quote):
            return pair.replace(quote, '')
    raise ValueError(f"Unable to determine base currency for pair: {pair}")


def process_orders(df):
    """
    Process the orders to calculate open and closed positions.
    """
    # Clean and prepare the data
    df['Price'] = df['Price'].astype(float)
    df['Executed'] = df['Executed'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
    df['Amount'] = df['Amount'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
    df['Date(UTC)'] = pd.to_datetime(df['Date(UTC)'])
    df['Base Currency'] = df['Pair'].apply(get_base_currency)
    
    # Sort by date to process orders chronologically
    df = df.sort_values(by='Date(UTC)').reset_index(drop=True)

    open_positions = {}
    closed_positions = []
    total_profit_usdt = 0
    total_profit_percentage = 0

    for _, row in df.iterrows():
        base_currency = row['Base Currency']
        pair = row['Pair']
        side = row['Side']
        price = row['Price']
        amount = row['Executed']
        date = row['Date(UTC)']

        if side == 'BUY':
            # Add to open positions
            if base_currency not in open_positions:
                open_positions[base_currency] = []
            open_positions[base_currency].append({'price': price, 'amount': amount, 'quote': pair.replace(get_base_currency(pair),''), 'date': date})
        elif side == 'SELL':
            # Process sell orders to close positions
            if base_currency in open_positions and open_positions[base_currency]:
                remaining_to_sell = amount
                while remaining_to_sell > 0 and open_positions[base_currency]:
                    open_order = open_positions[base_currency][0]
                    if open_order['amount'] <= remaining_to_sell:
                        # Fully close this open order
                        closed_amount = open_order['amount']
                        profit_loss_usdt = closed_amount * (price - open_order['price'])
                        profit_loss_percentage = (price - open_order['price']) / open_order['price'] * 100
                        closed_positions.append({
                            'Base Currency': base_currency,
                            'Open Price': open_order['price'],
                            'Close Price': price,
                            'Amount': closed_amount,
                            'Open Date': open_order['date'],
                            'Close Date': date,
                            'Quote Currency (Open)': open_order['quote'],
                            'Quote Currency (Close)': pair.replace(get_base_currency(pair),''),
                            'Profit/Loss USDT': profit_loss_usdt,
                            'Profit/Loss %': profit_loss_percentage
                        })
                        total_profit_usdt += profit_loss_usdt
                        total_profit_percentage += profit_loss_percentage
                        remaining_to_sell -= open_order['amount']
                        open_positions[base_currency].pop(0)  # Remove closed order
                    else:
                        # Partially close this open order
                        closed_amount = remaining_to_sell
                        profit_loss_usdt = closed_amount * (price - open_order['price'])
                        profit_loss_percentage = (price - open_order['price']) / open_order['price'] * 100
                        closed_positions.append({
                            'Base Currency': base_currency,
                            'Open Price': open_order['price'],
                            'Close Price': price,
                            'Amount': closed_amount,
                            'Open Date': open_order['date'],
                            'Close Date': date,
                            'Quote Currency (Open)': open_order['quote'],
                            'Quote Currency (Close)': pair[-4:],
                            'Profit/Loss USDT': profit_loss_usdt,
                            'Profit/Loss %': profit_loss_percentage
                        })
                        total_profit_usdt += profit_loss_usdt
                        total_profit_percentage += profit_loss_percentage
                        open_order['amount'] -= remaining_to_sell
                        remaining_to_sell = 0

    # Calculate mean price for open positions
    open_positions_summary = []
    for base_currency, positions in open_positions.items():
        total_amount = sum(position['amount'] for position in positions)
        mean_price = sum(position['price'] * position['amount'] for position in positions) / total_amount if total_amount > 0 else 0
        open_positions_summary.append({
            'Base Currency': base_currency,
            'Mean Price': mean_price,
            'Total Amount': total_amount,
            'Total Amount USD': total_amount*mean_price
        })

    return pd.DataFrame(open_positions_summary), pd.DataFrame(closed_positions), total_profit_usdt, total_profit_percentage

def main():
    folder_path = "data"
    try:
        df = load_csv_files(folder_path)
        open_positions_df, closed_positions_df, total_profit_usdt, total_profit_percentage = process_orders(df)

        print("\nOpen Positions:")
        print(open_positions_df.to_string(index=False))

        # Print column names with fixed-width formatting
        column_width = 20  # Set the width for each column
        print("\nClosed Positions:")
        print("".join(f"{col:<{column_width}}" for col in closed_positions_df.columns))

        # Print each row with fixed-width formatting
        for index, row in closed_positions_df.iterrows():
            print("".join(f"{str(value):<{column_width}}" for value in row.values))

        print(f"\nTotal Profit: {total_profit_usdt:.2f} USDT")
        print(f"Total Profit Percentage: {total_profit_percentage:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
