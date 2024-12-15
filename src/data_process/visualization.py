import mplfinance as mpf
import pandas as pd
import os, sys

# TODO: There are still problems in following visualziation
# TODO: 1. hard to visualize divergence in RSI. No options to add aline in addplot
# TODO: 2. hard to visualize TP, SL in graph it self. No annoatations are added.

path_splited = os.path.abspath('.').split('rsi_divergence_detector')[0]
PROJECT_PATH = os.path.join(path_splited, 'rsi_divergence_data')
sys.path.append(PROJECT_PATH)


class Visualizer:
    @staticmethod
    def plot_divergence(df, divergence_df, title='', save_fig=False):
        # Ensure 'datetime' is the index
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
        df.index = pd.to_datetime(df.index)  # Ensure index is datetime
        df.sort_index(inplace=True)

        # Prepare the data for mplfinance
        ohlc_df = df[['open', 'high', 'low', 'close', 'volume']]

        # Initialize lists to hold additional plots and lines
        apds = []
        add_lines = {
            'alines': [],
            'colors': [],
            'linewidths': [],
            'linestyle': [],
            # 'panel': []
        }
        hlines = {
            'hlines': [],
            'colors': [],
            'linestyle': [],
            'linewidths': [],
            # 'panel': []
        }

        # Plot RSI in a separate panel (panel=1)
        apds.append(mpf.make_addplot(df['rsi'], panel=1, color='orange', ylabel='RSI'))

        # Add horizontal lines for RSI levels at 70 and 30
        apds.append(mpf.make_addplot([70]*len(df), panel=1, color='grey', linestyle='--', linewidths=0.5))
        apds.append(mpf.make_addplot([30]*len(df), panel=1, color='grey', linestyle='--', linewidths=0.5))

        axes_annotations = []
        i = 0
        # Loop through divergence data and prepare annotations
        for idx, row in divergence_df.iterrows():
            try:
                start_datetime = idx
                end_datetime = pd.to_datetime(row['end_datetime'])
                entry_datetime = pd.to_datetime(row['entry_datetime'])
                previous_peak_datetime = pd.to_datetime(row['previous_peak_datetime'])
                tp = row['TP']
                sl = row['SL']
                divergence_type = row['divergence']

                # Ensure datetimes are in the index
                for dt in [start_datetime, end_datetime, previous_peak_datetime, entry_datetime]:
                    if dt not in df.index:
                        print(f"Datetime {dt} not in DataFrame index. Skipping.")
                        continue

                # Determine marker direction and color
                if divergence_type == 'Bullish Divergence':
                    color = 'g'
                    marker = '^'
                    price_name = 'low'
                    fib_start_value = df.loc[previous_peak_datetime, 'high']
                    fib_end_value = df.loc[end_datetime, 'low']
                elif divergence_type == 'Bearish Divergence':
                    color = 'r'
                    marker = 'v'
                    price_name = 'high'
                    fib_start_value = df.loc[previous_peak_datetime, 'low']
                    fib_end_value = df.loc[end_datetime, 'high']
                else:
                    print(f"Unknown divergence type: {divergence_type}")
                    color = 'b'  # Default color
                    price_name = 'close'  # Default price

                # Add divergence line on price chart using alines
                add_lines['alines'].append([
                    (start_datetime, df.loc[start_datetime, price_name]),
                    (end_datetime, df.loc[end_datetime, price_name])
                ])
                add_lines['colors'].append(color)
                add_lines['linewidths'].append(1.5)
                add_lines['linestyle'].append('--')
                # add_lines['panels'].append(0)  # Main panel

                # Add divergence line on RSI chart using alines
                add_lines['alines'].append([
                    (start_datetime, df.loc[start_datetime, 'rsi']),
                    (end_datetime, df.loc[end_datetime, 'rsi'])
                ])
                add_lines['colors'].append(color)
                add_lines['linewidths'].append(1.5)
                add_lines['linestyle'].append('--')
                # add_lines['panels'].append(1)  # RSI panel

                # Add Fibonacci line on price chart using alines (grey dotted line)
                add_lines['alines'].append([
                    (previous_peak_datetime, fib_start_value),
                    (end_datetime, fib_end_value)
                ])
                add_lines['colors'].append('grey')
                add_lines['linewidths'].append(1)
                add_lines['linestyle'].append(':')
                # add_lines['panels'].append(0)  # Main panel

                # Add TP, SL, and Entry Price lines using hlines
                entry_price = df.loc[entry_datetime, 'open']
                hlines['hlines'].extend([tp, sl, entry_price])
                hlines['colors'].extend(['g', 'r', 'b'])  # Use 'b' for entry price line
                hlines['linestyle'].extend(['-.', '-.', '--'])
                hlines['linewidths'].extend([1, 1, 1])
                # hlines['panels'].extend([0, 0, 0])  # Main panel

                # Add TP, SL, and entry annotations
                axes_annotations.append((entry_datetime, entry_price, f"Entry{i+1}: {entry_price:.2f}", 'blue'))
                axes_annotations.append((end_datetime, tp, f"TP{i+1}: {tp:.2f}", 'green'))
                axes_annotations.append((end_datetime, sl, f"SL{i+1}: {sl:.2f}", 'red'))
                i += 1

            except KeyError as e:
                print(f"KeyError: {e} for divergence at index {idx}. Skipping.")

            # Set up plot arguments
            kwargs = dict(
                type='candle',
                style='yahoo',
                title=title,
                ylabel='Price',
                volume=False,
                addplot=apds,
                panel_ratios=(3, 1),  # Adjust the panel ratio if needed
                figsize=(14, 10),
                tight_layout=True,
                alines=add_lines,
                hlines=hlines,
                xrotation=20,  # Rotate x-axis labels for better readability
            )


        # Plot the chart
        fig, axes = mpf.plot(ohlc_df, **kwargs, returnfig=True)


        # Add annotations for TP, SL, and Entry
        for _, value, text, color in axes_annotations:
            axes[0].annotate(
                text, 
                xy=(_, value),
                xytext=(_, value),
                arrowprops=dict(facecolor=color, shrink=0.05),
                fontsize=8, color=color
            )

        if save_fig:
            fig.savefig(f"{title.replace('/', '_')}.png")
            fig.close()
        else:
            fig.show()


if __name__ == "__main__":
    import sys
    import pandas as pd
    sys.path.append(f"{PROJECT_PATH}")
    df_15m_filter = pd.read_pickle(f"{PROJECT_PATH}/data/df_15m_filter_test.pickle")
    df_15m_filter[-5:]
    divergence_data = pd.read_pickle(f"{PROJECT_PATH}/data/divergence_data")
    divergence_15m = divergence_data['15m']
    divergence_15m[-2:]
    visualizer = Visualizer
    visualizer.plot_divergence(df_15m_filter, divergence_15m[-2:])
