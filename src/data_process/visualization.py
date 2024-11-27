import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_divergence(df, divergence_df, title='', save_fig=False):
        plt.figure(figsize=(14, 10))
        ax1 = plt.subplot(211)
        ax1.plot(df.index, df['close'], label='Price')
        ax1.set_title(f'{title} Price')
        ax1.legend()

        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(df.index, df['rsi'], label='RSI', color='orange')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='green', linestyle='--')
        ax2.set_title(f'{title} RSI')
        ax2.legend()

        for idx, row in divergence_df.iterrows():
            start_datetime = row['start_datetime']
            end_datetime = idx
            divergence_type = row['divergence']

            # Get the RSI values
            rsi_start = df.loc[start_datetime, 'rsi']
            rsi_end = df.loc[end_datetime, 'rsi']

            # Plot lines on RSI chart
            if divergence_type == 'Bullish Divergence':
                color = 'green'
                marker = '^'
                ax1.plot([start_datetime, end_datetime], [df.loc[start_datetime, 'close'], df.loc[end_datetime, 'close']], color=color, linewidth=2)
                ax1.scatter([end_datetime], [df.loc[end_datetime, 'close']], marker=marker, color=color, s=100)
                ax2.plot([start_datetime, end_datetime], [rsi_start, rsi_end], color=color, linewidth=2)
                ax2.scatter([end_datetime], [rsi_end], marker=marker, color=color, s=100)
            elif divergence_type == 'Bearish Divergence':
                color = 'red'
                marker = 'v'
                ax1.plot([start_datetime, end_datetime], [df.loc[start_datetime, 'close'], df.loc[end_datetime, 'close']], color=color, linewidth=2)
                ax1.scatter([end_datetime], [df.loc[end_datetime, 'close']], marker=marker, color=color, s=100)
                ax2.plot([start_datetime, end_datetime], [rsi_start, rsi_end], color=color, linewidth=2)
                ax2.scatter([end_datetime], [rsi_end], marker=marker, color=color, s=100)

        plt.tight_layout()
        if save_fig:
            plt.savefig(f"{title.replace('/', '_')}.png")
        plt.close()

