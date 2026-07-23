import pandas as pd
from trading.order_flow_signal import add_ofi_signals, ofi_entry_threshold
from backtest import generate_trades, compute_risk_stats

df = pd.read_parquet("data/kalshi_hf_panel.parquet")  # after rebuild
df = add_ofi_signals(df)  # adds ofi_6h, ofi_24h

for lb in [6, 24]:
    col = f"ofi_{lb}h"
    th = ofi_entry_threshold(df, col)
    for H in [1, 6, 12, 24]:
        trades = generate_trades(df, signal_col=col, horizon_hours=H, entry_threshold=th)
        if trades.empty:
            continue
        stats = compute_risk_stats(trades)
        print(f"{col} H={H}: n_trades={stats['n_trades']}, sharpe={stats['daily_sharpe_annualized']:.3f}, win_rate={stats['win_rate']:.3f}, total_pnl={stats['total_pnl']:.2f}, mean_return={stats['mean_return']:.4f}, daily_vol={stats['daily_vol']:.4f}, max_drawdown={stats['max_drawdown']:.4f}, turnover_per_day={stats['turnover_per_day']:.4f}, total_fees={stats['total_fees']:.2f}, sharpe_reliable={stats['sharpe_reliable']:.4f}")