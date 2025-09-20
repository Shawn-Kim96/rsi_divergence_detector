"""Utility helpers for visualising divergence-based trading logic with finplot.

The functions in this module are intentionally lightweight wrappers around
``finplot`` so that backtest routines can highlight interesting regions without
having to manually deal with common plotting boilerplate.

Example usage
-------------
>>> from visualization.finplot_utils import (
...     create_plot_context,
...     overlay_divergence_signals,
...     highlight_waves,
...     annotate_trade_levels,
...     show_plot,
... )
>>> ctx = create_plot_context(price_dataframe, indicator=price_dataframe["rsi"],
...                           title="BTC / USDT")
>>> overlay_divergence_signals(ctx, divergences)
>>> highlight_waves(ctx, wave_definitions)
>>> annotate_trade_levels(ctx, entry_time, entry_price,
...                       take_profit=tp_price, stop_loss=sl_price)
>>> show_plot()

The helper functions operate on :class:`PlotContext` objects, which store the
``finplot`` axes along with the price data used to render the chart.  All helper
functions keep side-effects local to the plot, so they can be composed freely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import pandas as pd

try:  # pragma: no cover - importing finplot requires GUI libraries in CI
    import finplot as _fplt
except Exception as exc:  # noqa: BLE001 - provide context for any import failure
    _FINPLOT_IMPORT_ERROR: Optional[Exception] = exc
    _fplt = None  # type: ignore[assignment]
else:
    _FINPLOT_IMPORT_ERROR = None


FpltType = Any
TimestampLike = Union[pd.Timestamp, str]
DivergencePoint = Tuple[pd.Timestamp, float]


@dataclass(slots=True)
class PlotContext:
    """Container describing the finplot objects created for a chart."""

    plot: FpltType
    price_ax: FpltType
    indicator_ax: Optional[FpltType]
    data: pd.DataFrame
    indicator: Optional[pd.Series]


def _require_finplot() -> None:
    """Ensure finplot is available before attempting to draw anything."""

    if _fplt is None:  # pragma: no cover - triggered only in missing GUI setups
        raise RuntimeError(
            "finplot is required for visualization but could not be imported. "
            "Install finplot and its GUI dependencies (PyQt6 and libGL) before "
            "calling visualization helpers."
        ) from _FINPLOT_IMPORT_ERROR


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _normalize_indicator(
    indicator: Union[str, pd.Series, None],
    data: pd.DataFrame,
    indicator_name: Optional[str],
) -> Tuple[Optional[pd.Series], Optional[str]]:
    if indicator is None:
        return None, None
    if isinstance(indicator, str):
        if indicator not in data.columns:
            raise KeyError(f"Indicator column '{indicator}' not found in data.")
        series = data[indicator]
        name = indicator
    elif isinstance(indicator, pd.Series):
        series = indicator.reindex(data.index)
        name = indicator.name or indicator_name or "Indicator"
    else:
        raise TypeError("indicator must be a column name or a pandas Series.")
    if indicator_name:
        name = indicator_name
    return series, name


def _prepare_price_frames(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = data[["open", "high", "low", "close", "volume"]].copy()
    base.reset_index(names="datetime", inplace=True)
    ohlc = base[["datetime", "open", "close", "high", "low"]]
    volume = base[["datetime", "open", "close", "volume"]]
    return base, ohlc, volume


def _color_with_opacity(color: str, opacity: float) -> str:
    opacity = min(max(opacity, 0.0), 1.0)
    alpha = round(opacity * 255)
    return f"{color}{alpha:02x}" if len(color) in {4, 7} else color


def _pick_divergence_color(kind: str) -> str:
    lowered = kind.lower()
    if "bear" in lowered:
        return "#ff1744"
    if "bull" in lowered:
        return "#00c853"
    return "#42a5f5"


def create_plot_context(
    data: pd.DataFrame,
    *,
    indicator: Union[str, pd.Series, None] = None,
    indicator_name: Optional[str] = None,
    title: str | None = None,
    init_zoom_periods: int | float = 200,
) -> PlotContext:
    """Create the base candlestick plot (optionally with an indicator panel).

    Parameters
    ----------
    data:
        DataFrame containing ``open``, ``high``, ``low``, ``close``, and
        ``volume`` columns. The index or a ``datetime`` column is interpreted as
        the candle timestamp.
    indicator:
        Optional indicator series (e.g., RSI) to plot on a secondary panel. It
        can be provided either as a column name from ``data`` or as a standalone
        :class:`pandas.Series`.
    indicator_name:
        Custom legend name for the indicator. When omitted the column/series
        name is used.
    title:
        Window title for the created plot.
    init_zoom_periods:
        Initial zoom window expressed in number of candles.

    Returns
    -------
    PlotContext
        Object describing the created plot and the data associated with it.
    """

    _require_finplot()

    data = _ensure_datetime_index(data)
    indicator_series, indicator_label = _normalize_indicator(
        indicator, data, indicator_name
    )
    rows = 2 if indicator_series is not None else 1
    axes = _fplt.create_plot(title=title or "Trading Chart", rows=rows, init_zoom_periods=init_zoom_periods)
    if isinstance(axes, (list, tuple)):
        price_ax = axes[0]
        indicator_ax = axes[1] if rows > 1 else None
    else:
        price_ax = axes
        indicator_ax = None

    _, ohlc, volume = _prepare_price_frames(data)
    _fplt.candlestick_ochl(ohlc, ax=price_ax)
    _fplt.volume_ocv(volume, ax=price_ax.overlay())

    indicator_ax_final: Optional[FpltType] = None
    if indicator_series is not None:
        indicator_ax_final = indicator_ax or price_ax.overlay()
        _fplt.plot(
            pd.Series(indicator_series.values, index=data.index),
            ax=indicator_ax_final,
            color="#ff9800",
            legend=indicator_label,
        )
        _fplt.add_horizontal_band(70, 70, color="#bdbdbd88", ax=indicator_ax_final)
        _fplt.add_horizontal_band(30, 30, color="#bdbdbd88", ax=indicator_ax_final)

    return PlotContext(
        plot=price_ax.vb.win,
        price_ax=price_ax,
        indicator_ax=indicator_ax_final,
        data=data,
        indicator=indicator_series,
    )


def overlay_divergence_signals(
    context: PlotContext,
    divergences: Iterable[Mapping[str, Any]],
    *,
    indicator_series: Optional[pd.Series] = None,
    marker_size: int = 6,
) -> None:
    """Draw divergence price/indicator lines and pivot markers.

    Each ``divergence`` mapping may contain the following keys:

    ``start`` / ``end``
        Required timestamps representing the divergence endpoints.
    ``kind``
        String categorising the signal (e.g. ``"bullish"`` or ``"bearish"``).
    ``price_points``
        Optional sequence of two ``(timestamp, price)`` tuples. When omitted the
        helper looks up prices from ``context.data`` based on ``kind``.
    ``indicator_points``
        Optional sequence of two ``(timestamp, value)`` tuples for the
        oscillator/indicator panel. Falls back to ``indicator_series`` or the
        indicator stored in the :class:`PlotContext`.
    ``label``
        Optional text annotation rendered near the most recent point.
    ``show_markers``
        Boolean flag controlling whether pivot circles are drawn on the price
        chart.
    """

    _require_finplot()

    data = context.data
    price_ax = context.price_ax
    indicator_ax = context.indicator_ax
    indicator_series = indicator_series or context.indicator

    for divergence in divergences:
        start = pd.to_datetime(divergence["start"])
        end = pd.to_datetime(divergence["end"])
        kind = divergence.get("kind", "divergence")
        color = divergence.get("color", _pick_divergence_color(kind))

        price_points: Sequence[DivergencePoint]
        custom_points = divergence.get("price_points")
        if custom_points:
            price_points = [
                (pd.to_datetime(t), float(p)) for t, p in custom_points  # type: ignore[arg-type]
            ]
        else:
            if "bear" in kind.lower():
                start_price = float(data.loc[start, "high"])
                end_price = float(data.loc[end, "high"])
            else:
                start_price = float(data.loc[start, "low"])
                end_price = float(data.loc[end, "low"])
            price_points = [(start, start_price), (end, end_price)]

        _fplt.add_line(price_points[0], price_points[1], color=color, width=2, style="dash", ax=price_ax)

        if divergence.get("show_markers", True):
            marker_series = pd.Series(
                [price for _, price in price_points],
                index=[time for time, _ in price_points],
            )
            _fplt.plot(marker_series, ax=price_ax, color=color, style="o", width=marker_size)

        indicator_points: Optional[Sequence[DivergencePoint]] = None
        custom_indicator = divergence.get("indicator_points")
        if custom_indicator:
            indicator_points = [
                (pd.to_datetime(t), float(v)) for t, v in custom_indicator  # type: ignore[arg-type]
            ]
        elif indicator_series is not None and indicator_ax is not None:
            indicator_points = [
                (start, float(indicator_series.loc[start])),
                (end, float(indicator_series.loc[end])),
            ]

        if indicator_points and indicator_ax is not None:
            _fplt.add_line(indicator_points[0], indicator_points[1], color=color, width=2, style="dash", ax=indicator_ax)

        label = divergence.get("label")
        if label:
            _fplt.add_text(price_points[-1], label, color=color, anchor=(0, -1), ax=price_ax)


def highlight_waves(
    context: PlotContext,
    waves: Iterable[Union[Tuple[TimestampLike, TimestampLike], Mapping[str, Any]]],
    *,
    default_color: str = "#1e88e5",
    default_opacity: float = 0.15,
) -> None:
    """Highlight price waves by shading their time span on the chart."""

    _require_finplot()

    data = context.data
    price_ax = context.price_ax

    for wave in waves:
        if isinstance(wave, Mapping):
            start = pd.to_datetime(wave["start"])
            end = pd.to_datetime(wave["end"])
            color = wave.get("color", default_color)
            opacity = wave.get("opacity", default_opacity)
            label = wave.get("label")
        else:
            start = pd.to_datetime(wave[0])
            end = pd.to_datetime(wave[1])
            color = default_color
            opacity = default_opacity
            label = None

        band_color = _color_with_opacity(color, opacity)
        _fplt.add_vertical_band(start, end, color=band_color, ax=price_ax)

        if label:
            segment = data.loc[start:end]
            if not segment.empty:
                y_value = float(segment["high"].max())
            else:
                y_value = float(data.loc[end, "high"])
            midpoint = start + (end - start) / 2
            _fplt.add_text((midpoint, y_value), label, color=color, anchor=(0.5, -1), ax=price_ax)


def annotate_trade_levels(
    context: PlotContext,
    entry_time: TimestampLike,
    entry_price: float,
    *,
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
    exit_time: Optional[TimestampLike] = None,
    label_prefix: str = "",
) -> None:
    """Annotate entry/exit information on the price axis."""

    _require_finplot()

    data = context.data
    price_ax = context.price_ax

    start = pd.to_datetime(entry_time)
    end = pd.to_datetime(exit_time) if exit_time is not None else data.index.max()

    def _plot_level(price: float, color: str, label: str) -> None:
        series = pd.Series([price, price], index=[start, end])
        _fplt.plot(series, ax=price_ax, color=color, style="dash")
        _fplt.plot(pd.Series([price], index=[start]), ax=price_ax, color=color, style="o", width=8)
        _fplt.add_text((start, price), label, color=color, anchor=(0, -1), ax=price_ax)

    prefix = f"{label_prefix} " if label_prefix else ""
    _plot_level(entry_price, "#1e88e5", f"{prefix}Entry @ {entry_price:.2f}")

    if take_profit is not None:
        _plot_level(take_profit, "#00c853", f"{prefix}TP @ {take_profit:.2f}")
    if stop_loss is not None:
        _plot_level(stop_loss, "#ff1744", f"{prefix}SL @ {stop_loss:.2f}")


def show_plot() -> None:
    """Display the finplot window (proxy for :func:`finplot.show`)."""

    _require_finplot()
    _fplt.show()
