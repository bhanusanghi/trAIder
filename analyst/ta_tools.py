import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import pandas_ta as pta
import requests
from smolagents import Tool
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Initialize rich console
console = Console()

BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
hyperliquid_api_endpoint = "https://api.hyperliquid.xyz/info"


def pretty_print_kline_data(df: pd.DataFrame, last_n: int = 5) -> None:
    """Pretty print the last n rows of kline data."""
    table = Table(title=f"Last {last_n} Kline Data Points")

    # Add columns
    for col in df.columns:
        table.add_column(col, justify="right")

    # Add rows
    for _, row in df.tail(last_n).iterrows():
        table.add_row(*[str(val) for val in row])

    console.print(table)


def pretty_print_technical_analysis(results: Dict[str, Any]) -> None:
    """Pretty print technical analysis results."""
    # Print metadata
    metadata = results.get("metadata", {})
    console.print(
        Panel.fit(
            "\n".join([f"{k}: {v}" for k, v in metadata.items()]),
            title="Analysis Metadata",
            border_style="blue",
        )
    )

    # Print RSI
    console.print(
        Panel(
            f"RSI: {results.get('rsi', 'N/A'):.2f}",
            title="RSI Analysis",
            border_style="green",
        )
    )

    # Print MACD
    macd_data = results.get("macd", {})
    macd_table = Table(title="MACD Analysis")
    macd_table.add_column("Component", style="cyan")
    macd_table.add_column("Value", justify="right", style="green")
    for k, v in macd_data.items():
        macd_table.add_row(k, f"{v:.6f}")
    console.print(macd_table)

    # Print Bollinger Bands
    bb_data = results.get("bollinger_bands", {})
    bb_table = Table(title="Bollinger Bands")
    bb_table.add_column("Band", style="cyan")
    bb_table.add_column("Value", justify="right", style="green")
    for k, v in bb_data.items():
        bb_table.add_row(k, f"{v:.2f}")
    console.print(bb_table)

    # Print Moving Averages
    ma_data = results.get("moving_averages", {})
    ma_table = Table(title="Moving Averages")
    ma_table.add_column("Type", style="cyan")
    ma_table.add_column("Value", justify="right", style="green")
    for k, v in ma_data.items():
        ma_table.add_row(k, f"{v:.2f}")
    console.print(ma_table)

    # Print Volume Analysis
    volume_data = results.get("volume", {})
    volume_table = Table(title="Volume Analysis")
    volume_table.add_column("Metric", style="cyan")
    volume_table.add_column("Value", justify="right", style="green")
    for k, v in volume_data.items():
        volume_table.add_row(k, f"{v:.2f}")
    console.print(volume_table)

    # Print Support/Resistance
    sr_data = results.get("support_resistance", {})
    sr_table = Table(title="Support & Resistance")
    sr_table.add_column("Level", style="cyan")
    sr_table.add_column("Value", justify="right", style="green")
    for k, v in sr_data.items():
        sr_table.add_row(k, f"{v:.2f}")
    console.print(sr_table)

    # Print Price Data
    price_data = results.get("price_data", {})
    price_table = Table(title="Price Analysis")
    price_table.add_column("Metric", style="cyan")
    price_table.add_column("Value", justify="right", style="green")
    for k, v in price_data.items():
        price_table.add_row(k, f"{v:.2f}")
    console.print(price_table)


def fetch_kline_data_binance(
    symbol: str,
    interval: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 500,
) -> List[List]:
    """
    Fetch kline data from Binance USDⓈ-M Futures API.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval (1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M)
        start_time: Start time in milliseconds
        end_time: End time in milliseconds
        limit: Number of records to fetch (max 1500)

    Returns:
        List of kline data with format:
        [
            [
                Open time,
                Open price,
                High price,
                Low price,
                Close price,
                Volume,
                Close time,
                Quote asset volume,
                Number of trades,
                Taker buy base asset volume,
                Taker buy quote asset volume,
                Ignore
            ],
            ...
        ]
    """
    endpoint = f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/klines"

    console.print(
        f"[bold blue]Fetching kline data for {symbol} on {interval} timeframe[/bold blue]"
    )

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": min(limit, 1500),
    }

    try:
        console.print(f"[dim]Request URL: {endpoint}[/dim]")
        console.print(f"[dim]Parameters: {params}[/dim]")

        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        console.print(f"[green]Successfully fetched {len(data)} kline records[/green]")
        return data

    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to fetch kline data: {str(e)}\nResponse: {response.text if 'response' in locals() else 'No response'}"
        console.print(f"[red bold]{error_msg}[/red bold]")
        raise Exception(error_msg)


def fetch_kline_data_hyperliquid(
    symbol: str,
    interval: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 500,
) -> List[List]:
    """
    Fetch kline data from Hyperliquid API.

    Args:
        symbol: Trading pair symbol (e.g., 'BTC')
        interval: Kline interval (1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M)
        start_time: Start time in milliseconds
        end_time: End time in milliseconds
        limit: Number of records to fetch (max 500 per request)

    Returns:
        List of kline data with format:
        [
            [
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                trades_count
            ],
            ...
        ]
    """

    console.print(
        f"[bold blue]Fetching Hyperliquid kline data for {symbol} on {interval} timeframe[/bold blue]"
    )

    # Convert interval to Hyperliquid format if needed
    interval_map = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "4h": "240",
        "1d": "1D",
    }
    hl_interval = interval_map.get(interval)
    if not hl_interval:
        raise ValueError(f"Unsupported interval {interval} for Hyperliquid")

    # Prepare request payload
    payload = {
        "type": "candles",
        "coin": symbol,
        "interval": hl_interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": min(limit, 500),  # Hyperliquid has a max limit of 500
    }

    try:
        console.print(f"[dim]Request URL: {hyperliquid_api_endpoint}[/dim]")
        console.print(f"[dim]Payload: {payload}[/dim]")

        response = requests.post(hyperliquid_api_endpoint, json=payload, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Transform data to match Binance format for compatibility
        transformed_data = []
        for candle in data:
            transformed_candle = [
                candle["t"],  # timestamp
                candle["o"],  # open
                candle["h"],  # high
                candle["l"],  # low
                candle["c"],  # close
                candle["v"],  # volume
                candle["t"] + (interval_to_milliseconds(interval) - 1),  # close time
                "0",  # quote asset volume (not provided by Hyperliquid)
                candle["n"],  # number of trades
                "0",  # taker buy base volume (not provided)
                "0",  # taker buy quote volume (not provided)
                "0",  # ignore
            ]
            transformed_data.append(transformed_candle)

        console.print(
            f"[green]Successfully fetched {len(transformed_data)} Hyperliquid kline records[/green]"
        )
        return transformed_data

    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to fetch Hyperliquid kline data: {str(e)}\nResponse: {response.text if 'response' in locals() else 'No response'}"
        console.print(f"[red bold]{error_msg}[/red bold]")
        raise Exception(error_msg)


def prepare_kline_data_hyperliquid(kline_data: List[List]) -> pd.DataFrame:
    """Convert kline data to pandas DataFrame with proper column names."""
    pass


def interval_to_milliseconds(interval: str) -> int:
    """Convert interval string to milliseconds."""
    unit = interval[-1]
    value = int(interval[:-1])

    if unit == "m":
        return value * 60 * 1000
    elif unit == "h":
        return value * 60 * 60 * 1000
    elif unit == "d":
        return value * 24 * 60 * 60 * 1000
    elif unit == "w":
        return value * 7 * 24 * 60 * 60 * 1000
    elif unit == "M":
        return value * 30 * 24 * 60 * 60 * 1000
    else:
        raise ValueError(f"Invalid interval unit: {unit}")


def prepare_kline_data_binance(kline_data: List[List]) -> pd.DataFrame:
    """Convert kline data to pandas DataFrame with proper column names."""
    try:
        console.print("[bold blue]Preparing kline data...[/bold blue]")

        df = pd.DataFrame(
            kline_data,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        # Convert numeric columns
        numeric_columns = ["open", "high", "low", "close", "volume"]
        df[numeric_columns] = df[numeric_columns].astype(float)

        # Set index to datetime
        df.index = pd.to_datetime(df["open_time"], unit="ms")

        console.print("[green]Successfully prepared kline data[/green]")
        pretty_print_kline_data(df)  # Print last few rows

        return df

    except Exception as e:
        error_msg = f"Failed to prepare kline data: {str(e)}"
        console.print(f"[red bold]{error_msg}[/red bold]")
        raise Exception(error_msg)


# Support and Resistance
def calculate_support_resistance(data: pd.DataFrame, periods: int = 20) -> tuple:
    recent_data = data.tail(periods)
    support = recent_data["low"].min()
    resistance = recent_data["high"].max()
    return support, resistance


class AnalyzeTechnicalIndicatorsTool(Tool):
    """
    Fetch kline data and analyze technical indicators for a futures trading pair.
    """

    name = "AnalyzeTechnicalIndicators"
    description = (
        "Fetch kline data and analyze technical indicators for a futures trading pair."
    )
    inputs = {
        "symbol": {
            "type": "string",
            "description": "Trading pair symbol (e.g., 'BTCUSDT')",
        },
        "interval": {
            "type": "string",
            "description": "Kline interval (1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M)",
            "nullable": True,
        },
        "lookback_days": {
            "type": "integer",
            "description": "Number of days of historical data to analyze",
            "nullable": True,
        },
        "rsi_period": {
            "type": "integer",
            "description": "Period for RSI calculation",
            "nullable": True,
        },
        "macd_fast": {
            "type": "integer",
            "description": "Fast period for MACD",
            "nullable": True,
        },
        "macd_slow": {
            "type": "integer",
            "description": "Slow period for MACD",
            "nullable": True,
        },
        "macd_signal": {
            "type": "integer",
            "description": "Signal period for MACD",
            "nullable": True,
        },
        "bb_period": {
            "type": "integer",
            "description": "Period for Bollinger Bands",
            "nullable": True,
        },
        "bb_std": {
            "type": "integer",
            "description": "Standard deviation multiplier for Bollinger Bands",
            "nullable": True,
        },
        "ma_periods": {
            "type": "string",
            "description": "List of periods for moving averages",
            "nullable": True,
        },
        "momentum_period": {
            "type": "integer",
            "description": "Period for momentum calculation",
            "nullable": True,
        },
        "support_resistance_periods": {
            "type": "integer",
            "description": "Periods to consider for support/resistance",
            "nullable": True,
        },
    }

    output_type = "string"

    def forward(
        self,
        symbol: str,
        interval: str = "1h",
        lookback_days: int = 30,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: int = 2,
        ma_periods: str = "7,25,99",
        momentum_period: int = 10,
        support_resistance_periods: int = 20,
    ) -> str:
        """
        Fetch kline data and analyze technical indicators for a futures trading pair.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSD_PERP')
            interval: Kline interval (1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M)
            lookback_days: Number of days of historical data to analyze
            rsi_period: Period for RSI calculation
            macd_fast: Fast period for MACD
            macd_slow: Slow period for MACD
            macd_signal: Signal period for MACD
            bb_period: Period for Bollinger Bands
            bb_std: Standard deviation multiplier for Bollinger Bands
            ma_periods: string of periods for moving averages separated by commas
            momentum_period: Period for momentum calculation
            support_resistance_periods: Periods to consider for support/resistance

        Returns:
            Dictionary containing calculated technical indicators as a JSON string using json.dumps()
            The result object is a string of JSON.dumps of of the following format:
            {
                "metadata": {
                    "symbol": "string",
                    "interval": "string",
                    "analysis_time": "string",
                    "data_start": "string",
                    "data_end": "string",
                    "total_periods": "int"
                },
                "rsi": "float",
                "macd": {
                    "macd": "float",
                    "signal": "float",
                    "histogram": "float"
                },
                "bollinger_bands": {
                    "upper": "float",
                    "middle": "float",
                    "lower": "float"
                },
                "moving_averages": {
                    "sma_period": "float",
                    "ema_period": "float"
                },
                "momentum": "float",
                "volume": {
                    "current_volume": "float",
                    "volume_ma": "float",
                    "volume_std": "float"
                },
                "support_resistance": {
                    "support": "float",
                    "resistance": "float"
                },
                "price_data": {
                    "current_price": "float",
                    "24h_high": "float",
                    "24h_low": "float",
                    "24h_volume": "float"
                }
            }
        """
        console.print(
            Panel.fit(
                f"Analyzing {symbol} on {interval} timeframe",
                title="Technical Analysis",
                border_style="blue",
            )
        )

        # Calculate time range
        end_time = int(time.time() * 1000)
        start_time = end_time - (lookback_days * 24 * 60 * 60 * 1000)

        # Fetch and prepare data
        kline_data = fetch_kline_data_binance(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=1500,
        )
        df = prepare_kline_data_binance(kline_data)

        console.print("[bold blue]Calculating technical indicators...[/bold blue]")

        # Create config from parameters
        config = {
            "rsi_period": rsi_period,
            "macd_fast": macd_fast,
            "macd_slow": macd_slow,
            "macd_signal": macd_signal,
            "bb_period": bb_period,
            "bb_std": bb_std,
            "ma_periods": ma_periods,
            "momentum_period": momentum_period,
            "support_resistance_periods": support_resistance_periods,
        }

        results = {}

        # Add metadata
        results["metadata"] = {
            "symbol": symbol,
            "interval": interval,
            "analysis_time": datetime.now().isoformat(),
            "data_start": df.index[0].isoformat(),
            "data_end": df.index[-1].isoformat(),
            "total_periods": len(df),
        }

        # Calculate all indicators
        try:
            # RSI
            rsi = RSIIndicator(df["close"], window=config["rsi_period"])
            results["rsi"] = rsi.rsi().iloc[-1]

            # MACD
            macd = pta.macd(
                df["close"],
                fast=config["macd_fast"],
                slow=config["macd_slow"],
                signal=config["macd_signal"],
            )
            results["macd"] = {
                "macd": macd[
                    "MACD_"
                    + str(config["macd_fast"])
                    + "_"
                    + str(config["macd_slow"])
                    + "_"
                    + str(config["macd_signal"])
                ].iloc[-1],
                "signal": macd[
                    "MACDs_"
                    + str(config["macd_fast"])
                    + "_"
                    + str(config["macd_slow"])
                    + "_"
                    + str(config["macd_signal"])
                ].iloc[-1],
                "histogram": macd[
                    "MACDh_"
                    + str(config["macd_fast"])
                    + "_"
                    + str(config["macd_slow"])
                    + "_"
                    + str(config["macd_signal"])
                ].iloc[-1],
            }

            # # Normalized MACD
            # # This implementation normalizes MACD values relative to price to reduce false signals
            # close_prices = df["close"]
            # norm_macd = pta.macd(
            #     close_prices,
            #     fast=config["macd_fast"],
            #     slow=config["macd_slow"],
            #     signal=config["macd_signal"],
            # )

            # # Get the raw MACD values
            # macd_raw = norm_macd[
            #     "MACD_" + str(config["macd_fast"]) + "_" + str(config["macd_slow"]) + "_" + str(config["macd_signal"])
            # ]

            # # Normalize MACD by dividing by a rolling standard deviation of prices
            # price_std = close_prices.rolling(window=config["macd_slow"]).std()
            # normalized_macd = macd_raw / price_std

            # # Calculate normalized signal and histogram
            # signal_raw = norm_macd[
            #     "MACDs_" + str(config["macd_fast"]) + "_" + str(config["macd_slow"]) + "_" + str(config["macd_signal"])
            # ]
            # normalized_signal = signal_raw / price_std
            # normalized_histogram = normalized_macd - normalized_signal

            # results["normalized_macd"] = {
            #     "macd": normalized_macd.iloc[-1],
            #     "signal": normalized_signal.iloc[-1],
            #     "histogram": normalized_histogram.iloc[-1]
            # }

            # Bollinger Bands
            bb = BollingerBands(
                df["close"], window=config["bb_period"], window_dev=config["bb_std"]
            )
            results["bollinger_bands"] = {
                "upper": bb.bollinger_hband().iloc[-1],
                "middle": bb.bollinger_mavg().iloc[-1],
                "lower": bb.bollinger_lband().iloc[-1],
            }

            # Moving Averages
            results["moving_averages"] = {}
            for period in config["ma_periods"].split(","):
                sma = SMAIndicator(df["close"], window=int(period))
                ema = EMAIndicator(df["close"], window=int(period))
                results["moving_averages"][f"sma_{period}"] = sma.sma_indicator().iloc[
                    -1
                ]
                results["moving_averages"][f"ema_{period}"] = ema.ema_indicator().iloc[
                    -1
                ]

                # Momentum
                momentum = pta.mom(df["close"], length=config["momentum_period"])
                results["momentum"] = momentum.iloc[-1]

                # Volume Analysis
                results["volume"] = {
                    "current_volume": df["volume"].iloc[-1],
                    "volume_ma": df["volume"].rolling(window=20).mean().iloc[-1],
                    "volume_std": df["volume"].rolling(window=20).std().iloc[-1],
                }

            support, resistance = calculate_support_resistance(
                df, config["support_resistance_periods"]
            )
            results["support_resistance"] = {
                "support": support,
                "resistance": resistance,
            }

            # Price data
            results["price_data"] = {
                "current_price": float(df["close"].iloc[-1]),
                "24h_high": float(df["high"].tail(24).max()),
                "24h_low": float(df["low"].tail(24).min()),
                "24h_volume": float(df["volume"].tail(24).sum()),
            }

            # Pretty print the results
            console.print(
                "[green]Successfully calculated all technical indicators[/green]"
            )
            pretty_print_technical_analysis(results)

            return json.dumps(results, indent=2)

        except Exception as e:
            error_msg = f"Failed to calculate technical indicators: {str(e)}"
            console.print(f"[red bold]{error_msg}[/red bold]")
            raise Exception(error_msg)
