# Trading Analysis Agent

An advanced technical analysis agent powered by LLMs that provides detailed trading insights and recommendations using multiple timeframe analysis and technical indicators.

## Features

1. **Multi-Exchange Support**
   - Binance Futures API integration
   - Hyperliquid API integration
   - Extensible for additional exchanges

2. **Advanced Technical Analysis**
   - Multiple timeframe analysis
   - Comprehensive indicator suite
   - Pattern recognition
   - Volume profile analysis

3. **AI-Powered Analysis**
   - Expert system prompts
   - Context-aware recommendations
   - Dynamic indicator configuration
   - Risk assessment

4. **Rich Output Formatting**
   - Detailed analysis reports
   - Pretty-printed data tables
   - Color-coded indicators
   - Clear action items

### Technical Analysis Capabilities
- Multi-timeframe analysis (15m, 1h, 4h, 1d) determined dynamically from LLM using user specific query if provided
- Advanced pattern recognition
- Support/resistance level identification
- Volume profile analysis
- Market structure evaluation
- Trend strength assessment

### Indicator Suite
- Momentum: RSI, MACD, Stochastic, Normalized MACD
- Trend: SMA, EMA, Bollinger Bands
- Volume: Volume Profile, OBV
- Volatility: ATR, Standard Deviation

### Risk Management
- Position sizing recommendations
- Clear entry/exit points
- Stop-loss and take-profit levels
- Risk/reward ratio calculations
- Volatility-adjusted positioning

## Prerequisites

- Python 3.10+
- Poetry (recommended) or pip
- OpenRouter API key or similar. Change the LLM model as per your wish.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bhanusanghi/trAIder.git
cd trAIder
```

2. Install dependencies:
```bash
poetry install
```

## Usage

```python
from analyst.runner import run_analyst

# Simple analysis
run_analyst("BTCUSDT", "1h")

# Analysis with specific query
run_analyst(
    "ETHUSDT",
    "4h",
    "Should I open a long position? What are the key support levels?"
)
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
