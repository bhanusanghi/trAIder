import os
from datetime import datetime
from typing import Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator

from dotenv import load_dotenv
import torch
from smolagents import (
    CodeAgent,
    LiteLLMModel,
    CODE_SYSTEM_PROMPT,
)
from rich.console import Console
from rich.panel import Panel
from rich.traceback import install
from trAIder.ta_tools import AnalyzeTechnicalIndicatorsTool
from trAIder.prompts import TRADING_EXPERT_AGENT_SYSTEM_PROMPT, AGENT_TASK_PROMPT

# Install rich traceback handler
install(show_locals=True)

# Initialize logging
console = Console()


class LogLevel(str, Enum):
    """Log levels for the application"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogMessage(BaseModel):
    """Structure for log messages"""

    timestamp: datetime = Field(default_factory=datetime.now)
    level: LogLevel
    step: str
    message: str
    error: Optional[Exception] = None
    context: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True


class DeviceConfig(BaseModel):
    """Configuration for device and batch size"""

    device: str = Field(default="cpu")
    batch_size: int = Field(default=16)

    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        valid_devices = ["cpu", "cuda", "mps"]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v):
        if v < 1:
            raise ValueError("Batch size must be positive")
        return v

    @classmethod
    def from_environment(cls) -> "DeviceConfig":
        """Create config from environment settings"""
        device = os.getenv("DEVICE") or (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        batch_size = int(
            os.getenv("BATCH_SIZE")
            or (64 if device == "mps" else 32 if device == "cuda" else 16)
        )
        return cls(device=device, batch_size=batch_size)


def log_event(
    level: LogLevel,
    step: str,
    message: str,
    error: Optional[Exception] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Unified logging function"""
    log_entry = LogMessage(
        level=level, step=step, message=message, error=error, context=context
    )

    # Format the message
    msg = f"[{log_entry.level}] {log_entry.step}: {log_entry.message}"

    # Add context if available
    if context:
        msg += f"\nContext: {context}"

    # Add error details if available
    if error:
        msg += f"\nError: {str(error)}"
        if hasattr(error, "__traceback__"):
            console.print_exception()

    # Print with appropriate styling
    style = {
        LogLevel.DEBUG: "dim",
        LogLevel.INFO: "white",
        LogLevel.WARNING: "yellow",
        LogLevel.ERROR: "red",
        LogLevel.CRITICAL: "bold red",
    }[level]

    console.print(Panel(msg, style=style))


def process_request(
    ticker: str, timeframe: str, extra_query: Optional[str] = None
) -> str:
    """
    Process a trading analysis request.

    Args:
        ticker: Trading pair symbol (e.g., 'BTCUSDT')
        timeframe: Trading timeframe (e.g., '1h', '4h', '1d')
        extra_query: Optional additional context or specific analysis request

    Returns:
        Dictionary containing the analysis results
    """
    # Load environment variables
    load_dotenv()

    # Validate API keys
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        log_event(
            LogLevel.CRITICAL,
            "Configuration",
            "OPENROUTER_API_KEY not found in environment variables",
        )
        raise ValueError("OPENROUTER_API_KEY is required")

    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        log_event(
            LogLevel.CRITICAL,
            "Configuration",
            "HF_API_TOKEN not found in environment variables",
        )
        raise ValueError("HF_API_TOKEN is required")

    print("using openrouter model, api key:", api_key)
    # Initialize model
    model = LiteLLMModel(
        # model_id="openrouter/deepseek/deepseek-r1",
        # model_id="openrouter/deepseek/deepseek-chat",
        model_id="openrouter/microsoft/phi-4",
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Initialize tools
    ta_tool = AnalyzeTechnicalIndicatorsTool()

    # Create agent with proper prompt and configuration
    agent = CodeAgent(
        model=model,
        tools=[ta_tool],
        max_steps=10,  # 2 iterations per phase (search, store) + (retrieve, analyze) + (predict)
        planning_interval=3,  # Plan after each phase
        verbose=True,
        additional_authorized_imports=[
            "json",
            "datetime",
            "ta",
            "pandas_ta",
            "rich",
            "pandas",
            "numpy",
            "requests",
        ],
        system_prompt=CODE_SYSTEM_PROMPT + TRADING_EXPERT_AGENT_SYSTEM_PROMPT,
    )

    # Format the task prompt with user parameters
    task_prompt = AGENT_TASK_PROMPT.format(
        symbol=ticker.upper(),  # Ensure symbol is uppercase
        timeframe=timeframe.lower(),  # Normalize timeframe format
        extra_data=(f"Additional Context: {extra_query}" if extra_query else ""),
    )

    # Run the agent and get results
    try:
        task_prompt = AGENT_TASK_PROMPT.format(
            **{
                "symbol": ticker.upper(),  # Ensure symbol is uppercase
                "timeframe": timeframe.lower(),  # Normalize timeframe format
                "extra_data": (
                    f"Additional Context: {extra_query}" if extra_query else ""
                ),
            }
        )

        result = agent.run(task_prompt)
        log_event(
            LogLevel.INFO,
            "Analysis",
            f"Successfully analyzed {ticker} on {timeframe} timeframe",
            context={"ticker": ticker, "timeframe": timeframe},
        )
        return result
    except Exception as e:
        log_event(
            LogLevel.ERROR,
            "Analysis",
            f"Failed to analyze {ticker}",
            error=e,
            context={"ticker": ticker, "timeframe": timeframe},
        )
        raise
