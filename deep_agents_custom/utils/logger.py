"""
Logging configuration for the deep search agents application.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[2;37m",  # Dim White
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset to default
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if colors are supported
        self.use_colors = self._supports_color()

    def _supports_color(self):
        """Check if the terminal supports ANSI color codes."""
        # Check for Windows Terminal, VS Code terminal, or Unix-like systems
        if os.getenv("WT_SESSION") or os.getenv("VSCODE_TERM_XTERM"):
            return True

        # Check if running in a terminal that supports colors
        if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
            # Enable colors for Windows 10+ with ANSI support
            if sys.platform == "win32":
                try:
                    # Enable ANSI escape sequence processing on Windows 10+
                    import ctypes

                    kernel32 = ctypes.windll.kernel32
                    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                    return True
                except (OSError, AttributeError, ImportError):
                    pass
            else:
                # Unix-like systems
                return True

        return False

    def format(self, record):
        # Get the original formatted message
        log_message = super().format(record)

        # Add color based on log level if colors are supported
        if self.use_colors:
            level_name = record.levelname
            if level_name in self.COLORS:
                # Color the entire message
                return f"{self.COLORS[level_name]}{log_message}{self.COLORS['RESET']}"

        return log_message


def setup_logging(log_level: str = "INFO", log_to_file: bool = True) -> None:
    """
    Setup logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file in addition to console
    """
    log_file = None

    # Create logs directory if it doesn't exist
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"deep_search_{timestamp}.log"

    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # Use colored formatter for console output
    console_formatter = ColoredFormatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if enabled)
    if log_to_file and log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log debug to file
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        root_logger.info("Logging to file: %s", log_file)

    # Set specific loggers to appropriate levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    root_logger.info("Logging system initialized")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        logging.Logger: Configured logger
    """
    return logging.getLogger(name)
