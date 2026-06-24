"""
Live run monitor for SkyDiscover solution discovery.

Provides a real-time WebSocket-powered dashboard that shows programs
appearing on a scatter plot as they're evaluated, with lineage arrows,
stats, and code inspection.
"""

import logging
import os
import time
from typing import Optional, Tuple

from skydiscover.extras.monitor.callback import create_external_callback, create_monitor_callback
from skydiscover.extras.monitor.server import MonitorServer

__all__ = [
    "MonitorServer",
    "create_monitor_callback",
    "create_external_callback",
    "start_monitor",
    "stop_monitor",
]

logger = logging.getLogger(__name__)


def start_monitor(
    config, output_dir: str
) -> Tuple[Optional[MonitorServer], Optional[object], Optional[object]]:
    """Start the live monitor server. Returns (server, callback, feedback_reader)."""
    monitor_server = None
    monitor_callback = None
    feedback_reader = None

    if not config.monitor.enabled:
        return monitor_server, monitor_callback, feedback_reader

    try:
        monitor_server = MonitorServer(
            host=config.monitor.host,
            port=config.monitor.port,
            max_solution_length=config.monitor.max_solution_length,
        )
        monitor_server.start()
        monitor_callback = create_external_callback(monitor_server, time.time())

        if config.monitor.summary_model:
            monitor_server.configure_summary(
                model=config.monitor.summary_model,
                api_key=config.monitor.summary_api_key or "",
                api_base=config.monitor.summary_api_base,
                top_k=config.monitor.summary_top_k,
                interval=config.monitor.summary_interval,
            )

        try:
            from skydiscover.context_builder.human_feedback import HumanFeedbackReader

            feedback_path = getattr(config, "human_feedback_file", None) or os.path.join(
                output_dir, "human_feedback.md"
            )
            feedback_mode = getattr(config, "human_feedback_mode", "append")
            feedback_reader = HumanFeedbackReader(feedback_path, mode=feedback_mode)
            monitor_server.set_feedback_reader(feedback_reader)
            logger.debug("Human feedback enabled — file: %s", feedback_path)
        except Exception as exc:
            logger.warning("Failed to set up human feedback: %s", exc)

        url = f"http://localhost:{monitor_server.port}/"
        print(f"\n  Live monitor: {url}\n", flush=True)
        logger.debug("Live monitor: %s", url)

    except Exception as exc:
        logger.warning("Failed to start monitor: %s", exc)

    return monitor_server, monitor_callback, feedback_reader


def stop_monitor(monitor_server: Optional[MonitorServer]) -> None:
    """Gracefully shut down the monitor server."""
    if monitor_server is None:
        return
    try:
        monitor_server.push_event({"type": "discovery_complete"})
        time.sleep(0.5)
        monitor_server.stop()
    except Exception:
        logger.debug("Failed to stop monitor server", exc_info=True)
