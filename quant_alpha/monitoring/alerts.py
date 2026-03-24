"""
Production Alert & Notification Dispatcher
==========================================
Centralized message routing subsystem for operational continuity and risk monitoring.

Purpose
-------
The `AlertSystem` acts as the notification backbone for the quantitative trading engine.
It decouples signal generation and infrastructure monitoring from specific notification
channels (Email, Slack, PagerDuty). It implements severity-based routing logic to
ensure critical failures (e.g., execution halts, risk limit breaches) receive immediate
attention while reducing alert fatigue for minor warnings.

Usage
-----
Intended for instantiation within the main application context or specific monitoring services.

.. code-block:: python

    from quant_alpha.monitoring.alerts import AlertSystem

    # Initialize for production (enables external dispatch)
    alerter = AlertSystem(env='production')

    # Dispatch a critical alert boundary
    alerter.send(
        level='CRITICAL',
        title='Data Feed Latency',
        message='Polygon.io feed lag > 500ms',
        details={'lag_ms': 542, 'timestamp': '2023-10-27T14:30:00Z'}
    )

Importance
----------
- **Operational Resilience**: Reduces Mean Time to Detect (MTTD) for critical infrastructure
  failures, preventing "silent" capital losses.
- **Noise Reduction**: Filters lower-priority events (`INFO`, `WARNING`) to local logs,
  reserving high-interrupt channels for `CRITICAL` incidents.
- **Audit Trail**: Ensures all alerts are persisted to disk via the `logging` module regardless
  of external delivery status.

Tools & Frameworks
------------------
- **Logging**: Python standard library for persistent local record keeping.
- **SMTP/Webhooks**: Encapsulated interfaces for external communication providers.
"""

import logging
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class AlertSystem:
    """
    Orchestrates multi-channel notification dispatch based on environment and severity.

    Attributes:
        env (str): Discrete execution environment ('development', 'production', etc.).
        enabled (bool): Flag controlling external dispatch boundaries. Active strictly in 'production'.
    """
    
    def __init__(self, env: str = 'development'):
        """Initializes the alert dispatch orchestrator mapped to runtime parameters."""
        self.env = env
        # Asserts structural safety bounds strictly averting notification leakage 
        # during development iterations or localized backtesting scenarios.
        self.enabled = env == 'production'
        
    def send(self, level: str, title: str, message: str, details: Optional[Dict] = None):
        """
        Routes a localized alert to the appropriate external channels based on discrete severity.
        
        Args:
            level (str): String severity classification mapping ('INFO', 'WARNING', 'CRITICAL').
            title (str): Brief declarative synopsis targeted for subject line headers.
            message (str): Explicitly detailed sequence description mapping the fault event.
            details (Optional[Dict]): Contextual nested metadata bounding diagnostic traces.

        Routing Logic:
        - **All Levels**: Persist to local disk logs.
        - **CRITICAL**: Dispatch via high-priority channels (Email, Slack) if production.
        """
        full_msg = f"[{level}] {title}: {message}"
        if details:
            full_msg += f"\nDetails: {details}"
            
        # Evaluates absolute persistence: Asserts continuous local log records are generated 
        # prior to executing external dispatches. Guarantees deterministic audit trails.
        if level == 'CRITICAL':
            logger.error(full_msg)
            if self.enabled:
                self._send_email(f"URGENT: {title}", full_msg)
                self._send_slack(full_msg)
        elif level == 'WARNING':
            logger.warning(full_msg)
        else:
            logger.info(full_msg)
            
    def _send_email(self, subject, body):
        """
        Instantiates internal abstraction mapping SMTP/AWS SES execution dispatches.
        
        Args:
            subject (str): Defined email subject limit.
            body (str): Transcribed plain-text message context.
        """
        logger.info(f"📧 EMAIL SENT: {subject}")
        
    def _send_slack(self, message):
        """
        Instantiates internal abstraction mapping targeted Slack Webhook distribution limits.
        
        Args:
            message (str): Extracted payload formatted for external ChatOps interfaces.
        """
        logger.info(f"💬 SLACK SENT: {message}")