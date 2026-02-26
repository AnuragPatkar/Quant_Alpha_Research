"""
Alert System
Handles notifications for production issues.
"""

import logging
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class AlertSystem:
    """
    Centralized alerting system.
    Supports: Logging (Default), Email (Mock), Slack (Mock)
    """
    
    def __init__(self, env: str = 'development'):
        self.env = env
        self.enabled = env == 'production'
        
    def send(self, level: str, title: str, message: str, details: Optional[Dict] = None):
        """
        Send alert based on severity level
        
        Args:
            level: 'INFO', 'WARNING', 'CRITICAL'
            title: Short summary
            message: Detailed description
        """
        full_msg = f"[{level}] {title}: {message}"
        if details:
            full_msg += f"\nDetails: {details}"
            
        # Always log to file/console
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
        # Placeholder: Use smtplib or boto3 here
        logger.info(f"📧 EMAIL SENT: {subject}")
        
    def _send_slack(self, message):
        # Placeholder: Use requests.post(webhook_url, ...)
        logger.info(f"💬 SLACK SENT: {message}")