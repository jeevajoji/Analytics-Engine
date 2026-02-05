"""
Notification Output Adapter.

Sends alerts via email, SMS, push notifications, webhooks.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import logging
import json

from ..lanes.ruleset_lane import OutputAdapter


logger = logging.getLogger(__name__)


class NotificationChannel(str, Enum):
    """Available notification channels."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"


@dataclass
class NotificationConfig:
    """Configuration for a notification channel."""
    channel: NotificationChannel
    enabled: bool = True
    # Email settings
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    from_email: Optional[str] = None
    to_emails: Optional[List[str]] = None
    # SMS settings (Twilio)
    twilio_sid: Optional[str] = None
    twilio_token: Optional[str] = None
    twilio_from: Optional[str] = None
    to_phones: Optional[List[str]] = None
    # Webhook settings
    webhook_url: Optional[str] = None
    webhook_headers: Optional[Dict[str, str]] = None
    # Slack settings
    slack_webhook: Optional[str] = None
    slack_channel: Optional[str] = None
    # Teams settings
    teams_webhook: Optional[str] = None


class NotificationAdapter(OutputAdapter):
    """
    Notification output adapter for alerts.
    
    Sends alerts through multiple channels based on event severity.
    
    Example:
        adapter = NotificationAdapter(
            channels=[
                NotificationConfig(
                    channel=NotificationChannel.EMAIL,
                    smtp_host="smtp.gmail.com",
                    from_email="alerts@company.com",
                    to_emails=["ops@company.com"]
                ),
                NotificationConfig(
                    channel=NotificationChannel.SLACK,
                    slack_webhook="https://hooks.slack.com/..."
                )
            ],
            severity_routing={
                "critical": ["email", "sms", "slack"],
                "error": ["email", "slack"],
                "warning": ["slack"],
            }
        )
        
        lane.add_output_adapter(adapter)
    """
    
    def __init__(
        self,
        channels: Optional[List[NotificationConfig]] = None,
        severity_routing: Optional[Dict[str, List[str]]] = None,
        rate_limit_seconds: int = 60,
        dedupe_window_seconds: int = 300,
    ):
        """
        Initialize notification adapter.
        
        Args:
            channels: List of channel configurations
            severity_routing: Map of severity to channel names
            rate_limit_seconds: Min seconds between notifications
            dedupe_window_seconds: Window for deduplication
        """
        self.channels = {c.channel.value: c for c in (channels or [])}
        self.severity_routing = severity_routing or {
            "critical": ["email", "sms", "slack", "webhook"],
            "error": ["email", "slack", "webhook"],
            "warning": ["slack", "webhook"],
            "info": ["webhook"],
        }
        self.rate_limit_seconds = rate_limit_seconds
        self.dedupe_window_seconds = dedupe_window_seconds
        
        self._last_sent: Dict[str, datetime] = {}
        self._sent_hashes: Dict[str, datetime] = {}
    
    async def connect(self) -> None:
        """Initialize notification channels."""
        logger.info(f"Notification adapter initialized with channels: {list(self.channels.keys())}")
    
    async def disconnect(self) -> None:
        """Cleanup notification channels."""
        pass
    
    async def write(self, record: Dict[str, Any]) -> None:
        """Send notification for an event."""
        severity = record.get("severity", "info")
        channels = self.severity_routing.get(severity, [])
        
        # Check deduplication
        event_hash = self._get_event_hash(record)
        if self._is_duplicate(event_hash):
            logger.debug(f"Skipping duplicate notification: {event_hash}")
            return
        
        for channel_name in channels:
            if channel_name in self.channels:
                config = self.channels[channel_name]
                if config.enabled:
                    await self._send_notification(config, record)
        
        self._sent_hashes[event_hash] = datetime.now()
    
    async def write_batch(self, records: List[Dict[str, Any]]) -> None:
        """Send notifications for multiple events."""
        for record in records:
            await self.write(record)
    
    def _get_event_hash(self, record: Dict[str, Any]) -> str:
        """Generate hash for deduplication."""
        key_fields = [
            record.get("event_type"),
            record.get("asset_id"),
            record.get("severity"),
        ]
        return ":".join(str(f) for f in key_fields if f)
    
    def _is_duplicate(self, event_hash: str) -> bool:
        """Check if event was recently sent."""
        if event_hash in self._sent_hashes:
            last_sent = self._sent_hashes[event_hash]
            if (datetime.now() - last_sent).total_seconds() < self.dedupe_window_seconds:
                return True
        return False
    
    async def _send_notification(
        self, 
        config: NotificationConfig, 
        record: Dict[str, Any]
    ) -> None:
        """Send notification through a specific channel."""
        try:
            if config.channel == NotificationChannel.EMAIL:
                await self._send_email(config, record)
            elif config.channel == NotificationChannel.SMS:
                await self._send_sms(config, record)
            elif config.channel == NotificationChannel.WEBHOOK:
                await self._send_webhook(config, record)
            elif config.channel == NotificationChannel.SLACK:
                await self._send_slack(config, record)
            elif config.channel == NotificationChannel.TEAMS:
                await self._send_teams(config, record)
                
        except Exception as e:
            logger.error(f"Failed to send {config.channel.value} notification: {e}")
    
    async def _send_email(self, config: NotificationConfig, record: Dict[str, Any]) -> None:
        """Send email notification."""
        import aiosmtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        subject = f"[{record.get('severity', 'ALERT').upper()}] {record.get('title', 'Alert')}"
        body = self._format_message(record)
        
        message = MIMEMultipart()
        message["From"] = config.from_email
        message["To"] = ", ".join(config.to_emails or [])
        message["Subject"] = subject
        message.attach(MIMEText(body, "html"))
        
        await aiosmtplib.send(
            message,
            hostname=config.smtp_host,
            port=config.smtp_port,
            username=config.smtp_user,
            password=config.smtp_password,
            use_tls=True,
        )
        
        logger.info(f"Email sent: {subject}")
    
    async def _send_sms(self, config: NotificationConfig, record: Dict[str, Any]) -> None:
        """Send SMS notification via Twilio."""
        from twilio.rest import Client
        
        client = Client(config.twilio_sid, config.twilio_token)
        
        message = f"[{record.get('severity', 'ALERT').upper()}] {record.get('title', 'Alert')}"
        if record.get("description"):
            message += f"\n{record['description'][:100]}"
        
        for phone in (config.to_phones or []):
            client.messages.create(
                body=message,
                from_=config.twilio_from,
                to=phone
            )
        
        logger.info(f"SMS sent to {len(config.to_phones or [])} numbers")
    
    async def _send_webhook(self, config: NotificationConfig, record: Dict[str, Any]) -> None:
        """Send webhook notification."""
        import aiohttp
        
        headers = {"Content-Type": "application/json"}
        if config.webhook_headers:
            headers.update(config.webhook_headers)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config.webhook_url,
                json=record,
                headers=headers
            ) as response:
                if response.status >= 400:
                    logger.warning(f"Webhook failed with status {response.status}")
                else:
                    logger.info(f"Webhook sent to {config.webhook_url}")
    
    async def _send_slack(self, config: NotificationConfig, record: Dict[str, Any]) -> None:
        """Send Slack notification."""
        import aiohttp
        
        severity = record.get("severity", "info")
        color_map = {
            "critical": "#FF0000",
            "error": "#FF6600",
            "warning": "#FFCC00",
            "info": "#0066FF",
        }
        
        payload = {
            "channel": config.slack_channel,
            "attachments": [{
                "color": color_map.get(severity, "#808080"),
                "title": record.get("title", "Alert"),
                "text": record.get("description", ""),
                "fields": [
                    {"title": "Type", "value": record.get("event_type", ""), "short": True},
                    {"title": "Severity", "value": severity.upper(), "short": True},
                    {"title": "Asset", "value": record.get("asset_id", "N/A"), "short": True},
                    {"title": "Time", "value": record.get("timestamp", ""), "short": True},
                ],
                "footer": "Analytics Engine",
                "ts": int(datetime.now().timestamp()),
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.slack_webhook, json=payload) as response:
                if response.status == 200:
                    logger.info("Slack notification sent")
                else:
                    logger.warning(f"Slack notification failed: {response.status}")
    
    async def _send_teams(self, config: NotificationConfig, record: Dict[str, Any]) -> None:
        """Send Microsoft Teams notification."""
        import aiohttp
        
        severity = record.get("severity", "info")
        color_map = {
            "critical": "FF0000",
            "error": "FF6600",
            "warning": "FFCC00",
            "info": "0066FF",
        }
        
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color_map.get(severity, "808080"),
            "summary": record.get("title", "Alert"),
            "sections": [{
                "activityTitle": record.get("title", "Alert"),
                "activitySubtitle": f"Severity: {severity.upper()}",
                "facts": [
                    {"name": "Event Type", "value": record.get("event_type", "")},
                    {"name": "Asset", "value": record.get("asset_id", "N/A")},
                    {"name": "Time", "value": record.get("timestamp", "")},
                ],
                "text": record.get("description", ""),
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.teams_webhook, json=payload) as response:
                if response.status == 200:
                    logger.info("Teams notification sent")
                else:
                    logger.warning(f"Teams notification failed: {response.status}")
    
    def _format_message(self, record: Dict[str, Any]) -> str:
        """Format record as HTML message."""
        severity = record.get("severity", "info")
        color_map = {
            "critical": "#FF0000",
            "error": "#FF6600",
            "warning": "#FFCC00",
            "info": "#0066FF",
        }
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background-color: {color_map.get(severity, '#808080')}; 
                        color: white; padding: 10px; border-radius: 5px;">
                <h2 style="margin: 0;">{record.get('title', 'Alert')}</h2>
                <p style="margin: 5px 0 0 0;">Severity: {severity.upper()}</p>
            </div>
            <div style="padding: 15px; background-color: #f5f5f5; margin-top: 10px;">
                <p><strong>Event Type:</strong> {record.get('event_type', 'N/A')}</p>
                <p><strong>Asset:</strong> {record.get('asset_id', 'N/A')}</p>
                <p><strong>Time:</strong> {record.get('timestamp', 'N/A')}</p>
                <p><strong>Description:</strong></p>
                <p>{record.get('description', 'No description provided.')}</p>
            </div>
            <div style="padding: 10px; font-size: 12px; color: #666;">
                Sent by Analytics Engine
            </div>
        </body>
        </html>
        """
