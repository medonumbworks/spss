"""CryptoPulse sentiment hub.

Collects crypto chatter from free sources (public Telegram web previews, the owner's own
Telegram account, crypto news RSS feeds and Reddit), scores every text with VADER (plus a crypto
slang overlay), FinBERT and CryptoBERT, aggregates per Bitpanda asset and writes one JSON feed
(schema ``cryptopulse.sentiment-feed/v1``) that the Android app and the MCP server download.

Run ``python tools/sentiment_hub/hub.py --help`` for the command line.
"""

__version__ = "1.0.0"
