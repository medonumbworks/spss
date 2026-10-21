#!/usr/bin/env python3
"""One-time interactive Telegram login for the sentiment hub.

    pip install telethon
    python tools/sentiment_hub/login.py            # prints the session string
    python tools/sentiment_hub/login.py --out session.txt   # writes it to a 0600 file instead
    python tools/sentiment_hub/login.py --check    # verifies TELEGRAM_* env vars

You need your own ``api_id`` and ``api_hash`` from https://my.telegram.org (API development
tools). The script asks for your phone number, the login code Telegram sends you and, if
enabled, your two-step verification password, then prints a Telethon ``StringSession`` and the
exact steps to store it as GitHub Actions secrets of medonumbworks/spss.

WARNING: the session string is a full login to your Telegram account. Never commit it, never
paste it into issues, chats or logs, and never share it. Revoke it any time in Telegram:
Settings -> Devices -> "CryptoPulse Sentiment Hub" -> Terminate session.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script: import as the `sentiment_hub` package and keep
    _here = Path(__file__).resolve().parent  # generic module names (common, net, feed) off sys.path
    sys.path[:] = [x for x in sys.path if Path(x or ".").resolve() != _here]
    sys.path.insert(0, str(_here.parent))

import argparse
import asyncio
import getpass
import os

REPO = "medonumbworks/spss"
SECRETS_URL = f"https://github.com/{REPO}/settings/secrets/actions"
VARIABLES_URL = f"https://github.com/{REPO}/settings/variables/actions"
DEVICE_MODEL = "CryptoPulse Sentiment Hub"

WARNING = """\
!!! SECURITY WARNING !!!
The session string below is a FULL LOGIN to your Telegram account: whoever has it can read
all your chats and send messages as you. Store it ONLY as a GitHub Actions secret.
Never commit it, never paste it into issues, chats, screenshots or logs, never share it.
Revoke it any time: Telegram -> Settings -> Devices -> "CryptoPulse Sentiment Hub" -> Terminate.
"""


def instructions(api_id: int | str, session_hint: str) -> str:
    return f"""
Add three repository secrets to {REPO}:

  1. Open {SECRETS_URL}
     (Settings -> Secrets and variables -> Actions -> "New repository secret")
  2. Name TELEGRAM_API_ID    value {api_id}
  3. Name TELEGRAM_API_HASH  value <the api_hash from https://my.telegram.org>
  4. Name TELEGRAM_SESSION   value {session_hint}

Or with the GitHub CLI (values are read from stdin, nothing ends up in your shell history):

  gh secret set TELEGRAM_API_ID   --repo {REPO}
  gh secret set TELEGRAM_API_HASH --repo {REPO}
  gh secret set TELEGRAM_SESSION  --repo {REPO} < session.txt     # file from --out

Then start the workflow once: Actions -> sentiment-hub -> Run workflow. In the run's summary the
source telegram_account should report "ok". Secrets are not passed to workflows started from
forks, and GitHub masks their values in the (public) logs.

Optional: the repository variable TELEGRAM_AUTO_JOIN=1 ({VARIABLES_URL}) lets the hub join up
to 5 public groups from channels.json per run. Without it the hub never joins anything.
"""


async def interactive_login(out: Path | None) -> int:
    try:
        from telethon import TelegramClient
        from telethon.errors import PhoneCodeInvalidError, SessionPasswordNeededError
        from telethon.sessions import StringSession
    except ImportError:
        print("telethon is not installed: pip install -r tools/sentiment_hub/requirements.txt", file=sys.stderr)
        return 2

    print(WARNING)
    print("Get api_id and api_hash at https://my.telegram.org -> API development tools.\n")
    raw_id = (os.environ.get("TELEGRAM_API_ID") or input("api_id: ")).strip()
    try:
        api_id = int(raw_id)
    except ValueError:
        print("api_id must be a number", file=sys.stderr)
        return 2
    api_hash = (os.environ.get("TELEGRAM_API_HASH") or getpass.getpass("api_hash (hidden): ")).strip()
    phone = input("phone number in international format (e.g. +38591...): ").strip()

    client = TelegramClient(StringSession(), api_id, api_hash, device_model=DEVICE_MODEL, receive_updates=False)
    await client.connect()
    try:
        await client.send_code_request(phone)
        for attempt in range(3):
            code = input("login code from Telegram: ").strip().replace(" ", "")
            try:
                await client.sign_in(phone=phone, code=code)
                break
            except PhoneCodeInvalidError:
                print("wrong code, try again", file=sys.stderr)
            except SessionPasswordNeededError:
                password = getpass.getpass("two-step verification password (hidden): ")
                await client.sign_in(password=password)
                break
        if not await client.is_user_authorized():
            print("login failed", file=sys.stderr)
            return 1
        me = await client.get_me()
        session = client.session.save()
    finally:
        await client.disconnect()

    print(f"\nLogged in as {getattr(me, 'first_name', '') or 'your account'} (id {getattr(me, 'id', '?')}).\n")
    if out is not None:
        # Mode 600 before anything is written (write_text + chmod would leave the secret
        # world-readable for a moment, also when the file already existed with wider permissions).
        fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        if hasattr(os, "fchmod"):  # POSIX (and Windows from Python 3.13 on)
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(session + "\n")
        print(f"Session string written to {out} (mode 600). Delete the file after adding the secret.")
        hint = f"<contents of {out}>"
    else:
        print(WARNING)
        print("----- BEGIN TELEGRAM_SESSION -----")
        print(session)
        print("----- END TELEGRAM_SESSION -----")
        hint = "<the string between the BEGIN/END lines>"
    print(instructions(api_id, hint))
    return 0


async def check_env() -> int:
    from sentiment_hub.telegram_account import default_client_factory, missing_env

    missing = missing_env(os.environ)
    if missing:
        print(f"missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2
    client = default_client_factory(
        int(os.environ["TELEGRAM_API_ID"]), os.environ["TELEGRAM_API_HASH"], os.environ["TELEGRAM_SESSION"]
    )
    await client.connect()
    try:
        if not await client.is_user_authorized():
            print("TELEGRAM_SESSION is NOT authorized; run login.py again", file=sys.stderr)
            return 1
        me = await client.get_me()
        print(f"session OK: user id {getattr(me, 'id', '?')}")
        return 0
    finally:
        await client.disconnect()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="login.py", description="Create a Telethon StringSession for the sentiment hub.")
    parser.add_argument("--out", type=Path, default=None, help="write the session string to this file (mode 600) instead of printing it")
    parser.add_argument("--check", action="store_true", help="verify TELEGRAM_API_ID/TELEGRAM_API_HASH/TELEGRAM_SESSION from the environment")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(check_env() if args.check else interactive_login(args.out))
    except KeyboardInterrupt:
        print("\naborted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
