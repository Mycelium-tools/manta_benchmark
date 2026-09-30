"""Check OpenRouter credits using OPENROUTER_API_KEY from the environment or .env.

Usage:
    uv run python -m experiments.openrouter_balance
    uv run python -m experiments.openrouter_balance --account  # requires a management key

API docs: https://openrouter.ai/docs/api/api-reference/api-keys/get-current-api-key
"""

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--account", action="store_true",
        help="Show account credit balance instead of key allowance (management key required).",
    )
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in your environment or the repository .env file.")

    endpoint = "credits" if args.account else "key"
    request = Request(
        f"https://openrouter.ai/api/v1/{endpoint}",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            data = json.load(response)["data"]

        if args.account:
            balance = data["total_credits"] - data["total_usage"]
            print(f"Account balance: ${balance:,.4f}")
            print(f"Total usage:     ${data['total_usage']:,.4f}")
        else:
            remaining = data["limit_remaining"]
            if remaining is None:
                print("Key allowance: no spending limit set (account credits still apply).")
            else:
                print(f"Key allowance remaining: ${remaining:,.4f}")
            print(f"Key usage (all time):    ${data['usage']:,.4f}")
            if data.get("limit_reset"):
                print(f"Key limit resets:       {data['limit_reset']}")
    except HTTPError as error:
        if error.code == 401:
            raise SystemExit("OpenRouter rejected OPENROUTER_API_KEY (HTTP 401).") from None
        if error.code == 403 and args.account:
            raise SystemExit("Account balance requires a management key in OPENROUTER_API_KEY (HTTP 403).") from None
        raise SystemExit(f"OpenRouter request failed (HTTP {error.code}).") from None
    except (URLError, OSError):
        raise SystemExit("Could not reach OpenRouter; check your connection and try again.") from None
    except (ValueError, KeyError, TypeError):
        raise SystemExit("OpenRouter returned an unexpected balance response.") from None


if __name__ == "__main__":
    main()
