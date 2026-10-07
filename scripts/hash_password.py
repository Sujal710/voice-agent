"""Create the DASHBOARD_PASSWORD_HASH value for .env.

Usage:  python -m scripts.hash_password
"""

import getpass
import secrets

from app.auth import hash_password


def main() -> None:
    password = getpass.getpass("New dashboard password (min 10 characters): ")
    if len(password) < 10:
        raise SystemExit("Please use at least 10 characters.")
    if password != getpass.getpass("Repeat password: "):
        raise SystemExit("Passwords don't match.")
    print("\nAdd these lines to your .env file:\n")
    print(f"DASHBOARD_PASSWORD_HASH={hash_password(password)}")
    print(f"SESSION_SECRET={secrets.token_urlsafe(48)}")


if __name__ == "__main__":
    main()
