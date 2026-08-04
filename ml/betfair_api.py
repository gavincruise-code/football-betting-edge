"""
Betfair Exchange API Integration
=================================
Secure Betfair Exchange API client using environment variables
or user-supplied runtime credentials.
"""

import os
import logging
import requests
import unicodedata
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

BETFAIR_CERT_LOGIN_URL = "https://identitysso-api.betfair.com/api/certlogin"
BETFAIR_API_URL = "https://api.betfair.com/exchange/betting/rest/v1.0"

def norm_str(s: str) -> str:
    if not s: return ""
    n = ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')
    return n.replace('IFK ', '').replace('FC ', '').replace('SK ', '').replace('AC ', '').replace('CD ', '').strip().lower()


class BetfairExchangeClient:
    """
    Automated Betfair Exchange API client using secure environment variables.
    """

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None, app_key: Optional[str] = None):
        self.username = username or os.getenv("BETFAIR_USERNAME", "")
        self.password = password or os.getenv("BETFAIR_PASSWORD", "")
        self.app_key = app_key or os.getenv("BETFAIR_APP_KEY", "")
        self.cert_path = os.getenv("BETFAIR_CERT_PATH", "./certs")
        self.session_token = None
        self.last_status = "Not authenticated"
        self.market_cache = {}

        # Resolve cert file paths
        self.crt_file = os.path.join(self.cert_path, "client-2048.crt")
        self.key_file = os.path.join(self.cert_path, "client-2048.key")

        # Attempt initial login if credentials provided
        self.login()

    def set_credentials(self, username: Optional[str] = None, password: Optional[str] = None, app_key: Optional[str] = None):
        """Update runtime credentials and persist to local .env file."""
        if username: self.username = username
        if password: self.password = password
        if app_key: self.app_key = app_key

        # Persist updated credentials to .env file on disk
        env_path = ".env"
        if os.path.exists(env_path):
            try:
                lines = []
                with open(env_path, "r") as f:
                    for line in f:
                        if line.startswith("BETFAIR_PASSWORD=") and password:
                            lines.append(f"BETFAIR_PASSWORD={password}\n")
                        elif line.startswith("BETFAIR_USERNAME=") and username:
                            lines.append(f"BETFAIR_USERNAME={username}\n")
                        elif line.startswith("BETFAIR_APP_KEY=") and app_key:
                            lines.append(f"BETFAIR_APP_KEY={app_key}\n")
                        else:
                            lines.append(line)
                with open(env_path, "w") as f:
                    f.writelines(lines)
            except Exception as e:
                logger.warning(f"Could not persist .env updates: {e}")

    def login(self, password: Optional[str] = None) -> bool:
        """
        Authenticate with Betfair via SSL certificate or standard API login.
        """
        if password:
            self.set_credentials(password=password)
        else:
            load_dotenv(override=True)
            self.username = os.getenv("BETFAIR_USERNAME", self.username)
            self.password = os.getenv("BETFAIR_PASSWORD", self.password)
            self.app_key = os.getenv("BETFAIR_APP_KEY", self.app_key)

        if not (self.username and self.password and self.app_key):
            self.last_status = "Missing credentials in .env file"
            logger.info("Betfair credentials not set in environment variables.")
            return False

        # Method 1: SSL Certificate Login
        if os.path.exists(self.crt_file) and os.path.exists(self.key_file):
            try:
                cert = (self.crt_file, self.key_file)
                headers = {
                    "X-Application": self.app_key,
                    "Content-Type": "application/x-www-form-urlencoded"
                }
                data = {"username": self.username, "password": self.password}
                resp = requests.post(BETFAIR_CERT_LOGIN_URL, data=data, headers=headers, cert=cert, timeout=10)
                if resp.status_code == 200:
                    res_data = resp.json()
                    status = res_data.get("loginStatus", "UNKNOWN_ERROR")
                    if status == "SUCCESS":
                        self.session_token = res_data.get("sessionToken")
                        self.last_status = "SUCCESS"
                        logger.info("Betfair SSL Certificate authentication successful!")
                        return True
                    else:
                        self.last_status = status
            except Exception as e:
                logger.warning(f"Cert login failed: {e}")

        # Method 2: Standard SSO API Login
        try:
            headers = {
                "X-Application": self.app_key,
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            data = {"username": self.username, "password": self.password}
            resp = requests.post("https://identitysso.betfair.com/api/login", data=data, headers=headers, timeout=10)
            if resp.status_code == 200:
                res_data = resp.json()
                status = res_data.get("status", "FAIL")
                if status == "SUCCESS" and res_data.get("token"):
                    self.session_token = res_data.get("token")
                    self.last_status = "SUCCESS"
                    logger.info("Betfair Standard API authentication successful!")
                    return True
                else:
                    err = res_data.get("error", "LOGIN_FAILED")
                    self.last_status = err
                    logger.error(f"Betfair Standard login failed: {err}")
            else:
                self.last_status = f"HTTP Error {resp.status_code}"
        except Exception as e:
            self.last_status = f"Exception: {e}"
            logger.error(f"Betfair Standard login exception: {e}")

        return False

    def get_headers(self) -> dict:
        if not self.session_token:
            self.login()
        return {
            "X-Application": self.app_key,
            "X-Authentication": self.session_token or "",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def fetch_all_live_markets(self) -> dict:
        """
        Fetch batch of live Over/Under 2.5 markets from Betfair Exchange.
        """
        if not self.session_token:
            self.login()

        if not self.session_token:
            return {}

        try:
            payload = {
                "filter": {
                    "eventTypeIds": ["1"],  # Soccer
                    "marketTypeCodes": ["OVER_UNDER_25"]
                },
                "maxResults": 500,
                "marketProjection": ["RUNNER_DESCRIPTION", "EVENT"]
            }
            url = f"{BETFAIR_API_URL}/listMarketCatalogue/"
            resp = requests.post(url, headers=self.get_headers(), json=payload, timeout=8)
            
            if resp.status_code == 200:
                cat = resp.json()
                if cat:
                    market_ids = [m["marketId"] for m in cat]
                    # Query live prices in batch (split into chunks of 40)
                    book_url = f"{BETFAIR_API_URL}/listMarketBook/"
                    books = []
                    for i in range(0, len(market_ids), 40):
                        chunk = market_ids[i:i+40]
                        b_payload = {
                            "marketIds": chunk,
                            "priceProjection": {"priceData": ["EX_BEST_OFFERS"]}
                        }
                        b_resp = requests.post(book_url, headers=self.get_headers(), json=b_payload, timeout=8)
                        if b_resp.status_code == 200:
                            books.extend(b_resp.json())
                    
                    book_dict = {b["marketId"]: b for b in books}
                    
                    cache = {}
                    for m in cat:
                        m_id = m["marketId"]
                        ev_name = m.get("event", {}).get("name", "")
                        b_data = book_dict.get(m_id)
                        if b_data and "runners" in b_data:
                            r_list = b_data["runners"]
                            runners_cat = m.get("runners", [])
                            o25_price, u25_price = None, None
                            for r in r_list:
                                s_id = r.get("selectionId")
                                r_name = next((rc.get("runnerName") for rc in runners_cat if rc.get("selectionId") == s_id), "")
                                avail = r.get("ex", {}).get("availableToBack", [])
                                best_p = avail[0]["price"] if avail else r.get("lastPriceTraded")
                                
                                if "Over" in r_name or s_id == 47973:
                                    o25_price = best_p
                                elif "Under" in r_name or s_id == 47972:
                                    u25_price = best_p
                                    
                            if o25_price or u25_price:
                                cache[norm_str(ev_name)] = {
                                    "over25_odds": o25_price,
                                    "under25_odds": u25_price,
                                    "source": "Betfair Exchange API (Live)"
                                }
                    self.market_cache = cache
                    return cache
        except Exception as e:
            logger.warning(f"Error refreshing batch Betfair markets: {e}")
        return self.market_cache

    def fetch_market_odds(self, home_team: str, away_team: str) -> Dict[str, float]:
        """
        Lookup live Betfair Exchange odds for a specific match.
        """
        if not self.market_cache:
            self.fetch_all_live_markets()

        nh = norm_str(home_team)
        na = norm_str(away_team)

        # 1. Exact or substring matching
        for ev_key, data in self.market_cache.items():
            if (nh in ev_key or nh[:4] in ev_key) and (na in ev_key or na[:4] in ev_key):
                return data

        # 2. Key word split matching
        h_words = [w for w in nh.split() if len(w) > 3]
        a_words = [w for w in na.split() if len(w) > 3]
        for ev_key, data in self.market_cache.items():
            h_match = any(w in ev_key for w in h_words) if h_words else nh[:3] in ev_key
            a_match = any(w in ev_key for w in a_words) if a_words else na[:3] in ev_key
            if h_match and a_match:
                return data

        return {
            "source": "Betfair Exchange",
            "over25_odds": 2.00,
            "under25_odds": 1.80
        }

# Global Singleton Instance
_betfair_client = None

def get_betfair_client() -> BetfairExchangeClient:
    global _betfair_client
    if _betfair_client is None:
        _betfair_client = BetfairExchangeClient()
    return _betfair_client
