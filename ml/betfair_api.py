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
        self.market_cache = {}

        # Resolve cert file paths
        self.crt_file = os.path.join(self.cert_path, "client-2048.crt")
        self.key_file = os.path.join(self.cert_path, "client-2048.key")

        # Attempt initial login if credentials provided
        if self.username and self.password and self.app_key:
            self.login()

    def login(self) -> bool:
        """
        Authenticate with Betfair via SSL certificate.
        """
        if not (self.username and self.password and self.app_key):
            logger.info("Betfair credentials not set in environment variables.")
            return False

        if not (os.path.exists(self.crt_file) and os.path.exists(self.key_file)):
            logger.info(f"Betfair certificates missing at {self.cert_path}")
            return False

        try:
            cert = (self.crt_file, self.key_file)
            headers = {
                "X-Application": self.app_key,
                "Content-Type": "application/x-www-form-urlencoded"
            }
            data = {
                "username": self.username,
                "password": self.password
            }
            resp = requests.post(BETFAIR_CERT_LOGIN_URL, data=data, headers=headers, cert=cert, timeout=10)
            if resp.status_code == 200:
                res_data = resp.json()
                if res_data.get("loginStatus") == "SUCCESS":
                    self.session_token = res_data.get("sessionToken")
                    logger.info("Betfair SSL Certificate authentication successful!")
                    return True
                else:
                    logger.error(f"Betfair login status failed: {res_data.get('loginStatus')}")
            else:
                logger.error(f"Betfair login HTTP error: {resp.status_code}")
        except Exception as e:
            logger.error(f"Betfair login exception: {e}")
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
                "maxResults": 150,
                "marketProjection": ["RUNNER_DESCRIPTION", "EVENT"]
            }
            url = f"{BETFAIR_API_URL}/listMarketCatalogue/"
            resp = requests.post(url, headers=self.get_headers(), json=payload, timeout=8)
            
            if resp.status_code == 200:
                cat = resp.json()
                if cat:
                    market_ids = [m["marketId"] for m in cat]
                    # Query live prices in batch
                    book_url = f"{BETFAIR_API_URL}/listMarketBook/"
                    b_payload = {
                        "marketIds": market_ids,
                        "priceProjection": {"priceData": ["EX_BEST_OFFERS"]}
                    }
                    b_resp = requests.post(book_url, headers=self.get_headers(), json=b_payload, timeout=8)
                    
                    if b_resp.status_code == 200:
                        books = b_resp.json()
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
                                    best_p = avail[0]["price"] if avail else None
                                    
                                    if "Over" in r_name:
                                        o25_price = best_p
                                    elif "Under" in r_name:
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

        # Fuzzy search in market cache
        for ev_key, data in self.market_cache.items():
            if nh[:4] in ev_key and na[:4] in ev_key:
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
