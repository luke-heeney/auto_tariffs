from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


PERCENT_COUNTRY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*([A-Za-z][A-Za-z\s/.\-&]*?)"
    r"(?=\s*\d+(?:\.\d+)?\s*%|[,;]|$)"
)


def _country_key(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00a0", " ").replace("\uff05", "%")
    text = re.sub(r"[\._]", " ", text)
    text = re.sub(r"[^A-Za-z0-9/&\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().upper()
    return text


COUNTRY_ALIASES: dict[str, str] = {
    # United States
    "UNITED STATES": "United States",
    "UNITED STATES OF AMERICA": "United States",
    "USA": "United States",
    "US": "United States",
    "U S": "United States",
    "U S A": "United States",
    # United Kingdom. Raw AALA/NHTSA files sometimes record this as UK, GB,
    # Great Britain, or the first token UNITED when parsed with older logic.
    "UNITED KINGDOM": "United Kingdom",
    "GREAT BRITAIN": "United Kingdom",
    "UK": "United Kingdom",
    "GB": "United Kingdom",
    "GBR": "United Kingdom",
    "UNITED": "United Kingdom",
    # Main source-country codes in the parts-content files.
    "MEXICO": "Mexico",
    "M": "Mexico",
    "MX": "Mexico",
    "JAPAN": "Japan",
    "J": "Japan",
    "JP": "Japan",
    "KOREA": "Korea",
    "SOUTH KOREA": "Korea",
    "K": "Korea",
    "KR": "Korea",
    "GERMANY": "Germany",
    "G": "Germany",
    "DE": "Germany",
    "DEU": "Germany",
    "CHINA": "China",
    "CH": "China",
    "CHN": "China",
    "CANADA": "Canada",
    "CN": "Canada",
    "CA": "Canada",
    "AUSTRIA": "Austria",
    "A": "Austria",
    "AU": "Austria",
    "AT": "Austria",
    "BELGIUM": "Belgium",
    "BE": "Belgium",
    "BRAZIL": "Brazil",
    "BR": "Brazil",
    "CZECHIA": "Czechia",
    "CZECH REPUBLIC": "Czechia",
    "CZ": "Czechia",
    "DENMARK": "Denmark",
    "DK": "Denmark",
    "FINLAND": "Finland",
    "FN": "Finland",
    "FRANCE": "France",
    "F": "France",
    "FR": "France",
    "HUNGARY": "Hungary",
    "H": "Hungary",
    "HUN": "Hungary",
    "INDIA": "India",
    "IN": "India",
    "IND": "India",
    "ITALY": "Italy",
    "I": "Italy",
    "NETHERLANDS": "Netherlands",
    "N": "Netherlands",
    "PHILIPPINES": "Philippines",
    "P": "Philippines",
    "POLAND": "Poland",
    "PL": "Poland",
    "PORTUGAL": "Portugal",
    "PO": "Portugal",
    "SLOVAKIA": "Slovakia",
    "SL": "Slovakia",
    "SPAIN": "Spain",
    "SP": "Spain",
    "ESP": "Spain",
    "SWEDEN": "Sweden",
    "SW": "Sweden",
    "SE": "Sweden",
    "THAILAND": "Thailand",
    "TH": "Thailand",
    "TURKEY": "Turkey",
    "T": "Turkey",
    "VIETNAM": "Vietnam",
    "SERBIA": "Serbia",
    "SRB": "Serbia",
    "SOUTH AFRICA": "South Africa",
    "REPUBLIC OF SOUTH AFRICA": "South Africa",
    "SA": "South Africa",
    "SAF": "South Africa",
    "AF": "South Africa",
    "AUSTRALIA": "Australia",
    "AUS": "Australia",
    "TAIWAN": "Taiwan",
    "TW": "Taiwan",
    "OTHER": "Other",
    "OT": "Other",
}


def normalize_country_name(value: Any, *, default: str | None = None) -> str | None:
    if value is None:
        return default
    raw_text = str(value).strip()
    if raw_text.upper() in {"", "<NA>", "NA", "NAN", "NONE", "NULL"}:
        return default
    key = _country_key(value)
    if not key:
        return default
    if key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[key]
    if default is not None:
        return default
    return re.sub(r"\s+", " ", raw_text)


def normalize_country_series(series):
    return series.map(lambda value: normalize_country_name(value))


def normalize_country_tariffs(country_tariffs: Mapping[str, float] | None) -> dict[str, float] | None:
    if country_tariffs is None:
        return None
    out: dict[str, float] = {}
    for country, rate in country_tariffs.items():
        canonical = normalize_country_name(country)
        if canonical is None:
            continue
        if canonical in out and float(out[canonical]) != float(rate):
            raise ValueError(
                f"Conflicting tariff rates for canonical country {canonical}: "
                f"{out[canonical]} versus {rate}"
            )
        out[canonical] = float(rate)
    return out


def is_us_country(value: Any) -> bool:
    return normalize_country_name(value) == "United States"


def extract_percent_country_pairs(value: Any) -> list[tuple[float, str]]:
    if value is None:
        return []
    text = str(value).replace("\uff05", "%")
    if not text.strip():
        return []
    pairs: list[tuple[float, str]] = []
    for pct, raw_country in PERCENT_COUNTRY_RE.findall(text):
        try:
            pct_value = float(pct)
        except ValueError:
            continue
        country = normalize_country_name(raw_country)
        if country is None:
            continue
        pairs.append((pct_value, country))
    return pairs


def extract_raw_percent_country_pairs(value: Any) -> list[tuple[float, str, str | None]]:
    if value is None:
        return []
    text = str(value).replace("\uff05", "%")
    if not text.strip():
        return []
    pairs: list[tuple[float, str, str | None]] = []
    for pct, raw_country in PERCENT_COUNTRY_RE.findall(text):
        try:
            pct_value = float(pct)
        except ValueError:
            continue
        raw = raw_country.strip()
        pairs.append((pct_value, raw, normalize_country_name(raw)))
    return pairs


def unmapped_country_values(values) -> list[str]:
    out: list[str] = []
    for value in values:
        key = _country_key(value)
        if key and key not in COUNTRY_ALIASES:
            out.append(str(value).strip())
    return sorted(set(out))
