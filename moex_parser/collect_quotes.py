from urllib.parse import urlencode
from scrapy.exceptions import IgnoreRequest
from typing import *

import scrapy
import pandas as pd
import json


params = {
    "iss.meta": "off",
    "iss.json": "extended",
    "callback": "JSON_CALLBACK",
    "lang": "ru",
    "security_collection": 202,
    "date": None,
    "start": 0,
    "limit": 100,
    "sort_column": "VALUE",
    "sort_order": "desc",
}


URL: str = (
    "https://iss.moex.com/iss/history/engines/stock/markets/bonds/boardgroups/58/securities.json"
)

START_DATE: str = "2017-01-01"
END_DATE: str = "2024-01-06"


class BondSpider(scrapy.Spider):

    name = "bond_spider"

    custom_settings = {"LOG_LEVEL": "INFO"}

    def __init__(self) -> Self:
        super().__init__()

    def start_requests(self) -> Iterable[scrapy.Request]:
        for date in pd.date_range(start=START_DATE, end=END_DATE, freq="D"):
            params["date"] = date
            yield scrapy.Request(
                url=f"{URL}?{urlencode(params)}", callback=self.parse_date
            )

    def parse_date(self, response) -> Iterable[Dict[str, Any]]:

        if response.status != 200:
            # ignore this request, do not attempt to parse it
            raise IgnoreRequest()

        data = json.loads(response.body)

        for bond in data[1]["history"]:
            yield bond
