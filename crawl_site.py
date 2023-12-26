#!python3

# Copied from -
# https://dev.to/pjcalvo/broken-links-checker-with-python-and-scrapy-webcrawler-1gom
# Execute via:
#    scrapy runspider linkchecker.py -o ~/tmp/broken-links.csv
# Use a webtool @ https://www.brokenlinkcheck.com/broken-links.php#status


from typing import Dict
import typer
from bs4 import BeautifulSoup

from icecream import ic
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.item import Item, Field
from scrapy import cmdline
import html2text
from scrapy import signals

app = typer.Typer()

site = "whatilearnedsofar.com"
crawled_artciles = set()
glossary: Dict[str, str] = {}


class MyItems(Item):
    referer = Field()  # where the link is extracted
    response = Field()  # url that was requested
    status = Field()  # status code received


def process_page(response):
    good_path = response.url.split(site)[1]
    if "?replytocom" in good_path:
        return

    # get content of site and pass to beautifulsoup
    soup = BeautifulSoup(response.text, "html.parser")
    # find the article content, and print it
    article = soup.find("article")
    # convert article to markdown
    converter = html2text.HTML2Text()
    converter.ignore_links = True
    article_md = converter.handle(str(article))

    def remove_usage_from_glossary(article_md):
        # remove the usage from the glossary
        items = article_md.split("#### Usage")
        if len(items) > 1:
            return items[0]
        return article_md

    article_md = remove_usage_from_glossary(article_md)

    if article_md in crawled_artciles:
        return

    if good_path.startswith("/glossary"):
        # first line of glossary is the # word, make that the key, and the rest the value
        lines = article_md.split("\n")
        key = lines[0].replace("# ", "")
        value = "\n".join(lines[1:])
        glossary[key] = value
        return

    crawled_artciles.add(article_md)
    ic(good_path)
    ic(article_md)

    # list of response codes that we want to include on the report, we know

    # that 404
    report_if = [404]
    if response.status in report_if:  # if the response matches then creates a MyItem
        item = MyItems()
        item["referer"] = response.request.headers.get("Referer", None)
        item["status"] = response.status
        item["response"] = response.url
        yield item
    yield None  # if the response did not match return emptyo


class MySpider(CrawlSpider):
    name = "test-crawler"
    site = "whatilearnedsofar.com"
    target_domains = [site]  # list of domains that will be allowed to be crawled
    start_urls = [f"https://{site}"]  # list of starting urls for the crawler
    handle_httpstatus_list = [
        404,
        410,
        500,
    ]  # only 200 by default. you can add more status to list

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    # Throttle crawl speed to prevent hitting site too hard
    custom_settings = {
        "CONCURRENT_REQUESTS": 20,  # Some requests timeout, so have plenty of threads.
        "DOWNLOAD_DELAY": 0.05,  # delay between requests
        "REDIRECT_ENABLED": True,
        "RETRY_ENABLED": False,
    }

    rules = [
        Rule(
            LinkExtractor(
                allow_domains=target_domains,
                deny=("patterToBeExcluded"),
                unique=("Yes"),
            ),
            callback=process_page,
            follow=True,
        ),
        # crawl external links but don't follow them
        # I don't follow what don't follow means - seems like it's actually crawling
        # Rule(
        # LinkExtractor(allow=(""), deny=("patterToBeExcluded"), unique=("Yes")),
        # callback=parse_my_url,
        # follow=False,
        # ),
    ]

    def spider_closed(self, spider):
        print("spider closed")
        ic(glossary)


@app.command()
def un():
    from unstructured.partition.html import partition_html
    from unstructured.chunking.title import chunk_by_title

    url = "https://idvork.in/manager-book"
    elements = partition_html(url=url)
    chunks = chunk_by_title(elements)

    for chunk in chunks:
        print(chunk)
        print("\n\n" + "-" * 80)
        input()


@app.command()
def crawl():
    cmdline.execute("scrapy runspider crawl_site.py".split())


if __name__ == "__main__":
    app()
