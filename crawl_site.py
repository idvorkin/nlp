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
from scrapy import cmdline
import html2text
from scrapy import signals
from pathlib import Path

site = "whatilearnedsofar.com"
crawled_articles = set()
glossary: Dict[str, str] = {}
output_root = Path().home() / "tmp/whatilearnedsofar"


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
    converter.body_width = 0  # no wrapping
    article_md = converter.handle(str(article))

    def remove_usage_from_glossary(article_md):
        # remove the usage from the glossary
        items = article_md.split("#### Usage")
        if len(items) > 1:
            return items[0]
        return article_md

    article_md = remove_usage_from_glossary(article_md)

    if article_md in crawled_articles:
        return

    if good_path.startswith("/glossary"):
        # first line of glossary is the # word, make that the key, and the rest the value
        lines = article_md.split("\n")
        key = lines[0].replace("# ", "")
        value = "\n".join(lines[1:])
        glossary[key] = value
        yield

    crawled_articles.add(article_md)
    ic(good_path)
    ic(article_md)

    # write the file to disk
    output_path = output_root / (good_path[1:-1] + ".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(article_md)
    yield


class MySpider(CrawlSpider):
    name = "whatilearnedsofar"
    start_urls = [f"https://{site}"]  # list of starting urls for the crawler

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(crawl_complete, signal=signals.spider_closed)
        return spider

    rules = [
        Rule(
            LinkExtractor(
                allow_domains=[site],
                unique=True,
            ),
            callback=process_page,
            follow=True,
        ),
    ]


def crawl_complete(spider):  # noqa
    ic(glossary)
    ic(spider)
    glossary_path = output_root / "glossary.md"
    glossary_path.write_text("\n".join([f"## {k}\n{v}" for k, v in glossary.items()]))


app = typer.Typer()


@app.command()
def crawl():
    cmdline.execute("scrapy runspider crawl_site.py".split())


if __name__ == "__main__":
    app()
