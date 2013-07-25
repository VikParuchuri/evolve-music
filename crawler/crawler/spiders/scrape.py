from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.spider import BaseSpider
from scrapy.selector import HtmlXPathSelector
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import HtmlXPathSelector
from scrapy.item import Item, Field
import re

import logging
log = logging.getLogger(__name__)

lfm_electronic_base_url = "http://www.last.fm/music/+free-music-downloads/electronic"
lfm_classical_base_url = "http://www.last.fm/music/+free-music-downloads/classical"

lfme_urls = [lfm_electronic_base_url+"/?page={0}".format(i) for i in xrange(1,25)]
lfmc_urls = [lfm_classical_base_url+"/?page={0}".format(i) for i in xrange(1,25)]

class Link(Item):
    url = Field()
    link = Field()
    ltype = Field()

class MusicSpider(BaseSpider):
    name = "lfm"
    allowed_domains = ['www.last.fm', 'last.fm']
    start_urls = [lfm_electronic_base_url, lfm_classical_base_url] + lfme_urls + lfmc_urls

    def parse(self, response):
        x = HtmlXPathSelector(response)
        links = []
        url = response.url
        ltype = url.split('/')[-1]
        music_links = x.select('//a[@class="lfmButton lfmBigButton lfmFreeDownloadButton"]/@href').extract()
        for l in music_links:
            link = Link()
            link['url'] = url
            link['ltype'] = ltype
            link['link'] = l
            links.append(link)
        return links