from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.spider import BaseSpider
from scrapy.selector import HtmlXPathSelector
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import HtmlXPathSelector
from scrapy.item import Item, Field
import re
import os
import requests

MIDI_MUSIC_PATH = "/media/vik/FreeAgent GoFlex Drive/Music/evolve/midi"

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

def join_path(p1,p2):
    return os.path.abspath(os.path.join(p1,p2))

class MidiSpider(CrawlSpider):
    ltype = "none"
    name = "midi"
    allowed_domains = ['www.freemidi.org', 'freemidi.org']
    start_urls = ["http://freemidi.org/genre-hip-hop-rap"]
    rules = [
        Rule(SgmlLinkExtractor(allow=["/directory-\d+"])),
        Rule(SgmlLinkExtractor(allow=['download-\d+-\d+']), 'parse_midi')
    ]

    def parse_midi(self, response):
        url = response.url
        ltypes = ["electronic","hiphop"]
        for l in ltypes:
            jp = os.path.abspath(os.path.join(MIDI_MUSIC_PATH,l))
            if not os.path.isdir(jp):
                os.mkdir(jp)

        try:
            x = HtmlXPathSelector(response)
            link = Link()
            music_link = x.select('//span[@class="download-button-one"]/a/@href').extract()[0]
            artist = x.select("//strong/text()").extract()[1].replace(" Free Midi","").replace("Other  ","")
            name = x.select('//span[@class="download-button-one"]/a/text()').extract()[0].replace("Download ","")
            link['url'] = url
            link['ltype'] = self.ltype
            link['link'] = "http://freemidi.org/" + music_link
            link['name'] = name.strip().replace(" ","_")
            link['artist'] = artist.strip().replace(" ","_")
            fpath = join_path(join_path(MIDI_MUSIC_PATH,self.ltype),link['artist'] + link['name'])
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.31 (KHTML, like Gecko) Chrome/26.0.1410.63 Safari/537.31'
                }
                headers.update({"Referer" : response.url})
                if not os.path.isfile(fpath):
                    r = requests.get(link['link'],headers=headers)
                    f = open(fpath, 'wb')
                    f.write(r.content)
                    f.close()
            except Exception:
                log.exception("Could not get music file.")
                return None
            link['path'] = fpath
            return link
        except Exception:
            log.exception("Could not parse url {0}".format(url))
            return None

class HipHopSpider(MidiSpider):
    ltype = "hiphop"
    name = "midih"
    start_urls = ["http://freemidi.org/genre-hip-hop-rap"]

class DanceSpider(MidiSpider):
    ltype = "electronic"
    name = "midid"
    start_urls = ["http://freemidi.org/genre-dance"]

class MWSpider(BaseSpider):
    name = "mws"
    ltype = "classical"
    allowed_domains = ["www.midiworld.com","midiworld.com"]
    start_urls = ["http://www.midiworld.com/classic.htm"]

    def parse(self, response):
        x = HtmlXPathSelector(response)
        links = []
        url = response.url
        music_links = x.select('//ul/li/a/@href').extract()
        music_links = [m for m in music_links if m.endswith(".mid")]
        for l in music_links:
            link = Link()
            link['url'] = url
            link['ltype'] = self.ltype
            link['link'] = l
            links.append(link)
        return links

class MASpider(BaseSpider):
    name = "mas"
    ltype = "modern"
    allowed_domains = ["midi-archive.com","www.midi-archive.com"]
    start_urls = ["http://midi-archive.com/"]

    def parse(self, response):
        x = HtmlXPathSelector(response)
        links = []
        url = response.url
        music_links = x.select("//td/a/@href").extract()
        music_links = [m for m in music_links if m.endswith(".mid")]
        for l in music_links:
            link = Link()
            link['url'] =  url
            link['ltype'] = self.ltype
            link['link'] = "http://midi-archive.com/" + l
            links.append(link)
        return links
