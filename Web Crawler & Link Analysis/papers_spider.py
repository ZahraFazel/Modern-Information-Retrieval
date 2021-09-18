from scrapy import Request, Spider
from twisted.internet import reactor
from scrapy.crawler import CrawlerRunner
from re import findall
from json import dump
from math import ceil, log10


data_path = 'D:\\University\\Modern Information Retrieval\\Project\\Phase 3\\data\\data.json'


class PaperSpider(Spider):
    name = 'paper_spider'
    seed = []
    number_of_papers = 0
    papers = []
    custom_settings = {'DEPTH_LIMIT': 2}

    def start_requests(self):
        for url in self.seed:
            yield Request(url=url, callback=self.parse)

    def parse(self, response):
        if len(self.papers) < self.number_of_papers:
            info = response.css('pre.bibtex-citation::text').get()
            title = findall('title={.*}', info)[0].replace('title={', '').replace('}', '')
            authors = findall('author={.*}', info)[0].replace('author={', '').replace('}', '').split(' and ')
            year = findall('year={.*}', info)[0].replace('year={', '').replace('}', '') \
                if len(findall('year={.*}', info)) > 0 else ''
            abstract = response.xpath("//meta[@name='description']/@content")[0].extract() \
                if len(response.xpath("//meta[@name='description']/@content")) > 0 else ''
            references = ['https://www.semanticscholar.org' + reference.css('h2.citation__title a::attr(href)').extract()[0]
                          for reference in response.css('div.references').css('div.paper-citation')]
            self.papers.append({'id': response.url.split("/")[-1], 'title': title, 'authors': authors, 'date': year,
                                'abstract': abstract, 'references': references})
            for reference in references:
                yield Request(url=reference, callback=self.parse)


def run(seed, number_of_papers=2000):
    PaperSpider.seed = seed
    PaperSpider.number_of_papers = number_of_papers
    PaperSpider.custom_settings['DEPTH_LIMIT'] = ceil(log10(number_of_papers / len(seed))) + 2
    runner = CrawlerRunner()
    d = runner.crawl(PaperSpider)
    d.addBoth(lambda _: reactor.stop())  # @UndefinedVariable
    reactor.run()  # @UndefinedVariable
    with open(data_path, 'w', encoding='utf8') as outfile:
        dump(PaperSpider.papers, outfile, ensure_ascii=False)
