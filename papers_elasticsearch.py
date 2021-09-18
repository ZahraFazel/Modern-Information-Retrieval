from elasticsearch import Elasticsearch
from json import load
import numpy


def index(data_path, elasticsearch_host, elasticsearch_port):
    es = Elasticsearch([{'host': elasticsearch_host, 'port': elasticsearch_port}])
    es.indices.create(index='paper-index', ignore=400)
    with open(data_path, 'r', encoding='utf8') as infile:
        papers_data = load(infile)
    for i in range(len(papers_data)):
        es.index(index='paper_index', id=i, body={'paper': papers_data[i]})


def delete_index(elasticsearch_host, elasticsearch_port):
    es = Elasticsearch([{'host': elasticsearch_host, 'port': elasticsearch_port}])
    es.indices.delete(index='paper_index', ignore=[400, 404])


def calculate_page_rank(alpha, elasticsearch_host, elasticsearch_port):
    es = Elasticsearch([{'host': elasticsearch_host, 'port': elasticsearch_port}])
    total = es.count(index='paper_index', body={'query': {'match_all': {}}})['count']
    responses = es.search(index='paper_index', body={'size': total, 'query': {'match_all': {}}})['hits']['hits']
    web_graph = {}
    pages = {}
    for response in responses:
        page_id = response['_source']['paper']['id']
        pages[page_id] = int(response['_id'])
    for response in responses:
        page_id = response['_source']['paper']['id']
        web_graph[pages[page_id]] = []
        for reference in response['_source']['paper']['references']:
            reference_id = reference.split('/')[-1]
            if reference_id in pages.keys():
                web_graph[pages[page_id]].append(pages[reference_id])
    probability_matrix = [[alpha / len(pages) for _ in range(len(pages))] for __ in range(len(pages))]
    for page in web_graph.keys():
        for reference in web_graph[page]:
            probability_matrix[page][reference] += (1 - alpha) * (1 / len(web_graph[page]))
    probability_matrix = numpy.asarray(probability_matrix)
    page_rank = numpy.transpose([1 for _ in range(len(pages))])
    eigenvalue = numpy.dot(numpy.transpose(page_rank), probability_matrix.dot(page_rank)) / \
                 numpy.dot(numpy.transpose(page_rank), page_rank)
    converge = False
    iteration = 0
    while not converge and iteration < 200000:
        page_rank_new = probability_matrix.dot(page_rank)
        page_rank_new /= numpy.linalg.norm(page_rank_new)
        eigenvalue_new = numpy.dot(numpy.transpose(page_rank_new), probability_matrix.dot(page_rank_new)) / \
                         numpy.dot(numpy.transpose(page_rank_new), page_rank_new)
        converge = abs(eigenvalue_new - eigenvalue) / eigenvalue_new <= 10 ** (-9)
        iteration += 1
        eigenvalue = eigenvalue_new
        page_rank = page_rank_new
    for id in pages.values():
        es.update(index='paper_index', id=id, body={'doc': {'paper': {'page_rank': page_rank[id]}}})
    print(page_rank)
    print(numpy.sum(page_rank))


def search(title, abstract, date, elasticsearch_host, elasticsearch_port, use_page_rank):
    es = Elasticsearch([{'host': elasticsearch_host, 'port': elasticsearch_port}])
    query = {'query': {
        'function_score': {'query': {'bool': {'should': [
                {'match': {"paper.title": {"query": title['phrase'], "boost": title['weight']}}},
                {'match': {"paper.abstract": {"query": abstract['phrase'], "boost": abstract['weight']}}},
                {'range': {"paper.date": {"gte": date['from'], "boost": date['weight']}}}
            ]}},
            'functions': [{"field_value_factor": {
                            "field": "paper.page_rank",
                            "factor": 10000 if use_page_rank else 0,
                            "modifier": "ln1p",
                            "missing": 0
            }}],
            "boost_mode": "sum"
        }
    }}
    response = es.search(index='paper_index', body=query)['hits']['hits']
    return response


def run_hits(number_of_authors, elasticsearch_host, elasticsearch_port):
    es = Elasticsearch([{'host': elasticsearch_host, 'port': elasticsearch_port}])
    total = es.count(index='paper_index', body={'query': {'match_all': {}}})['count']
    responses = es.search(index='paper_index', body={'size': total, 'query': {'match_all': {}}})['hits']['hits']
    authors = {}
    id_authors = {}
    id_references = {}
    id = 0
    for response in responses:
        paper_id = response['_source']['paper']['id']
        for author in response['_source']['paper']['authors']:
            if author not in authors.keys():
                authors[author] = id
                id += 1
        id_authors[paper_id] = response['_source']['paper']['authors']
        id_references[paper_id] = [reference.split('/')[-1] for reference in response['_source']['paper']['references']]
    reference_graph = [[0 for _ in range(len(authors))] for __ in range(len(authors))]
    for id in id_authors.keys():
        for author in id_authors[id]:
            for reference in id_references[id]:
                if reference in id_authors.keys():
                    for reference_author in id_authors[reference]:
                        reference_graph[authors[author]][authors[reference_author]] = 1
    hub = [1 for _ in range(len(authors))]
    authority = [1 for _ in range(len(authors))]
    for i in range(5):
        new_hub = [0 for _ in range(len(authors))]
        new_authority = [0 for _ in range(len(authors))]
        for id_author in authors.values():
            for id_reference_author in authors.values():
                new_hub[id_author] += authority[id_reference_author] * reference_graph[id_author][id_reference_author]
                new_authority[id_author] += hub[id_reference_author] * reference_graph[id_reference_author][id_author]
        hub = new_hub
        authority = new_authority
    hub = list(numpy.asarray(hub) / numpy.linalg.norm(numpy.asarray(hub)))
    authority = list(numpy.asarray(authority) / numpy.linalg.norm(numpy.asarray(authority)))
    authority = {i: authority[i] for i in range(len(authority))}
    authority = sorted(authority.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    best_authors = {}
    for i in range(number_of_authors):
        index = authority[i][0]
        for author_name in authors.keys():
            if authors[author_name] == index:
                best_authors[author_name] = authority[i][1]
    return best_authors
