from crawler.crawler.spiders.papers_spider import run, data_path
from papers_elasticsearch import *
import ast
import re

print("Help:"
      "\n\t* Pay attention to whitespaces!"
      "\n\tTo run:"
      "\n\t\tpart 1 enter: \'crawl seed:[%link 1%, %link 2%, ..., %link n%] %number of pages%\' or "
      "\'crawl seed:[%link 1%, %link 2%, ..., %link n%]\'"
      "\n\t\tpart 2 to load data in elasticsearch enter: \'index %data_path% %elasticsearch_host% %elasticsearch_port%\'"
      "\n\t\tpart 2 to delete index enter: \'delete index %elasticsearch_host% %elasticsearch_port%\'"
      "\n\t\tpart 3 enter: \'rank pages %alpha% %elasticsearch_host% %elasticsearch_port%\'"
      "\n\t\tpart 4 enter: \'search title:{'phrase': %\'phrase\'%, 'weight': %weight%} "
      "abstract:{'phrase': %\'phrase\'%, 'weight': %weight%} date:{'from': %\'year\'%, 'weight': %weight%} "
      "%True if page rank should be used else False% %elasticsearch_host% %elasticsearch_port%\'"
      "\n\t\tpart 5 enter: \'run HITS %number of authors% %elasticsearch_host% %elasticsearch_port%"
      "\n\tEnter 'exit' to exit")
while True:
    command = input()
    command_parts = command.split(' ')
    if command_parts[0] == 'exit':
        break
    elif command_parts[0] == 'crawl':
        if len(command_parts) > 2:
            run(ast.literal_eval(command_parts[1].replace('seed:', '')), int(command_parts[2]))
        else:
            run(ast.literal_eval(command_parts[1].replace('seed:', '')))
        print("Crawler finished. Data stored in \'" + data_path + '\'')
    elif command_parts[0] == 'index':
        path = command_parts[1]
        for i in range(2, len(command_parts) - 2):
            path += " " + command_parts[i]
        index(path, command_parts[-2], int(command_parts[-1]))
        print("Index created successfully.")
    elif command_parts[0] == 'delete':
        delete_index(command_parts[2], int(command_parts[3]))
        print("Index deleted successfully.")
    elif command_parts[0] == 'rank':
        calculate_page_rank(float(command_parts[2]), command_parts[3], int(command_parts[4]))
        print("Calculation of page rank completed.")
    elif command_parts[0] == 'search':
        title_pattern = re.compile('title:{.*} *abstract')
        abstract_pattern = re.compile('abstract:{.*} *date')
        title = ast.literal_eval(re.sub(' *abstract', '', title_pattern.search(command).group(0)).replace('title:', ''))
        abstract = ast.literal_eval(re.sub(' *date', '', abstract_pattern.search(command).group(0)).replace('abstract:', ''))
        date = ast.literal_eval(re.findall('date:{.*}', command)[0].replace('date:', ''))
        use_page_rank = True if command_parts[-3] == 'True' else False
        results = search(title, abstract, date, command_parts[-2], int(command_parts[-1]), use_page_rank)
        print('Results are:')
        for i in range(len(results)):
            print('\t' + str(i + 1) + ':')
            print('\t\tTitle: ' + results[i]['_source']['paper']['title'])
            print('\t\tAuthors: ')
            for author in results[i]['_source']['paper']['authors']:
                print('\t\t\t' + author)
            print('\t\tDate: ' + results[i]['_source']['paper']['date'])
            print('\t\tAbstract: \n\t\t\t' + results[i]['_source']['paper']['abstract'].replace('\n', '\n\t\t\t'))
    elif command_parts[0] == 'run':
        results = run_hits(int(command_parts[2]), command_parts[3], int(command_parts[4]))
        print('Top ' + command_parts[2] + ' authors are:')
        i = 0
        for author, authority in results.items():
            print('\t' + str(i + 1) + ':')
            print('\t\tَAuthor: ' + author)
            print('\t\tAuthority: ' + str(authority))
            i += 1
