import requests

from wikiwho.wikiwho import Wikiwho
from wikiwho.utils import iter_rev_tokens


def process_api_output(page_id):
    # you check here for the explanation of the api call
    # https://www.mediawiki.org/wiki/API:Revisions
    url = 'https://en.wikipedia.org/w/api.php'
    params = {'pageids': page_id, 'action': 'query', 'prop': 'revisions',
              'rvprop': 'content|ids|timestamp|sha1|comment|flags|user|userid',
              'rvlimit': 'max', 'format': 'json', 'continue': '', 'rvdir': 'newer'}

    # gets only first 50 revisions of given page
    result = requests.get(url=url, params=params).json()
    if 'error' in result:
        raise Exception('Wikipedia API returned the following error:' + str(result['error']))

    pages = result['query']['pages']
    if "-1" in pages:
        raise Exception('The article ({}) you are trying to request does not exist!'.format(page_id))

    _, page = result['query']['pages'].popitem()
    wikiwho = Wikiwho(page['title'])
    wikiwho.analyse_article(page.get('revisions', []))
    wikiwho.rvcontinue = result['continue']['rvcontinue']
    return wikiwho

if __name__ == '__main__':
    page_id = 6187
    wikiwho_obj = process_api_output(page_id)
    print(wikiwho_obj.title)
    print(wikiwho_obj.ordered_revisions)
    for token in iter_rev_tokens(wikiwho_obj.revisions[wikiwho_obj.ordered_revisions[0]]):
        print(token.value, token.token_id, token.origin_rev_id)
