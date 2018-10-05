import requests
from WikiWho.wikiwho import Wikiwho
from WikiWho.utils import iter_rev_tokens


def get_wiki_page(page_id: int) -> dict:
    """
    Call the Wikipedia API for a specific article
    
    You can check here for the explanation of the api call
    https://www.mediawiki.org/wiki/API:Revisions
    
    Args:
        page_id (int): the id of the article
    
    Returns:
        dict: the JSON (as a dict) with the contents of the API request
    
    Raises:
        Exception: whenever an unknown error occurs or a the pages was not found
    """
    url = 'https://en.wikipedia.org/w/api.php'
    params = {'pageids': page_id, 'action': 'query', 'prop': 'revisions',
              'rvprop': 'content|ids|timestamp|sha1|comment|flags|user|userid',
              'rvlimit': 'max', 'format': 'json', 'continue': '', 'rvdir': 'newer'}

    # gets only first 50 revisions of given page
    result = requests.get(url=url, params=params).json()
    if 'error' in result:
        raise Exception(
            'Wikipedia API returned the following error:' + str(result['error']))

    pages = result['query']['pages']
    if "-1" in pages:
        raise Exception(
            'The article ({}) you are trying to request does not exist!'.format(page_id))

    _, page = result['query']['pages'].popitem()
    if 'missing' in page:
        raise Exception(
            'The article ({}) you are trying to request does not exist!'.format(page_id))

    import ipdb; ipdb.set_trace()  # breakpoint be970539 //


    return page


def process_wiki_page(page: dict) -> Wikiwho:
    """Find the original author of each token in a wikipedia page. It
    uses Wikiwho for finding such article.
    
    Args:
        page (dict): the JSON (as a dict) with the contents of the API request 
    
    Returns:
        Wikiwho: The object that contains the authorship of each token in the
            article.
    """
    wikiwho = Wikiwho(page['title'])
    wikiwho.analyse_article(page.get('revisions', []))
    wikiwho.rvcontinue = result['continue']['rvcontinue']
    return wikiwho


if __name__ == '__main__':
    # Wikipedia article id (e.g. 6187 for Cologne)
    page = get_wiki_page(6187)

    # Process the page to find the authorships of tokens
    wikiwho_obj = process_wiki_page(page)

    print(wikiwho_obj.title)
    print(wikiwho_obj.ordered_revisions)

    for token in iter_rev_tokens(wikiwho_obj.revisions[wikiwho_obj.ordered_revisions[0]]):
        print(token.value, token.token_id, token.origin_rev_id)
