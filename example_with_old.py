import requests
from WikiWho.wikiwho_old import Wikiwho
from WikiWho.utils import iter_rev_tokens, browse_dict, Timer


class WikiPage(object):

    def __init__(self, page_id: int, lng: str='en', start_from: str=None):
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

        self.url = f'https://{lng}.wikipedia.org/w/api.php'
        self.params = {
            'pageids': page_id,
            'action': 'query',
            'prop': 'revisions',
            'rvprop': 'content|ids|timestamp|sha1|comment|flags|user|userid',
            'rvlimit': 'max',
            'format': 'json',
            'continue': '',
            'rvdir': 'newer',
        }

        if start_from is not None:

            self.params['rvstart'] = start_from

        self.result = self.request(self.url, self.params)
        self.page = self.get_page(self.result)

    def request(self, url, params) -> dict:
        # gets only first 50 revisions of given page
        result = requests.get(url=url, params=params).json()

        if 'error' in result:
            raise Exception(
                'Wikipedia API returned the following error:' + str(result['error']))

        return result

    def get_page(self, result):
        pages = result['query']['pages']
        if "-1" in pages:
            raise Exception(
                'The article ({}) you are trying to request does not exist!'.format(page_id))

        _, page = result['query']['pages'].popitem()
        if 'missing' in page:
            raise Exception(
                'The article ({}) you are trying to request does not exist!'.format(page_id))

        print(f"Loading revisions starting from {page['revisions'][-1]['revid']} "
              f"({page['revisions'][-1]['timestamp']})")

        return page

    def get_title(self):
        return self.page['title']

    def revisions(self):
        yield self.page.get('revisions', [])

        while 'continue' in self.result:
            self.params['rvcontinue'] = self.result['continue']['rvcontinue']
            self.result = self.request(self.url, self.params)
            self.page = self.get_page(self.result)
            yield self.page.get('revisions', [])


def process_wiki_page(wiki_page: WikiPage) -> Wikiwho:
    """Find the original author of each token in a wikipedia page. It
    uses Wikiwho for finding such article.

    Args:
        wiki_page (WikiPage): a representation of the wikipedia page that contains all
            the revisions

    Returns:
        Wikiwho: The object that contains the authorship of each token in the
            article.
    """
    wikiwho = Wikiwho(wiki_page.get_title())

    for revisions in wiki_page.revisions():
        with Timer():
            wikiwho.analyse_article(revisions)

    return wikiwho


if __name__ == '__main__':
    # Wikipedia article id (e.g. 6187 for Cologne)
    wiki_page = WikiPage(6187) #, start_from='2017-08-19T18:23:42Z')

    # Process the page to find the authorships of tokens
    wikiwho_obj = process_wiki_page(wiki_page)

    import ipdb; ipdb.set_trace()  # breakpoint 632ac25c //


    print(wikiwho_obj.title)
    print(wikiwho_obj.ordered_revisions)

    for token in iter_rev_tokens(wikiwho_obj.revisions[wikiwho_obj.ordered_revisions[0]]):
        print(token.value, token.token_id, token.origin_rev_id)
