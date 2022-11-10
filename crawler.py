import datasets


# model composition website
main_page_url = "https://studymoose.com"

# get urls for pages with different themes
page_urls = datasets.get_page_urls(main_page_url)

# might fail dut to memory restriction or loss of web connection
# therefore, separate the task into several manageable subtasks instead
# page_urls page_urls[0:4] for instance
page_urls = page_urls[31:33]

# a list of articles
articles = []

# get articles in a list
for page_url in page_urls:
    num_subpage = datasets.get_num_subpage(page_url)
    subpage_urls = datasets.get_subpage_urls(page_url, num_subpage)
    article_urls = datasets.get_article_urls(subpage_urls)
    transformed_article_urls = datasets.transform_article_urls(article_urls)
    for transformed_article_url in transformed_article_urls:
        datasets.write2json(page_url[35:], datasets.get_articles(transformed_article_url), "./dataset/dataset.json", "a")
