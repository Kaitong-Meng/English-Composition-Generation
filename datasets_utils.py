# use "requests" "parsel" to scrape model
# compositions from www.studymoose.com
# use "json" to store and manipulate the dataset in .json format
import requests
import parsel
import json


def get_page_urls(main_page_url):
    main_page = requests.get(main_page_url)
    page_selector = parsel.Selector(main_page.text)
    page_urls = page_selector.xpath("/html/body/section[3]/div/div[2]/div/a/@href").getall()
    return page_urls


def get_num_subpage(page_url):
    main_subpage = requests.get(page_url)
    num_subpage_selector = parsel.Selector(main_subpage.text)
    num_subpage = int(
        num_subpage_selector.xpath(
            "/html/body/section/div[2]/div/div[2]/div/div[2]/div[1]/div[2]/text()"
        ).getall()[1][5:]
    )
    return num_subpage


def get_subpage_urls(page_url, num_subpage):
    subpage_urls = []
    for i in range(num_subpage):
        if i == 0:
            subpage_urls.append(page_url)
        else:
            subpage_urls.append(page_url + "/page/" + str(i + 1))
    return subpage_urls


def get_article_urls(subpage_urls):
    article_urls = []
    for subpage_url in subpage_urls:
        subpage = requests.get(subpage_url)
        article_selector = parsel.Selector(subpage.text)
        article_urls.append(
            article_selector.xpath(
                "//*[@class='border-block inner-border-block_type-3 last-papers_paper category_post post_info']/div/a/@href"
            ).getall()
        )
    return article_urls


def transform_article_urls(article_urls):
    transformed_article_urls = []
    for i in range(len(article_urls)):
        for j in range(len(article_urls[i])):
            transformed_article_urls.append(article_urls[i][j])
    return transformed_article_urls


def get_articles(transformed_article_url):
    article = requests.get(transformed_article_url)
    articles_selector = parsel.Selector(article.text)
    articles = articles_selector.xpath("//*[@class='typography-content']/p/text()").getall()
    return articles


def write2json(topic, articles, filename, mode):
    dataset = []
    for i in range(len(articles)):
        dataset.append({"topic": topic, "article": articles[i]})
    location = filename
    with open(location, mode, newline="", encoding="utf-8") as f:
        for i in range(len(dataset)):
            json.dump(dataset[i], f)
            f.write("\n")


def read_json(filename):
    dataset = [json.loads(line) for line in open(filename, "r", newline="", encoding="utf-8")]
    return dataset


def clean_data(dataset):
    cleaned_dataset = []
    for i in range(len(dataset)):
        if "http" in dataset[i]["article"] or "https" in dataset[i]["article"]:
            continue
        if len(dataset[i]["article"].split()) < 50:
            continue
        dataset[i]["article"] = dataset[i]["article"].strip()
        dataset[i]["article"] = dataset[i]["article"].lstrip(".")
        dataset[i]["article"] = dataset[i]["article"].lstrip()
        cleaned_dataset.append(dataset[i])
    return cleaned_dataset


def build_subset(subset, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        for i in range(len(subset)):
            json.dump(subset[i], f)
            f.write("\n")


if __name__ == "__main__":
    main_page_url = "https://studymoose.com"
    page_urls = get_page_urls(main_page_url)
    print(len(page_urls))
    num_subpage = get_num_subpage(page_urls[0])
    subpage_urls = get_subpage_urls(page_urls[0], num_subpage)
    article_urls = get_article_urls(subpage_urls)
    transformed_article_urls = transform_article_urls(article_urls)
    articles = get_articles(transformed_article_urls[0])
    print(len(articles))
    write2json(page_urls[0][35:], articles, "dataset", "w")
    print(read_json("dataset"))
