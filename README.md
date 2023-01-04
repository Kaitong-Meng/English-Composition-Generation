# English-Composition-Generation
An English composition generator based on GPT-2 implemented with Gradio
<br>
This is a brief introduction of my final year project for my undergraduate degree. The finetuned GPT-2 model can generate English composition based on keywords defined by users. For reduction of computational complexity, a small version of open-sourced GPT-2 model by Hugging Face is chosen for this project. The dataset of model composition is scraped on a website called studymoose. The final demostration is implemented with Gradio.
<br>
<br>
As for model parameters and related files, readers can refer to Hugging Face for more details.
<br>
<br>
The project architecture can be devided into three main parts, namely keywords extraction, essay generation, and essay assessment. A keywords extractor based on part of speech analysis is implemented, whose output can be combined with user-defined prompt as input for essay generators. Essay generators consist of three subparts, which are GPT-2 by OpenAI, Bloom by BigScience, OPT by Meta, with versions chosen on HUgging Face. As for assessment, a DeBerta model is implemented to score the essays based on vocabulary and syntax, phrasology and grammar, conventions and cohesion. The winner will be corrected grammarly afterwards. A content analysis is performed at last, including part of speech analysis, sentiment analysis, content summarization and feature extraction based content similarity on cosine, as well as vocabulary diversity based on quasi-TF-IDF.
<br>
