# NLP on Webscraped Wikipedia articles 

This short folder illustrates how to answer the Technical Test given by Dathena. To run it, please have all the packages in the requirements text file and run a command file in the src folder and write :'python Code_Python_Dathena.py'. It needs time to webscrap 50 texts.

The idea behind this project was to webscrap 50 texts, and run some clustering on the texts. Before reaching that goal, removing stopwords and stemming was used in order to perform tf-idf vectoriser, and then find clusters through KMeans. I've found KMeans to be quite robust in the past once sentences are vectorised.

I was too lazy to find internet sites that had phone numbers and were of different languages, or fetch 50 internet sites manually. So I used Wikipedia's API, randomly chose a language between French, English and Spanish for the articles, but I made sure the first wikipedia page had phone numbers by starting the search on the webpage 'Telephone Numbers'. Since it's an API, there's a maximum amount of pages we can scrap during a given time, so one shouldn't run the code several times in a row.

Since the wescraped pages will be different then those I did my analysis on, I added a jupyter notebook with all my results (Notebook.ipynb).

Overall, I'd suggest to run the code like explained above, and have a look on the analysis I did in the jupyter notebook. Check out everything it out !
