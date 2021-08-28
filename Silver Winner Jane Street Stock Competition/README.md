

# Kaggle Jane Street Stock Competition, Silver Medal ( top 3% winner )

* The data is blinded, you have data but no descriptions of what the data represents along with 5 returns one for each quarter and a final return

* Data also contains a meta file. I graphed and tried using each grouping of meta data in a model. All were almost as accurate as using the full feature set.

* Two files in this set are for exploratory data analysis Janestreet-first-feature_look.ipynb and graph-and-plot-feature-correlation-v2.ipynb.

* One model bottleneck-neuralnet.ipynb cleans the data removing outlines, uses a bottleneck model to filter noise and then trains on batches of data by date. 

*The second notebook flat-ffn.ipynb lets the network figure it all out and shuffles the data to prevent it from getting too comfortable with current market trends. This one was the best of my scores but both scores ended up closer than I expected.



