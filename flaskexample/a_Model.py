## Defining a function to predict rank based on user inputted keyword and probability matrix output from lda
def top_n_papers_lda(keyword, probab_topics_docs, n_topics, top_n = 5):
    import pickle
    import numpy as np
    import pandas as pd
    with open(r"./flaskexample/lda_model.pickle", "rb") as input_file:
        lda_model = pickle.load(input_file)
    get_keyword_topic_probabilities = lda_model.get_term_topics(keyword, minimum_probability=0.0000)
    probab_word_topics = np.zeros(n_topics)
    for i in range(n_topics):
        probab_word_topics[i] = get_keyword_topic_probabilities[i][1]
    dist_metric_keyword_abstracts = np.dot(probab_word_topics,probab_topics_docs)
    pd.set_option('mode.chained_assignment', None)
    ## Converting into list
    dist_metric_keyword_abstracts_list = dist_metric_keyword_abstracts.tolist()
    ## indices of the 5 highest elements of the list of distance metric
    top_n_indices = np.array(dist_metric_keyword_abstracts_list).argsort()[::-1][:top_n]
    rows_to_be_removed = [x+1 for x in list(range(len(dist_metric_keyword_abstracts_list))) if x not in top_n_indices]
    data_subset_read = pd.read_csv('./flaskexample/df_subset_nan_removed.csv', skiprows=rows_to_be_removed)
    #data_sort_by_probab = data.sort_values('probab', ascending=False)
    data_display = data_subset_read.loc[:,"Index":"Abstract"].iloc[0:top_n]
    print(type(data_display))
    return(data_display)
    
## Defining a function that takes a list of 5 nodes and creates a subnetwork plot
def subnetwork_generator(top_n_nodes):
    import pickle
    import pandas as pd
    import networkx as nx
    from bokeh.io import show, output_file
    from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, BoxZoomTool, ResetTool
    from bokeh.models.graphs import from_networkx
    from bokeh.palettes import Spectral4
    
    ## loading the total edgelist
    with open(r"./flaskexample/total_edgelist.pickle", "rb") as input_file:
        total_edgelist = pickle.load(input_file)
    ## Creating the subnetwork edgelist
    subnet_edgelist = []
    for nodeID in top_n_nodes:
        for i, edge in enumerate(total_edgelist):
            if edge[0]==nodeID or edge[1]==nodeID:
                subnet_edgelist.append(edge)
    ## Initializing the graph
    G=nx.Graph()
    ## Adding the nodes and edges
    for edge in subnet_edgelist:
        G.add_edge(edge[0],edge[1])
    df_index_title = pd.read_csv('./flaskexample/df_index_title.csv')
    df_subnet = df_index_title[df_index_title['Index'].isin(list(G.nodes))]
    print(df_subnet)
    
    ## Getting title dictionary based on filtered dataframe over the subnetwork
    dict_title = {}
    keys = list(G.nodes)
    for i in keys:
        dict_title[i] = 'Missing'
        for j in range(df_subnet.shape[0]):
            if str(df_subnet.iloc[j,0])==i:
                dict_title[i] = df_subnet.iloc[j,1]
    nx.set_node_attributes(G, dict_title, 'title')
    # Show with Bokeh
    plot = Plot(plot_width=400, plot_height=400, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
    #plot.title.text = "Subnetwork of recommended articles"
    node_hover_tool = HoverTool(tooltips=[("Title", "@title")])
    plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())
    graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))
    graph_renderer.node_renderer.glyph = Circle(size=8, fill_color=Spectral4[0])
    #graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.8, line_width=1)
    plot.renderers.append(graph_renderer)
    return(plot)

## defining a function to get complete data structure for term topic probabilities
def get_complete_list(keyword_topic_probabilities, n_topic = 9):
    complete_list = []
    for i in range(n_topic):
        complete_list.append((i,0))
    k=0
    for i,val in enumerate(keyword_topic_probabilities):
        complete_list[val[0]] = (val[0],val[1])
        k+=1
    return(complete_list)
    
## defining a function to calculate word to word distance
def word_to_word_distance(keyword1, keyword2, n_topics = 9):
    import numpy as np
    import pickle
    with open(r"./flaskexample/lda_model.pickle", "rb") as input_file:
        lda_model = pickle.load(input_file)
    ## keyword 1
    get_keyword1_topic_probabilities = lda_model.get_term_topics(keyword1, minimum_probability=-0.1)
    get_keyword1_topic_probabilities = get_complete_list(keyword_topic_probabilities = get_keyword1_topic_probabilities, n_topic = 9)
    probab_keyword1_topics = np.zeros(n_topics)
    for i in range(n_topics):
        probab_keyword1_topics[i] = get_keyword1_topic_probabilities[i][1]
    ## keyword 2
    get_keyword2_topic_probabilities = lda_model.get_term_topics(keyword2, minimum_probability=-0.1)
    get_keyword2_topic_probabilities = get_complete_list(keyword_topic_probabilities = get_keyword2_topic_probabilities, n_topic = 9)
    probab_keyword2_topics = np.zeros(n_topics)
    for i in range(n_topics):
        probab_keyword2_topics[i] = get_keyword2_topic_probabilities[i][1]
    ## calculating distance
    distance = np.dot(probab_keyword1_topics, probab_keyword2_topics)
    return(distance)

## defining a function to get top 10 words
def jargon_recommender(keyword, dictionary_tokens, n_topics = 9, top_n = 11):
    import numpy as np
    dist_keyword_to_vocab = np.zeros(len(dictionary_tokens))
    for i, word in enumerate(dictionary_tokens):
        #print(word)
        dist_keyword_to_vocab[i] = word_to_word_distance(keyword1 = keyword, keyword2 = word, n_topics = 9)
    top_n_indices = dist_keyword_to_vocab.argsort()[::-1][:top_n]
    top_n_indices_list = [int(w) for w in list(top_n_indices)]
    top_n_words = [dictionary_tokens[i] for i in top_n_indices_list]
    return(top_n_words)

def subnetwork_depth_generator(top_n_nodes, depth = 2):
    with open(r"./flaskexample/full_network_with_titles.pickle", "rb") as input_file:
        full_network_with_titles = pickle.load(input_file)
    sub_network_set = {key for node in top_n_nodes for key in nx.single_source_shortest_path(full_network_with_titles,node,cutoff=depth).keys()}
    sub_network_with_titles = full_network_with_titles.subgraph(sub_network_set)
    # Show with Bokeh
    plot = Plot(plot_width=400, plot_height=400, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
    # plot.title.text = "Citation Network"
    node_hover_tool = HoverTool(tooltips=[("Title", "@title")])
    plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())
    # plot.add_tools(BoxZoomTool(), ResetTool())
    graph_renderer = from_networkx(sub_network_with_titles, nx.spring_layout, scale=1, center=(0, 0))
    graph_renderer.node_renderer.glyph = Circle(size=4, fill_color=Spectral4[0])
    # graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.8, line_width=1)
    plot.renderers.append(graph_renderer)
    return(plot)
