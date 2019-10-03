from flask import render_template
from flaskexample import app
import numpy as np
import pandas as pd
import pickle
from flask import request
from flaskexample.a_Model import top_n_papers_lda, subnetwork_generator, get_complete_list, word_to_word_distance, jargon_recommender
from bokeh.embed import components

@app.route('/')
def SciNet_input():
    return render_template("input.html")

@app.route('/output')
def SciNet_output():
    #pull 'keyword_input' from input field and store it
    keyword = request.args.get('keyword_input')
    
    with open(r"./flaskexample/dictionary_tokens.pickle", "rb") as input_file:
           dictionary_tokens = pickle.load(input_file)
    
    if keyword in dictionary_tokens:
        ## Loading the pandas dataframe for processed data
        ##with open(r"./flaskexample/df_subset_nan_removed.pickle", "rb") as input_file:
        ##  df_subset_nan_removed = pickle.load(input_file)
        
        ## Loading the topic document probabilities from lds
        probab_topics_docs = np.load('./flaskexample/probab_topics_docs.npy')
        
        ## Running the metric calculation and sorting the vector
        top_n_result = top_n_papers_lda(keyword = keyword, probab_topics_docs = probab_topics_docs, n_topics = 9, top_n = 5)
        
        with open(r"./flaskexample/top_n_result.pickle", "wb") as output_file:
            pickle.dump(top_n_result, output_file)
        
        top_n_result_list = []
        for i in range(0,top_n_result.shape[0]):
            top_n_result_list.append(dict(title=top_n_result.iloc[i]['Title'], author=top_n_result.iloc[i]['Author'],  year=top_n_result.iloc[i]['Year'], journal=top_n_result.iloc[i]['Journal'], abstract=top_n_result.iloc[i]['Abstract']))
            
        #print(top_n_result_list[0])
        top_n_nodes = top_n_result.Index.values.tolist()
        top_n_nodes = [str(x) for x in top_n_nodes]
        #print(top_n_nodes)
        
        ## Getting the Bokeh network plot
        plot = subnetwork_generator(top_n_nodes = top_n_nodes)
        plot_script, plot_div = components(plot)
        kwargs = {'plot_script': plot_script, 'plot_div': plot_div}
        return render_template("output.html", top_n_result_list = top_n_result_list, **kwargs)
    else:
        return render_template("outputerror.html")

@app.route('/AnalysisLDA')
def SciNet_outputLDA():
     return render_template("AnalysisLDA.html")

@app.route('/inputJargon')
def SciNet_inputJargon():
    return render_template("inputJargon.html")

@app.route('/outputJargon')
def SciNet_outputJargon():
    #pull 'keyword_input' from input field and store it
    keyword2 = request.args.get('keyword_input2')
    
    with open(r"./flaskexample/dictionary_tokens.pickle", "rb") as input_file:
           dictionary_tokens = pickle.load(input_file)
           
    if keyword2 in dictionary_tokens:
        jargon_list = jargon_recommender(keyword = keyword2, dictionary_tokens = dictionary_tokens)[1:]
        top_n_words_list = []
        for i in range(10):
            top_n_words_list.append(dict(jargons=jargon_list[i]))
        #print(top_n_words_list)
        return render_template("outputJargon.html", top_n_words_list = top_n_words_list)
    else:
        return render_template("outputerror.html")


#@app.route('/network')
#def SciNet_network():
#
#    #pull 'keyword_input' from input field and store it
#    keyword2 = request.args.get('keyword_input2')
#
#    ## Loading the pandas dataframe for processed data
#    ##with open(r"./flaskexample/df_subset_nan_removed.pickle", "rb") as input_file:
#    ##  df_subset_nan_removed = pickle.load(input_file)
#
#    ## Loading the topic document probabilities from lds
#    probab_topics_docs2 = np.load('./flaskexample/probab_topics_docs.npy')
#
#    ## Running the metric calculation and sorting the vector
#    top_n_result = top_n_papers_lda(keyword = keyword2, probab_topics_docs = probab_topics_docs2, n_topics = 9, top_n = 5)
#
#    #with open(r"./flaskexample/top_n_result.pickle", "rb") as input_file:
#        #top_n_result = pickle.load(input_file)
#
#    print(top_n_result)
#
#    top_n_nodes = top_n_result.Index.values.tolist()
#    top_n_nodes = [str(x) for x in top_n_nodes]
#    #print(top_n_nodes)
#
#    ## Getting the Bokeh network plot
#    plot = subnetwork_generator(top_n_nodes = top_n_nodes)
#    plot_script, plot_div = components(plot)
#    kwargs = {'plot_script': plot_script, 'plot_div': plot_div}
#    return render_template("network.html", **kwargs)
