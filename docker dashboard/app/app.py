#import packages
import pandas as pd
import gensim
from gensim import corpora
import pyLDAvis.gensim_models
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash import dash_table
from dash.dependencies import Input
from dash.dependencies import Output
from flask import Flask

# Bestanden importeren van github
url_uiteindelijke_tekst = "https://raw.githubusercontent.com/WouterMulders1999/ecommercetopicanalysis/main/uiteindelijke_tekst.csv"
uiteindelijke_tekst = pd.read_csv(url_uiteindelijke_tekst, index_col = 0, squeeze = True, quotechar='"')

url_df_topics_wegschrijven = "https://raw.githubusercontent.com/WouterMulders1999/ecommercetopicanalysis/main/dftopicswegschrijven.csv"
df_topics_wegschrijven = pd.read_csv(url_df_topics_wegschrijven)
df_topics_wegschrijven = df_topics_wegschrijven.drop(columns = ['Unnamed: 0'])
df_topics_wegschrijven.index += 1
df_topics_wegschrijven['Topic'] += 1

uiteindelijke_tekst_goed = uiteindelijke_tekst

for i in range(0,len(uiteindelijke_tekst_goed)):
    uiteindelijke_tekst_goed[i] = uiteindelijke_tekst[i][1:-1].replace("'", '').split(", ")

dictionary = corpora.Dictionary(uiteindelijke_tekst)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in uiteindelijke_tekst]
topics = df_topics_wegschrijven['Topic'].unique().tolist()

Lda = gensim.models.ldamodel.LdaModel

# Dashboard maken
server = Flask(__name__)
app = dash.Dash(server=server)
app.title = 'Dashboard'

n_words = 10

app.layout = html.Div(
    children=[html.Img(src= app.get_asset_url('inin_logo.jpg')),
        html.A(html.Button('Herlaad visualisatie'),href='/'),
        dcc.Slider(1, 50, id = "slider", step = 1, persistence = True, value = 20),
        dcc.Dropdown(
            id="filter_dropdown",
            options=[{"label": tp, "value": tp} for tp in topics],
            placeholder="Selecteer een topic",
            multi=True,
            value = df_topics_wegschrijven.Topic.unique(),
        ),
        dash_table.DataTable(id = "table-container", 
                             data = df_topics_wegschrijven.to_dict('records'), 
                             columns =  [{"name": i, "id": i} for i in df_topics_wegschrijven.columns],
                            ),
        
        html.Iframe(src = app.get_asset_url('lda.html'), id = "lda", style = {"width":"100%", "height":"1000px"}),
        #html.Meta(httpEquiv="refresh",content="5") #refreshed pagina iedere 5 seconden, maar is wel irritant
        
    ]
    
)  

@app.callback(
    [Output("lda", "data"),
    Output("filter_dropdown", "value")],
    Input("slider", "value")
)
def update_lda(slider):
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics = slider , id2word = dictionary, passes=40,\
            iterations=200,  chunksize = 10000, eval_every = None, random_state=0)

    topic_data =  pyLDAvis.gensim_models.prepare(ldamodel, doc_term_matrix, dictionary, mds = "mmds")
    pyLDAvis.save_html(topic_data, 'assets/lda.html')

    top_words_per_topic = []
    for t in range(ldamodel.num_topics):
        top_words_per_topic.extend([(t, ) + x for x in ldamodel.show_topic(t, topn = n_words)])


    df_topics_wegschrijven = pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P'])
    df_topics_wegschrijven.index += 1
    df_topics_wegschrijven['Topic'] += 1

    topics = df_topics_wegschrijven['Topic'].unique().tolist()
    value = df_topics_wegschrijven.Topic.unique()
    return pyLDAvis.save_html(topic_data, 'assets/lda.html'), value


@app.callback(
    Output("table-container", "data"),
    # Output("filter_dropdown", "value")], 
    [Input("filter_dropdown", "value"),
    Input("slider", "value")]
    
)
def display_table(filter_dropdown, slider):
    ldamodel = Lda(doc_term_matrix, num_topics = slider, id2word = dictionary, passes=40,\
            iterations=200,  chunksize = 10000, eval_every = None, random_state=0)

    topic_data =  pyLDAvis.gensim_models.prepare(ldamodel, doc_term_matrix, dictionary, mds = "mmds")

    top_words_per_topic = []
    for t in range(ldamodel.num_topics):
        top_words_per_topic.extend([(t, ) + x for x in ldamodel.show_topic(t, topn = n_words)])

    df_topics_wegschrijven = pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P'])
    df_topics_wegschrijven.index += 1
    df_topics_wegschrijven['Topic'] += 1
    df_topics_wegschrijven = df_topics_wegschrijven[:slider*n_words][df_topics_wegschrijven.Topic.isin(filter_dropdown)]
    data = df_topics_wegschrijven.to_dict('records')
    return data

@app.callback(
    Output("filter_dropdown", "options"),
    Input("slider", "value")
)
def update_options(slider):
    options = [{"label": tp, "value": tp} for tp in range(1, slider + 1)]
    return options

if __name__ == '__main__':
    app.run_server()#(debug=False, dev_tools_hot_reload=True)#use_reloader = True)#, dev_tools_hot_reload=True)