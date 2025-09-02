import json
import pandas as pd
import plotly.express as px
from plotly.colors import sample_colorscale


def visualisation(coordinates, videos, method, experiment_name, gold):

    """
    Create an HTML file to visualise the dimensionality reduction results.
    
    Args:
        coordinates (np.ndarray): the coordinates of the reduced dimensions.
        videos (list): List of video names.
        method (str): The method used for similarity measurement.
    """

    if gold:  
        print(f"Using gold standard from {gold} to colour videos.")
        with open(gold, 'r') as f:
            gold_groups = json.load(f)

        video_to_group = {}
        for group_ID, group in gold_groups.items():
            for video in group:
                video_to_group[video.split(".")[0]] = group_ID
    
    else:
        video_to_group = {video: 0 for video in videos}
 
    df = pd.DataFrame(coordinates, columns=["PC1", "PC2"])
    df["video"] = videos
    df["group"] = df["video"].map(video_to_group)  
    df["gif"] = df["video"].apply(lambda x: f"gifs/{x}.gif")

    unique_groups = df["group"].unique()
    num_groups = len(unique_groups)
    color_palette = sample_colorscale("Rainbow", [i / num_groups for i in range(num_groups)])
    color_map = {group: color_palette[i] for i, group in enumerate(unique_groups)}


    fig = px.scatter(df, x="PC1", y="PC2", 
                     color="group", 
                     hover_data={"video": True, "gif": True},
                     color_discrete_map=color_map)  

    fig.update_layout(legend=dict(bgcolor="darkgray", bordercolor="black", borderwidth=2))


    gif_update_script = """
    document.addEventListener('DOMContentLoaded', function() {
        var myPlot = document.querySelector('.js-plotly-plot');
        
        myPlot.on('plotly_hover', function(event) {
            var points = event.points;
            if (points.length > 0) {
                var gifUrl = points[0].customdata[1];  
                var xPos = points[0].x;  // Get X position
                var yPos = points[0].y;  // Get Y position

                var imgElement = document.getElementById('hover-gif');
                if (!imgElement) {
                    imgElement = document.createElement('img');
                    imgElement.id = 'hover-gif';
                    imgElement.style.position = 'absolute';
                    imgElement.style.width = '150px';  
                    imgElement.style.height = '150px';
                    imgElement.style.pointerEvents = 'none'; 
                    document.body.appendChild(imgElement);
                }
                imgElement.src = gifUrl;
                
            
                var rect = myPlot.getBoundingClientRect();
                var xScreen = rect.left + (points[0].xaxis.l2p(xPos));
                var yScreen = rect.top + (points[0].yaxis.l2p(yPos));

                imgElement.style.left = xScreen + 'px';
                imgElement.style.top = (yScreen - 110) + 'px'; 
                imgElement.style.display = 'block';
            }
        });

        myPlot.on('plotly_unhover', function(event) {
            var imgElement = document.getElementById('hover-gif');
            if (imgElement) {
                imgElement.style.display = 'none';
            }
        });
    });
    """
  
    fig.write_html(f"simcheck/DIMREDUC/results/visualisations/{experiment_name}_{method}_visualisation.html", include_plotlyjs="cdn", post_script=gif_update_script)
    print(f"Visualisation saved to simcheck/DIMREDUC/results/visualisations/{experiment_name}_{method}_visualisation.html")
