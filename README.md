# Vegetation-Predictor
This project represents the full pipeline to the prediction of the density of vegetation, that is used to morph a given repartition of vegetation (supposedly given by an artist) to some topography.

For that, we first of all generate a matrix representing the likelihood of presence of vegetation in each pixel of the topographical map, and then, we populate this map with the artist's sketch, keeping the relationships between its different elements.

The dataset is made of aerial images from the Alpes region in France. These images are then treated by calculating the "NDVI" index, that we refined to obtain better prediction results.

The data is then fed to a U-Net model, wich learns to map the topographical data (a matrix of heights) to the density (a matrix of "refined" NDVI)

And then the output of the neural network is used with a generalization algorithm to generalize the artist's sketch.
