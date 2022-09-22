# Toxic/Rash Plant Classification

## iNaturalist
`./notebooks/scrape-iNaturalist.ipynb` - 
data scraping and cleaning procedure to acquire training/testing images for a computer vision classification task (identifying toxic plants from nontoxic plants).  
- Complete dataset on Kaggle: https://www.kaggle.com/datasets/hanselliott/toxic-plant-classification  

`./iNaturalist/nontoxic_images/` & `./iNaturalist/toxic_images/` are empty directories to which images are sent by the scraping procedure in the notebook above.  

`./iNaturalist/saved_urls/` - lists of image URLs which were scraped from iNaturalist.org and used to download images for each class.  

`./data/herb22meta_data.csv` - metadata exported from the [Herbarium 2022 Kaggle Challenge](https://www.kaggle.com/c/herbarium-2022-fgvc9) used for matching the plants used in this project to their metadata in the Herbarium 2022 dataset.    

## Streamlit
`./streamlit/` - code used for deploying a trained toxic plant classifier as a Streamlit app.  
App: https://hans-elliott99-toxic-plant-classification-streamlitapp-egloqy.streamlitapp.com/

---
---
### Google Images (Old)
Initally, plant images were scraped from google image searches. This was an unreliable method (poor image quality, inaccurate labels).  

`./google_images/` - (outdated) urls and scraping procedure for scraping google images for plant images.  
`./notebooks/upload-images-and-process.ipynb` - (outdated) code used for scraping google images for plant images. 
