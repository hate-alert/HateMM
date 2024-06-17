# Detail of the Shared Code

#To Extract the Text-based Features

1.FastTextEmb_and_LASEREmbExtraction.py 
2.BERTandHateXPlainEmbedding.py

# To Extract the Audio Based Features

3.AudioMFCC_Feat_andSpectrumGen.py  
4.AudioVGG19andInceptionFeat.py

# To Extract the Video Based Features

4.AudioVGG19andInceptionFeat.py 
5.Model-ViT_featureExtract.py

# To Run all the Unimodal Models

8. UnimodalANN_foldWise.py

# To Run the unimodal Vision Based models

6.Vision+lstm_foldWise.py   
7.3DCNN_withFolds.py

# To Run the Multimodal Model
       
9. MultiModalFusionModelfoldWise.py

# To extract all the video frames.
frameExtract.py

# Extraction of transcript

The 'all__video_vosk_audioMap.p' has to be generated using the Vosk speech recognition toolkit(https://alphacephei.com/vosk/). The format of the file is in JSON format like the below:

{
  "video_name1": "transcript1",
  "video_name2": "transcript2",
  ...
  "video_name3": "transcript3"
}

