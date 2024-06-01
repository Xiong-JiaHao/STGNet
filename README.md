## STGNet: Spatio-Temporal Graph Neural Network in Camera-based Remote Photoplethysmography
* Color Space：RGB
* Dataset：UBFC-rPPG and PURE
* STGNet
  * Stem Module：Stem_Block + Self_Attention_Block ([B, 3, 160, 64, 64] -> [B, 64, 40, 8, 8])
  * Spatio-Temporal Graph Module * 4 ([B, 64, 40, 8, 8] -> [B, 64, 20, 4, 4])
    * Spatio-Temporal Graph Denoise Block * M
      * KNN + Average Relative GraphConv + MLP
    * ROI Selection Block
      * 3D Conv + Attention Mask + 3D Conv
  * BVP Prediction Module：3D Transpose Conv + 3D AvgPool ([B, 64, 20, 4, 4] -> [B, 160])
* Loss： NegPearson

### Stem Module (model/Stem_Block.py)

The Stem Block divides the video into blocks and uses Diff Block to extract the frame difference for the video and connect it to the original input into the subsequent module.

### Spatio-Temporal Graph Module (model/ST_Graph_Module.py)

The spatio-temporal graph module utilizes the inherent property that physiological signals of the same person are always similar in different facial regions across adjacent time segments for feature extraction using spatio-temporal graph neural networks.

#### Spatio-Temporal Graph Denoise Block (model/ST_Graph_Denoise_Block.py)

The spatio-temporal graph denoise block is implemented based on the inherent properties of physiological signals that the same individual's physiological signals remain similar in neighboring facial regions across adjacent time segments.

#### ROI Selection Block (model/ROI_Selection_Block.py)

The ROI Selection block allows the model to selectively focus on facial regions abundant in physiological signals, while ignoring or reducing less informative regions such as the eyes or hair. 

### BVP Prediction Module (model/BVP_Prediction_Module.py)

The BVP prediction module filter and regenerate the time dimension of the extracted blood volume pulse (BVP) features.
