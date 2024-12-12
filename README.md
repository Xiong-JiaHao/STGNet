## STGNet: Spatio-Temporal Graph Neural Network in Camera-based Remote Photoplethysmography
### [STGNet.pdf](https://www.sciencedirect.com/science/article/pii/S1746809424007481)
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

### Abstract
As the global population continues to age, the demand for convenient monitoring of physiological signals has grown significantly. Camera-based remote photoplethysmography (rPPG) has emerged as promising due to its non-contact characteristic, especially with the integration of deep learning methods. However, existing deep learning-based rPPG methods often overlook the inherent properties of physiological signals, such as periodicity and consistency, resulting in inaccurate physiological signal estimation. In this paper, we propose a novel remote photoplethysmography model using the spatio-temporal graph neural networks and the attention mask, named STGNet, which is based on the inherent properties that physiological signals are always similar in different facial regions across adjacent time segments of the same individual. We utilize the spatio-temporal graph neural networks to denoise physiological signals according to their periodicity and consistency. Additionally, we use the attention mask to filter the facial region abundant in physiological signals, thereby preventing excessive interference from surrounding information. Our proposed STGNet model shows comparable and even superior performance compared with state-of-the-art methods, as validated on the UBFC-rPPG and PURE datasets. To the best of our knowledge, this is the first work to estimate physiological signal using spatio-temporal graph neural networks.

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
