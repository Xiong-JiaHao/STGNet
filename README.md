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
