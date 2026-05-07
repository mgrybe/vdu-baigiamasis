# Suvokiamos kokybės optimizavimas vieno vaizdo raiškos didinimo uždaviniuose

[![Demo](https://img.shields.io/badge/Demo-Gyvai-brightgreen)](https://mgrybe.github.io/vdu-baigiamasis/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Šioje repozitorijoje pateikiamas magistro baigiamojo darbo **„Suvokiamos kokybės optimizavimas vieno vaizdo raiškos
didinimo uždaviniuose, integruojant HAT architektūrą bei kombinuotas nuostolių funkcijas“** (autorius Marius Grybė,
Vytauto Didžiojo universitetas) programinis kodas ir tyrimų duomenys.

## Santrauka

> Šiame magistro darbe nagrinėjamas suvokiamos kokybės optimizavimas vieno vaizdo raiškos didinimo (SISR) uždaviniuose.
> Darbo tikslas - ištirti ir pritaikyti modernius giliųjų neuroninių tinklų apmokymo metodus, įgalinančius pasiekti aukštą
> vaizdo tikroviškumą. Atlikus literatūros analizę, bazine architektūra pasirinktas hibridinis dėmesio transformeris(HAT).
> Dėl savo gebėjimo efektyviai modeliuoti globalų vaizdo kontekstą ir į rekonstrukcijos procesą įtraukti didesnį pikselių
> kiekį, šis modelis demonstravo aukščiausią rekonstrukcijos tikslumą (vertinant pagal PSNR ir SSIM metrikas) bei
> subjektyvią vizualinę kokybę. Tyrimo metu buvo eksperimentuojama su įvairiais nuostolių funkcijų deriniais. Nustatyta,
> kad tradicinis pikselių lygmens nuostolis (L1) sukelia vaizdų susiliejimą, todėl jį naudinga pakeisti AESOP
> autoenkoderiu grįsta funkcija, atskiriančia struktūrinį tikslumą nuo natūralios tekstūrų variacijos. Eksperimentų
> rezultatai parodė, kad optimaliausią suvokimo ir iškraipymo kompromisą užtikrina AESOP turinio ir UNET priešpriešinio (
> GAN) nuostolių derinys. Pritaikius šį subalansuotą nuostolių funkcijų derinį HAT tinklui ir apmokius modelį su
> sudėtingomis realaus pasaulio vaizdų degradacijomis, pavyko reikšmingai pagerinti vizualinę kokybę: vaizdų ryškumas (
> pagal Laplaso dispersiją) padidėjo 35,57%, o natūralumas (pagal NIQE metriką) pagerėjo 4,34%, nesukuriant dirbtinių
> artefaktų.

## Architektūra ir Metodika

Projektas remiasi **hibridiniu dėmesio transformeriu (HAT)**, kurį sudaro išplėstiniai moduliai:

- **Kanalo dėmesio sutelkimo blokas (CAB)** – suteikiantis modeliui galimybę išnaudoti globalią informaciją.
- **Langu grįstas vietinis dėmesys (HAB)** – efektyviam lokalių priklausomybių modeliavimui.
- **Persidengiantis kryžminio dėmesio mechanizmas (OCA)** – pagerinantis informacijos sklidimą tarp atskirų langų.

Modelio apmokymui naudota kombinuota tikslo funkcija, kurios svoriai optimizuoti siekiant išvengti gradientų
disbalanso ($L_{bendra}=0,1*L_{AESOP}+0,1*L_{VGG19}+0,1*L_{UNET}$). Implementacija sukurta
naudojant [BasicSR](https://github.com/XPixelGroup/BasicSR) karkasą.

## Projekto struktūra

- `src/method/`: Pagrindinė modelių, architektūrų, nuostolių funkcijų ir metrikų realizacija.
    - `models/`: Aukšto lygio modelių logika (RealHATGAN, AESOP integracijos, MoE variantai).
    - `archs/`: Tinklų architektūros (HAT, DRCT, SwinIR).
    - `losses/`: Specifinės nuostolių funkcijos (AESOP, GAN nuostoliai).
- `options/`: YAML konfigūraciniai failai apmokymui ir testavimui.
- `notebooks/`: Jupyter užrašinės statistinei analizei ir vizualizacijai.
- `data/`: Eksperimentų rezultatai ir žurnalai (metrikos, apmokymo kreivės).
- `papers/`: Susiję moksliniai straipsniai ir jų metaduomenys.
- `demo/`: Projekto demonstracinio puslapio resursai.

## Literatūra

[1] G. M. James, "[Variance and Bias for General Loss Functions](papers/L1-decomposition-A_1022899518027.pdf)," *Machine
Learning*, vol. 51, pp. 115–135, 2003.

[2] Z. Chen et
al., "[NTIRE 2025 Challenge on Image Super-Resolution (x4): Methods and Results](papers/Ntire_2025_SR.pdf)," *arXiv
preprint arXiv:2504.14582*, 2025.

[3] J. Cai, H. Zeng, H. Yong, Z. Cao, and L.
Zhang, "[Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model](papers/RealSR.pdf)," 2019.

[4] M. Lee, S. Hyun, W. Jun, and J.
Heo, "[Auto-Encoded Supervision for Perceptual Image Super-Resolution](papers/aesop-2412.00124v2.pdf)," *arXiv preprint
arXiv:2412.00124*, 2025.

[5] M. Lee, S. Hyun, W. Jun, and J.
Heo, "[Auto-Encoded Supervision for Perceptual Image Super-Resolution](papers/aesop.pdf)," in *Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[6] J. Park, S. Son, and K. M. Lee, "[Content-Aware Local GAN for Photo-Realistic Super-Resolution](papers/calgan.pdf),"
in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.

[7] W. Lai, J. Huang, N. Ahuja, and M.
Yang, "[Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](papers/charbonnier-loss-1704.03915v2.pdf),"
*arXiv preprint arXiv:1704.03915*, 2017.

[8] J. Wang, K. C. K. Chan, and C. C.
Loy, "[Exploring CLIP for Assessing the Look and Feel of Images](papers/clipiqa-2207.12396v2.pdf)," *arXiv preprint
arXiv:2207.12396*, 2022.

[9] E. J. Nunn, P. Khadivi, and S.
Samavi, "[Compound Fréchet Inception Distance for Quality Assessment of GAN Created Images](papers/compound-fid-2106.08575v1.pdf),"
*arXiv preprint arXiv:2106.08575*, 2021.

[10] L. Xie et
al., "[DeSRA: Detect and Delete the Artifacts of GAN-based Real-World Super-Resolution Models](papers/desra-2307.02457v1.pdf),"
in *Proceedings of the 40th International Conference on Machine Learning (ICML)*, 2023.

[11] K. Ding, K. Ma, S. Wang, and E. P.
Simoncelli, "[Image Quality Assessment: Unifying Structure and Texture Similarity](papers/dists-2004.07728v3.pdf),"
*IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2020.

[12] C. Hsu, C. Lee, and Y.
Chou, "[DRCT: Saving Image Super-Resolution away from Information Bottleneck](papers/drct-2404.00722v5.pdf)," *arXiv
preprint arXiv:2404.00722*, 2024.

[13] W. Shi et
al., "[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](papers/espcn-1609.05158v2.pdf),"
*arXiv preprint arXiv:1609.05158*, 2016.

[14] A. Aitken et
al., "[Checkerboard artifact free sub-pixel convolution: A note on sub-pixel convolution, resize convolution and convolution resize](papers/espcn-checkboard-1707.02937v1.pdf),"
*arXiv preprint arXiv:1707.02937*, 2017.

[15] X. Wang et
al., "[ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](papers/esrgan-1809.00219v2.pdf)," *arXiv
preprint arXiv:1809.00219*, 2018.

[16] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S.
Hochreiter, "[GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](papers/fid-1706.08500v6.pdf),"
*arXiv preprint arXiv:1706.08500*, 2018.

[17] I. J. Goodfellow et al., "[Generative Adversarial Nets](papers/gan-1406.2661v1.pdf)," *arXiv preprint arXiv:
1406.2661*, 2014.

[18] X. Chen et al., "[HAT: Hybrid Attention Transformer for Image Restoration](papers/hat-2309.05239v3.pdf)," *IEEE
Transactions on Pattern Analysis and Machine Intelligence*, 2025.

[19] J. Liang, H. Zeng, and L.
Zhang, "[Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution](papers/ldl-2203.09195v1.pdf),"
*arXiv preprint arXiv:2203.09195*, 2022.

[20] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O.
Wang, "[The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](papers/lpips-1801.03924v2.pdf)," *arXiv
preprint arXiv:1801.03924*, 2018.

[21] S. Yang et
al., "[MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment](papers/maniqa-2204.08958v2.pdf),"
*arXiv preprint arXiv:2204.08958*, 2022.

[22] J. Ke, Q. Wang, Y. Wang, P. Milanfar, and F.
Yang, "[MUSIQ: Multi-scale Image Quality Transformer](papers/musiq-2108.05997v1.pdf)," *arXiv preprint arXiv:
2108.05997*, 2021.

[23] A. Mittal, R. Soundararajan, and A. C.
Bovik, "[Making a 'Completely Blind' Image Quality Analyzer](papers/niqe.pdf)," *IEEE Signal Processing Letters*, 2013.

[24] M. Lee and J.
Heo, "[Noise-free Optimization in Early Training Steps for Image Super-Resolution](papers/noise-free-2312.17526v1.pdf),"
*arXiv preprint arXiv:2312.17526*, 2023.

[25] Y. Blau and T. Michaeli, "[The Perception-Distortion Tradeoff](papers/pdt-1711.06077v4.pdf)," *arXiv preprint
arXiv:1711.06077*, 2017.

[26] J. Johnson, A. Alahi, and L.
Fei-Fei, "[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](papers/perceptual-loss-1603.08155v1.pdf),"
*arXiv preprint arXiv:1603.08155*, 2016.

[27] Y. Zhang et
al., "[Image Super-Resolution Using Very Deep Residual Channel Attention Networks](papers/rcan-1807.02758v2.pdf),"
*arXiv preprint arXiv:1807.02758*, 2018.

[28] L. Xie, X. Wang, C. Dong, and Y.
Shan, "[Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](papers/real-esrgan-2107.10833v2.pdf),"
*arXiv preprint arXiv:2107.10833*, 2021.

[29] J. Cai, H. Zeng, H. Yong, Z. Cao, and L.
Zhang, "[Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model](papers/realsr-1904.00523v1.pdf),"
*arXiv preprint arXiv:1904.00523*, 2019.

[30] K. He, X. Zhang, S. Ren, and J.
Sun, "[Deep Residual Learning for Image Recognition](papers/resnet-1512.03385v1.pdf)," *arXiv preprint arXiv:
1512.03385*, 2015.

[31] J. Yang, J. Wright, Y. Ma, and T.
Huang, "[Image Super-Resolution as Sparse Representation of Raw Image Patches](papers/sparse-coding-v1.pdf)," in *CVPR*,

2008.

[32] J. Yang, J. Wright, T. Huang, and Y.
Ma, "[Image Super-Resolution via Sparse Representation](papers/sparse-coding-v2.pdf)," *IEEE Transactions on Image
Processing*, 2010.

[33] T. Miyato, T. Kataoka, M. Koyama, and Y.
Yoshida, "[SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS](papers/spectral-norm-gan-1802.05957v1.pdf)," in
*ICLR*, 2018.

[34] C. Dong, C. C. Loy, K. He, and X.
Tang, "[Image Super-Resolution Using Deep Convolutional Networks](papers/srcnn-1501.00092v3.pdf)," *arXiv preprint
arXiv:1501.00092*, 2015.

[35] C. Ledig et
al., "[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](papers/srgan-1609.04802v5.pdf),"
*arXiv preprint arXiv:1609.04802*, 2017.

[36] J. Nilsson and T. Akenine-Möller, "[Understanding SSIM](papers/ssim-2006.13846v2.pdf)," *arXiv preprint arXiv:
2006.13846*, 2020.

[37] Z. Liu et
al., "[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](papers/swin-2103.14030v2.pdf)," *arXiv
preprint arXiv:2103.14030*, 2021.

[38] J. Liang et al., "[SwinIR: Image Restoration Using Swin Transformer](papers/swinir-2108.10257v1.pdf)," *arXiv
preprint arXiv:2108.10257*, 2021.

[39] W. Fedus, B. Zoph, and N.
Shazeer, "[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](papers/swith-transformer-2101.03961v3.pdf),"
*Journal of Machine Learning Research*, vol. 23, pp. 1-40, 2022.

[40] C. Chen et
al., "[TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment](papers/topiq.pdf)," *arXiv
preprint arXiv:2308.03060*, 2023.

[41] A. Vaswani et al., "[Attention Is All You Need](papers/transformers-1706.03762v7.pdf)," *arXiv preprint arXiv:
1706.03762*, 2017.

[42] A. Dosovitskiy et
al., "[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](papers/vit-2010.11929v2.pdf)," in
*ICLR*, 2021.
