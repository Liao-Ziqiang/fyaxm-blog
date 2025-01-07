### Supplementary (补充材料)

#### A. 概述

This supplementary material provides more details on experiments in the main paper and includes more experiments to validate and analyze our proposed method.  

In Sec B, we provide specific network architectures used for experiments in the main paper and also describe details in data preparation and training.  In Sec C, we show more experimental results including benchmark performance on part segmentation and analysis on neighborhood query, sensitivity to sampling randomness, and time-space complexity.

本补充材料提供了主文中实验的更多细节，并包括更多实验来验证和分析我们提出的方法。

在 Sec B 中，我们提供了主文实验中使用的具体网络架构，并描述了数据准备和训练的细节。在 Sec C 中，我们展示了更多实验结果，包括在部件分割任务上的基准性能、对邻域查询的分析、对采样随机性的敏感性以及时间和空间复杂度分析。

---

#### B. 实验细节

**Architecture Protocol.** We use the following notations to describe our network architecture.  

$SA(K, r, [l_1, ..., l_d])$ is a set abstraction (SA) level with $K$ local regions of ball radius $r$ using PointNet of $d$ fully connected layers with width $l_i$ $(i = 1, ..., d)$.  $SA([l_1, ...l_d])$ is a global set abstraction level that converts set to a single vector.  In multi-scale setting (as in MSG), we use $SA(K, [r^{(1)}, ..., r^{(m)}], [[l^{(1)}_1, ..., l^{(1)}_d], ..., [l^{(m)}_1, ..., l^{(m)}_d]])$ to represent MSG with $m$ scales.  

$FC(l, dp)$ represents a fully connected layer with width $l$ and dropout ratio $dp$.  $FP(l_1, ..., l_d)$ is a feature propagation (FP) level with $d$ fully connected layers. It is used for updating features concatenated from interpolation and skip link.  All fully connected layers are followed by batch normalization and ReLU except for the last score prediction layer.

**架构协议。** 我们使用以下符号来描述网络架构。

$SA(K, r, [l_1, ..., l_d])$ 表示一个集合抽象（SA）层，它包含 $K$ 个以球半径 $r$ 为局部区域的点集合，并使用 $d$ 个宽度为 $l_i$ $(i = 1, ..., d)$ 的全连接层组成的 PointNet。$SA([l_1, ...l_d])$ 是一个全局集合抽象层，它将集合转换为单个向量。在多尺度设置（如 MSG）中，我们用 $SA(K, [r^{(1)}, ..., r^{(m)}], [[l^{(1)}_1, ..., l^{(1)}_d], ..., [l^{(m)}_1, ..., l^{(m)}_d]])$ 表示具有 $m$ 个尺度的 MSG。

$FC(l, dp)$ 表示一个宽度为 $l$，丢弃率为 $dp$ 的全连接层。$FP(l_1, ..., l_d)$ 是一个特征传播（FP）层，包含 $d$ 个全连接层，用于更新从插值和跳跃连接中拼接的特征。除最后的分数预测层外，所有全连接层后面都跟有批归一化和 ReLU。

---

##### B.1 网络架构

对于所有分类实验，我们使用以下架构（Ours SSG），并根据不同的 $K$（类别数）进行调整：

$$
\begin{array}{l}
    SA(512, 0.2, [64, 64, 128]) \rightarrow SA(128, 0.4, [128, 128, 256]) \rightarrow SA([256, 512, 1024]) \rightarrow  \\
    FC(512, 0.5) \rightarrow FC(256, 0.5) \rightarrow FC(K)
\end{array}
$$

多尺度分组（MSG）网络（PointNet++）架构如下：

$$
\begin{array}{l}
    SA(512, [0.1, 0.2, 0.4], [[32, 32, 64], [64, 64, 128], [64, 96, 128]]) \rightarrow   \\
    SA(128, [0.2, 0.4, 0.8], [[64, 64, 128], [128, 128, 256], [128, 128, 256]]) \rightarrow  \\
    SA([256, 512, 1024]) \rightarrow FC(512, 0.5) \rightarrow FC(256, 0.5) \rightarrow FC(K)
\end{array}
$$

跨层多分辨率分组（MRG）网络的架构包含三个分支：

- **分支 1**：$$ SA(512, 0.2, [64, 64, 128]) \rightarrow SA(64, 0.4, [128, 128, 256]) $$  
- **分支 2**：$$ SA(512, 0.4, [64, 128, 256]) $$ 使用 $r = 0.4$ 的原始点区域。  
- **分支 3**：$$ SA(64, 128, 256, 512) $$ 使用所有的原始点。  
- **分支 4**：$$ SA(256, 512, 1024) $$  

分支 1 和分支 2 的输出被拼接后输入到分支 4。分支 3 和分支 4 的输出被拼接后输入到：$FC(512, 0.5) \rightarrow FC(256, 0.5) \rightarrow FC(K)$ 用于分类。

用于语义场景标注的网络（FP 的最后两层全连接层后跟有丢弃率为 0.5 的 dropout 层）如下：  

$$
\begin{array}{l}
    SA(1024, 0.1, [32, 32, 64]) \rightarrow SA(256, 0.2, [64, 64, 128]) \rightarrow \\
    SA(64, 0.4, [128, 128, 256]) \rightarrow SA(16, 0.8, [256, 256, 512]) \rightarrow \\
    FP(256, 256) \rightarrow FP(256, 256) \rightarrow FP(256, 128) \rightarrow FP(128, 128, 128, 128, K)
\end{array}
$$

用于语义和部件分割的网络（FP 的最后两层全连接层后跟有丢弃率为 0.5 的 dropout 层）如下：

$$
\begin{array}{l}
    SA(512, 0.2, [64, 64, 128]) \rightarrow SA(128, 0.4, [128, 128, 256]) \rightarrow SA([256, 512, 1024]) \rightarrow  \\
    FP(256, 256) \rightarrow FP(256, 128) \rightarrow FP(128, 128, 128, 128, K)
\end{array}
$$

---

<img src="E:/OneDrive/大学课程/深度学习/pointnet++/img/fig_9.jpg" alt="fig_9" style="zoom:18%;" />

**Figure 9:** Virtual scan generated from ScanNet.

**图9：**从ScanNet生成的虚拟扫描。

------------

##### B.2 虚拟扫描生成

In this section, we describe how we generate labeled virtual scans with non-uniform sampling density from ScanNet scenes.  For each scene in ScanNet, we set the camera location 1.5m above the centroid of the floor plane and rotate the camera orientation in the horizontal plane evenly in 8 directions. In each direction, we use an image plane with size $100px \times 75px$ and cast rays from the camera through each pixel to the scene. This gives a way to select visible points in the scene.  We could then generate 8 virtual scans for each test scene, and an example is shown in Fig. 9. Notice that point samples are denser in regions closer to the camera.

在本节中，我们描述如何从 ScanNet 场景中生成具有非均匀采样密度的标注虚拟扫描。对于 ScanNet 中的每个场景，我们将摄像机位置设置为地面平面中心点上方 1.5m，并在水平平面上将摄像机方向均匀地旋转 8 个方向。在每个方向上，我们使用尺寸为 $100px \times 75px$ 的图像平面，并从摄像机通过每个像素向场景投射光线。这种方法用于选择场景中的可见点。由此，我们可以为每个测试场景生成 8 个虚拟扫描，图 9 显示了一个示例。值得注意的是，离摄像机越近的区域，点样本越密集。

---

##### B.3 MNIST 和 ModelNet40 实验细节

For MNIST images, we firstly normalize all pixel intensities to range $[0, 1]$ and then select all pixels with intensities larger than $0.5$ as valid digit pixels. Then we convert digit pixels in an image into a 2D point cloud with coordinates within $[-1, 1]$, where the image center is the origin point.  Augmented points are created to add the point set up to a fixed cardinality (512 in our case). We jitter the initial point cloud (with random translation of Gaussian distribution $N(0, 0.01)$ and clipped to 0.03) to generate the augmented points.  For ModelNet40, we uniformly sample $N$ points from CAD models surfaces based on face area.  

对于 MNIST 图像，我们首先将所有像素强度归一化到 $[0, 1]$ 范围内，然后选择强度大于 $0.5$ 的像素作为有效数字像素。接着，我们将图像中的数字像素转换为一个二维点云，坐标范围在 $[-1, 1]$，其中图像中心为原点。为了使点集达到固定的大小（在我们的实验中为 512），我们增加了增强点。增强点是通过对初始点云进行抖动（以均值为 0、标准差为 0.01 的高斯分布随机平移，并限制在 0.03 的范围内）生成的。对于 ModelNet40，我们根据 CAD 模型表面的面面积均匀采样 $N$ 个点。

------------

For all experiments, we use Adam [9] optimizer with learning rate 0.001 for training. For data augmentation, we randomly scale objects, perturb the object location as well as point sample locations. We also follow [21] to randomly rotate objects for ModelNet40 data augmentation.  We use TensorFlow and GTX 1080, Titan X for training. All layers are implemented in CUDA to run on GPU. It takes around 20 hours to train our model to convergence.

在所有实验中，我们使用 Adam [9] 优化器，学习率为 0.001 进行训练。对于数据增强，我们随机缩放物体，扰动物体位置以及点采样位置。我们还按照 [21] 的方法对 ModelNet40 数据进行随机旋转增强。我们使用 TensorFlow 并在 GTX 1080 和 Titan X 上进行训练。所有层均在 CUDA 上实现以便在 GPU 上运行。将模型训练到收敛大约需要 20 小时。

-------------

##### B.4 ScanNet 实验细节

To generate training data from ScanNet scenes, we sample $1.5\text{m} \times 1.5\text{m} \times 3\text{m}$ cubes from the initial scene and then keep the cubes where $\geq 2\%$ of the voxels are occupied and $\geq 70\%$ of the surface voxels have valid annotations (this is the same set up in [5]). We sample such training cubes on the fly and random rotate it along the up-right axis. Augmented points are added to the point set to make a fixed cardinality (8192 in our case). 

为了从 ScanNet 场景中生成训练数据，我们从初始场景中采样 $1.5\text{m} \times 1.5\text{m} \times 3\text{m}$ 的立方体，然后保留满足以下条件的立方体：至少有 $2\%$ 的体素被占据，且至少 $70\%$ 的表面体素具有有效标注（与 [5] 设置相同）。这些训练立方体是在训练时动态采样的，并沿竖直方向随机旋转。为了使点集达到固定的大小（在我们的实验中为 8192），我们增加了增强点。

------

During test time, we similarly split the test scene into smaller cubes and get label prediction for every point in the cubes first, then merge label prediction in all the cubes from the same scene. If a point gets different labels from different cubes, we will just conduct a majority voting to get the final point label prediction.

在测试时，我们同样将测试场景划分为较小的立方体，并首先为每个立方体中的每个点生成标签预测，然后将同一场景中所有立方体的标签预测进行合并。如果某个点从不同的立方体中获得了不同的标签，我们将进行多数投票来确定该点的最终标签预测。

---

##### B.5 SHREC15 实验细节

We randomly sample 1024 points on each shape both for training and testing. To generate the input intrinsic features, we extract 100-dimensional WKS, HKS, and multiscale Gaussian curvature respectively, leading to a 300-dimensional feature vector for each point. Then we conduct PCA to reduce the feature dimension to 64. We use an 8-dimensional embedding following [23] to mimic the geodesic distance, which is used to describe our non-Euclidean metric space while choosing the point neighborhood.

我们在每个形状上随机采样 1024 个点，分别用于训练和测试。为了生成输入的内在特征，我们分别提取 100 维的 WKS、HKS 和多尺度高斯曲率，从而为每个点生成一个 300 维的特征向量。接着，我们使用 PCA 将特征维度降到 64。我们按照 [23] 的方法使用一个 8 维嵌入来模拟测地距离，该嵌入用于描述我们的非欧几里得度量空间，同时选择点邻域。

---

#### C. 更多实验

In this section, we provide more experiment results to validate and analyze our proposed network architecture.

在本节中，我们提供更多实验结果以验证和分析我们提出的网络架构。

---

##### C.1 语义部件分割

Following the setting in [32], we evaluate our approach on the part segmentation task assuming category label for each shape is already known. Taken shapes represented by point clouds as input, the task is to predict a part label for each point. The dataset contains 16,881 shapes from 16 classes, annotated with 50 parts in total. We use the official train-test split following [4].

按照 [32] 中的设置，我们在部件分割任务中评估我们的方法，假设每个形状的类别标签已知。以点云表示的形状作为输入，任务是为每个点预测一个部件标签。该数据集包含来自 16 个类别的 16,881 个形状，共标注了 50 个部件。我们使用 [4] 提供的官方训练-测试划分。

-------------------

<img src="E:/OneDrive/大学课程/深度学习/pointnet++/img/table_4.jpg" alt="table_4" style="zoom:25%;" />

**表4：**ShapeNet部件数据集的分割结果。

----------

We equip each point with its normal direction to better depict the underlying shape. This way we could get rid of hand-crafted geometric features as used in [32, 33]. We compare our framework with traditional learning-based techniques [32], as well as state-of-the-art deep learning approaches [20, 33] in Table 4. Point intersection over union (IoU) is used as the evaluation metric, averaged across all part classes. Cross-entropy loss is minimized during training. 

我们为每个点添加其法向量，以更好地描述底层形状。这使我们能够摆脱 [32, 33] 中使用的手工设计几何特征。我们在表 4 中将我们的框架与传统的基于学习的技术 [32] 以及最新的深度学习方法 [20, 33] 进行了比较。我们使用部件类别的平均交并比（IoU）作为评估指标。在训练中最小化交叉熵损失。

-----------

On average, our approach achieves the best performance. In comparison with [20], our approach performs better on most of the categories, which proves the importance of hierarchical feature learning for detailed semantic understanding. Notice our approach could be viewed as implicitly building proximity graphs at different scales and operating on these graphs, thus is related to graph CNN approaches such as [33]. Thanks to the flexibility of our multi-scale neighborhood selection as well as the power of set operation units, we could achieve better performance compared with [33]. Notice our set operation unit is much simpler compared with graph convolution kernels, and we do not need to conduct expensive eigen decomposition as opposed to [33]. These make our approach more suitable for large-scale point cloud analysis.

总体而言，我们的方法取得了最佳性能。与 [20] 相比，我们的方法在大多数类别上表现更好，这证明了分层特征学习对于详细语义理解的重要性。值得注意的是，我们的方法可以被视为在不同尺度上隐式构建邻接图并在这些图上操作，因此与图 CNN 方法（如 [33]）相关。得益于我们多尺度邻域选择的灵活性以及集合操作单元的强大功能，与 [33] 相比，我们取得了更好的性能。而且，我们的集合操作单元比图卷积核简单得多，不需要像 [33] 那样执行昂贵的特征值分解。这使得我们的方法更适合于大规模点云分析。

------------

<img src="E:/OneDrive/大学课程/深度学习/pointnet++/img/table_5.jpg" alt="table_5" style="zoom:22%;" />

**表5：**邻域选择的影响。评估指标是ModelNet 40测试集上的分类准确率（%）。

---

##### C.2 邻域查询：kNN vs. Ball Query

Here we compare two options to select a local neighborhood. We used radius-based ball query in our main paper. Here we also experiment with kNN-based neighborhood search and also play with different search radius and $k$. In this experiment, all training and testing are on ModelNet40 shapes with uniform sampling density. 1024 points are used. As seen in Table 5, radius-based ball query is slightly better than kNN-based methods. However, we speculate in very non-uniform point sets, kNN-based query will result in worse generalization ability. Also, we observe that a slightly larger radius is helpful for performance, probably because it captures richer local patterns.

在这里，我们比较了选择局部邻域的两种方式。在我们的主论文中，我们采用了基于半径的球查询（radius-based ball query）。此外，我们还尝试了基于 k 近邻（kNN）的邻域搜索，并探索了不同的搜索半径和 $k$ 的效果。在该实验中，所有训练和测试均基于点密度均匀采样的 ModelNet40 数据集，使用了 1024 个点。如表 5 所示，基于半径的球查询略优于基于 kNN 的方法。然而，我们推测在点分布非常不均匀的情况下，基于 kNN 的查询可能会导致较差的泛化能力。此外，我们观察到，稍大的半径有助于提升性能，这可能是因为它捕获了更丰富的局部模式。

----------------

<img src="E:/OneDrive/大学课程/深度学习/pointnet++/img/table_6.jpg" alt="table_6" style="zoom:16%;" />

**表6：**FPS中随机性的影响（使用ModelNet40）。

---

##### C.3 最远点采样中随机性的影响

For the Sampling layer in our set abstraction level, we use farthest point sampling (FPS) for point set subsampling. However, the FPS algorithm is random and the subsampling depends on which point is selected first. Here we evaluate the sensitivity of our model to this randomness. In Table 6, we test our model trained on ModelNet40 for feature stability and classification stability. 

在我们的集合抽象层的采样模块中，我们使用最远点采样（Farthest Point Sampling, FPS）进行点集的子采样。然而，FPS 算法具有随机性，其采样结果依赖于初始选择的点。在此，我们评估了模型对这种随机性的敏感性。在表 6 中，我们测试了模型在 ModelNet40 上的特征稳定性和分类稳定性。

------------

To evaluate feature stability, we extract global features of all test samples 10 times with different random seeds. Then we compute mean features for each shape across the 10 samplings. Then we compute the standard deviation of the norms of features’ difference from the mean feature. At last, we average all standard deviations in all feature dimensions as reported in the table. Since features are normalized into 0 to 1 before processing, the $0.021$ difference means a $2.1\%$ deviation of feature norm.

为了评估特征稳定性，我们对所有测试样本在 10 个不同随机种子下提取全局特征。然后，我们计算每个形状在 10 次采样中的平均特征，进一步计算特征与平均特征差异的范数的标准差。最后，我们对所有特征维度的标准差取平均值，并将结果报告在表中。由于特征在处理前被归一化到 0 到 1 的范围内，$0.021$ 的差异意味着特征范数有 $2.1\%$ 的偏差。

-----------

For classification, we observe only a $0.17\%$ standard deviation in test accuracy on all ModelNet40 test shapes, which is robust to sampling randomness.

对于分类任务，我们观察到所有 ModelNet40 测试形状的测试准确率标准差仅为 $0.17\%$，表明模型对采样随机性具有较强的鲁棒性。

---------------

<img src="E:/OneDrive/大学课程/深度学习/pointnet++/img/table_7.jpg" alt="table_7" style="zoom:21%;" />

**表7：** 几个网络的模型大小和推理时间（前向传播）。

---

##### C.4 时间和空间复杂度

Table 7 summarizes comparisons of time and space cost between a few point set-based deep learning methods. We record forward time with a batch size of 8 using TensorFlow 1.1 with a single GTX 1080. The first batch is neglected since there is some preparation for the GPU. While PointNet (vanilla) [20] has the best time efficiency, our model without density adaptive layers achieved the smallest model size with fair speed.

表 7 总结了几种基于点集的深度学习方法在时间和空间成本上的对比。我们在单个 GTX 1080 上使用 TensorFlow 1.1，批大小为 8，记录了前向计算时间。忽略第一批次的时间，因为 GPU 有一些初始化准备工作。虽然 PointNet（vanilla）[20] 具有最佳的时间效率，但我们不带密度自适应层的模型实现了最小的模型尺寸，同时具有较好的计算速度。

---------

It’s worth noting that our MSG, while it has good performance in non-uniformly sampled data, is $2 \times$ more expensive than the SSG version due to the multi-scale region feature extraction. Compared with MSG, MRG is more efficient since it uses regions across layers.

需要注意的是，我们的 MSG 模型虽然在非均匀采样数据中表现良好，但由于多尺度区域特征提取，其计算成本是 SSG 版本的 $2 \times$。与 MSG 相比，MRG 更高效，因为它在跨层次的区域中进行特征提取。

----------

