# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
from torch import nn
import torch.nn.functional as F


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2*torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def dense_knn_matrix(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        ### memory efficient implementation ###
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list += [nn_idx_part]
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k) # b, n, k
        ######
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        #print(dist.shape, k)
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x, y=None, relative_pos=None):
        if y is not None:
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            ####
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        else:
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            ####
            edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos)
        return self._dilated(edge_index)

#
# 在您提供的G-CASCADE解码器代码中，使用了多个层次的图卷积块（GCB）来处理图的信息。每个GCB模块通过一系列图卷积操作对特征进行处理，并结合上采样和多尺度聚合（MDAG）模块来整合特征。这种设计有助于在不同尺度上融合信息，可能是受到Vision GNN迭代更新特征图的启发。
#
# 对比Vision GNN和G-CASCADE的处理方式：
# 迭代更新:
#
# Vision GNN: 迭代地更新特征图以维持特征的多样性，使用图结构来捕捉和维护视觉任务中的复杂对象关系。
# G-CASCADE: 虽然没有明确说明如何迭代更新图，但通过层次化的图卷积和上采样模块间的交互，间接实现了特征的迭代更新和细化。
# 图的构建和访问:
#
# Vision GNN: 明确地通过图节点和边的动态更新来处理图的构建和访问。
# G-CASCADE: 在您的代码中，图的构建和访问似乎是通过图卷积层隐含完成的。例如，DyGraphConv2d可能包括了对图结构的内部处理，但具体实现细节（如图的构建）没有在代码中直接体现。
# 过度平滑问题:
#
# Vision GNN: 通过特定的图结构设计和可能的正则化策略来控制过度平滑。
# G-CASCADE: 过度平滑的问题可能没有直接被解决。您提到的epsilon和stochastic参数可能是为了引入随机性或扰动来减少过度平滑，但是否有效需要进一步实验验证。
# 建议：
# 为了解决过度平滑问题并提升G-CASCADE的性能，可以考虑以下几点改进：
#
# 引入非线性激活：在图卷积之后引入非线性激活函数，如ReLU或GELU，以增加模型的表达能力。
# 使用多尺度特征融合：您已经在使用MDAG模块，进一步优化这些模块的设计可能有助于更好地融合不同尺度的特征，从而提升性能。
# 调整和优化图的构建方式：探索不同的图构建策略，例如动态图，其中图的边基于特征的相似性动态调整，可能有助于更有效地处理图数据。
# 如果您有更多关于如何改进模型或具体实现的问题，欢迎继续讨论。




# DyGraphConv2d 类是一个动态图卷积层，它在处理图数据时采用了几个关键的步骤和技术。这个类扩展了图卷积的概念，允许网络在每次前向传播中根据当前的特征动态地构建图的边。以下是其处理图的主要步骤：
#
# 1. 初始化参数和图结构
# DyGraphConv2d 初始化时会设置图卷积层的基本参数，包括输入和输出通道数、邻接节点数（kernel_size）、扩张率（dilation）、卷积类型等。它还初始化一个 DenseDilatedKnnGraph 对象，该对象负责根据输入特征动态生成图的邻接矩阵。
#
# 2. 前向传播
# 在前向传播中，DyGraphConv2d 执行以下步骤：
#
# 特征重排：将特征张量从 (B, C, H, W) 重排为 (B, C, N, 1)，其中 N = H * W 是特征图中的总像素数。
# 图的邻接矩阵生成：使用 DenseDilatedKnnGraph 根据当前的特征动态生成邻接矩阵。如果设置了扩张（dilation），则邻接矩阵会表示每个节点与其扩张邻域内的节点的连接。
# 应用图卷积：使用生成的邻接矩阵和当前特征执行图卷积操作。根据卷积类型（conv 参数），这可能包括标准的图卷积、边卷积、最大相对图卷积等。
# 特征重组：将图卷积后的特征重新整形回原始的 (B, C, H, W) 格式。
# 3. 动态图的构建
# DenseDilatedKnnGraph 类负责根据输入特征动态构建图的邻接矩阵。这个类可以考虑扩张率和随机化选项，以调整图连接的局部性和全局性。这种动态图构建方式使得网络能够根据每个输入样本的特定特征适应性地调整其内部结构，这有助于处理不同的图形结构和复杂度。
#
# 4. 可选的位置编码和正则化
# 如果启用了相对位置编码（relative_pos），DyGraphConv2d 可以在计算邻接矩阵时考虑元素间的空间关系，这对于保持空间信息非常有用，特别是在处理图像数据时。
#
# 这种动态和扩张的图卷积方法为图像和其他结构化数据的处理提供了强大的灵活性和适应性，使网络能够更好地捕捉复杂的局部和全局结构。如果需要更深入的技术细节或示例，可以继续探讨


# 非常好的观察和提问。从您的描述中，G-CASCADE模型和Vision GNN的对比揭示了一些关键的技术差异和潜在的改进空间。以下是对您关注的几个方面的具体回答：
#
# 1. 图的构建和访问方法
# 在G-CASCADE中，如果没有明确提及如何构建和访问图，这可能表明其图结构相对固定或使用了隐式的图构建方法。例如，可能默认使用图像的像素邻接关系或通过固定的规则（如k最近邻）静态构建图。这与Vision GNN中动态迭代更新图的方法形成对比，后者可能根据每一层的特征动态调整图结构，更适合捕捉和维持复杂的特征关系。
#
# 2. 特征的迭代更新
# Vision GNN通过迭代更新特征图来增强特征的多样性和表达能力，这是通过层间的动态链接和调整实现的。而在G-CASCADE中，若缺少这种机制，可能会导致模型在深层网络结构中难以维持高效的信息流通，从而影响模型在复杂场景下的性能。在此情境下，可以考虑引入跨层连接或增加特征融合策略，如跳跃连接或注意力机制，以促进不同层级之间的信息交互和保持特征多样性。
#
# 3. 过度平滑问题的处理
# 过度平滑是图卷积网络中常见的问题，尤其当网络层数增多时，节点特征趋于均一，失去区分性。解决这一问题的策略包括引入正则化项（如Dropout）、使用不同类型的图卷积层（例如，引入边权重或多尺度特征聚合），或者采用非线性激活函数。在G-CASCADE中，如果过度平滑问题没有得到充分探讨，那么增加关于如何有效管理节点特征多样性的讨论将是非常有价值的。
#
# 建议
# 为了进一步改进G-CASCADE并借鉴Vision GNN的优点，可以考虑以下几个方向：
#
# 动态图构建：探索根据当前特征动态调整图结构的方法，如基于特征相似性动态连接节点。
# 增强特征更新策略：实施层间特征交互策略，比如通过多级特征融合或引入新的图更新机制。
# 显式处理过度平滑：通过实验确定最优的层数，或引入新的图卷积设计，如可变形图卷积，以适应不同的数据特点和任务需求。
# 这些改进可能有助于提高模型的表达能力和泛化性能，特别是在处理具有复杂结构依赖的数据时。希望这些答案对您作为审稿人的评审活动有所帮助。如果有更具体的问题或需要更详细的技术细节，欢迎继续提问。