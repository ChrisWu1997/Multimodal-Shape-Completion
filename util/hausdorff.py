import torch


def directed_hausdorff(point_cloud1:torch.Tensor, point_cloud2:torch.Tensor, reduce_mean=True):
    """

    :param point_cloud1: (B, 3, N)
    :param point_cloud2: (B, 3, M)
    :return: directed hausdorff distance, A -> B
    """
    n_pts1 = point_cloud1.shape[2]
    n_pts2 = point_cloud2.shape[2]

    pc1 = point_cloud1.unsqueeze(3)
    pc1 = pc1.repeat((1, 1, 1, n_pts2)) # (B, 3, N, M)
    pc2 = point_cloud2.unsqueeze(2)
    pc2 = pc2.repeat((1, 1, n_pts1, 1)) # (B, 3, N, M)

    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)

    shortest_dist, _ = torch.min(l2_dist, dim=2)

    hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )

    if reduce_mean:
        hausdorff_dist = torch.mean(hausdorff_dist)

    return hausdorff_dist


if __name__ == '__main__':
    import numpy as np
    u = np.array([
        [
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1]
        ],
        [
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1]
        ]
    ])

    v = np.array([
        [
            [2, 0],
            [0, 2],
            [-2, 0],
            [0, -4]
        ],
        [
            [2, 0],
            [0, 2],
            [-2, 0],
            [0, -4]
        ]
    ])
    # u_tensor = tf.constant(u, dtype=tf.float32)
    # u_tensor = tf.tile(u_tensor, (1, 500, 1))
    # v_tensor = tf.constant(v, dtype=tf.float32)
    # v_tensor = tf.tile(v_tensor, (1, 500, 1))
    u_tensor = torch.tensor(u, dtype=torch.float32).transpose(1, 2)[:1]
    v_tensor = torch.tensor(v, dtype=torch.float32).transpose(1, 2)[:1]
    print(u_tensor.shape)
    print(v_tensor.shape)
    distances = directed_hausdorff(u_tensor, v_tensor)
    distances1 = directed_hausdorff(v_tensor, u_tensor)
    print(distances)
    print(distances1)
