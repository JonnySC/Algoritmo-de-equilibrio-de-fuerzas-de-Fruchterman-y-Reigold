import numpy as np
from scipy.sparse import coo_matrix


def sparse_fruchterman_reingold(A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-5, dim=3):
    # np.random.seed(1)
    nodes_num = A.shape[0]
    A = A.tolil()
    A = A.astype('float')
    if pos is None:
        pos = np.asarray(np.random.rand(nodes_num, dim), dtype=A.dtype)
        print('Init pos', pos)
    else:
        pos = np.array(pos)
        pos = pos.astype(A.dtype)

    if fixed is None:
        fixed = []

    if k is None:
        k = np.sqrt(1.0 / nodes_num)
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    dt = t / float(iterations + 1)
    displacement = np.zeros((dim, nodes_num))
    for iteration in range(iterations):
        displacement *= 0
        for i in range(A.shape[0]):
            if i in fixed:
                continue
            delta = (pos[i] - pos).T
            distance = np.sqrt((delta ** 2).sum(axis=0))
            distance = np.where(distance < 0.01, 0.01, distance)
            Ai = np.asarray(A.getrowview(i).toarray())
            print('Ai', Ai)
            displacement[:, i] += \
                (delta * (k * k / distance ** 2 - Ai * distance / k)).sum(axis=1)
        # update positions
        length = np.sqrt((displacement ** 2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = (displacement * t / length).T
        pos += delta_pos
        # cool temperature
        t -= dt
        err = np.linalg.norm(delta_pos) / nodes_num
        if err < threshold:
            break

    return pos


if __name__ == '__main__':

    row = np.array([0, 0, 1, 2, 3, 3])
    col = np.array([1, 3, 0, 3, 0, 2])
    data = np.array([1, 1, 1, 1, 1, 1])
    coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    print(coo_matrix((data, (row, col)), shape=(4, 4)))
    a = sparse_fruchterman_reingold(coo_matrix((data, (row, col)), shape=(4, 4)))
    print(a)
