# !/usr/bin/python3
import math
import networkx as nx


import gym_pybullet_drones.utils.utils as utils
import numpy as np



class MathUtils():

    EPSILON = 0.1
    H = 0.2
    A, B = 5, 5
    C = np.abs(A-B)/np.sqrt(4*A*B)  # phi

    R = 40
    D = 40

    @staticmethod
    def sigma_1(z):
        return z / np.sqrt(1 + z**2)

    @staticmethod
    def sigma_norm(z, e=EPSILON):
        return (np.sqrt(1 + e * np.linalg.norm(z, axis=-1, keepdims=True)**2) - 1) / e

    @staticmethod
    def sigma_norm_grad(z, e=EPSILON):
        return z/np.sqrt(1 + e * np.linalg.norm(z, axis=-1, keepdims=True)**2)

    @staticmethod
    def bump_function(z, h=H):
        ph = np.zeros_like(z)
        ph[z <= 1] = (1 + np.cos(np.pi * (z[z <= 1] - h)/(1 - h)))/2
        ph[z < h] = 1
        ph[z < 0] = 0
        return ph

    @staticmethod
    def phi(z, a=A, b=B, c=C):
        return ((a + b) * MathUtils.sigma_1(z + c) + (a - b)) / 2

    @staticmethod
    def phi_alpha(z, r=R, d=D):
        r_alpha = MathUtils.sigma_norm([r])
        d_alpha = MathUtils.sigma_norm([d])
        return MathUtils.bump_function(z/r_alpha) * MathUtils.phi(z-d_alpha)

    @staticmethod
    def normalise(v, pre_computed=None):
        n = pre_computed if pre_computed is not None else math.sqrt(
            v[0]**2 + v[1]**2)
        if n < 1e-13:
            return np.zeros(2)
        else:
            return np.array(v) / n


class MathematicalFlock(): #Removed Behavior inheritance
    C1_alpha = 3
    C2_alpha = 2 * np.sqrt(C1_alpha)
    C1_gamma = 5
    C2_gamma = 0.2 * np.sqrt(C1_gamma)

    C1_beta = 20
    C2_beta = 2 * np.sqrt(C1_beta)

    ALPHA_RANGE = 1.2
    ALPHA_DISTANCE = 1.2
    ALPHA_ERROR = 0.4
    BETA_RANGE = 1
    BETA_DISTANCE = 1

    def __init__(self, follow_cursor: bool,
                 sensing_range: float,
                 danger_range: float,
                 initial_consensus: np.ndarray):

        # self._herds = []
        # self._shepherds = []
    
        self._sample_t = 0
        self._pause_agents = np.zeros(1)

        self._follow_cursor = follow_cursor
        self._sensing_range = sensing_range
        self._danger_range = danger_range
        self._consensus_pose = np.array(initial_consensus)

        #Replaced in _flockupdate()
        # self._enable_flocking = True
        # self._flocking_condition = 0
        # self._dt = 0.2
        # self._dt_sqr = 0.1

        # self._contour_agents = []
        # self._plot_voronoi = False

        # Clusters
        self._total_clusters = 0
        self._clusters = []
        #self._plot_cluster = False ---Not used, was used in display()

    #REMOVED add_herd()
    #REMOVED add_shepherd()
    #REMOVED add_obstacle()
    #REMOVED set_consensus()
    #REMOVED get_herd_mean()
    #RELOCATED update() to flockupdate in BaseAviary.py
    #REMOVED display()

    ###MODIFIED###
    # Mathematical model of flocking
    def _flocking(self, cattle_states: np.ndarray, drone_states: np.ndarray) -> np.ndarray:
        u = np.zeros((cattle_states.shape[0], 2))
        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(cattle_states, r=self._sensing_range)
        #removed beta_adjacency_matrix
        delta_adjacency_matrix = self._get_delta_adjacency_matrix(cattle_states, drone_states, r=self._sensing_range)

        for idx in range(cattle_states.shape[0]):
            # Flocking terms
            neighbor_idxs = alpha_adjacency_matrix[idx]
            u_alpha = self._calc_flocking_control(idx=idx, neighbors_idxs=neighbor_idxs, cattle_states=cattle_states)

            #removed obstalce avoidance term

            # Shepherd
            drones_idxs = delta_adjacency_matrix[idx]
            u_delta = self._calc_shepherd_interaction_control(idx=idx, drones_idxs=drones_idxs,
                                                            delta_adj_matrix=delta_adjacency_matrix,
                                                            cattle_states=cattle_states, drone_states=drone_states)

            # Ultimate flocking model
            u[idx] = u_alpha + u_delta # u_alpha *10
        return u
    
    def _herd_density(self, herd_states: np.ndarray,
                      shepherd_states: np.ndarray):
        herd_densities = np.zeros((herd_states.shape[0], 2))
        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(herd_states, r=self._sensing_range)
        for idx in range(herd_states.shape[0]):
            # Density
            neighbor_idxs = alpha_adjacency_matrix[idx]
            density = self._calc_density(idx=idx, neighbors_idxs=neighbor_idxs, herd_states=herd_states)
            herd_densities[idx] = density
        return herd_densities

    def _global_clustering(self, cattle_states: np.ndarray, drone_states: np.ndarray) -> np.ndarray:
        u = np.zeros((cattle_states.shape[0], 2))
        for idx in range(cattle_states.shape[0]):
            qi = cattle_states[idx, :2]
            pi_full = cattle_states[idx, 10:13]
            pi = pi_full[:2]

            target = self._consensus_pose
            u_gamma = self._calc_group_objective_control(target=target, qi=qi, pi=pi)
            u[idx] = u_gamma
        return u



    def _local_clustering(self, cattle_states: np.ndarray, drone_states: np.ndarray, k: float) -> np.ndarray:
        adj_matrix = self._get_alpha_adjacency_matrix(cattle_states=cattle_states, r=self._sensing_range * 1.)
        graph = nx.Graph(adj_matrix)

        clusters_idxs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

        self._total_clusters = len(clusters_idxs)
        self._clusters = []

        clusters = []
        cluster_indx_list = []
        for cluster_idxs in clusters_idxs:
            cluster = []
            cluster_indx = []

            for cluster_edge in cluster_idxs.edges:
                cluster.append(cattle_states[cluster_edge, :])

            self._clusters.append(cluster)

            cluster_nodes = []
            if len(cluster_idxs.nodes) == 1:
                continue

            for cluster_node in cluster_idxs.nodes:
                cluster_nodes.append(cattle_states[cluster_node, :])
                cluster_indx.append(cluster_node)
            clusters.append(cluster_nodes)
            cluster_indx_list.append(cluster_indx)

        # Perform local flocking with local cluster
        all_gamma = np.zeros((cattle_states.shape[0], 2))
        for cluster_indx, cluster in enumerate(clusters):
            if len(clusters) == 1:
                continue

            local_cluster_states = np.empty((0, 4))
            for cluster_node in cluster:
                local_cluster_states = np.vstack((local_cluster_states, cluster_node))

            for idx in range(local_cluster_states.shape[0]):
                qi = local_cluster_states[idx, :2]
                pi = local_cluster_states[idx, 2:4]

                this_indx = cluster_indx_list[cluster_indx][idx]

                # Group consensus term
                cluster_mean = np.sum(local_cluster_states[:, :2], axis=0) / local_cluster_states.shape[0]

                target = cluster_mean
                u_gamma = k * self._calc_group_objective_control(target=target, qi=qi, pi=pi)
                all_gamma[this_indx, :] = u_gamma
        return all_gamma

    def _calc_remain_in_boundary_control(self, cattle_states, boundary: np.ndarray, k: float):
        x_min = boundary['x_min']; x_max = boundary['x_max']
        y_min = boundary['y_min']; y_max = boundary['y_max']

        u = np.zeros((cattle_states.shape[0], 2))  # XY control
        for idx in range(cattle_states.shape[0]):
            qi = cattle_states[idx, :2]
            if qi[0] < x_min:
                u[idx, :] += k * np.array([1.0, 0.0])
            elif qi[0] > x_max:
                u[idx, :] += k * np.array([-1.0, 0.0])

            if qi[1] < y_min:
                u[idx, :] += k * np.array([0.0, 1.0])
            elif qi[1] > y_max:
                u[idx, :] += k * np.array([0.0, -1.0])
        return u


    def _calc_flocking_control(self, idx: int, neighbors_idxs: np.ndarray, cattle_states: np.ndarray):
        qi = cattle_states[idx, :2]
        pi_full = cattle_states[idx, 10:13]
        pi = pi_full[:2]
        u_alpha = np.zeros(2)
        if sum(neighbors_idxs) > 0:
            qj = cattle_states[neighbors_idxs, :2]
            pj_full = cattle_states[neighbors_idxs, 10:13]
            pj = pj_full[:, :2]

            alpha_grad = self._gradient_term(
                c=MathematicalFlock.C2_alpha, qi=qi, qj=qj,
                r=MathematicalFlock.ALPHA_RANGE,
                d=MathematicalFlock.ALPHA_DISTANCE)

            alpha_consensus = self._velocity_consensus_term(
                c=MathematicalFlock.C2_alpha,
                qi=qi, qj=qj,
                pi=pi, pj=pj,
                r=MathematicalFlock.ALPHA_RANGE)
            u_alpha = alpha_grad + alpha_consensus
        return u_alpha

    
    #REMOVED _calc_obstacle_avoidance_control()

    def _calc_group_objective_control(self, target: np.ndarray, qi: np.ndarray, pi: np.ndarray):
        u_gamma = self._group_objective_term(c1=MathematicalFlock.C1_gamma,
                                            c2=MathematicalFlock.C2_gamma,
                                            pos=target,
                                            qi=qi,
                                            pi=pi)
        return u_gamma

    def _calc_shepherd_interaction_control(self, idx: int,
                                       drones_idxs: np.ndarray,
                                       delta_adj_matrix: np.ndarray,
                                       cattle_states: np.ndarray,
                                       drone_states: np.ndarray):
        qi = cattle_states[idx, :2]                 # 2D position
        pi_full = cattle_states[idx, 10:13]         # 3D velocity
        pi = pi_full[:2]                            # use 2D velocity for flocking
        u_delta = np.zeros(2)

        if np.any(drones_idxs):
            delta_in_radius = np.where(delta_adj_matrix[idx] > 0)[0]
            if len(delta_in_radius) > 0:
                delta_agents = []
                for d_idx in delta_in_radius:
                    yk = drone_states[d_idx, :2]
                    pk_full = drone_states[d_idx, 10:13]
                    pk = pk_full[:2]

                    diff = qi - yk
                    d = np.linalg.norm(diff) + 1e-6
                    r = self._r if hasattr(self, "_r") else 1.0
                    mu = min(d / r, 1.0)

                    ak = diff / d
                    P = np.eye(2) - np.outer(ak, ak)

                    qik = mu * qi + (1 - mu) * yk
                    pik = mu * (P @ pi)   # projected bull velocity component

                    delta_agents.append(np.hstack((qik, pik)))

                delta_agents = np.array(delta_agents)  # shape (M,4)
                qid = delta_agents[:, :2]
                pid = delta_agents[:, 2:4]             # fixed slice

                delta_grad = self._gradient_term(c=MathematicalFlock.C2_beta,
                                                qi=qi, qj=qid,
                                                r=MathematicalFlock.BETA_RANGE,
                                                d=MathematicalFlock.BETA_DISTANCE)

                delta_consensus = self._velocity_consensus_term(c=MathematicalFlock.C2_beta,
                                                                qi=qi, qj=qid,
                                                                pi=pi, pj=pid,
                                                                r=MathematicalFlock.BETA_RANGE)

                u_delta = delta_grad + delta_consensus

        u_delta += self._predator_avoidance_term(si=qi,
                                                r=self._danger_range,
                                                k=650000,
                                                drone_states=drone_states)
        return u_delta



    def _gradient_term(self, c: float, qi: np.ndarray, qj: np.ndarray, r: float, d: float):
        n_ij = self._get_n_ij(qi, qj)
        return c * np.sum(MathUtils.phi_alpha(MathUtils.sigma_norm(qj-qi), r=r, d=d)*n_ij, axis=0)

    def _velocity_consensus_term(self, c: float, qi: np.ndarray,
                                qj: np.ndarray, pi: np.ndarray,
                                pj: np.ndarray, r: float):
        # a_ij should be (M,) or (M,1) broadcasting to (M,2)
        a_ij = self._get_a_ij(qi, qj, r)   # ensure this returns shape (M,) or (M,1)
        arr = pj - pi                       # pj: (M,2), pi: (2,)
        return c * np.sum(a_ij * arr, axis=0)


    def _group_objective_term(self, c1: float, c2: float, pos: np.ndarray, qi: np.ndarray, pi: np.ndarray):
        return -c1 * MathUtils.sigma_1(qi - pos) - c2 * (pi)

    def _predator_avoidance_term(self, si: np.ndarray, r: float, k: float, drone_states: np.ndarray):
        si_dot = np.zeros(2)
        for di in drone_states[:, 0:2]: 
            if np.linalg.norm(di - si) <= r:
                si_dot += -k * (di - si)/(np.linalg.norm(di - si))**3
        return si_dot

    def _get_alpha_adjacency_matrix(self, cattle_states: np.ndarray, r: float) -> np.ndarray:
        adj_matrix = np.array([np.linalg.norm(cattle_states[i, :2]-cattle_states[:, :2], axis=-1) <= r
                               for i in range(len(cattle_states))])
        np.fill_diagonal(adj_matrix, False)
        return adj_matrix

    def _get_delta_adjacency_matrix(self, cattle_states: np.ndarray, drone_states: np.ndarray, r: float) -> np.ndarray:
        adj_matrix = np.array([]).reshape((0, len(drone_states)))
        for i in range(len(cattle_states)):
            adj_vec = []
            for delta_agent in drone_states:
                adj_vec.append(self.in_entity_radius(delta_agent[:2], cattle_states[i, :2], r=r))
            adj_matrix = np.vstack((adj_matrix, np.array(adj_vec)))
        return adj_matrix

    def _get_a_ij(self, q_i, q_js, range):
        r_alpha = MathUtils.sigma_norm([range])
        return MathUtils.bump_function(MathUtils.sigma_norm(q_js-q_i)/r_alpha)

    def in_entity_radius(self, drone_pose, qi: np.ndarray, r: float) -> bool:
        _r = 2
        return np.linalg.norm(drone_pose - qi) <= (r + _r)
    
    def _get_n_ij(self, q_i, q_js):
        return MathUtils.sigma_norm_grad(q_js - q_i)
    
    def _density(self, si: np.ndarray, sj: np.ndarray, k: float):
        w_sum = np.zeros(2).astype(np.float64)
        for i in range(sj.shape[0]):
            sij = si - sj[i, :]
            w = (1/(1 + k * np.linalg.norm(sij))) * utils.unit_vector(sij)
            w_sum += w
        return w_sum