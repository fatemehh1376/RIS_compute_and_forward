import numpy as np


class RIS_Environment(object):
    def __init__(self, num_antennas, num_RIS_elements, num_users, Variance_AWGN=1,
                 Variance_Channel_Noise=1, Channel_Estmation_Error=False):
        self.K = 1
        self.N = num_users
        self.R = num_RIS_elements

        self.Variance_AWGN = Variance_AWGN
        self.Variance_Channel_Noise = Variance_Channel_Noise
        self.Channel_Estimation_Error = Channel_Estmation_Error

        Power_Size = self.K
        Channel_Size = 2 * (self.K * self.R + self.R * self.N + self.K * self.N)
        # Channel_Size = 2 * (self.K * self.R + self.R * self.N)

        self.Action_Dimension = 2 * (self.R)
        # self.Action_Dimension = 2 * (self.K * self.N + self.R)


        # self.State_Dimension = Channel_Size + self.Action_Dimension
        self.State_Dimension = self.Action_Dimension
        # self.State_Dimension = Power_Size + Channel_Size + self.Action_Dimension
        # self.State_Dimension = Channel_Size + self.Action_Dimension

        self.SNR = [1, 1.99, 3.98, 7.94, 15.85, 31.62, 63.09, 125.89, 251.19, 501.19, 1000]
        self.G_user = None
        self.H_Direct = None
        self.H_s = None
        self.Phi = np.eye(self.R, dtype=complex)
        for u in range(self.R):
            for v in range(self.R):
                if u == v:
                    angles = np.random.uniform(0, 2 * np.pi, 1)
                    # print("angles:", angles)
                    self.Phi[u][v] = np.exp(1.0j * (angles))

        self.state = None
        self.Done_status = None
        self.Episode_trials = None

    def Channel_compute(self):
        return (self.G_user @ self.Phi @ self.H_s + self.H_Direct)

    def reset(self):
        self.Episode_trials = 0
        self.Phi = np.eye(self.R, dtype=complex)
        for u in range(self.R):
            for v in range(self.R):
                if u == v:
                    angles = np.random.uniform(0, 2 * np.pi, 1)
                    # print("angles:", angles)
                    self.Phi[u][v] = np.exp(1.0j * (angles))

        self.G_user = np.random.rayleigh(1.0, (self.N, self.R)) + 1j * np.random.rayleigh(1.0,
                                                                                          (self.N, self.R))
        self.H_s = np.random.rayleigh(1.0, (self.R, self.K)) + 1j * np.random.rayleigh(1.0,
                                                                                       (self.R, self.K))
        self.H_Direct = np.random.rayleigh(1.0, (self.N, self.K)) + 1j * np.random.rayleigh(1.0,
                                                                                            (self.N, self.K))

        initial_action_Phi = np.hstack(
            (np.real(np.diag(self.Phi)).reshape(1, -1), np.imag(np.diag(self.Phi)).reshape(1, -1)))

        initial_action = initial_action_Phi

        Phi_real = initial_action[:, 0:self.R]
        Phi_imag = initial_action[:, self.R:]

        self.Phi = np.eye(self.R, dtype=complex) * (Phi_real + 1j * Phi_imag)

        # Power_at_transmitter = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        H_2_tilde = self.Channel_compute()
        Power_at_receiver = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2

        G_u_real, G_u_imag = np.real(self.G_user).reshape(1, -1), np.imag(self.G_user).reshape(1, -1)
        H_s_real, H_s_imag = np.real(self.H_s).reshape(1, -1), np.imag(self.H_s).reshape(1, -1)
        H_Direct_real, H_Direct_imag = np.real(self.H_Direct).reshape(1, -1), np.imag(self.H_Direct).reshape(1, -1)

        # print(len(initial_action[0]))
        # # print(len(Power_at_transmitter[0]))
        # print(len(Power_at_receiver[0]))
        # print(len(G_u_real[0]))
        # print(len(G_u_imag[0]))
        # print(len(H_s_real[0]))
        # print(len(H_s_imag[0]))
        # print(len(H_Direct_real[0]))
        # print(len(H_Direct_imag[0]))

        # self.state = np.hstack(
        #     (initial_action, G_u_real, G_u_imag, H_s_real, H_s_imag,
        #      H_Direct_real, H_Direct_imag))

        self.state = initial_action
        # self.State_Dimension = len(self.state[0])

        # print("len (self.state):", len(self.state), len(self.state[0]))

        return self.state

    def compute_reward(self, Phi, SNR_num):
        alpha = []
        for n in range(self.N):
            alpha.append(1)

        # h_s_n = self.H_s[:, n].reshape(-1, 1)
        # g_n = self.G[:, n].reshape(-1, 1)
        # h_direct_n = g_n.T @ self.H_Direct[:, n].reshape(-1, 1)

        # G_removed = np.delete(self.G, n, axis=1)
        # h_interference = h_s_n.T @ Phi @ self.G_user @ G_removed
        #
        # interference_total = np.sum(np.abs(h_interference ** 2))
        # Noise = interference_total + (self.N - 1) * self.Variance_AWGN
        # SNR_n = signal_abs / Noise
        # SNR_n_optimal = signal_abs / (self.N - 1) * self.Variance_AWGN
        #
        # print("SNR_n:", SNR_n)

        h_effective = self.G_user @ Phi @ self.H_s + self.H_Direct
        signal_abs = np.sum(np.abs(h_effective) ** 2)
        SNR = self.SNR[SNR_num]

        h_effective_hermitian = np.conjugate(np.transpose(h_effective))
        Beta = (SNR * (h_effective_hermitian @ alpha)) / (1 + SNR * signal_abs)
        # print("Beta:", Beta)

        Reward = max(np.log2(
            (SNR) / (np.abs(Beta) ** 2 + SNR * np.sum(np.abs((Beta * (h_effective.reshape([self.N, 1]))) - 1) ** 2))),
            0)

        print("Reward:", Reward)

        # Optimal_Reward = Optimal_Reward + np.abs(np.log(
        #     SNR_n_optimal / (np.abs(Beta) ** 2 + SNR_n_optimal * np.abs(Beta * h_effective - 1) ** 2)))
        # print("Optimal Reward:", Optimal_Reward)
        # print("----------------------------------")

        return Reward

    def step(self, action, SNR_num):
        self.Episode_trials = self.Episode_trials + 1
        action = action.reshape(1, -1)

        # G_Real = action[:, :self.K ** 2]
        # G_Imaginary = action[:, self.K ** 2:2 * self.K ** 2]

        Phi_Real = action[:, 0:self.R]
        Phi_Imaginary = action[:, self.R:]

        # self.G = G_Real.reshape(self.K, self.N) + 1j * G_Imaginary.reshape(self.K, self.N)
        self.Phi = np.eye(self.R, dtype=complex) * (Phi_Real + 1j * Phi_Imaginary)

        # Power_at_transmitter = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        Channel = self.Channel_compute()

        Power_at_receiver = np.linalg.norm(Channel, axis=0).reshape(1, -1) ** 2

        G_u_real, G_u_imag = np.real(self.G_user).reshape(1, -1), np.imag(self.G_user).reshape(1, -1)
        H_s_real, H_s_imag = np.real(self.H_s).reshape(1, -1), np.imag(self.H_s).reshape(1, -1)
        H_Direct_real, H_Direct_imag = np.real(self.H_Direct).reshape(1, -1), np.imag(self.H_Direct).reshape(1, -1)

        # self.state = np.hstack(
        #     (action, G_u_real, G_u_imag, H_s_real, H_s_imag, H_Direct_real,
        #      H_Direct_imag))


        self.state = action

        # print("**********************************")
        # print("len self.state:", len(self.state[0]))
        # print("self.State_Dimension:", self.State_Dimension)
        # print("**********************************")

        Reward = self.compute_reward(self.Phi, SNR_num)
        Done = Reward == (Reward - 0.1)

        return self.state, Reward, Done, None
