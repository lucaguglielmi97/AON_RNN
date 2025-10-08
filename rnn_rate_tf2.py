
import numpy as np
import tensorflow as tf

def moving_average(tensor, window_size=4):
    w = int(window_size)
    t = tf.shape(tensor)[0]
    new_t = tf.math.floordiv(t, w)
    trimmed = tensor[: new_t * w]
    reshaped = tf.reshape(trimmed, (new_t, w, tf.shape(tensor)[1]))
    return tf.reduce_mean(reshaped, axis=1)

class RNNRateTF2(tf.Module):
    def __init__(self, N, N1, N2, N3, w_dist='gaus', gain=1.0, apply_dale=True, Inib=None, name=None):
        super().__init__(name=name)
        self.N = int(N); self.N1 = int(N1); self.N2 = int(N2); self.N3 = int(N3)
        self.w_dist = str(w_dist); self.gain = float(gain); self.apply_dale = bool(apply_dale)
        if Inib is None:
            raise ValueError('Inhibitory mask (Inib) must be provided.')
        inib = np.asarray(Inib).astype(np.float32).reshape(-1)
        assert inib.shape[0] == self.N, f'Inhibitory mask length {inib.shape[0]} != N={self.N}'
        self.Inib = (inib > 0.5).astype(np.float32).reshape(-1,1)
        self.inh_mask = tf.constant(self.Inib, dtype=tf.float32)
        self.W = tf.Variable(self._init_W_numpy(), trainable=True, name='W')
        self.W_in  = tf.Variable(tf.random.normal([self.N, 10], stddev=0.05, dtype=tf.float32), name='W_in')
        self.W_in2 = tf.Variable(tf.random.normal([self.N, 10], stddev=0.05, dtype=tf.float32), name='W_in2')
        self.I_ext = tf.Variable(tf.zeros([self.N, 1], dtype=tf.float32), name='I_ext')
        self.taus_param = tf.Variable(tf.random.normal([self.N, 1], stddev=0.1, dtype=tf.float32), name='taus_param')

    def _init_W_numpy(self):
        rng = np.random.default_rng()
        N = self.N
        if self.w_dist == 'gaus':
            W_abs = np.abs(rng.normal(0.0, 1.0, size=(N, N))).astype(np.float32)
        elif self.w_dist == 'gamma':
            W_abs = rng.gamma(2.0, 1.0, size=(N, N)).astype(np.float32)
        else:
            raise ValueError("w_dist must be 'gaus' or 'gamma'")
        np.fill_diagonal(W_abs, 0.0)
        sign = np.ones((N, N), dtype=np.float32)
        inh_idx = np.where(self.Inib.squeeze() > 0.5)[0]
        sign[inh_idx, :] = -1.0
        W = (self.gain / np.sqrt(N)) * (W_abs * sign)
        return W

    def project_dale(self):
        W = self.W; N = self.N
        inh = tf.cast(self.inh_mask > 0.5, tf.float32); exc = 1.0 - inh
        W_proj = tf.where(inh > 0, -tf.abs(W),  W)
        W_proj = tf.where(exc > 0,  tf.abs(W_proj), W_proj)
        W_proj = tf.linalg.set_diag(W_proj, tf.zeros([N], dtype=W_proj.dtype))
        self.W.assign(W_proj)

    def effective_weights(self):
        N1, N2, N3 = self.N1, self.N2, self.N3
        ww = self.W
        AIP_AIP = ww[0:N1, 0:N1]
        AIP_F5  = tf.nn.relu(ww[0:N1, N1:N1+N2])
        AIP_F6  = tf.nn.relu(ww[0:N1, N1+N2:N1+N2+N3])
        F5_AIP  = tf.nn.relu(ww[N1:N1+N2, 0:N1])
        F5_F5   = ww[N1:N1+N2, N1:N1+N2]
        F5_F6   = tf.nn.relu(ww[N1:N1+N2, N1+N2:N1+N2+N3])
        F6_AIP  = tf.nn.relu(ww[N1+N2:N1+N2+N3, 0:N1])
        F6_F5   = tf.nn.relu(ww[N1+N2:N1+N2+N3, N1:N1+N2])
        F6_F6   = ww[N1+N2:N1+N2+N3, N1+N2:N1+N2+N3]
        AIP = tf.concat([AIP_AIP, AIP_F5, AIP_F6], axis=1)
        F5  = tf.concat([F5_AIP,  F5_F5,  F5_F6], axis=1)
        F6  = tf.concat([F6_AIP,  F6_F5,  F6_F6], axis=1)
        ww  = tf.concat([AIP, F5, F6], axis=0)
        return ww

    def taus_sigmoid(self):
        tau_min = tf.constant(1.0, tf.float32)
        tau_max = tf.constant(100.0, tf.float32)
        s = tf.sigmoid(self.taus_param)  # [N,1]
        return s * (tau_max - tau_min) + tau_min  # [N,1]

    @tf.function
    def simulate_one(self, stim_seq, IC_r0, IC_x0, DeltaT, taus_sig, w_in_sel):
        T = tf.shape(stim_seq)[1]
        ww = self.effective_weights()
        def body(t, x_prev, r_prev, r_ta):
            inp = tf.matmul(w_in_sel, tf.expand_dims(stim_seq[:, t-1], 1))
            core = tf.matmul(ww, r_prev) + inp
            next_x = (1.0 - DeltaT/taus_sig) * x_prev + (DeltaT/taus_sig) * core
            next_r = tf.math.sigmoid(next_x + self.I_ext)
            r_ta = r_ta.write(t, tf.squeeze(next_r, axis=-1))
            return t+1, next_x, next_r, r_ta
        r_ta = tf.TensorArray(tf.float32, size=T, clear_after_read=False)
        r_ta = r_ta.write(0, tf.squeeze(IC_r0, axis=-1))
        _, _, _, r_ta = tf.while_loop(lambda t, *_: t < T, body,
                                      loop_vars=[tf.constant(1, tf.int32), IC_x0, IC_r0, r_ta],
                                      parallel_iterations=1)
        return r_ta.stack()  # [T,N]

    @tf.function
    def forward(self, stim, IC_r, IC_x, DeltaT):
        """
        stim: [6, 10, T_total]
        IC_r, IC_x: [12, N, 1]
        returns: rates [12, T_total, N]
        """
        taus_sig = self.taus_sigmoid()

        def run_group(w_in_sel, offset):
            ta = tf.TensorArray(tf.float32, size=6, clear_after_read=False)
            for j in tf.range(6):
                r_seq = self.simulate_one(
                    stim[j],              # [10, T]
                    IC_r[offset + j],     # [N, 1]
                    IC_x[offset + j],     # [N, 1]
                    DeltaT,
                    taus_sig,             # [N, 1]
                    w_in_sel              # [N, 10]
                )                         # -> [T, N]
                ta = ta.write(j, r_seq)   # accumula [T, N]
            return ta.stack()              # [6, T, N]

        group1 = run_group(self.W_in, 0)   # [6, T, N]
        group2 = run_group(self.W_in2, 6)  # [6, T, N]

        rates = tf.concat([group1, group2], axis=0)  # [12, T, N]
        return rates


    def loss(self, rates, exp_data, thermal=100, window_size=4, loss_type='l2'):
        """
        Implements the article's objective:

            L = sqrt( sum_{task=1..12} sum_{t} sum_{i} (r_{i,t,task} - rhat_{i,t,task})^2 )

        - `rates`: Tensor [12, T_total, N] produced by `forward(...)`.
        - Experimental FRs are expected to be [N, 175] (already at 0.20s bins).
        - Synthetic trajectories are simulated at 0.05s; after warmup (thermal) they
        have 700 steps and are downsampled to 175 via moving_average(window=4).
        """
        loss_sq = tf.constant(0.0, tf.float32)  # accumulator for SUM of squared errors

        for idx in range(12):
            # Model side: discard warmup and downsample ONLY synthetic rates
            R = rates[idx]                            # [T_total, N]
            R = R[thermal:]                           # -> [700, N] (by construction)
            Rm = moving_average(R, window_size=4)     # -> [175, N]
            RmT = tf.transpose(Rm, perm=[1, 0])       # -> [N, 175]

            # Experimental side: DO NOT touch (already [N, 175])
            FR = tf.cast(exp_data[f'FR{idx+1}'], tf.float32)  # [N, 175]

            # Strict shape check (temporal length must match)
            tf.debugging.assert_equal(
                tf.shape(RmT)[1], tf.shape(FR)[1],
                message='Time length mismatch (model 175 vs FR 175)'
            )

            if loss_type.lower() == 'l1':
                # Optional: L1 routed through squared sum to keep same global form
                diff = tf.abs(RmT - FR)
                loss_sq += tf.reduce_sum(tf.square(diff))
            else:
                diff_sq = tf.math.squared_difference(RmT, FR)
                loss_sq += tf.reduce_sum(diff_sq)

        # Main data term: sqrt of the sum of squared errors
        data_term = tf.sqrt(loss_sq)

        # Optional small L2 regularization (kept from previous version)
        reg = (tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.W_in) + tf.nn.l2_loss(self.W_in2))
        reg_term = 1e-3 * tf.sqrt(2.0 * reg)

        return data_term + reg_term