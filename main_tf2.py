
import os, time, json
import argparse
from pathlib import Path
import numpy as np
import scipy.io as sio
import scipy.ndimage as ndi
import tensorflow as tf

from rnn_rate_tf2 import RNNRateTF2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(42); np.random.seed(42)

def save_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_inhibitory_mask(path):
    D = sio.loadmat(path)
    inh = D.get('inhibitory')
    if inh is None:
        raise KeyError("Key 'inhibitory' not found in InhibitoryMask.mat")
    inh = np.asarray(inh).astype(np.float32).reshape(-1)
    return inh

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--firing_rates_mat', type=str, required=True)
    p.add_argument('--inhibitory_mask_mat', type=str, required=True)
    p.add_argument('--mode', choices=['train','eval'], default='train')
    p.add_argument('--n_trials', type=int, default=100000)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--loss_fn', type=str, default='l2')
    p.add_argument('--output_dir', type=str, default='out_tf2')
    p.add_argument('--N1', type=int, default=86)
    p.add_argument('--N2', type=int, default=106)
    p.add_argument('--N3', type=int, default=163)
    p.add_argument('--thermal', type=int, default=100)
    return p.parse_args()

def load_exp_data(fr_path):
    mat = sio.loadmat(fr_path)
    Exp = {}
    for i in range(1,13):
        key = f'FR{i}'
        if key not in mat:
            raise KeyError(f'{key} missing in {fr_path}')
        Exp[key] = np.asarray(mat[key]).astype(np.float32)
    return Exp, mat

def build_stim_and_IC(FR_mat, N, thermal, inputF5=0.5):
    if 'FR1' not in FR_mat:
        raise KeyError('FR1 missing in .mat; cannot infer T')
    
    T_model = 700   
    T_total = int(thermal) + T_model

    sound_on = 100 + thermal
    sound_dur = 360
    light_on = 260 + thermal
    delay_sound = 20
    delay_sound_off = 20
    delay_object = 20

    stim = np.zeros((6, 10, T_total), dtype=np.float32)

    cue_on  = int(np.ceil(sound_on + delay_sound))
    cue_off = int(np.ceil(sound_on + sound_dur + delay_sound_off))
    obj_on  = int(np.ceil(light_on + delay_object))

    stim[0:2, 0, cue_on:cue_off] = 1.0
    stim[3:5, 4, cue_on:cue_off] = 1.0
    stim[0:2, 5, cue_on:cue_off] = inputF5
    stim[3:5, 9, cue_on:cue_off] = inputF5

    stim[0, 1, obj_on:] = 1.0; stim[3, 1, obj_on:] = 1.0
    stim[1, 2, obj_on:] = 1.0; stim[4, 2, obj_on:] = 1.0
    stim[2, 3, obj_on:] = 1.0; stim[5, 3, obj_on:] = 1.0

    stim[0, 6, obj_on:] = inputF5; stim[3, 6, obj_on:] = inputF5
    stim[1, 7, obj_on:] = inputF5; stim[4, 7, obj_on:] = inputF5
    stim[2, 8, obj_on:] = inputF5; stim[5, 8, obj_on:] = inputF5

    sigma_gauss = 20
    stim = ndi.gaussian_filter1d(stim, sigma_gauss, axis=-1)
    for j in range(10):
        for c in range(6):
            stim[c, j, :] = ndi.gaussian_filter1d(stim[c, j, :], sigma_gauss)

    stim += np.random.normal(0.0, 0.005, size=stim.shape).astype(np.float32)

    IC_r = []; IC_x = []
    for i in range(1, 13):
        key_r = f'IC_FR{i}'; key_x = f'IC_X{i}'
        if key_r in FR_mat and key_x in FR_mat:
            r0 = np.asarray(FR_mat[key_r]).astype(np.float32).reshape(N,1)
            x0 = np.asarray(FR_mat[key_x]).astype(np.float32).reshape(N,1)
        else:
            r0 = np.zeros((N,1), dtype=np.float32)
            x0 = np.zeros((N,1), dtype=np.float32)
        IC_r.append(r0); IC_x.append(x0)
    IC_r = np.stack(IC_r, axis=0); IC_x = np.stack(IC_x, axis=0)
    return stim, IC_r, IC_x

def main():
    args = parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir/'config.json', vars(args))

    Exp_Data_np, raw = load_exp_data(args.firing_rates_mat)
    N = int(Exp_Data_np['FR1'].shape[0])

    Inib = load_inhibitory_mask(args.inhibitory_mask_mat)
    if Inib.shape[0] != N:
        raise ValueError(f'Inhibitory mask length {Inib.shape[0]} != N={N}')

    stim_np, IC_r_np, IC_x_np = build_stim_and_IC(raw, N, args.thermal)

    Exp_Data = {k: tf.convert_to_tensor(v) for k,v in Exp_Data_np.items()}
    stim = tf.convert_to_tensor(stim_np)     # [6,10,T_total]
    IC_r = tf.convert_to_tensor(IC_r_np)     # [12,N,1]
    IC_x = tf.convert_to_tensor(IC_x_np)     # [12,N,1]

    model = RNNRateTF2(N, args.N1, args.N2, args.N3, w_dist='gaus', gain=1.0, apply_dale=True, Inib=Inib)
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    DeltaT = tf.constant(1.0, tf.float32)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            rates = model.forward(stim, IC_r, IC_x, DeltaT)
            loss = model.loss(
                rates, Exp_Data,
                thermal=args.thermal, window_size=4, loss_type=args.loss_fn
            )
        vars_ = [model.W, model.W_in, model.W_in2, model.I_ext, model.taus_param]
        grads = tape.gradient(loss, vars_)

        # --- Gradient clipping (recommended default) ---
        CLIP_NORM = 1.0
        grads, _ = tf.clip_by_global_norm(grads, CLIP_NORM)

        opt.apply_gradients(zip(grads, vars_))

        # Enforce Dale's law after the update
        if model.apply_dale:
            model.project_dale()

        return loss

    @tf.function
    def eval_step():
        rates = model.forward(stim, IC_r, IC_x, DeltaT)
        loss = model.loss(rates, Exp_Data, thermal=args.thermal, window_size=4, loss_type=args.loss_fn)
        return loss

    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=opt, model=model)
    manager = tf.train.CheckpointManager(ckpt, directory=str(outdir), max_to_keep=3)


    if args.mode == 'train':
        best = None
        # --- Early stopping settings (no CLI flags by design) ---
        TARGET_LOSS = 1.00   # stop immediately once we reach this value
        PATIENCE = 200       # stop if no improvement for this many steps
        patience_left = PATIENCE

        for it in range(1, args.n_trials + 1):
            l = train_step()
            lval = float(l.numpy())
            ckpt.step.assign_add(1)

            improved = (best is None) or (lval < best - 1e-6)
            if improved:
                best = lval
                patience_left = PATIENCE
                # Save occasionally when improving (keeps I/O reasonable)
                if it % 50 == 0:
                    manager.save()
            else:
                patience_left -= 1

            if it % 10 == 0:
                print(f"[{it}/{args.n_trials}] loss={lval:.6f} best={best:.6f}", flush=True)

            # --- Early stop rules ---
            if lval <= TARGET_LOSS:
                print(f"Early stop: target loss {TARGET_LOSS} reached at step {it}.")
                break
            if patience_left <= 0:
                print(f"Early stop: no improvement for {PATIENCE} steps (best={best:.6f}).")
                break

        # Final checkpoint on exit
        manager.save()

if __name__ == '__main__':
    main()
