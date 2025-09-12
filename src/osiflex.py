import numpy as np
import tensorflow as tf
from src.config import OSIFlexConfig


class OSIFlex:
    """
    OSI-Flex phase retrieval algorithm.

    Attributes:
        shearing_images (tf.Tensor): Input shearing data tensor of shape [N_images, H, W].
        config (OSIFlexConfig): Configuration object for algorithm parameters.
        phase_est (tf.Variable): Current estimate of the phase [H, W].
        shears (tf.Variable): Shear parameters tensor [N_shears, 2].
        optimizer (tf.keras.optimizers.Optimizer): Optimizer for phase and shear updates.
    """

    def __init__(self, shearing_data: np.ndarray, config: OSIFlexConfig) -> None:
        """
        Initialize OSI-Flex algorithm.

        Args:
            shearing_data (np.ndarray): Array of sheared images, shape [N_images, H, W].
            config (OSIFlexConfig): Configuration object with algorithm parameters.
        """
        self.shearing_images: tf.Tensor = tf.convert_to_tensor(shearing_data, dtype=tf.float32)
        self.config: OSIFlexConfig = config

        # Phase initialization
        self.phase_est: tf.Variable = tf.Variable(tf.zeros_like(shearing_data[0, ...]), dtype=tf.float32)

        # Shears
        self.shears: tf.Variable = tf.Variable(
            config.shears, dtype=tf.float32, trainable=(config.shear_lr > 0.0), name="shears"
        )
        self.tune_shear: bool = config.shear_lr > 0.0
        self.shear_lr: float = config.shear_lr

        # Optimizer
        self.optimizer: tf.keras.optimizers.Optimizer = self._create_optimizer(config.opti_method, config.learning_rate)

        # Regularization
        self.sign_reg_coeff: float = config.sign_reg_coeff

        if self.sign_reg_coeff > 0:
            self._sign_reg = lambda img: nnc_reg(img, self.sign_reg_coeff)
        elif self.sign_reg_coeff < 0:
            self._sign_reg = lambda img: npc_reg(img, -self.sign_reg_coeff)
        else:
            self._sign_reg = lambda img: 0.0

        self.tv_coeff: float = config.tv_coeff
        if self.tv_coeff > 0:
            self._tv_reg = lambda img: tv_reg(img, self.tv_coeff)
        else:
            self._tv_reg = lambda img: 0.0

    def _create_optimizer(self, method: str, lr: float) -> tf.keras.optimizers.Optimizer:
        """
        Create TensorFlow optimizer.

        Args:
            method (str): Optimizer method ("sgd", "adam", "rmsprop").
            lr (float): Learning rate.

        Returns:
            tf.keras.optimizers.Optimizer: Initialized optimizer.
        """
        method = method.strip().lower()
        if method == "sgd":
            return tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True)
        elif method == "adam":
            return tf.keras.optimizers.Adam(lr)
        elif method == "rmsprop":
            return tf.keras.optimizers.RMSprop(lr, rho=0.9, epsilon=1e-7)
        else:
            raise ValueError(f"Invalid optimizer: {method}")

    @tf.function
    def _generate_difference_images_diffshift(self, img: tf.Tensor, shifts: tf.Tensor) -> tf.Tensor:
        """
        Differentiable forward model: computes image differences with respect to shifts using bilinear interpolation.

        Args:
            img (tf.Tensor): Phase image of shape [H, W].
            shifts (tf.Tensor): Shear shifts, shape [N, 2].

        Returns:
            tf.Tensor: Differences between input image and shifted images, shape [1, H, W].
        """
        H, W = tf.shape(img)[0], tf.shape(img)[1]
        N = tf.shape(shifts)[0]

        y = tf.range(H, dtype=tf.float32)
        x = tf.range(W, dtype=tf.float32)
        grid_y, grid_x = tf.meshgrid(y, x, indexing='ij')
        grid = tf.expand_dims(tf.stack([grid_y, grid_x], axis=-1), axis=0)  # [1, H, W, 2]

        shifts = tf.reshape(shifts, [N, 1, 1, 2])
        shifted_grid = grid - shifts
        y_shift, x_shift = shifted_grid[..., 0], shifted_grid[..., 1]

        y0, x0 = tf.floor(y_shift), tf.floor(x_shift)
        y1, x1 = y0 + 1, x0 + 1
        wy, wx = y_shift - y0, x_shift - x0

        y0 = tf.clip_by_value(tf.cast(y0, tf.int32), 0, H - 1)
        y1 = tf.clip_by_value(tf.cast(y1, tf.int32), 0, H - 1)
        x0 = tf.clip_by_value(tf.cast(x0, tf.int32), 0, W - 1)
        x1 = tf.clip_by_value(tf.cast(x1, tf.int32), 0, W - 1)

        def get_vals(y_idx: tf.Tensor, x_idx: tf.Tensor) -> tf.Tensor:
            indices = tf.stack([y_idx, x_idx], axis=-1)
            flat_idx = tf.reshape(indices, [N * H * W, 2])
            vals = tf.gather_nd(img, flat_idx)
            return tf.reshape(vals, [N, H, W])

        Ia, Ib, Ic, Id = get_vals(y0, x0), get_vals(y0, x1), get_vals(y1, x0), get_vals(y1, x1)
        wa, wb, wc, wd = (1 - wy)*(1 - wx), (1 - wy)*wx, wy*(1 - wx), wy*wx
        shifted = wa*Ia + wb*Ib + wc*Ic + wd*Id
        return tf.expand_dims(img, axis=0) - shifted

    @tf.function
    def _data_mismatch(self, img1: tf.Tensor, img2: tf.Tensor) -> tf.Tensor:
        """
        Compute mean squared error between two image tensors.

        Args:
            img1 (tf.Tensor): Tensor of shape [N, H, W].
            img2 (tf.Tensor): Tensor of shape [N, H, W].

        Returns:
            tf.Tensor: Scalar tensor representing data mismatch (MSE).
        """
        return tf.reduce_mean(tf.square(img1 - img2))

    @tf.function
    def _total_cost(self) -> tf.Tensor:
        """
        Compute total cost: data fidelity + TV regularization + sign constraint.

        Returns:
            tf.Tensor: Scalar tensor representing total cost.
        """
        shearing_images_est = self._generate_difference_images_diffshift(self.phase_est, self.shears)
        fidelity = self._data_mismatch(shearing_images_est, self.shearing_images)
        tv_term = self._tv_reg(self.phase_est)
        sign_term = self._sign_reg(self.phase_est)
        return fidelity + tv_term + sign_term

    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the OSI-Flex optimization.

        Returns:
            tuple:
                - phase_est (np.ndarray): Estimated phase, shape [H, W].
                - shears (np.ndarray): Estimated shears, shape [N_shears, 2].
                - costs (np.ndarray): Array of cost values per iteration.
        """
        max_iter = self.config.max_iter
        costs = np.full(max_iter, np.nan)

        patience, min_delta = 10, 1e-9
        best_loss, steps_since_improvement = float('inf'), 0

        print("=== Starting OSI-flex algorithm ===")
        if self.tune_shear:
            print("→ Enabling shear fine-tuning")
        else:
            print("→ Phase estimation only (no shear fine-tuning)")

        for iter in range(max_iter):
            with tf.GradientTape() as tape:
                cost = self._total_cost()

            if self.tune_shear:
                grads = tape.gradient(cost, [self.phase_est, self.shears])
                self.optimizer.apply_gradients([
                    (grads[0], self.phase_est),
                    (self.shear_lr * grads[1], self.shears)
                ])
            else:
                grads = tape.gradient(cost, self.phase_est)
                self.optimizer.apply_gradients([(grads, self.phase_est)])

            current_loss = cost.numpy()
            costs[iter] = current_loss
            if iter % 10 == 0:
                print(f"Iteration {iter:3d}: Cost = {current_loss:.6e}")

            # Early stopping
            if current_loss < best_loss - min_delta:
                best_loss, steps_since_improvement = current_loss, 0
            else:
                steps_since_improvement += 1
            if steps_since_improvement >= patience:
                print(f"Stopping early at iteration {iter}: no improvement for {patience} steps.")
                break

        phase_est = self.phase_est.numpy()
        phase_est -= np.median(phase_est)
        return phase_est, self.shears.numpy(), costs


@tf.function
def nnc_reg(img: tf.Tensor, coeff: float) -> tf.Tensor:
    """
    Non-negativity constraint: penalizes negative values.

    Args:
        img (tf.Tensor): Tensor of arbitrary shape [H, W] or [batch, H, W].
        coeff (float): Positive regularization coefficient.

    Returns:
        tf.Tensor: Scalar penalty tensor.
    """
    neg_elements = tf.where(img < 0, img, tf.zeros_like(img))
    neg_sum = -tf.reduce_sum(neg_elements)
    return coeff * neg_sum


@tf.function
def npc_reg(img: tf.Tensor, coeff: float) -> tf.Tensor:
    """
    Non-positivity constraint: penalizes positive values.

    Args:
        img (tf.Tensor): Tensor of arbitrary shape [H, W] or [batch, H, W].
        coeff (float): Positive regularization coefficient.

    Returns:
        tf.Tensor: Scalar penalty tensor.
    """
    pos_elements = tf.where(img > 0, img, tf.zeros_like(img))
    pos_sum = tf.reduce_sum(pos_elements)
    return coeff * pos_sum


@tf.function
def tv_reg(img: tf.Tensor, coeff: float) -> tf.Tensor:
    """
    Total variation regularization.

    Args:
        img (tf.Tensor): Tensor of shape [H, W].
        coeff (float): Regularization coefficient.

    Returns:
        tf.Tensor: Scalar penalty tensor.
    """
    d0 = img[1:, :-1] - img[:-1, :-1]
    d1 = img[:-1, 1:] - img[:-1, :-1]
    epsilon = 1e-6
    tv = tf.sqrt(tf.square(d0) + tf.square(d1) + epsilon)
    return coeff * tf.reduce_sum(tv)
