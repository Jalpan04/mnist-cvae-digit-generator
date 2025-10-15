# MNIST Conditional Variational Autoencoder

A generative model that learns to produce handwritten digits conditioned on specific digit labels.

![digits](https://github.com/user-attachments/assets/70c60869-9276-4524-a939-4ee77371dc79)

## What is a CVAE?

A Conditional Variational Autoencoder extends the standard VAE by conditioning both encoding and decoding on labels. This allows controlled generation: specify a digit (0-9) and the model generates it in varying handwriting styles.

**Components:**
- **Encoder**: Maps image x and label y to latent distribution parameters μ and σ
- **Decoder**: Generates image from latent vector z and label y
- **Latent Space**: Random sampling creates style variation

## Mathematical Framework

### Model Distributions

**Encoder:**
```
q_φ(z|x,y) = N(z; μ_φ(x,y), σ²_φ(x,y))
```

**Decoder:**
```
p_θ(x|z,y) = Bernoulli(x; f_θ(z,y))
```

**Prior:**
```
p(z) = N(0, I)
```

### Loss Function

The model maximizes the Evidence Lower Bound (ELBO):

```
L = E[log p_θ(x|z,y)] - KL[q_φ(z|x,y) || p(z)]
```

**Reconstruction Loss** (binary cross-entropy):
```
L_recon = Σ_i [x_i log(x̂_i) + (1-x_i)log(1-x̂_i)]
```

**KL Divergence** (closed form):
```
L_KL = -1/2 Σ_j [1 + log(σ²_j) - μ²_j - σ²_j]
```

**Reparameterization Trick:**
```
z = μ + σ ⊙ ε, where ε ~ N(0,I)
```

This enables backpropagation through stochastic sampling.

## Architecture

**Encoder:**
- Input: 784 pixels + 10-dim one-hot label
- Hidden layers: 794 → 512 → 256
- Output: μ and log(σ²) vectors (latent_dim = 2)

**Decoder:**
- Input: latent_dim + 10-dim label
- Hidden layers: 12 → 256 → 512
- Output: 784 pixels with sigmoid (reshaped to 28×28)

## Training

**Dataset:** MNIST (60k training images, normalized to [0,1])

**Process:**
1. Encode (x,y) → μ, σ
2. Sample z = μ + σ⊙ε
3. Decode (z,y) → x̂
4. Compute loss and backpropagate

**Hyperparameters:** Batch size 128, learning rate 0.001, 50-100 epochs

## Generation

**Single Digit:**
1. Choose digit label y
2. Sample z ~ N(0,I)
3. Generate x = decoder(z,y)

Different random samples produce different handwriting styles for the same digit.

**Multi-Digit Sequences:**
Generate each digit independently with separate latent samples.

<img width="1569" height="151" alt="generated_num" src="https://github.com/user-attachments/assets/a94dc1bc-8b9c-4900-bbd8-68f9fc520109" />


## Key Properties

**Latent Space:**
- 2D enables visualization
- Higher dimensions (10-20) capture richer variations
- Continuous: nearby points produce similar images

**Style vs Identity:**
- Label controls digit class (identity)
- Latent vector controls handwriting style
- Separation enables controlled generation

**Advantages over Standard VAE:**
- Direct control over digit class
- Latent space focuses on style variation
- More efficient generation

## Limitations

- Generated images may be blurry (reconstruction loss encourages averaging)
- Posterior collapse: decoder may ignore latent code
- Limited to MNIST distribution characteristics

## Applications

- Data augmentation for training classifiers
- Synthetic dataset generation
- Style exploration and visualization
- Anomaly detection via reconstruction error
