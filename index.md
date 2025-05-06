---
layout: default
title: Home
nav_order: 1
---

## About

This project demonstrates integration of **two approaches** to movie-genre classification:

- **Text model** – NLP on film synopses  
- **Vision model** – CNN on poster imagery  

Together they outperform single-modality baselines.

---

## Data and Dataset Explanation

[Collection](#how-did-we-collect-data) · [EDA](#exploratory-data-analysis) · [Text Prep](#label-encoding-and-text-processing)

### How did we collect data? 
{:#how-did-we-collect-data}

We started from the [title.basics.tsv.gz](https://developer.imdb.com/non-commercial-datasets/) IMDb dump (~4 M titles, 28 genres).  
Using [box-office statistics](https://www.statista.com/statistics/188658/movie-genres-in-north-america-by-box-office-revenue-since-1995),  
we kept the **9 highest-grossing genres**.

After filtering, the dataset was heavily skewed towards *Drama* and *Comedy* (Figure 1).

<figure class="centered-figure">
  <img src="assets/images/initial_distribution.png" alt="Initial distribution" />
  <figcaption style="text-align: center;"><em>Figure 1 · Genre counts after revenue-based filtering.</em></figcaption>

</figure>

To balance the dataset, we capped each genre at ≈ **4,000 films** (Figure 2).

<figure class="centered-figure">
  <img src="assets/images/after_filtering.png" alt="After balancing" />
  <figcaption style="text-align: center;"><em>Figure 2 · Balanced distribution after capping.</em></figcaption>
  
</figure>

Next, the problem we faced was that the dataset we used contained only IMDb IDs, titles, and genres. It did not include posters or plots for each movie. So, we used the TMDB API to retrieve plot summaries and poster URLs. One can obtain their own API KEY for retrieving the plots and poster URLs by following the steps on the [TMDB API docs](https://developer.themoviedb.org/reference/intro/getting-started).

Using the TMDB API, we retrieved poster URLs and plots and added them to our dataset in the columns `plot` and `img`. One can directly download our filtered dataset from [Kaggle](https://www.kaggle.com/datasets/kumaramara/movies-with-poster-urls-and-plots).

After downloading the dataset, use this [code](download_posters.html) to download the posters. Make sure you set up your TMDB API key before running the script.

<figure class="centered-figure">
  <img src="assets/images/textual_data.png" alt="Textual sample" />
  <figcaption style="text-align: center;"><em>Figure 3 · Sample rows after enrichment (`plot`, `img`).</em></figcaption>
</figure>

We saved all posters using their IMDb IDs for easy access.

<figure class="centered-figure">
  <img src="assets/images/posters.png" alt="Poster sample" />
  <figcaption style="text-align: center;"><em>Figure 4 · Random movie posters downloaded via TMDB.</em></figcaption>
</figure>


### Exploratory Data Analysis
{:#exploratory-data-analysis}

*Placeholder for visualizations – genre heatmaps, histogram plots, etc.*

---

### Label Encoding & Text Processing{:#label-encoding-and-text-processing }

Each movie has **1–3 genres**, which makes this a **multi-label classification** task.  
We represent genre combinations using **multi-hot encoding** (binary vector):

![Multi-hot example](/assets/images/Multi-Hot.png)  
*Figure 5 · Multi-hot binary label matrix.*

---

We applied the following steps to clean the plot descriptions:

- Tokenization of sentences
- Punctuation and noise removal
- Conversion of accented characters (e.g., “Léon” → “Leon”)
- Stopword removal (NLTK)
- Lemmatization for GloVe embedding compatibility

![Before/after description processing](/assets/images/Description-Processing.png)  
*Figure 6 · Text cleaning pipeline.*


<!-- ============ 3. CLASSIFIER ======================================== -->
<section id="classifier" class="section">
  <h2>Classifier (text-only baseline)</h2>

  <ul>
    <li><strong>Model</strong>: ELECTRA-small-discriminator</li>
    <li><strong>Frozen layers</strong>: bottom&nbsp;4&nbsp;/&nbsp;12</li>
  </ul>

  <h3>Hyper-parameters</h3>

  <p>epochs 5 batch 8 accum 2</p>
  <p>max_len 384 lr_head 2e-5 lr_backbone 5e-6</p>

  <p><strong>Validation</strong>: <code>P 0.580 R 0.668 F1 0.620</code> (threshold 0.30)</p>

  <details>
    <summary>Training command</summary>

    ```bash
    python train_electra_transfer.py \
      --tsv final_data.tsv \
      --epochs 5 --batch 8 --accum 2 \
      --lr_head 2e-5 --lr_backbone 5e-6 \
      --max_len 384 --freeze_layers 4 \
      --warmup_ratio 0.1 --out electra_best.pth
    warmup 10 %
    ```
  </details>
</section>

<!-- ============ 4. WEB APP / DEMO ==================================== -->
<section id="web-app" class="section">
  <h2>Web App ♟️</h2>
  <p>A minimal Flask API wraps the fusion model and a small JS front-end consumes it.</p>
  
  <figure>
    <img src="/assets/images/demo.gif" alt="Web-app demo">
    <figcaption>Interactive demo: paste a plot → top-3 predicted genres with probabilities.</figcaption>
  </figure>

  <h3>Run locally</h3>
  <pre>
    cd src/webapp
    pip install -r requirements.txt   # Flask + torch
    python app.py                     # http://127.0.0.1:5000/
  </pre>
</section>

<!-- ============ 5. TOPIC MODELLING / EXPLAIN ======================== -->
<section id="topic-modelling" class="section">
  <h2>Topic Modelling &amp; Visual Explainability</h2>
  <ul>
    <li><strong>BERTopic</strong> on synopses &rarr; latent clusters (“heist”, “courtroom drama”, “alien invasion”).</li>
    <li><strong>Grad-CAM</strong> on poster CNN &rarr; genre-specific regions highlighted (e.g. explosions for <em>Action</em>).</li>
  </ul>
  <p><em>(Insert screenshots/plots here.)</em></p>
</section>

<!-- ============ 6. REPRODUCIBILITY ================================== -->
<section id="run-code" class="section">
  <h2>Run Code 🛠️</h2>
  
  <pre>
  # 1 · Clone
  git clone https://github.com/abhirams303/mm-genre-classifier.git
  cd mm-genre-classifier

  # 2 · Create env
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt        # torch, transformers, pandas …

  # 3 · Re-train
  python train_electra_transfer.py --tsv final_data.tsv …

  # 4 · Generate site assets
  python notebooks/plots.py
  </pre>

  <p>The repository root doubles as this website’s source. A GitHub Actions workflow rebuilds and pushes the static site on every commit to <code>main</code>.</p>
</section>

<!-- ============ 7. CONTRIBUTORS ==================================== -->
<section id="contributors" class="section">
  <h2>Contributors</h2>

  <table>
    <thead>
      <tr>
        <th>Name</th>
        <th>Role</th>
        <th>GitHub</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Abhiram S</td>
        <td>Text modelling and Website Building</td>
        <td><a href="https://github.com/abhirams303" target="_blank">@abhirams303</a></td>
      </tr>
      <tr>
        <td>Satish Amara</td>
        <td>EDA and Image modelling </td>
        <td>—</td>
      </tr>
    </tbody>
  </table>

  <p><em>Course: Basics of AI · Prof. Jue Guo, Spring 2025</em></p>
</section>



<!-- ─────────────────────  TEXT MODEL ───────────────────── -->

{: #text-model .section}
## 2 Text Model Experiments

| Model                        | Frozen Layers | Val F1 | Notes                                      |
|-----------------------------|---------------|--------|--------------------------------------------|
| Bi-LSTM (GloVe-6B-100d)     | –             | 0.43   | 2-layer, 120 hidden, 70% dropout           |
| DistilBERT-base-uncased     | 3 / 6         | 0.57   | AdamW 3e-5, max_len = 384                  |
| ELECTRA-small-discriminator | 4 / 12        | 0.62   | best run → saved as `electra_best.pth`     |

---

### Why ELECTRA outperformed the rest

- **Transformers vs LSTM**  Transformers like ELECTRA attend to full plot context simultaneously, making them better suited for long, nuanced synopses. LSTMs struggle with long-range dependencies.
- **ELECTRA vs DistilBERT/BERT**  ELECTRA’s generator-discriminator pretraining gives it finer-grained token understanding. This helped capture subtleties in our movie plot dataset better than DistilBERT.
- **Partial fine-tuning**  Unfreezing only the top 8 layers let ELECTRA specialize on our domain while retaining strong general language representations.

---

<details>
<summary>Training command</summary>

```bash
python train_electra_transfer.py \
  --tsv data/final_data.tsv \
  --epochs 5 --batch 8 --accum 2 \
  --lr_head 2e-5 --lr_backbone 5e-6 \
  --max_len 384 --freeze_layers 4 \
  --warmup_ratio 0.1 --out electra_best.pth
```
</details> 

---

<!-- ──────────────────────── VISION MODEL ───────────────────────── -->

{: #vision-model .section}
## Vision Model

Fine-tuned **EfficientNet-B0** on movie posters for multi-label prediction.  
(Mixup, random crop, colour-jitter augmentations.)

---

{: #webapp .section}
## Web Application

Flask micro-service → loads `fusion_model.pt` and returns top-3 genres.  
Front-end page accepts a plot or poster upload and calls `/predict` with fetch/AJAX.

---

{: #topicmodelling .section}
## Topic Modelling

* **BERTopic** clusters plots into themes (“heist”, “alien invasion”, “courtroom drama”).  
* **Grad-CAM** reveals poster regions driving the CNN (e.g. explosions ⇒ *Action*).

---

{: #code .section}
## Running the Code

```bash
git clone https://github.com/abhirams303/mm-genre-classifier.git
cd mm-genre-classifier

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# reproduce ELECTRA baseline
python train_electra_transfer.py --tsv data/final_data.tsv ...

# launch demo
cd src/webapp && python app.py
