---
layout: default               # keep Sleek’s default template
title:  Home
nav_order: 1                  # makes the nav bar order match
---

<!-- ───────────────────────────── HERO ──────────────────────────── -->

# Multi-Modal Movie Genre Classification
*A dual-pipeline approach to multi-label prediction*  

---

<!-- ───────────────────────────  ABOUT ──────────────────────────── -->

{: #about .section}
## About

This project demonstrates **two complementary approaches** to movie-genre classification:

* **Text model** – NLP on film synopses  
* **Vision model** – CNN on poster imagery  

Together they outperform single-modality baselines.

---

<!-- ────────────────────────  2. DATA  ───────────────────────────── -->
{: #data .section}
## Data and Dataset Explanation
<div class="sub-nav">
  <a href="#how-did-we-collect-data">Collection</a> ·
  <a href="#exploratory-data-analysis">EDA</a> ·
  <a href="#label-encoding-and-text-processing">Text Prep</a>
</div>


### How did we collect data?

We decided to use the [title.basics.tsv.gz](https://developer.imdb.com/non-commercial-datasets/) from the IMDB website.  
Initially, this data contained ~4 million datapoints. Based on the [online statistics](https://www.statista.com/statistics/188658/movie-genres-in-north-america-by-box-office-revenue-since-1995), we cut down the number of genres to 9. We selected the genres with the highest revenue in the statistics provided. After filtering the number of genres to 9, we got a dataset which was mostly skewed towards Drama and Comedy, as shown in the below figure.

<figure>
  <img class="dataset centered" src="assets/Images/initial_distribution.png" />
  <figcaption>Figure 1: Data distribution per genre after filtering.</figcaption>
</figure>

To reduce skewness in the dataset, we leveraged the number of movies available per genre. Since the number of movies per genre was large, we capped the number of movies in each genre to approximately 4,000, as shown below.

<figure>
  <img class="dataset centered" src="assets/Images/after_filtering.png" />
  <figcaption>Figure 2: Data distribution per genre after capping.</figcaption>
</figure>

Next, the problem we faced was that the dataset we used contained only IMDb IDs, titles, and genres. It did not include posters or plots for each movie. So, we used the [TMDB API](https://developer.themoviedb.org/reference/intro/getting-started) to retrieve plot summaries and poster URLs while capping the number of images. One can obtain their own API KEY for retrieving the plots and poster URLs by following the steps on the [TMDB API docs](https://developer.themoviedb.org/reference/intro/getting-started).

We added the poster URLs and plots of each movie in the dataset in the columns `plot` and `img`. One can download the posters using the [code](download_posters.html).

<figure>
  <img class="dataset centered" src="assets/Images/textual_data.png" />
  <figcaption>Figure 3: Random rows from movies_with_posters.csv</figcaption>
</figure>

We saved the posters using their IMDb IDs for easy access.

<figure>
  <img class="dataset centered" src="../assets/Images/posters.png" />
  <figcaption>Figure 4: Random movie posters.</figcaption>
</figure>

---

### Exploratory Data Analysis

*(Placeholder — insert visualizations or commentary here.)*

---

## Label Encoding and Text Processing

Each film has between one and three genres, allowing us to perform multi-label classification. Unlike multi-class, where only one output is given, multi-label allows multiple predictions to be made at once. Therefore, to represent the multiple combinations of labels in a way that the classifier can understand, we used multi-hot encoding. This is a binary representation where `1` signifies that a description belongs to a genre and `0` means that it does not.

<figure>
  <img class="multihot centered" src="Assets/Images/Multi-Hot.png" />
  <figcaption>Figure 5: Multi-hot label encoding of the genres.</figcaption>
</figure>

Before a film description is given as input to the classifier, the text must first be converted to a canonical form. It is therefore processed in the following ways:

- Tokenization to separate the words within sentences
- Removal of punctuation and bad characters
- Conversion of accented characters to non-accented form (e.g., “Léon” → “Leon”)
- Removal of stop words using the NLTK stop word list
- Lemmatization to convert words into a form compatible with GloVe word embeddings

<figure>
  <img class="descriptions-processed centered" src="Assets/Images/Description-Processing.png" />
  <figcaption>Figure 6: Processing the film descriptions before and after.</figcaption>
</figure>


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
