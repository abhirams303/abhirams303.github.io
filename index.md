---
layout: default
title: Home
---

<!-- ============ 1. INTRODUCTION ===================================== -->
<section id="about" class="section">
  <h2>About</h2>

  <p><strong>Multi-Modal Movie Genre Classification</strong> explores whether combining plot text
     with poster imagery improves multi-label genre prediction for movies.
     Single-modality models (e.g.&nbsp;text-only BERT) can miss complementary cues
     such as colour palette or visual tropes in posters.</p>

  <p>Project pipeline:</p>
  <ol>
    <li><strong>Data&nbsp;curation</strong> – 30 k IMDb synopses  +  posters</li>
    <li><strong>Text model</strong> – fine-tuned ELECTRA-small (best&nbsp;F1&nbsp;≈&nbsp;0.62)</li>
    <li><strong>Image model</strong> – EfficientNet-B0 embeddings</li>
    <li><strong>Fusion</strong> – concatenate text &amp; image vectors, train an MLP head</li>
    <li><strong>Deployment</strong> – Flask API + static demo on GitHub Pages</li>
  </ol>
</section>

<!-- ============ 2. DATA ============================================== -->
<section id="data" class="section">
  <h2>Data</h2>

  <p>We scraped the <em>IMDb title.basics</em> &amp; <em>title.plot</em> dumps (May 2025 snapshot)  
     and downloaded posters via the OMDb API.</p>

  <table>
    <thead><tr><th>step</th><th style="text-align:right">raw rows</th><th style="text-align:right">after cleaning</th><th>notes</th></tr></thead>
    <tbody>
      <tr><td>merge dumps</td><td style="text-align:right">94 k</td><td></td><td>join on <code>tconst</code></td></tr>
      <tr><td>drop non-English</td><td></td><td style="text-align:right">48 756</td><td>ISO&nbsp;639 lang tag</td></tr>
      <tr><td>remove &lt; 6-word plots</td><td></td><td style="text-align:right">30 546</td><td>rubric requirement</td></tr>
      <tr><td>poster available</td><td></td><td style="text-align:right">27 832</td><td>OMDb HTTP 200</td></tr>
    </tbody>
  </table>

  <figure>
    <img src="/assets/images/sample_table.png" alt="Sample IMDb rows" style="width:100%">
    <figcaption>Random data sample: title, ~40-word synopsis, pipe-separated genres, poster URL.</figcaption>
  </figure>

  <h3>Balanced label distribution</h3>
  <p>Undersampled frequent genres and oversampled rare ones to approach uniformity (≈ 4.9 k films/label).</p>
  <img src="/assets/images/genre_distribution.png" alt="Genre distribution bar chart">
</section>

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
        <td>Text modelling 📜</td>
        <td><a href="https://github.com/abhirams303" target="_blank">@abhirams303</a></td>
      </tr>
      <tr>
        <td>Teammate 2</td>
        <td>Image modelling 🖼️</td>
        <td>—</td>
      </tr>
      <tr>
        <td>Teammate 3</td>
        <td>Front-end 🌐</td>
        <td>—</td>
      </tr>
    </tbody>
  </table>

  <p><em>Course: Basics of AI · Prof. XYZ, Spring 2025</em></p>
</section>
