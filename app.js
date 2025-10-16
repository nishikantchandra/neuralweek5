// app.js
export class App {
  constructor() {
    this.loader = new DataLoader();
    this.model  = new GRUModel();
    this.data   = null;
    this.abort  = null;           // AbortController for cancellation
  }

  init() {
    const fileIn  = document.getElementById('csvFile');
    const trainBtn=document.getElementById('trainBtn');
    const abortBtn=document.getElementById('abortBtn');
    const prog    = document.getElementById('progress');
    const status  = document.getElementById('status');

    fileIn.addEventListener('change', async (ev)=>{
      const file = ev.target.files[0];
      if (!file) return;
      status.textContent = 'Parsing CSV…';
      try {
        this.data = await this.loader.load(file);
        status.textContent = `Loaded ${this.data.X_train.shape[0]} train / ${this.data.X_test.shape[0]} test samples`;
        trainBtn.disabled = false;
      } catch (e) {
        status.textContent = 'Error: ' + e;
      }
    });

    trainBtn.addEventListener('click', async ()=>{
      this.abort = new AbortController();
      trainBtn.disabled = true;  abortBtn.style.display='inline';
      await this._train(prog, status);
      trainBtn.disabled = false; abortBtn.style.display='none';
    });

    abortBtn.addEventListener('click', ()=> this.abort.abort());
  }

  /* ---------- training with yield points ------------------------------- */
  async _train(prog, status) {
    const {X_train, y_train, X_test, y_test, symbols} = this.data;
    this.model.build(symbols);
    status.textContent = 'Training…';  prog.value = 0;

    const epochs = 20;
    for (let e = 0; e < epochs; e++) {
      if (this.abort.signal.aborted) { status.textContent='Aborted'; return; }
      const h = await this.model.model.fit(X_train, y_train, {
        epochs: 1,  batchSize: 64,
        validationData: [X_test, y_test]
      });
      prog.value = (e+1)/epochs;
      status.textContent = `Epoch ${e+1}/${epochs} – loss=${h.history.loss[0].toFixed(4)}`;
      await tf.nextFrame();          // yield to browser
    }

    status.textContent = 'Evaluating (this may take a few seconds)…';
    await tf.nextFrame();

    const evals = await this._evaluateBatched(X_test, y_test, symbols);
    this._renderEval(evals.stockAcc, symbols);
    await this._renderTimelineBatched(X_test, y_test, symbols);
    status.textContent = 'Done.';
  }

  /* ---- batched evaluation (non-blocking) ----------------------------- */
  async _evaluateBatched(X_test, y_test, symbols) {
    const preds = this.model.predict(X_test);
    const N     = preds.shape[0];
    const stockAcc = new Array(symbols.length).fill(0);
    const batch = 256;                 // tune if needed

    for (let b = 0; b < N; b += batch) {
      if (this.abort.signal.aborted) break;
      const end = Math.min(b + batch, N);
      const pBatch = preds.slice([b,0], [end-b, 30]).round();
      const yBatch = y_test.slice([b,0], [end-b, 30]);
      const cmp    = pBatch.equal(yBatch);  // bool tensor
      const correct= cmp.sum(0);            // sum over batch
      const arr    = await correct.data();
      for (let s = 0; s < symbols.length; s++) {
        for (let d = 0; d < 3; d++) stockAcc[s] += arr[s*3 + d];
      }
      tf.dispose([pBatch, yBatch, cmp, correct]);
      await tf.nextFrame();                // keep UI alive
    }
    stockAcc.forEach((_,i)=>stockAcc[i]/=(N*3));
    preds.dispose();
    return {stockAcc};
  }

  /* ---- batched timeline plot ----------------------------------------- */
  async _renderTimelineBatched(X_test, y_test, symbols) {
    const preds = this.model.predict(X_test);
    const N = preds.shape[0];
    const container = document.getElementById('timeline');
    container.innerHTML = '';
    const batch = 256;

    for (let s = 0; s < symbols.length; s++) {
      const cv = [];                       // correctness vector
      for (let b = 0; b < N; b += batch) {
        if (this.abort.signal.aborted) break;
        const end = Math.min(b + batch, N);
        const pBatch = preds.slice([b, s*3], [end-b, 3]).round();
        const yBatch = y_test.slice([b, s*3], [end-b, 3]);
        const right  = pBatch.equal(yBatch).all(1).cast('float32'); // [batch]
        cv.push(...Array.from(await right.data()));
        tf.dispose([pBatch, yBatch, right]);
        await tf.nextFrame();
      }
      /* small canvas per stock */
      const canvas = document.createElement('canvas');
      canvas.height = 40; canvas.width = 600;
      container.appendChild(canvas);
      new Chart(canvas, {
        type: 'line',
        data: {
          labels: Array.from({length: cv.length}, (_,i)=>i),
          datasets: [{
            label: symbols[s],
            data: cv,
            segment: { borderColor: ctx => ctx.p0.parsed.y > 0.5 ? 'green' : 'red' },
            pointRadius: 0, borderWidth: 1
          }]
        },
        options: { scales:{y:{min:0,max:1}}, plugins:{legend:{display:false}} }
      });
    }
    preds.dispose();
  }

  /* ------------------ unchanged helpers ------------------------------- */
  _renderEval(stockAcc, symbols) {
    const container = document.getElementById('accChart');
    container.innerHTML = '';
    const sorted = symbols.map((s,i)=>({s, acc: stockAcc[i]}))
                          .sort((a,b)=>b.acc-a.acc);
    const canvas = document.createElement('canvas');
    container.appendChild(canvas);
    new Chart(canvas, {
      type: 'bar',
      data: {
        labels: sorted.map(x=>x.s),
        datasets: [{
          label: 'Accuracy',
          data: sorted.map(x=>x.acc),
          backgroundColor: 'rgba(54,162,235,0.8)'
        }]
      },
      options: { indexAxis: 'y', plugins:{ title:{display:true,text:'Per-stock accuracy (best→worst)'}} }
    });
  }
}
