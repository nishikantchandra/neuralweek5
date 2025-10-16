// app.js
/* ES6 module – UI glue & visualisations
   Requires: DataLoader, GRUModel
*/

import {DataLoader} from './data-loader.js';
import {GRUModel}  from './gru.js';

export class App {
  constructor() {
    this.loader = new DataLoader();
    this.model  = new GRUModel();
    this.data   = null;
  }

  /* wire UI -------------------------------------------------------------- */
  init() {
    const fileIn = document.getElementById('csvFile');
    const trainBtn= document.getElementById('trainBtn');
    const prog   = document.getElementById('progress');
    const status = document.getElementById('status');

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
      trainBtn.disabled = true;
      await this._train(prog, status);
      trainBtn.disabled = false;
    });
  }

  /* training ------------------------------------------------------------- */
  async _train(prog, status) {
    const {X_train, y_train, X_test, y_test, symbols} = this.data;
    this.model.build(symbols);
    status.textContent = 'Training…';
    prog.value = 0;
    await this.model.train(X_train, y_train, X_test, y_test, 25, 64, (epoch, logs) => {
      prog.value = (epoch+1)/25;
      status.textContent = `Epoch ${epoch+1} – loss=${logs.loss.toFixed(4)} acc=${logs.binaryAccuracy.toFixed(4)}`;
    });
    status.textContent = 'Evaluating…';
    const evals = this.model.evaluate(X_test, y_test);
    this._renderEval(evals.stockAcc, symbols);
    this._renderTimeline(X_test, y_test, symbols);
    status.textContent = 'Done.';
  }

  /* accuracy bar chart --------------------------------------------------- */
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
      options: {indexAxis: 'y', plugins:{title:{display:true,text:'Per-stock accuracy (best→worst)'}}}
    });
  }

  /* timeline plot per stock ---------------------------------------------- */
  _renderTimeline(X_test, y_test, symbols) {
    const preds = this.model.predict(X_test);
    const N = X_test.shape[0];
    const container = document.getElementById('timeline');
    container.innerHTML = '';

    for (let s = 0; s < symbols.length; s++) {
      const cv = []; // correct vector
      for (let i = 0; i < N; i++) {
        let correct = 0;
        for (let d = 0; d < 3; d++) {
          const idx = s*3 + d;
          const p = preds.slice([i,idx],[1,1]).round().dataSync()[0];
          const y = y_test.slice([i,idx],[1,1]).dataSync()[0];
          correct += (p===y)?1:0;
        }
        cv.push(correct/3);
      }
      const canvas = document.createElement('canvas');
      container.appendChild(canvas);
      new Chart(canvas, {
        type: 'line',
        data: {
          labels: Array.from({length:N},(_,i)=>i),
          datasets: [{
            label: symbols[s],
            data: cv,
            segment: {
              borderColor: ctx => ctx.p0.parsed.y > 0.5 ? 'green' : 'red'
            },
            pointRadius: 0,
            borderWidth: 1
          }]
        },
        options: {scales:{y:{min:0,max:1}}, plugins:{legend:{display:true}}}
      });
    }
    preds.dispose();
  }
}
