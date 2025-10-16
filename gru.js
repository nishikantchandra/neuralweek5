// gru.js
/* ES6 module – GRU model definition, training, evaluation, save/load
   Exports: GRUModel class
*/

export class GRUModel {
  constructor() {
    this.model = null;
    this.symbols = null;
  }

  /* build model ---------------------------------------------------------- */
  build(symbols) {
    this.symbols = symbols;
    const numStocks = symbols.length;
    this.model = tf.sequential({
      layers: [
        tf.layers.gru({units: 64, returnSequences: true, inputShape: [12, 20]}),
        tf.layers.gru({units: 32}),
        tf.layers.dense({units: 30, activation: 'sigmoid'})
      ]
    });
    this.model.compile({
      optimizer: tf.train.adam(1e-3),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });
  }

  /* train ---------------------------------------------------------------- */
  async train(X_train, y_train, X_val, y_val, epochs=20, batchSize=64, onEpochEnd) {
    if (!this.model) throw 'Call build() first';
    const history = await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      validationData: [X_val, y_val],
      callbacks: {onEpochEnd}
    });
    return history;
  }

  /* predict -------------------------------------------------------------- */
  predict(X) {
    return this.model.predict(X);
  }

  /* evaluate ------------------------------------------------------------- */
  evaluate(X_test, y_test) {
    const preds = this.predict(X_test); // [N,30]
    const yTrue = y_test;               // [N,30]
    const N = preds.shape[0];
    const stockAcc = new Array(this.symbols.length).fill(0);
    const dayAcc   = new Array(3).fill(0);

    for (let i = 0; i < N; i++) {
      const p = preds.slice([i,0], [1,30]).round();
      const y = yTrue.slice([i,0], [1,30]);
      const cmp = p.mul(y).add(p.neg().add(1).mul(y.neg().add(1))); // 1 if equal
      for (let s = 0; s < this.symbols.length; s++) {
        let sc = 0;
        for (let d = 0; d < 3; d++) {
          const idx = s*3 + d;
          const correct = cmp.slice([0, idx], [1,1]).dataSync()[0];
          stockAcc[s] += correct;
          dayAcc[d]   += correct;
        }
      }
      p.dispose(); y.dispose(); cmp.dispose();
    }
    stockAcc.forEach((_,i)=>stockAcc[i]/=(N*3));
    dayAcc.forEach((_,i)=>dayAcc[i]/=(N*this.symbols.length));

    preds.dispose();
    return {stockAcc, dayAcc};
  }

  /* save/load ----------------------------------------------------------- */
  async save() {
    const artifacts = await this.model.save('localstorage://gru-stock');
    return artifacts;
  }

  async load() {
    this.model = await tf.loadLayersModel('localstorage://gru-stock');
    // assume symbols unchanged
  }
}
