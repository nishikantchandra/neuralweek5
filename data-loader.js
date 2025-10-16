// data-loader.js
/* ES6 module – loads & prepares CSV entirely in browser
   Exports: DataLoader class with async load(file) -> {X_train, y_train, X_test, y_test, symbols}
*/

const EPS = 1e-7;

export class DataLoader {
  constructor() {
    this.symbols = null;          // sorted array of tickers
    this.dateMap = null;          // Map: dateISO -> {SYMBOL:{open,close}}
    this.dates = null;            // sorted unique dates
    this.stockMin = {};           // per-stock min for scaling
    this.stockMax = {};           // per-stock max for scaling
  }

  /* public API ----------------------------------------------------------- */
  async load(file, trainSplit = 0.8) {
    if (!file.name.endsWith('.csv')) throw 'Please upload a .csv file';
    const text = await file.text();
    this._parseCSV(text);
    this._sortDates();
    this._forwardFillGaps();
    this._computeMinMax(trainSplit); // only on train portion
    const {X, y} = this._makeWindows();
    return this._split(X, y, trainSplit);
  }

  /* CSV parsing ---------------------------------------------------------- */
  _parseCSV(text) {
    const lines = text.trim().split(/\r?\n/).filter(l => l);
    const hdr = lines[0].split(',').map(h => h.trim());
    const dateIdx = hdr.indexOf('Date');
    const symIdx  = hdr.indexOf('Symbol');
    const openIdx = hdr.indexOf('Open');
    const closeIdx= hdr.indexOf('Close');
    [dateIdx, symIdx, openIdx, closeIdx].forEach(i => {
      if (i < 0) throw 'CSV must contain Date,Symbol,Open,Close';
    });

    const map = new Map(); // date -> symbol -> {open,close}
    const symSet = new Set();

    for (let i = 1; i < lines.length; i++) {
      const row = this._splitLine(lines[i]);
      const date = row[dateIdx];
      const sym  = row[symIdx];
      const open = parseFloat(row[openIdx]);
      const close= parseFloat(row[closeIdx]);
      if (isNaN(open) || isNaN(close)) continue;
      symSet.add(sym);
      if (!map.has(date)) map.set(date, {});
      map.get(date)[sym] = {open, close};
    }

    this.dateMap = map;
    this.symbols  = Array.from(symSet).sort();
    this.dates    = Array.from(map.keys()).sort();
  }

  /* handle comma inside quotes */
  _splitLine(line) {
    const res = [];
    let cur = '', inQuote = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (c === '"') { inQuote = !inQuote; continue; }
      if (c === ',' && !inQuote) { res.push(cur); cur = ''; continue; }
      cur += c;
    }
    res.push(cur);
    return res;
  }

  /* sort dates ascending */
  _sortDates() {
    this.dates.sort((a,b) => new Date(a) - new Date(b));
  }

  /* forward-fill missing symbols per date */
  _forwardFillGaps() {
    for (let i = 1; i < this.dates.length; i++) {
      const prev = this.dateMap.get(this.dates[i-1]);
      const cur  = this.dateMap.get(this.dates[i]);
      for (const s of this.symbols) {
        if (!cur[s] && prev[s]) cur[s] = {...prev[s]};
      }
    }
  }

  /* MinMax on train period only */
  _computeMinMax(trainFrac) {
    const trainEnd = Math.floor(this.dates.length * trainFrac);
    for (const s of this.symbols) {
      let minO =  Infinity, maxO = -Infinity;
      let minC =  Infinity, maxC = -Infinity;
      for (let i = 0; i < trainEnd; i++) {
        const day = this.dateMap.get(this.dates[i]);
        if (!day[s]) continue;
        const {open, close} = day[s];
        minO = Math.min(minO, open);  maxO = Math.max(maxO, open);
        minC = Math.min(minC, close); maxC = Math.max(maxC, close);
      }
      this.stockMin[s] = {open: minO, close: minC};
      this.stockMax[s] = {open: maxO, close: maxC};
    }
  }

  /* scale helper */
  _scale(val, min, max) {
    return (val - min) / (max - min + EPS);
  }

  /* sliding windows 12->3 */
  _makeWindows() {
    const X = [], y = [];
    const L = 12, H = 3;
    for (let i = L; i <= this.dates.length - H; i++) {
      const windowDates = this.dates.slice(i-L, i+H);
      const baseDate = this.dates[i-1];
      const baseClose = {};
      for (const s of this.symbols) {
        baseClose[s] = this.dateMap.get(baseDate)[s]?.close;
      }

      // build input tensor slice [12, 20]
      const inp = [];
      let skip = false;
      for (let k = 0; k < L; k++) {
        const day = this.dateMap.get(windowDates[k]);
        if (!day) { skip = true; break; }
        const step = [];
        for (const s of this.symbols) {
          const rec = day[s];
          if (!rec) { skip = true; break; }
          const so = this._scale(rec.open , this.stockMin[s].open , this.stockMax[s].open);
          const sc = this._scale(rec.close, this.stockMin[s].close, this.stockMax[s].close);
          step.push(so, sc);
        }
        if (skip) break;
        inp.push(step);
      }
      if (skip) continue;

      // build output binary [30]
      const out = [];
      for (let h = 1; h <= H; h++) {
        const futDay = this.dateMap.get(windowDates[L-1+h]);
        if (!futDay) { skip = true; break; }
        for (const s of this.symbols) {
          const futClose = futDay[s]?.close;
          if (futClose == null) { skip = true; break; }
          out.push(futClose > baseClose[s] ? 1 : 0);
        }
      }
      if (skip) continue;

      X.push(inp);
      y.push(out);
    }
    const Xtensor = tf.tensor3d(X);
    const ytensor = tf.tensor2d(y, [y.length, this.symbols.length*3]);
    return {X: Xtensor, y: ytensor};
  }

  /* train/test split */
  _split(X, y, trainFrac) {
    const n = X.shape[0];
    const split = Math.floor(n * trainFrac);
    return {
      X_train: X.slice([0,0,0], [split, 12, 20]),
      y_train: y.slice([0,0], [split, 30]),
      X_test : X.slice([split,0,0], [n-split, 12, 20]),
      y_test : y.slice([split,0], [n-split, 30]),
      symbols: this.symbols
    };
  }
}
