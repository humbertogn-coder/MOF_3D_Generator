const refs = require("./refs.js");
const blocks = [
  { title: "Highlights" },
  { plain: "Active learning for low-band-gap three-dimensional metal–organic frameworks: prospective DFT validation of a chemically audited hypothetical catalogue" },
  { items: [
    "Audited PORMAKE catalogue of 131 hypothetical 3D MOFs built on charge-balanced nodes",
    "Three prospective DFT batches, 18 labels: all 13 open-shell frameworks below 1 eV",
    "QMOF-trained surrogate: 0.23 eV error for open-shell 3d, 0.96 eV for Zn4O frameworks",
    "Metal d band edges fix the gap; linkers tune it only when the band edges are organic",
    "Periodic GFN1-xTB on CHGNet geometries checks closed-shell gaps without training data",
  ] },
];
module.exports = { blocks, refs, docTitle: "Highlights", lineNumbers: false };
