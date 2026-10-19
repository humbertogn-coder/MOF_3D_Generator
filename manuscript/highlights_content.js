const refs = require("./refs.js");
const blocks = [
  { title: "Highlights" },
  { plain: "Active learning guided screening of three-dimensional metal-organic frameworks: prospective DFT validation and the limits of band gap prediction" },
  { items: [
    "Model guided screening of 131 audited hypothetical 3D MOFs, validated prospectively",
    "27 frameworks computed, 18 labels: all open-shell Fe, Co and Cu frameworks below 1 eV",
    "Guided picks give 3 to 4 times more low gap hits than random; retraining adds little",
    "Fe and Co gaps are set by metal d states; Zn4O gaps follow the linker; Cu in between",
    "Periodic GFN1-xTB checks closed-shell predictions without any training data",
  ] },
];
module.exports = { blocks, refs, docTitle: "Highlights", lineNumbers: false };
