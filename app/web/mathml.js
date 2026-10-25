/* mathml.js - tiny MathML builders shared by the browser demos (the browser renders MathML natively, no library).
   MathML.I identifier, N number, O operator, T text, R row, sub, sup, frac, sum(under, body), ind(...) indicator 1[...], M(...) display block. */
(function () {
  const I = s => `<mi>${s}</mi>`, N = s => `<mn>${s}</mn>`, O = s => `<mo>${s}</mo>`, T = s => `<mtext>${s}</mtext>`;
  const R = (...x) => `<mrow>${x.join("")}</mrow>`;
  const sub = (a, b) => `<msub>${a}${b}</msub>`, sup = (a, b) => `<msup>${a}${b}</msup>`, frac = (a, b) => `<mfrac>${a}${b}</mfrac>`;
  const sum = (u, b) => R(`<munder>${O("∑")}${u}</munder>`, b);
  const ind = (...x) => R(`<mi mathvariant="double-struck">1</mi>`, O("["), ...x, O("]"));
  const M = (...x) => `<math display="block">${x.join("")}</math>`;
  window.MathML = { I, N, O, T, R, sub, sup, frac, sum, ind, M };
})();
