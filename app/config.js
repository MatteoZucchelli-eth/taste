// Deployment settings. Locally, serve.py maps /img/ to data/lapis/images.
// On Vercel, point imgBase at the Blob store, e.g. "https://<store>.public.blob.vercel-storage.com/lapis/".
window.TASTE_CONFIG = {
  imgBase: "/img/",
  nTrain: 50,      // questions used to learn the person (first nRandom are random, the rest infogain)
  nRandom: 5,
  nTest: 20,       // test pairs where the frozen model predicts first
  nSure: 5,       // test pairs the model is most sure about; URL ?sure=N
  nContra: 5,     // test pairs the model is most sure about AND the crowd model would get wrong; URL ?contra=N (rest random)
  nCand: 3000,     // candidate pairs scored per infogain question
  sideBiasSd: 0.5, // prior sd of the per-person top/bottom (or left/right) bias term
  minViewMs: 450,  // taps faster than this after the pair appears are ignored
};
