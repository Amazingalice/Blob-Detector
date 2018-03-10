# Blob-Detector
Implemented a Laplacian blob detector for Scale Space Blob Detection
1. Generate a Laplacian of Gaussian filter.
2. Build a Laplacian scale space, starting with some initial scale and going for n iterations:
(a) Filter image with scale-normalized Laplacian at current scale.
(b) Save square of Laplacian response for current level of scale space.
(c) Increase scale by a factor k.
3. Perform nonmaximum suppression in scale space.
4. Display resulting circles at their characteristic scales.
