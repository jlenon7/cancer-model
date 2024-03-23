# Cancer data

## Feature Columns

The mean, standard error, and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features.  For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

- `radius` - (mean of distances from center to points on the perimeter)
- `texture` - (standard deviation of gray-scale values)
- `perimeter`
- `area`
- `smoothness` - (local variation in radius lengths)
- `compactness` - (perimeter^2 / area - 1.0)
- `concavity` - (severity of concave portions of the contour)
- `concave points` - (number of concave portions of the contour)
- `symmetry`
- `fractal dimension` - ("coastline approximation" - 1)
