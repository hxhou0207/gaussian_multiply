# gaussian_multiply

## Prerequisites

- Python 3.x
- `matplotlib` library (`pip install matplotlib`)

## How to Run

1. **Clone or download the script.**

2. **Run the script from the command line:**

   ```bash
   python gaussian_multiply.py [--mu1 <mean1>] [--sigma1 <stddev1>] [--mu2 <mean2>] [--sigma2 <stddev2>]
   ```

   - `--mu1`: Mean of the first Gaussian
   - `--sigma1`: Standard deviation of the first Gaussian
   - `--mu2`: Mean of the second Gaussian
   - `--sigma2`: Standard deviation of the second Gaussian

   **Example:**

   ```bash
   python gaussian_multiply.py --mu1 0 --sigma1 1 --mu2 2 --sigma2 1
   ```

   If you omit any of the arguments, the script will use the default values.

## Output

The script will display a plot with the two input Gaussian distributions and the resulting Gaussian distribution after multiplication.