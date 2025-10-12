import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline
import numpy.polynomial.chebyshev as cheb
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

# 🎯 Target function we're trying to approximate
def target_function(x):
    return np.exp(-x**2/20) * np.cos(5*x)

# 📈 Derivative of our target function (for Hermite interpolation)
def target_derivative(x):
    return (-x/10 * np.cos(5*x) - 5 * np.sin(5*x)) * np.exp(-x**2/20)

# 🌟 Generate special Chebyshev nodes that help avoid Runge's phenomenon
def generate_chebyshev_nodes(n, a=-10, b=10):
    """Generate n Chebyshev nodes on interval [a,b] - these are magic! ✨"""
    k = np.arange(n)
    nodes = np.cos((2*k + 1) * np.pi / (2*n))  # On [-1,1]
    return (a + b)/2 + (b - a)/2 * nodes  # Map to [a,b]

# 🔧 More stable implementation of Lagrange interpolation
def stable_lagrange_interpolation(x_nodes, y_nodes, x_eval):
    """A more numerically stable Lagrange interpolation implementation"""
    n = len(x_nodes)
    result = np.zeros_like(x_eval)
    
    # For each evaluation point, build the Lagrange polynomial
    for i, x in enumerate(x_eval):
        L = 0
        for j in range(n):
            basis = 1
            # Build the j-th Lagrange basis polynomial
            for k in range(n):
                if k != j:
                    basis *= (x - x_nodes[k]) / (x_nodes[j] - x_nodes[k])
            L += y_nodes[j] * basis
        result[i] = L
    
    return result

# 📊 Discrete least squares using Chebyshev polynomials as basis functions
def discrete_least_squares_chebyshev_basis(n_nodes, degree, x_eval):
    """Fit using Chebyshev polynomials - much better than monomials! 🚀"""
    # Generate Chebyshev nodes for sampling
    x_nodes = generate_chebyshev_nodes(n_nodes)
    y_nodes = target_function(x_nodes)
    
    # Map nodes to [-1, 1] where Chebyshev polynomials live
    t_nodes = x_nodes / 10
    
    # Build design matrix with Chebyshev basis functions
    A = np.zeros((n_nodes, degree + 1))
    for i in range(n_nodes):
        for j in range(degree + 1):
            A[i, j] = cheb.chebval(t_nodes[i], [0]*j + [1])
    
    # Solve normal equations A^T A c = A^T y
    coeffs = linalg.solve(A.T @ A, A.T @ y_nodes)
    
    # Evaluate our beautiful approximation
    t_eval = x_eval / 10
    result = np.zeros_like(x_eval)
    for i, t_val in enumerate(t_eval):
        result[i] = 0
        for j in range(degree + 1):
            result[i] += coeffs[j] * cheb.chebval(t_val, [0]*j + [1])
    
    return result, x_nodes, y_nodes

# 🧮 Continuous least squares using Chebyshev polynomials
def continuous_least_squares_chebyshev(degree, x_eval):
    """Continuous least squares - integrating over the whole domain! 📐"""
    # Map x to [-1,1] interval
    t_eval = x_eval / 10
    
    # Compute Chebyshev coefficients via numerical integration
    coeffs = np.zeros(degree + 1)
    for k in range(degree + 1):
        # Use Gauss-Chebyshev quadrature for accurate integration
        t_nodes, weights = cheb.chebgauss(100)
        integrand = target_function(10*t_nodes) * cheb.chebval(t_nodes, [0]*k + [1])
        coeffs[k] = np.sum(weights * integrand)
        
        # Normalize by the inner product
        denom_integrand = cheb.chebval(t_nodes, [0]*k + [1])**2
        denom = np.sum(weights * denom_integrand)
        coeffs[k] /= denom
    
    # Return our smooth approximation
    return cheb.chebval(t_eval, coeffs)

# 🎪 Hermite interpolation using Newton's divided differences
def hermite_newton_divided_difference_correct(x_nodes, x_eval):
    """Hermite interpolation - we know both function values AND derivatives! 🎯"""
    n = len(x_nodes) - 1
    
    # Create extended node sequence - each point appears twice
    z = np.zeros(2*(n+1))
    for i in range(n+1):
        z[2*i] = x_nodes[i]
        z[2*i+1] = x_nodes[i]
    
    # Initialize divided difference table
    size = 2*(n+1)
    Q = np.zeros((size, size))
    
    # Fill first column with function values
    for i in range(size):
        Q[i, 0] = target_function(z[i])
    
    # Compute divided differences with special handling for repeated nodes
    for j in range(1, size):
        for i in range(size - j):
            if abs(z[i] - z[i+j]) < 1e-10:  # Repeated node - use derivative!
                Q[i, j] = target_derivative(z[i])
            else:
                # Regular divided difference
                Q[i, j] = (Q[i+1, j-1] - Q[i, j-1]) / (z[i+j] - z[i])
    
    # Build the Hermite interpolation polynomial
    result = np.zeros_like(x_eval)
    for idx, x_val in enumerate(x_eval):
        # Newton form: H(x) = f[z_0] + Σ f[z_0,...,z_k] * (x-z_0)...(x-z_{k-1})
        poly = Q[0, 0]  # Constant term
        product = 1.0
        
        for k in range(1, size):
            product *= (x_val - z[k-1])
            poly += Q[0, k] * product
        
        result[idx] = poly
    
    return result

# ============================================================================
# 🎨 VISUALIZATION SECTION - Let's make some beautiful plots! 🌈
# ============================================================================

# Fine grid for smooth plotting
x_fine = np.linspace(-10, 10, 1000)
y_true = target_function(x_fine)

print("Starting plot generation... 🌠")

# ============================================================================
# 📈 FIGURES 1-2: Lagrange interpolation with equidistant nodes
# ============================================================================

print("Generating Figures 1-2: Lagrange interpolation with equidistant nodes")
plt.figure(figsize=(15, 6))

# Figure 1: 20 equidistant nodes
plt.subplot(1, 2, 1)
x_nodes_20 = np.linspace(-10, 10, 20)
y_nodes_20 = target_function(x_nodes_20)
y_lagrange_20 = stable_lagrange_interpolation(x_nodes_20, y_nodes_20, x_fine)

plt.plot(x_fine, y_true, 'b-', label='True Function', linewidth=2)
plt.plot(x_fine, y_lagrange_20, 'r--', label='Lagrange Interpolation', linewidth=2)
plt.plot(x_nodes_20, y_nodes_20, 'ko', markersize=6, label='Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Lagrange Interpolation with 20 Equidistant Nodes')
plt.legend()
plt.grid(True, alpha=0.3)
plt.minorticks_on()
plt.grid(True, which='minor', alpha=0.2)

# Figure 2: 50 equidistant nodes - watch out for Runge's phenomenon! ⚠️
plt.subplot(1, 2, 2)
x_nodes_50 = np.linspace(-10, 10, 50)
y_nodes_50 = target_function(x_nodes_50)
y_lagrange_50 = stable_lagrange_interpolation(x_nodes_50, y_nodes_50, x_fine)

plt.plot(x_fine, y_true, 'b-', label='True Function', linewidth=2)
plt.plot(x_fine, y_lagrange_50, 'r--', label='Lagrange Interpolation', linewidth=2)
plt.plot(x_nodes_50, y_nodes_50, 'ko', markersize=4, label='Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Lagrange Interpolation with 50 Equidistant Nodes')
plt.legend()
plt.grid(True, alpha=0.3)
plt.minorticks_on()
plt.grid(True, which='minor', alpha=0.2)

plt.tight_layout()
plt.savefig('degree20_50.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 🌟 FIGURE 3: Chebyshev nodes to the rescue! 🦸
# ============================================================================

print("Generating Figure 3: Chebyshev node interpolation")
plt.figure(figsize=(12, 8))
node_counts = [20, 30, 40, 50]
colors = ['red', 'green', 'orange', 'purple']

plt.plot(x_fine, y_true, 'b-', label='True Function', linewidth=3)

for i, n in enumerate(node_counts):
    # Chebyshev nodes are our heroes against Runge's phenomenon! 💪
    x_cheb = generate_chebyshev_nodes(n)
    y_cheb = target_function(x_cheb)
    
    y_interp = stable_lagrange_interpolation(x_cheb, y_cheb, x_fine)
    
    plt.plot(x_fine, y_interp, '--', color=colors[i], linewidth=1.5, 
             label=f'Chebyshev Nodes (n={n})')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Lagrange Interpolation with Chebyshev Nodes')
plt.legend()
plt.grid(True, alpha=0.3)
plt.minorticks_on()
plt.grid(True, which='minor', alpha=0.2)
plt.savefig('che.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 🔄 FIGURE 4: Cubic spline interpolation - smooth and stable! ✨
# ============================================================================

print("Generating Figure 4: Cubic spline interpolation")
plt.figure(figsize=(12, 8))
node_counts_spline = [5, 10, 20, 40]
colors_spline = ['red', 'green', 'orange', 'purple']

plt.plot(x_fine, y_true, 'b-', label='True Function', linewidth=3)

for i, n in enumerate(node_counts_spline):
    x_nodes = np.linspace(-10, 10, n)
    y_nodes = target_function(x_nodes)
    
    # Cubic splines - piecewise polynomials that join smoothly 🎢
    cs = CubicSpline(x_nodes, y_nodes)
    y_spline = cs(x_fine)
    
    plt.plot(x_fine, y_spline, '--', color=colors_spline[i], linewidth=1.5,
             label=f'Cubic Spline (n={n})')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Cubic Spline Interpolation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.minorticks_on()
plt.grid(True, which='minor', alpha=0.2)
plt.savefig('cluie.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 🎯 FIGURES 5-8: Hermite interpolation with equidistant nodes
# ============================================================================

print("Generating Figures 5-8: Hermite interpolation with equidistant nodes")
node_counts_hermite = [10, 15, 20, 30]
fig, axes = plt.subplots(4, 2, figsize=(20, 24))
fig.subplots_adjust(hspace=0.6, wspace=0.3)

for i, n in enumerate(node_counts_hermite):
    # Left subplot: Function and interpolation
    ax1 = axes[i, 0]
    x_nodes = np.linspace(-10, 10, n)
    
    # Hermite interpolation using divided differences
    y_hermite = hermite_newton_divided_difference_correct(x_nodes, x_fine)
    
    ax1.plot(x_fine, y_true, 'b-', label='True Function', linewidth=2)
    ax1.plot(x_fine, y_hermite, 'r--', label='Hermite Interpolation', linewidth=2)
    ax1.plot(x_nodes, target_function(x_nodes), 'ko', markersize=4, label='Nodes')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_ylim(-1, 1)
    ax1.set_title(f"Hermite Interpolation (n={n} equidistant nodes)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.minorticks_on()
    ax1.grid(True, which='minor', alpha=0.2)
    
    # Right subplot: Error analysis 📊
    ax2 = axes[i, 1]
    error = np.abs(y_hermite - y_true)
    ax2.plot(x_fine, error, 'g-', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Error')
    ax2.set_title(f'Error of Hermite Interpolation (n={n})')
    ax2.grid(True, alpha=0.3)
    ax2.minorticks_on()
    ax2.grid(True, which='minor', alpha=0.2)

plt.savefig('hermite_equidistant.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 🌟 FIGURES 9-12: Hermite interpolation with Chebyshev nodes
# ============================================================================

print("Generating Figures 9-12: Hermite interpolation with Chebyshev nodes")
node_counts_hermite_cheb = [20, 30, 40, 50]
fig, axes = plt.subplots(4, 2, figsize=(20, 24))
fig.subplots_adjust(hspace=0.6, wspace=0.3)

for i, n in enumerate(node_counts_hermite_cheb):
    # Left subplot: Function and interpolation
    ax1 = axes[i, 0]
    
    # Chebyshev nodes + Hermite = Double awesomeness! 🚀
    x_cheb = generate_chebyshev_nodes(n)
    y_hermite_cheb = hermite_newton_divided_difference_correct(x_cheb, x_fine)
    
    ax1.plot(x_fine, y_true, 'b-', label='True Function', linewidth=2)
    ax1.plot(x_fine, y_hermite_cheb, 'r--', label='Hermite Interpolation', linewidth=2)
    ax1.plot(x_cheb, target_function(x_cheb), 'ko', markersize=4, label='Chebyshev Nodes')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_ylim(-1, 1)
    ax1.set_title(f"Hermite Interpolation (n={n} Chebyshev nodes)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.minorticks_on()
    ax1.grid(True, which='minor', alpha=0.2)
    
    # Right subplot: Error analysis
    ax2 = axes[i, 1]
    error = np.abs(y_hermite_cheb - y_true)
    ax2.plot(x_fine, error, 'g-', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Error')
    ax2.set_title(f'Error of Hermite Interpolation (n={n} Chebyshev nodes)')
    ax2.grid(True, alpha=0.3)
    ax2.minorticks_on()
    ax2.grid(True, which='minor', alpha=0.2)

plt.savefig('hermite_chebyshev.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 📐 FIGURES 13-14: Discrete least squares approximation
# ============================================================================

print("Generating Figures 13-14: Discrete least squares with Chebyshev basis")
plt.figure(figsize=(15, 6))

# Figure 13: Degree 50 polynomial approximation
plt.subplot(1, 2, 1)
n_nodes = 100
degree = 50

y_ls_50, x_nodes_ls_50, y_nodes_ls_50 = discrete_least_squares_chebyshev_basis(n_nodes, degree, x_fine)

plt.plot(x_fine, y_true, 'b-', label='True Function', linewidth=2)
plt.plot(x_fine, y_ls_50, 'r--', label=f'Discrete LS (degree={degree})', linewidth=2)
plt.plot(x_nodes_ls_50, y_nodes_ls_50, 'ko', markersize=2, alpha=0.5, label='Chebyshev Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.ylim(-1, 1)
plt.title('Discrete Least Squares with Chebyshev Basis (Degree 50)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.minorticks_on()
plt.grid(True, which='minor', alpha=0.2)

# Figure 14: Degree 60 polynomial approximation - pushing the limits! 🔥
plt.subplot(1, 2, 2)
degree = 60

y_ls_60, x_nodes_ls_60, y_nodes_ls_60 = discrete_least_squares_chebyshev_basis(n_nodes, degree, x_fine)

plt.plot(x_fine, y_true, 'b-', label='True Function', linewidth=2)
plt.plot(x_fine, y_ls_60, 'r--', label=f'Discrete LS (degree={degree})', linewidth=2)
plt.plot(x_nodes_ls_60, y_nodes_ls_60, 'ko', markersize=2, alpha=0.5, label='Chebyshev Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.ylim(-1, 1)
plt.title('Discrete Least Squares with Chebyshev Basis (Degree 60)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.minorticks_on()
plt.grid(True, which='minor', alpha=0.2)

plt.tight_layout()
plt.savefig('discrete_least_squares.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 🎯 FIGURE 15: Continuous least squares approximation
# ============================================================================

print("Generating Figure 15: Continuous least squares")
plt.figure(figsize=(12, 8))
degrees = [40, 50, 60]
colors_cheb = ['red', 'green', 'orange']

plt.plot(x_fine, y_true, 'b-', label='True Function', linewidth=3)

for i, degree in enumerate(degrees):
    y_cheb_fit = continuous_least_squares_chebyshev(degree, x_fine)
    plt.plot(x_fine, y_cheb_fit, '--', color=colors_cheb[i], linewidth=1.5,
             label=f'Chebyshev LS (degree={degree})')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Continuous Least Squares with Chebyshev Polynomials')
plt.legend()
plt.grid(True, alpha=0.3)
plt.minorticks_on()
plt.grid(True, which='minor', alpha=0.2)
plt.savefig('chebshey.png', dpi=300, bbox_inches='tight')
plt.show()

print("All plots generated successfully! 🎉✨")