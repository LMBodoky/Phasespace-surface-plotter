# 5D Polynomial Ray Tracing Toolbox (Electron Optics Version)
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Utility functions
# ============================================================

def monomial_multiply(exp1, exp2):
    return tuple(e1 + e2 for e1, e2 in zip(exp1, exp2))

def monomial_order(exp):
    return sum(exp)

def zero_exp(dim=5):
    return (0,) * dim

# ============================================================
# Polynomial Map Class (5D) with CORRECT composition
# ============================================================

class PolynomialMap:
    def __init__(self, order=3):
        self.order = order
        self.dim = 5  # (x, xp, y, yp, delta)
        self.map = [dict() for _ in range(self.dim)]

    # ----------------------------
    # Identity map
    # ----------------------------
    @staticmethod
    def identity(order=3):
        M = PolynomialMap(order)
        for i in range(5):
            exp = [0] * 5
            exp[i] = 1
            M.map[i][tuple(exp)] = 1.0
        return M

    # ----------------------------
    # Add polynomial term
    # ----------------------------
    def add_term(self, output_index, coefficient, exponent):
        exponent = tuple(exponent)
        if monomial_order(exponent) <= self.order:
            self.map[output_index][exponent] = self.map[output_index].get(exponent, 0.0) + coefficient

    # ----------------------------
    # Apply map to ray
    # ----------------------------
    def apply(self, ray):
        out = np.zeros(5)
        for i in range(5):
            for exp, coef in self.map[i].items():
                term = coef
                for var_index in range(5):
                    p = exp[var_index]
                    if p != 0:
                        term *= ray[var_index] ** p
                out[i] += term
        return out

    # ============================================================
    # Internal truncated polynomial algebra (dict: exp -> coef)
    # ============================================================

    def _poly_add(self, A, B):
        out = dict(A)
        for e, c in B.items():
            out[e] = out.get(e, 0.0) + c
            if out[e] == 0.0:
                del out[e]
        return out

    def _poly_mul(self, A, B):
        out = {}
        for ea, ca in A.items():
            for eb, cb in B.items():
                en = monomial_multiply(ea, eb)
                if monomial_order(en) <= self.order:
                    out[en] = out.get(en, 0.0) + ca * cb
        # drop exact zeros (optional)
        out = {e: c for e, c in out.items() if c != 0.0}
        return out

    def _poly_pow(self, P, n):
        if n < 0:
            raise ValueError("Only nonnegative integer powers supported.")
        if n == 0:
            return {zero_exp(self.dim): 1.0}
        if n == 1:
            return dict(P)

        # exponentiation by squaring
        result = {zero_exp(self.dim): 1.0}
        base = dict(P)
        k = n
        while k > 0:
            if k & 1:
                result = self._poly_mul(result, base)
            k >>= 1
            if k:
                base = self._poly_mul(base, base)
        return result

    # ----------------------------
    # Compose maps (self ∘ other) [CORRECT]
    # ----------------------------
    def compose(self, other):
        """
        Return self ∘ other, i.e. apply 'other' first, then 'self'.
        Correct truncated polynomial substitution up to self.order.
        """
        if self.dim != other.dim:
            raise ValueError("Map dimensions do not match.")

        result = PolynomialMap(self.order)

        for i in range(self.dim):
            Pi = self.map[i]
            composed_poly = {}

            for exp_self, coef_self in Pi.items():
                # Start as constant polynomial coef_self
                term_poly = {zero_exp(self.dim): coef_self}

                # Multiply by Π_k (Q_k(u))^(exp_self[k])
                for k in range(self.dim):
                    p = exp_self[k]
                    if p == 0:
                        continue
                    Qk_pow = self._poly_pow(other.map[k], p)
                    term_poly = self._poly_mul(term_poly, Qk_pow)
                    if not term_poly:
                        break

                composed_poly = self._poly_add(composed_poly, term_poly)

            result.map[i] = composed_poly

        return result

# ============================================================
# Optical Elements
# ============================================================

def drift(L, order=3):
    M = PolynomialMap.identity(order)
    # x -> x + L*x'
    M.add_term(0, L, (0, 1, 0, 0, 0))
    # y -> y + L*y'
    M.add_term(2, L, (0, 0, 0, 1, 0))
    return M

def thin_lens(f, Cs=0.0, Cc=0.0, order=3):
    M = PolynomialMap.identity(order)

    # Linear focusing: x' -> x' - x/f, y' -> y' - y/f
    M.add_term(1, -1.0 / f, (1, 0, 0, 0, 0))
    M.add_term(3, -1.0 / f, (0, 0, 1, 0, 0))

    # --- Rotational spherical aberration (toy model in slopes) ---
    # x' -= Cs (x'^2 + y'^2) x'
    # y' -= Cs (x'^2 + y'^2) y'
    if Cs != 0.0:
        M.add_term(1, -Cs, (0, 3, 0, 0, 0))  # x'^3
        M.add_term(1, -Cs, (0, 1, 0, 2, 0))  # x'*y'^2
        M.add_term(3, -Cs, (0, 0, 0, 3, 0))  # y'^3
        M.add_term(3, -Cs, (0, 2, 0, 1, 0))  # y'*x'^2

    # --- Chromatic aberration (toy model) ---
    # x' -= (Cc/f) * x * delta
    # y' -= (Cc/f) * y * delta
    if Cc != 0.0:
        M.add_term(1, -Cc / f, (1, 0, 0, 0, 1))
        M.add_term(3, -Cc / f, (0, 0, 1, 0, 1))

    return M

# ============================================================
# Beam Generation
# ============================================================

def generate_rays(
    n=10000,
    sigma_r=2e-4,
    max_angle=5e-2,
    energy_spread=5e-2
):
    rays = []
    for _ in range(n):
        # Gaussian spatial distribution
        x = np.random.normal(0, sigma_r)
        y = np.random.normal(0, sigma_r)

        # Uniform angular disk (area-uniform)
        r = max_angle * np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()
        xp = r * np.cos(theta)
        yp = r * np.sin(theta)

        # Uniform chromatic spread
        delta = energy_spread * (2 * np.random.rand() - 1)

        rays.append(np.array([x, xp, y, yp, delta]))
    return rays

# ============================================================
# Plotting
# ============================================================

labels = ["x", "xp", "y", "yp", "delta"]

def plot_slice(rays, i, j, title="", xlim=None, ylim=None):
    xs = [ray[i] for ray in rays]
    ys = [ray[j] for ray in rays]
    colors = [ray[4] for ray in rays]  # color by delta

    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys, s=2, c=colors, cmap="viridis")
    plt.xlabel(labels[i])
    plt.ylabel(labels[j])
    plt.title(title)
    plt.colorbar(label="delta")

    if xlim is not None:
        plt.xlim(-xlim, xlim)
    if ylim is not None:
        plt.ylim(-ylim, ylim)

    # plt.tight_layout()
    plt.show()

# ============================================================
# Example Execution
# ============================================================

if __name__ == "__main__":
    order = 3

    # Strongly aberrated system
    system = (
    drift(0.3, order)
    .compose(thin_lens(f=10, Cs=5000, Cc=10, order=order))   # lens 1
    .compose(drift(0.05, order))                            # spacing between lenses
    .compose(thin_lens(f=6, Cs=20, Cc=5, order=order))      # lens 2
    .compose(drift(0.02, order))
)

    # Generate beam
    rays_in = generate_rays(
        n=5000,
        sigma_r=3e-4,
        max_angle=1e-2,
        energy_spread=5e-2
    )

    # Propagate
    rays_out = [system.apply(ray) for ray in rays_in]

    # Plots
    plot_slice(rays_out, 0, 1, "x - x' phase space (spherical curvature)")
    plot_slice(rays_out, 0, 2, "x - y spot diagram")
    plot_slice(rays_out, 0, 4, "x - delta (chromatic spread)")
    