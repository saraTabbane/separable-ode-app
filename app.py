import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Separable ODE Solver", page_icon="🧠", layout="centered")
st.title("🧠 Separable Differential Equation Solver")
st.write("Solves **only separable** first-order ODEs and shows the steps.")

x, y = sp.symbols("x y")
C = sp.Symbol("C")

transformations = standard_transformations + (implicit_multiplication_application,)

ALLOWED_FUNCS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "sec": sp.sec,
    "csc": sp.csc,
    "cot": sp.cot,
    "exp": sp.exp,
    "ln": sp.log,
    "log": sp.log,
    "sqrt": sp.sqrt,
}
ALLOWED_SYMBOLS = {"x": x, "y": y, **ALLOWED_FUNCS}

examples = {
    "Basic: x*y": "x*y",
    "Trig: y*tan(x)": "y*tan(x)",
    "Nonlinear: exp(x)*y**2": "exp(x)*y**2",
    "Fraction: (x**2+1)/(y-3)": "(x**2+1)/(y-3)",
    "Reciprocal: sin(x)/y": "sin(x)/y",
}

# ----------------------------
# Session state init
# ----------------------------
if "rhs_input" not in st.session_state:
    st.session_state["rhs_input"] = "x*y"

if "example_choice" not in st.session_state:
    st.session_state["example_choice"] = list(examples.keys())[0]


# ----------------------------
# Callbacks (SAFE)
# ----------------------------
def load_example():
    choice = st.session_state["example_choice"]
    st.session_state["rhs_input"] = examples[choice]


def clear_input():
    st.session_state["rhs_input"] = ""


# ----------------------------
# Helpers
# ----------------------------
def safe_parse(expr_str: str) -> sp.Expr:
    expr_str = expr_str.strip()
    if not expr_str:
        raise ValueError("Empty input.")
    return parse_expr(expr_str, local_dict=ALLOWED_SYMBOLS, transformations=transformations)


def validate_expression(expr: sp.Expr) -> None:
    extras = expr.free_symbols - {x, y}
    if extras:
        extras_str = ", ".join(sorted(str(s) for s in extras))
        raise ValueError(f"Unsupported variable(s): {extras_str}. Use only x and y.")


def separate_rhs(rhs_expr: sp.Expr):
    """
    Returns (ok, g(x), h(y)) if rhs = g(x)*h(y)
    """
    try:
        sep = sp.separatevars(rhs_expr, symbols=[x, y], dict=True, force=True)
        if isinstance(sep, dict) and (x in sep) and (y in sep):
            gx = sp.simplify(sep[x])
            hy = sp.simplify(sep[y])
            return True, gx, hy
    except Exception:
        pass
    return False, None, None


def latex_simpl(expr: sp.Expr) -> str:
    return sp.latex(sp.simplify(expr))


# ----------------------------
# UI
# ----------------------------
left, right = st.columns([2, 1], vertical_alignment="center")

with left:
    st.text_input(
        "Enter the RHS in:  dy/dx = f(x, y)",
        key="rhs_input",
        help="Examples: x*y, y*tan(x), (x**2+1)/(y-3). Use ** for powers (y**2).",
    )

with right:
    st.selectbox("Try an example", list(examples.keys()), key="example_choice")
    a, b = st.columns(2)
    with a:
        st.button("Load", use_container_width=True, on_click=load_example)
    with b:
        st.button("Clear", use_container_width=True, on_click=clear_input)

show_explicit = st.toggle("Try to solve explicitly for y", value=True)
show_why = st.toggle("Show why it is separable", value=True)

solve_clicked = st.button("Solve", type="primary", use_container_width=True)

# ----------------------------
# Solve
# ----------------------------
if solve_clicked:
    st.divider()

    rhs_str = st.session_state["rhs_input"]

    # Validation message: empty input
    if not rhs_str.strip():
        st.error("⚠️ Please enter the RHS. Example: `x*y` or `sin(x)/y`.")
        st.stop()

    # Parse & validate
    try:
        rhs = safe_parse(rhs_str)
        validate_expression(rhs)
    except Exception as e:
        st.error("⚠️ Invalid input.")
        st.caption("Tips: use only x and y, powers like `y**2`, and functions like `sin(x)` / `exp(x)`.")
        st.code(str(e))
        st.stop()

    # Show equation
    st.latex(r"\frac{dy}{dx} = " + sp.latex(rhs))

    # Check separable
    ok, gx, hy = separate_rhs(rhs)
    if not ok:
        st.error("❌ Not separable (or I couldn't separate it automatically).")
        st.info("A separable ODE has the form: dy/dx = g(x)·h(y).")
        st.stop()

    st.success("✅ Separable! We can write the RHS as g(x)·h(y).")

    # Why separable
    if show_why:
        st.markdown("## Why this equation is separable")
        st.write("Because the RHS can be factored into:")
        st.latex(r"g(x) = " + latex_simpl(gx))
        st.latex(r"h(y) = " + latex_simpl(hy))
        st.write("So:")
        st.latex(r"\frac{dy}{dx} = " + latex_simpl(gx) + r"\cdot " + latex_simpl(hy))
        st.caption("One part depends only on x and the other only on y → variables can be separated.")

    # Steps
    st.markdown("## Step-by-step solution")

    st.markdown("### Step 1 — Separate variables")
    st.latex(r"\frac{1}{h(y)}\,dy = g(x)\,dx")

    left_side = sp.simplify(1 / hy)
    st.latex(latex_simpl(left_side) + r"\,dy = " + latex_simpl(gx) + r"\,dx")

    st.markdown("### Step 2 — Integrate both sides")
    Iy = sp.integrate(left_side, y)
    Ix = sp.integrate(gx, x)

    st.latex(r"\int " + latex_simpl(left_side) + r"\,dy = \int " + latex_simpl(gx) + r"\,dx")
    st.latex(latex_simpl(Iy) + " = " + latex_simpl(Ix) + r" + C")

    st.markdown("### Step 3 — Final implicit solution")
    implicit_eq = sp.Eq(Iy, Ix + C)
    st.latex(sp.latex(implicit_eq))

    if show_explicit:
        st.markdown("### Step 4 (optional) — Solve for y")
        try:
            sols = sp.solve(sp.Eq(Iy, Ix + C), y)
            if sols:
                for sol in sols:
                    st.latex(r"y = " + sp.latex(sp.simplify(sol)))
                # --- Step 5 — Rewrite exp(C) as a simple constant ---
                st.markdown("### Step 5 — Rewrite the constant simply")

                simplified_solutions = []
                # Change the C to "y_0" if you prefer that notation
                new_const = sp.Symbol("C")

                for sol in sols:
                    # Replace exp(C) with a new constant symbol
                    sol_simpl = sol.replace(sp.exp(C), new_const)
                    simplified_solutions.append(sp.simplify(sol_simpl))

                st.write("We use the fact that:")
                st.latex(r"e^C \;\text{is just another constant, so we rename it}\; C.")
                st.markdown("#### Final simplified form:")
                for sol in simplified_solutions:
                    st.latex(r"y = " + sp.latex(sol))
            else:
                st.info("Could not isolate y explicitly. The implicit solution is still correct.")
        except Exception:
            st.info("Could not isolate y explicitly. The implicit solution is still correct.")

st.caption("Note: This app is intentionally limited to separable first-order ODEs.")