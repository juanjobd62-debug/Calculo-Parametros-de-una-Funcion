import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO
# ============================================================================
st.set_page_config(page_title="Calculadora de Parámetros", layout="wide", page_icon="📐")

# Forzamos tema oscuro en Matplotlib coherente con Streamlit dark mode
mpl.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#cdd6f4",
    "axes.labelcolor": "#cdd6f4",
    "text.color": "#cdd6f4",
    "xtick.color": "#cdd6f4",
    "ytick.color": "#cdd6f4",
    "grid.color": "#45475a",
    "grid.alpha": 0.5,
    "font.family": "monospace",
})

x, a, b, c, d, e = sp.symbols("x a b c d e")
PARAM_SYMBOLS = [a, b, c, d, e]
LOCALS = {"x": x, "a": a, "b": b, "c": c, "d": d, "e": e}

# ============================================================================
# LÓGICA MATEMÁTICA (Adaptada de tu script original)
# ============================================================================
def _expr_indeterminada(expr):
    try:
        return expr.has(sp.oo, -sp.oo, sp.zoo, sp.nan, sp.sign, sp.Piecewise)
    except Exception:
        return True

def _eq_desde_spec(spec, conocidos):
    if spec["modo"] == "directa":
        return spec["eq"].subs(conocidos)
    expr = spec["expr"].subs(conocidos)
    try:
        val = sp.limit(expr, x, sp.oo)
    except Exception:
        return None
    if _expr_indeterminada(val):
        return None
    return sp.Eq(val, spec["valor"])

def resolver_sistema(especificaciones, func, params):
    conocidos = {}
    max_vueltas = len(params) + len(especificaciones) + 2
    for _ in range(max_vueltas):
        pendientes = [p for p in params if p not in conocidos]
        if not pendientes:
            break
        eqs_vuelta = []
        for spec in especificaciones:
            eq = _eq_desde_spec(spec, conocidos)
            if eq is not None:
                eqs_vuelta.append(eq)
        sol_vuelta = sp.solve(eqs_vuelta, pendientes, dict=True)
        sol_vuelta = [s for s in sol_vuelta if all(not v.free_symbols for v in s.values())]
        if not sol_vuelta:
            continue
        nuevo = sol_vuelta[0]
        progreso = False
        for k, v in nuevo.items():
            if k not in conocidos:
                conocidos[k] = sp.nsimplify(v)
                progreso = True
        if not progreso:
            break
            
    if not all(p in conocidos for p in params):
        return None
        
    soluciones = [conocidos]
    def es_real(sol):
        try:
            return all(sp.simplify(sp.im(val)) == 0 for val in sol.values())
        except Exception:
            return True
    reales = [s for s in soluciones if es_real(sol)]
    return reales if reales else soluciones

# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================
st.title("📐 Calculadora de Parámetros de Funciones")
st.caption("Traducción a Streamlit del cuaderno de Mathematica/Python original")

# --- PASO 1: Definir Función ---
st.header("1. Definición de la Función")
func_txt = st.text_input(
    "Introduce f(x) en términos de x, a, b, c, d, e:",
    value="a*x**2 + b*x + c",
    help="Sintaxis SymPy. Ej: a*exp(b*x)+c, (a*x+b)/(c*x+d)"
)

try:
    func = sp.sympify(func_txt, locals=LOCALS)
    params_detected = [p for p in PARAM_SYMBOLS if p in func.free_symbols]
    st.success(f"✅ Función válida. Parámetros detectados: {params_detected}")
except Exception as ex:
    st.error(f"❌ Error en la función: {ex}")
    st.stop()

# --- PASO 2: Condiciones ---
st.header("2. Condiciones del Sistema")
if "condiciones" not in st.session_state:
    st.session_state.condiciones = []

col_add, col_list = st.columns([1, 2])

with col_add:
    st.subheader("Añadir condición")
    tipo_cond = st.selectbox("Tipo de condición", [
        "Punto (x, y)", "Extremo relativo", "Punto de inflexión",
        "Tangencia con pendiente", "Asíntota vertical",
        "Asíntota horizontal", "Asíntota oblicua (límites)", "Integral definida"
    ])
    
    # Inputs dinámicos según tipo
    vals = {}
    c1, c2 = st.columns(2)
    with c1:
        if tipo_cond in ["Punto (x, y)", "Extremo relativo", "Punto de inflexión", 
                          "Tangencia con pendiente", "Asíntota vertical"]:
            vals["x0"] = st.text_input("x₀", value="0", key="inp_x0")
        if tipo_cond in ["Punto (x, y)"]:
            vals["y0"] = st.text_input("y₀", value="0", key="inp_y0")
        if tipo_cond == "Tangencia con pendiente":
            vals["m"] = st.text_input("Pendiente m", value="1", key="inp_m")
        if tipo_cond == "Asíntota horizontal":
            vals["y_as"] = st.text_input("y asíntota", value="0", key="inp_yas")
        if tipo_cond == "Asíntota oblicua (límites)":
            vals["m_ob"] = st.text_input("Pendiente m", value="1", key="inp_mob")
            vals["n_ob"] = st.text_input("Ordenada n", value="0", key="inp_nob")
        if tipo_cond == "Integral definida":
            vals["x1"] = st.text_input("Lím. inferior", value="0", key="inp_x1")
            vals["x2"] = st.text_input("Lím. superior", value="1", key="inp_x2")
            vals["val_int"] = st.text_input("Valor integral", value="1", key="inp_valint")
    with c2:
        conoce_y = False
        if tipo_cond in ["Extremo relativo", "Punto de inflexión", "Tangencia con pendiente"]:
            conoce_y = st.checkbox("¿Conoce la coordenada y?", key="chk_conoce_y")
            if conoce_y:
                vals["y0"] = st.text_input("y₀", value="0", key="inp_y0_extra")

    if st.button("➕ Añadir condición"):
        try:
            spec_list = []
            desc_list = []
            meta = {"tipo": tipo_cond}
            
            # Parseo seguro
            def p(k): return sp.sympify(vals.get(k, "0"), locals=LOCALS)
            
            if tipo_cond == "Punto (x, y)":
                x0, y0 = p("x0"), p("y0")
                spec_list = [{"modo": "directa", "eq": sp.Eq(func.subs(x, x0), y0)}]
                desc_list = [f"f({x0})={y0}"]
                meta.update({"x": x0, "y": y0})
                
            elif tipo_cond == "Extremo relativo":
                x0 = p("x0")
                df = sp.diff(func, x)
                spec_list = [{"modo": "directa", "eq": sp.Eq(df.subs(x, x0), 0)}]
                desc_list = [f"f'({x0})=0"]
                meta["x"] = x0
                if conoce_y:
                    y0 = p("y0")
                    spec_list.append({"modo": "directa", "eq": sp.Eq(func.subs(x, x0), y0)})
                    desc_list.append(f"f({x0})={y0}")
                    meta["y"] = y0
                    
            elif tipo_cond == "Punto de inflexión":
                x0 = p("x0")
                d2f = sp.diff(func, x, 2)
                spec_list = [{"modo": "directa", "eq": sp.Eq(d2f.subs(x, x0), 0)}]
                desc_list = [f"f''({x0})=0"]
                meta["x"] = x0
                if conoce_y:
                    y0 = p("y0")
                    spec_list.append({"modo": "directa", "eq": sp.Eq(func.subs(x, x0), y0)})
                    desc_list.append(f"f({x0})={y0}")
                    meta["y"] = y0
                    
            elif tipo_cond == "Tangencia con pendiente":
                x0, m = p("x0"), p("m")
                df = sp.diff(func, x)
                spec_list = [{"modo": "directa", "eq": sp.Eq(df.subs(x, x0), m)}]
                desc_list = [f"f'({x0})={m}"]
                meta.update({"x": x0, "m": m})
                if conoce_y:
                    y0 = p("y0")
                    spec_list.append({"modo": "directa", "eq": sp.Eq(func.subs(x, x0), y0)})
                    desc_list.append(f"f({x0})={y0}")
                    meta["y"] = y0
                    
            elif tipo_cond == "Asíntota vertical":
                x0 = p("x0")
                _, den = sp.fraction(sp.together(func))
                spec_list = [{"modo": "directa", "eq": sp.Eq(den.subs(x, x0), 0)}]
                desc_list = [f"den({x0})=0"]
                meta["x"] = x0
                
            elif tipo_cond == "Asíntota horizontal":
                y0 = p("y_as")
                spec_list = [{"modo": "limite", "expr": func, "valor": y0}]
                desc_list = [f"lim(x→∞)f(x)={y0}"]
                meta["y"] = y0
                
            elif tipo_cond == "Asíntota oblicua (límites)":
                m_val, n_val = p("m_ob"), p("n_ob")
                spec_list = [
                    {"modo": "limite", "expr": func/x, "valor": m_val},
                    {"modo": "limite", "expr": func - m_val*x, "valor": n_val}
                ]
                desc_list = [f"lim f(x)/x={m_val}", f"lim (f(x)-{m_val}x)={n_val}"]
                meta.update({"m": m_val, "n": n_val})
                
            elif tipo_cond == "Integral definida":
                x1, x2, vi = p("x1"), p("x2"), p("val_int")
                integral = sp.integrate(func, (x, x1, x2))
                spec_list = [{"modo": "directa", "eq": sp.Eq(integral, vi)}]
                desc_list = [f"∫[{x1},{x2}]f(x)dx={vi}"]
                meta.update({"x1": x1, "x2": x2, "valor": vi})

            st.session_state.condiciones.append({
                "specs": spec_list, "descs": desc_list, "meta": meta
            })
            st.rerun()
        except Exception as ex:
            st.error(f"Error al añadir condición: {ex}")

with col_list:
    st.subheader(f"Condiciones actuales ({len(st.session_state.condiciones)})")
    if st.session_state.condiciones:
        for i, cond in enumerate(st.session_state.condiciones):
            desc = "; ".join(cond["descs"])
            cols_item = st.columns([4, 1])
            cols_item[0].markdown(f"**{i+1}.** [{cond['meta']['tipo']}] `{desc}`")
            if cols_item[1].button("🗑️", key=f"del_{i}", help="Eliminar condición"):
                st.session_state.condiciones.pop(i)
                st.rerun()
    else:
        st.info("No hay condiciones añadidas todavía.")

# --- PASO 3: Resolver y Graficar ---
st.header("3. Resolución y Gráfica")
if st.button("🚀 Resolver Sistema", type="primary"):
    if not st.session_state.condiciones:
        st.warning("Añade al menos una condición antes de resolver.")
    else:
        specs = [s for c in st.session_state.condiciones for s in c["specs"]]
        metas = [c["meta"] for c in st.session_state.condiciones]
        
        with st.spinner("Resolviendo sistema simbólico..."):
            soluciones = resolver_sistema(specs, func, params_detected)
        
        if soluciones is None or len(soluciones) == 0:
            st.error("No se encontró solución. Revisa las condiciones o la función.")
        else:
            st.success(f"✅ Se encontraron {len(soluciones)} solución(es)")
            
            # Tabla de resultados
            for idx, sol in enumerate(soluciones):
                st.subheader(f"Solución {idx+1}")
                tabla_data = {
                    "Parámetro": [str(p) for p in params_detected],
                    "Valor Exacto": [str(sp.nsimplify(sol.get(p, sp.Symbol("?")))) for p in params_detected],
                    "Valor Aprox.": [f"{float(sol.get(p, 0)):.6g}" if sol.get(p, sp.Symbol("?")).free_symbols == set() else "-" for p in params_detected]
                }
                st.table(tabla_data)
                
                func_final = sp.simplify(func.subs(sol))
                st.latex(f"f(x) = {sp.latex(func_final)}")
                
                # Gráfica
                fig, ax = plt.subplots(figsize=(10, 6))
                f_lamb = sp.lambdify(x, func_final, modules=["numpy"])
                
                # Rango automático basado en condiciones
                xs_rel = []
                for m in metas:
                    for k in ("x", "x1", "x2"):
                        if k in m and hasattr(m[k], 'is_real') and m[k].is_real:
                            try: xs_rel.append(float(m[k]))
                            except: pass
                xmin = min(xs_rel) - 5 if xs_rel else -10
                xmax = max(xs_rel) + 5 if xs_rel else 10
                
                xs_plot = np.linspace(xmin, xmax, 2000)
                try:
                    ys_plot = f_lamb(xs_plot)
                    ys_plot = np.asarray(ys_plot, dtype=float)
                    ys_plot[np.abs(ys_plot) > 1e4] = np.nan
                    ax.plot(xs_plot, ys_plot, color="#89b4fa", linewidth=2, label="f(x)")
                except Exception as plot_ex:
                    st.warning(f"No se pudo graficar: {plot_ex}")
                
                # Marcar puntos y asíntotas
                for m in metas:
                    if m["tipo"] in ["Punto (x, y)", "Extremo relativo", "Punto de inflexión", "Tangencia con pendiente"]:
                        if "x" in m:
                            try:
                                px = float(m["x"])
                                py = float(m.get("y", func_final.subs(x, m["x"])))
                                ax.plot(px, py, "o", color="#f9e2af", markersize=8, zorder=5)
                            except: pass
                    if m["tipo"] == "Asíntota vertical" and "x" in m:
                        ax.axvline(float(m["x"]), color="#f38ba8", ls="--", alpha=0.8, label="As. Vertical")
                    if m["tipo"] == "Asíntota horizontal" and "y" in m:
                        ax.axhline(float(m["y"]), color="#a6e3a1", ls="--", alpha=0.8, label="As. Horizontal")
                    if m["tipo"] == "Asíntota oblicua (límites)":
                        mm, nn = float(m["m"]), float(m["n"])
                        ax.plot(xs_plot, mm*xs_plot+nn, color="#a6e3a1", ls="--", alpha=0.8, label="As. Oblicua")
                
                ax.axhline(0, color="#585b70", lw=0.8)
                ax.axvline(0, color="#585b70", lw=0.8)
                ax.grid(True, alpha=0.3)
                ax.legend(facecolor="#0e1117", edgecolor="#45475a")
                st.pyplot(fig)
                plt.close(fig)
