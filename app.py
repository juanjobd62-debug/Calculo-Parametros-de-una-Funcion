import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go

# ============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO
# ============================================================================
st.set_page_config(page_title="Calculadora de Parámetros", layout="wide", page_icon="📐")

x, a, b, c, d, e = sp.symbols("x a b c d e")
PARAM_SYMBOLS = [a, b, c, d, e]
LOCALS = {"x": x, "a": a, "b": b, "c": c, "d": d, "e": e}


# ============================================================================
# LÓGICA MATEMÁTICA
# ============================================================================
def _expr_indeterminada(expr):
    """True si expr todavía contiene infinitos/signos/Piecewise sin resolver."""
    try:
        return expr.has(sp.oo, -sp.oo, sp.zoo, sp.nan, sp.sign, sp.Piecewise)
    except Exception:
        return True


def _eq_desde_spec(spec, conocidos):
    """Construye la ecuación de una spec, sustituyendo los valores conocidos."""
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
    """Resuelve el sistema de forma progresiva (límites diferidos)."""
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
        sol_vuelta = [
            s for s in sol_vuelta
            if all(not v.free_symbols for v in s.values())
        ]

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

    def es_real(sol_dict):
        """Comprueba si todos los valores de la solución son reales."""
        try:
            return all(sp.simplify(sp.im(val)) == 0 for val in sol_dict.values())
        except Exception:
            return True

    reales = [s for s in soluciones if es_real(s)]
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
        "Punto (x, y)",
        "Extremo relativo",
        "Punto de inflexión",
        "Tangencia con pendiente",
        "Asíntota vertical",
        "Asíntota horizontal",
        "Asíntota oblicua (límites)",
        "Integral definida"
    ])

    vals = {}
    c1, c2 = st.columns(2)

    with c1:
        if tipo_cond in [
            "Punto (x, y)", "Extremo relativo", "Punto de inflexión",
            "Tangencia con pendiente", "Asíntota vertical"
        ]:
            vals["x0"] = st.text_input("x₀", value="0", key="inp_x0")

        if tipo_cond == "Punto (x, y)":
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

            def p(k):
                return sp.sympify(vals.get(k, "0"), locals=LOCALS)

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
                    {"modo": "limite", "expr": func / x, "valor": m_val},
                    {"modo": "limite", "expr": func - m_val * x, "valor": n_val},
                ]
                desc_list = [
                    f"lim f(x)/x={m_val}",
                    f"lim (f(x)-{m_val}x)={n_val}"
                ]
                meta.update({"m": m_val, "n": n_val})

            elif tipo_cond == "Integral definida":
                x1, x2, vi = p("x1"), p("x2"), p("val_int")
                integral = sp.integrate(func, (x, x1, x2))
                spec_list = [{"modo": "directa", "eq": sp.Eq(integral, vi)}]
                desc_list = [f"∫[{x1},{x2}]f(x)dx={vi}"]
                meta.update({"x1": x1, "x2": x2, "valor": vi})

            st.session_state.condiciones.append({
                "specs": spec_list,
                "descs": desc_list,
                "meta": meta
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

# --- PASO 3: Resolver y Graficar (INTERACTIVO CON PLOTLY) ---
st.header("3. Resolución y Gráfica")

# Inicializar estado de soluciones si no existe
if "soluciones_resultado" not in st.session_state:
    st.session_state.soluciones_resultado = None
if "metas_resultado" not in st.session_state:
    st.session_state.metas_resultado = None

# Botones de acción global
col_resolve, col_reset = st.columns([1, 4])
with col_resolve:
    resolve_clicked = st.button("🚀 Resolver Sistema", type="primary")
with col_reset:
    reset_all_clicked = st.button("🗑️ Resetear todo", type="secondary")

# Lógica de reseteo total
if reset_all_clicked:
    keys_to_delete = [k for k in st.session_state.keys()
                      if k.startswith(("xmin_", "xmax_", "ymin_", "ymax_", "reset_"))]
    for k in keys_to_delete:
        del st.session_state[k]
    st.session_state.condiciones = []
    st.session_state.soluciones_resultado = None
    st.session_state.metas_resultado = None
    st.rerun()

# Lógica de resolución
if resolve_clicked:
    if not st.session_state.condiciones:
        st.warning("Añade al menos una condición antes de resolver.")
        st.session_state.soluciones_resultado = None
    else:
        specs = [s for c in st.session_state.condiciones for s in c["specs"]]
        metas = [c["meta"] for c in st.session_state.condiciones]

        with st.spinner("Resolviendo sistema simbólico..."):
            soluciones = resolver_sistema(specs, func, params_detected)

        if soluciones is None or len(soluciones) == 0:
            st.error("No se encontró solución. Revisa las condiciones o la función.")
            st.session_state.soluciones_resultado = None
        else:
            st.session_state.soluciones_resultado = soluciones
            st.session_state.metas_resultado = metas

# === MOSTRAR RESULTADOS (fuera del botón para persistir entre reruns) ===
if st.session_state.soluciones_resultado is not None:
    soluciones = st.session_state.soluciones_resultado
    metas = st.session_state.metas_resultado

    st.success(f"✅ Se encontraron {len(soluciones)} solución(es)")

    for idx, sol in enumerate(soluciones):
        st.subheader(f"Solución {idx + 1}")

        # Tabla de resultados
        tabla_data = {
            "Parámetro": [str(p) for p in params_detected],
            "Valor Exacto": [
                str(sp.nsimplify(sol.get(p, sp.Symbol("?"))))
                for p in params_detected
            ],
            "Valor Aprox.": [
                f"{float(sol[p]):.6g}"
                if p in sol and sol[p].free_symbols == set()
                else "-"
                for p in params_detected
            ]
        }
        st.table(tabla_data)

        func_final = sp.simplify(func.subs(sol))
        st.latex(f"f(x) = {sp.latex(func_final)}")

        # === RANGO INTERACTIVO ===
        xs_rel = []
        for m in metas:
            for k in ("x", "x1", "x2"):
                if k in m and hasattr(m[k], "is_real") and m[k].is_real:
                    try:
                        xs_rel.append(float(m[k]))
                    except Exception:
                        pass

        auto_xmin = min(xs_rel) - 5 if xs_rel else -10.0
        auto_xmax = max(xs_rel) + 5 if xs_rel else 10.0

        col_r1, col_r2, col_r3, col_r4, col_btn = st.columns([1, 1, 1, 1, 1])
        with col_r1:
            xmin_user = st.number_input(
                "X mín", value=float(auto_xmin), format="%.2f",
                key=f"xmin_{idx}"
            )
        with col_r2:
            xmax_user = st.number_input(
                "X máx", value=float(auto_xmax), format="%.2f",
                key=f"xmax_{idx}"
            )
        with col_r3:
            ymin_user = st.number_input(
                "Y mín (vacío=auto)", value=None, format="%.2f",
                key=f"ymin_{idx}"
            )
        with col_r4:
            ymax_user = st.number_input(
                "Y máx (vacío=auto)", value=None, format="%.2f",
                key=f"ymax_{idx}"
            )
        with col_btn:
            reset_range = st.button("🔄 Auto", key=f"reset_{idx}")

        if reset_range:
            for k in [f"xmin_{idx}", f"xmax_{idx}", f"ymin_{idx}", f"ymax_{idx}"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

        # Validar rango X
        if xmin_user >= xmax_user:
            st.error("X mín debe ser menor que X máx")
            continue

        # === GENERAR GRÁFICA PLOTLY ===
        f_lamb = sp.lambdify(x, func_final, modules=["numpy"])
        xs_plot = np.linspace(xmin_user, xmax_user, 3000)

        try:
            ys_plot = f_lamb(xs_plot)
            ys_plot = np.asarray(ys_plot, dtype=float)
            ys_plot[np.abs(ys_plot) > 1e4] = np.nan
        except Exception as plot_ex:
            st.warning(f"No se pudo evaluar la función: {plot_ex}")
            continue

        fig = go.Figure()

        # Curva principal
        fig.add_trace(go.Scatter(
            x=xs_plot, y=ys_plot, mode="lines",
            name="f(x)", line=dict(color="#89b4fa", width=2.5)
        ))

        # Anotaciones sobre la gráfica
        for m in metas:
            if m["tipo"] in [
                "Punto (x, y)", "Extremo relativo",
                "Punto de inflexión", "Tangencia con pendiente"
            ] and "x" in m:
                try:
                    px = float(m["x"])
                    py = float(m.get("y", func_final.subs(x, m["x"])))
                    fig.add_trace(go.Scatter(
                        x=[px], y=[py], mode="markers",
                        marker=dict(size=12, color="#f9e2af", symbol="circle"),
                        name=f"{m['tipo']} ({px:.2g}, {py:.2g})"
                    ))
                except Exception:
                    pass

            if m["tipo"] == "Asíntota vertical" and "x" in m:
                xv = float(m["x"])
                fig.add_vline(
                    x=xv, line_dash="dash", line_color="#f38ba8",
                    annotation_text=f"x={xv:.2g}", annotation_position="top right"
                )

            if m["tipo"] == "Asíntota horizontal" and "y" in m:
                yh = float(m["y"])
                fig.add_hline(
                    y=yh, line_dash="dash", line_color="#a6e3a1",
                    annotation_text=f"y={yh:.2g}", annotation_position="bottom right"
                )

            if m["tipo"] == "Asíntota oblicua (límites)":
                mm, nn = float(m["m"]), float(m["n"])
                yy_ob = mm * xs_plot + nn
                fig.add_trace(go.Scatter(
                    x=xs_plot, y=yy_ob, mode="lines",
                    name=f"As. oblicua y={mm:.2g}x+{nn:.2g}",
                    line=dict(color="#a6e3a1", width=1.5, dash="dash")
                ))

            if m["tipo"] == "Tangencia con pendiente" and "x" in m and "m" in m:
                try:
                    x0_t = float(m["x"])
                    m_t = float(m["m"])
                    y0_t = float(m.get("y", func_final.subs(x, m["x"])))
                    yy_tan = m_t * (xs_plot - x0_t) + y0_t
                    fig.add_trace(go.Scatter(
                        x=xs_plot, y=yy_tan, mode="lines",
                        name=f"Tangente en x={x0_t:.2g}",
                        line=dict(color="#fab387", width=1.5, dash="dot")
                    ))
                except Exception:
                    pass

        # Estilo oscuro coherente
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#cdd6f4", family="monospace"),
            xaxis=dict(
                title="x", gridcolor="#45475a", zerolinecolor="#585b70",
                range=[xmin_user, xmax_user]
            ),
            yaxis=dict(
                title="f(x)", gridcolor="#45475a", zerolinecolor="#585b70"
            ),
            legend=dict(
                bgcolor="rgba(14,17,23,0.8)",
                bordercolor="#45475a", borderwidth=1
            ),
            margin=dict(l=60, r=30, t=40, b=60),
            height=550
        )

        # Aplicar rango Y personalizado
        y_range = [None, None]
        if ymin_user is not None:
            y_range[0] = ymin_user
        if ymax_user is not None:
            y_range[1] = ymax_user
        if y_range != [None, None]:
            fig.update_yaxes(range=y_range)

        st.plotly_chart(fig, use_container_width=True)
