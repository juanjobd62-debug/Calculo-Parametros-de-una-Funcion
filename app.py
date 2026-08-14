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
# Símbolos auxiliares para asíntotas
m_obl, n_obl = sp.symbols("m_obl n_obl")
ah_lim, av_x = sp.symbols("ah_lim av_x") # ah_lim: asíntota horizontal límite, av_x: valor x de asíntota vertical
ALL_PARAM_SYMBOLS = PARAM_SYMBOLS + [m_obl, n_obl, ah_lim, av_x]
LOCALS = {
    "x": x, "a": a, "b": b, "c": c, "d": d, "e": e,
    "m_obl": m_obl, "n_obl": n_obl, "ah_lim": ah_lim, "av_x": av_x
}


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

    # Manejo de límites para asíntotas oblicuas
    if spec["modo"] in ["limite_oblicuo_m", "limite_oblicuo_n", "limite_horiz"]:
        expr = spec["expr"].subs(conocidos)
        try:
            val = sp.limit(expr, x, sp.oo)
        except Exception:
            return None

        if _expr_indeterminada(val):
            return None

        # El valor límite debe ser igual al valor objetivo (puede ser un símbolo o un número)
        return sp.Eq(val, spec["valor_objetivo"])

    # Manejo de límites para asíntotas verticales (comportamiento en x -> av_x)
    if spec["modo"] == "limite_vert":
        # Esta condición se maneja de forma distinta, ya que implica que el denominador tiende a 0
        # Se manejará en la creación de la condición directamente como antes, pero si se quiere
        # hacerlo con un límite, sería más complejo y no es tan directo.
        # Mantenemos la lógica de la fracción para AV por ahora.
        # Si se quisiera hacer con límites, por ejemplo lim_{x->x0} |f(x)| = oo, es más complejo.
        # La implementación directa con denominador=0 es más robusta para AV.
        # Por lo tanto, no se procesa aquí como límite genérico.
        pass 

    # Caso general para otros límites
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
            # Solo procesar ecuaciones que no dependan de variables no resueltas aún
            # (Esto es un manejo básico, SymPy solve hará el trabajo pesado)
            eq = _eq_desde_spec(spec, conocidos)
            if eq is not None:
                eqs_vuelta.append(eq)

        if not eqs_vuelta:
             # Si no hay ecuaciones en esta vuelta, intentar resolver las pendientes restantes con todas las ecuaciones acumuladas
            sol_vuelta = sp.solve(eqs_vuelta, pendientes, dict=True)
        else:
            sol_vuelta = sp.solve(eqs_vuelta, pendientes, dict=True)

        sol_vuelta = [
            s for s in sol_vuelta
            if all(not v.free_symbols for v in s.values()) # Asegura que todas las variables de la solución estén resueltas
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
            # Verifica si la parte imaginaria de cada valor es 0
            return all(sp.simplify(sp.im(val)).is_zero for val in sol_dict.values())
        except Exception:
            # Si falla la simplificación, asumimos verdadero
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
        "Asíntota vertical (posición x)",
        "Asíntota vertical (posición x, flexible)",
        "Asíntota horizontal (límite y)",
        "Asíntota horizontal (límite y, flexible)",
        "Asíntota oblicua (m y n flexibles)",
        "Integral definida"
    ])

    vals = {}
    c1, c2 = st.columns(2)

    with c1:
        if tipo_cond in [
            "Punto (x, y)", "Extremo relativo", "Punto de inflexión",
            "Tangencia con pendiente", "Asíntota vertical (posición x)"
        ]:
            vals["x0"] = st.text_input("x₀", value="0", key="inp_x0")

        if tipo_cond == "Punto (x, y)":
            vals["y0"] = st.text_input("y₀", value="0", key="inp_y0")

        if tipo_cond == "Tangencia con pendiente":
            vals["m"] = st.text_input("Pendiente m", value="1", key="inp_m")

        # Nueva opción para asíntota horizontal flexible
        if tipo_cond in ["Asíntota horizontal (límite y)", "Asíntota horizontal (límite y, flexible)"]:
            if tipo_cond == "Asíntota horizontal (límite y)":
                 vals["y_as"] = st.text_input("y asíntota (dato)", value="0", key="inp_yas_fija")
            else: # Flexible
                vals["ah_tipo"] = st.selectbox("Tipo de límite y", ["Dato (número)", "Incógnita"], key="inp_ah_tipo")
                if vals["ah_tipo"] == "Dato (número)":
                    vals["ah_valor"] = st.text_input("Valor de límite y", value="0", key="inp_ah_valor")

        # Nueva opción para asíntota vertical flexible
        if tipo_cond in ["Asíntota vertical (posición x)", "Asíntota vertical (posición x, flexible)"]:
            if tipo_cond == "Asíntota vertical (posición x)":
                 vals["x_as"] = st.text_input("x asíntota (dato)", value="0", key="inp_xas_fija")
            else: # Flexible
                vals["av_tipo"] = st.selectbox("Tipo de posición x", ["Dato (número)", "Incógnita"], key="inp_av_tipo")
                if vals["av_tipo"] == "Dato (número)":
                    vals["av_valor"] = st.text_input("Valor de posición x", value="0", key="inp_av_valor")


        # Nueva opción para asíntota oblicua flexible
        if tipo_cond == "Asíntota oblicua (m y n flexibles)":
            # Configurar m
            vals["m_tipo"] = st.selectbox("Tipo de m", ["Dato (número)", "Incógnita"], key="inp_m_tipo")
            if vals["m_tipo"] == "Dato (número)":
                vals["m_valor"] = st.text_input("Valor de m", value="1", key="inp_m_valor")

            # Configurar n
            vals["n_tipo"] = st.selectbox("Tipo de n", ["Dato (número)", "Incógnita"], key="inp_n_tipo")
            if vals["n_tipo"] == "Dato (número)":
                vals["n_valor"] = st.text_input("Valor de n", value="0", key="inp_n_valor")


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

            elif tipo_cond == "Asíntota vertical (posición x)":
                x0 = p("x_as") # Usamos x_as en lugar de x0
                _, den = sp.fraction(sp.together(func))
                spec_list = [{"modo": "directa", "eq": sp.Eq(den.subs(x, x0), 0)}]
                desc_list = [f"den({x0})=0"]
                meta["x"] = x0 # Se guarda como 'x' para visualización

            # Nueva condición: Asíntota vertical flexible
            elif tipo_cond == "Asíntota vertical (posición x, flexible)":
                if vals["av_tipo"] == "Dato (número)":
                    x_val = p("av_valor")
                    x_symbol_or_value = x_val
                else: # Incógnita -> Usamos el símbolo av_x
                    x_symbol_or_value = av_x
                
                # La condición sigue siendo que el denominador es 0 en x = x_symbol_or_value
                _, den = sp.fraction(sp.together(func))
                spec_list = [{"modo": "directa", "eq": sp.Eq(den.subs(x, x_symbol_or_value), 0)}]
                desc_list = [f"den(x={x_symbol_or_value})=0"]
                meta["x_simbolo"] = str(x_symbol_or_value) # Guardamos el símbolo o valor usado
                meta["av_tipo"] = vals["av_tipo"]


            elif tipo_cond == "Asíntota horizontal (límite y)":
                y0 = p("y_as") # Usamos y_as en lugar de y0
                spec_list = [{"modo": "limite_horiz", "expr": func, "valor_objetivo": y0}]
                desc_list = [f"lim(x→∞)f(x)={y0}"]
                meta["y"] = y0 # Se guarda como 'y' para visualización

            # Nueva condición: Asíntota horizontal flexible
            elif tipo_cond == "Asíntota horizontal (límite y, flexible)":
                if vals["ah_tipo"] == "Dato (número)":
                    y_val = p("ah_valor")
                    y_symbol_or_value = y_val
                else: # Incógnita -> Usamos el símbolo ah_lim
                    y_symbol_or_value = ah_lim
                
                # La ecuación es lim f(x) = y_symbol_or_value
                spec_list = [{"modo": "limite_horiz", "expr": func, "valor_objetivo": y_symbol_or_value}]
                desc_list = [f"lim(x→∞)f(x)={y_symbol_or_value}"]
                meta["y_simbolo"] = str(y_symbol_or_value) # Guardamos el símbolo o valor usado
                meta["ah_tipo"] = vals["ah_tipo"]


            # Asintota oblicua (mantiene lógica anterior)
            elif tipo_cond == "Asíntota oblicua (m y n flexibles)":
                # Procesar m
                if vals["m_tipo"] == "Dato (número)":
                    m_val = p("m_valor")
                    m_symbol_or_value = m_val
                else: # Incógnita -> Usamos el símbolo m_obl
                    m_symbol_or_value = m_obl
                
                # Procesar n
                if vals["n_tipo"] == "Dato (número)":
                    n_val = p("n_valor")
                    n_symbol_or_value = n_val
                else: # Incógnita -> Usamos el símbolo n_obl
                    n_symbol_or_value = n_obl

                # Crear las ecuaciones de límite
                # lim (f(x) / x) = m
                spec_list.append({
                    "modo": "limite_oblicuo_m",
                    "expr": func / x,
                    "valor_objetivo": m_symbol_or_value
                })
                desc_list.append(f"lim f(x)/x = m ({m_symbol_or_value})")

                # lim (f(x) - m*x) = n
                expr_n = func - m_symbol_or_value * x
                spec_list.append({
                    "modo": "limite_oblicuo_n",
                    "expr": expr_n,
                    "valor_objetivo": n_symbol_or_value
                })
                desc_list.append(f"lim (f(x) - m*x) = n ({n_symbol_or_value})")
                
                meta.update({
                    "m_tipo": vals["m_tipo"],
                    "m_valor_o_simbolo": str(m_symbol_or_value),
                    "n_tipo": vals["n_tipo"],
                    "n_valor_o_simbolo": str(n_symbol_or_value)
                })

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
        
        # Detectar parámetros necesarios para resolver, incluyendo los auxiliares
        params_finales = set(params_detected) # Copia los parámetros base (a, b, c, d, e)
        for spec in specs:
            # Chequear si la ecuación de límite involucra símbolos auxiliares
            if spec["modo"] in ["limite_oblicuo_m", "limite_oblicuo_n", "limite_horiz"]:
                if spec["valor_objetivo"] == m_obl:
                    params_finales.add(m_obl)
                if spec["valor_objetivo"] == n_obl:
                    params_finales.add(n_obl)
                if spec["valor_objetivo"] == ah_lim:
                    params_finales.add(ah_lim)
            
            # Chequear si la ecuación directa (AV) involucra símbolos auxiliares
            if spec["modo"] == "directa":
                # Esta es la condición AV flexible: Eq(den(subs(x, av_x)), 0)
                # Buscamos si av_x está en la ecuación
                if av_x in spec["eq"].free_symbols:
                     params_finales.add(av_x)
        
        params_finales = list(params_finales) # Convertir a lista para resolver

        metas = [c["meta"] for c in st.session_state.condiciones]

        with st.spinner("Resolviendo sistema simbólico..."):
            soluciones = resolver_sistema(specs, func, params_finales)

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

        # Determinar todos los parámetros que deben mostrarse (base + auxiliares si están en la solución)
        all_params_in_sol = set(params_detected)
        for p in sol.keys():
            if p in ALL_PARAM_SYMBOLS: # Solo mostrar símbolos reconocidos
                all_params_in_sol.add(p)
        
        # Filtrar solo los que aparecen en la solución
        params_to_show = [p for p in ALL_PARAM_SYMBOLS if p in all_params_in_sol]

        # Tabla de resultados
        tabla_data = {
            "Parámetro": [str(p) for p in params_to_show],
            "Valor Exacto": [
                str(sp.nsimplify(sol.get(p, sp.Symbol("?"))))
                for p in params_to_show
            ],
            "Valor Aprox.": [
                f"{float(sol[p]):.6g}"
                if p in sol and sol[p].free_symbols == set() and hasattr(sol[p], '__float__')
                else "-"
                for p in params_to_show
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
            # También recoger x_simbolo si es incógnita resuelta para AV flexible
            if m["tipo"] == "Asíntota vertical (posición x, flexible)" and m["av_tipo"] == "Incógnita":
                if av_x in sol:
                    try:
                        xs_rel.append(float(sol[av_x]))
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
            # Limitar valores extremos para graficar
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
        for m_meta in metas:
            if m_meta["tipo"] in [
                "Punto (x, y)", "Extremo relativo",
                "Punto de inflexión", "Tangencia con pendiente"
            ] and "x" in m_meta:
                try:
                    px = float(m_meta["x"])
                    py = float(m_meta.get("y", func_final.subs(x, m_meta["x"])))
                    fig.add_trace(go.Scatter(
                        x=[px], y=[py], mode="markers",
                        marker=dict(size=12, color="#f9e2af", symbol="circle"),
                        name=f"{m_meta['tipo']} ({px:.2g}, {py:.2g})"
                    ))
                except Exception:
                    pass

            # Asintota vertical (fija)
            if m_meta["tipo"] == "Asíntota vertical (posición x)" and "x" in m_meta:
                xv = float(m_meta["x"])
                fig.add_vline(
                    x=xv, line_dash="dash", line_color="#f38ba8",
                    annotation_text=f"x={xv:.2g}", annotation_position="top right"
                )
            
            # Asintota vertical (flexible)
            if m_meta["tipo"] == "Asíntota vertical (posición x, flexible)":
                av_tipo = m_meta.get("av_tipo")
                av_orig_str = m_meta.get("x_simbolo")
                if av_tipo == "Incógnita":
                    # Buscar en la solución el valor de av_x
                    if av_x in sol:
                        xv = float(sol[av_x])
                        fig.add_vline(
                            x=xv, line_dash="dash", line_color="#f38ba8",
                            annotation_text=f"x={xv:.2g}", annotation_position="top right"
                        )
                else: # Dato (número)
                    xv = float(sp.sympify(av_orig_str))
                    fig.add_vline(
                        x=xv, line_dash="dash", line_color="#f38ba8",
                        annotation_text=f"x={xv:.2g}", annotation_position="top right"
                    )


            # Asintota horizontal (fija)
            if m_meta["tipo"] == "Asíntota horizontal (límite y)" and "y" in m_meta:
                yh = float(m_meta["y"])
                fig.add_hline(
                    y=yh, line_dash="dash", line_color="#a6e3a1",
                    annotation_text=f"y={yh:.2g}", annotation_position="bottom right"
                )
            
            # Asintota horizontal (flexible)
            if m_meta["tipo"] == "Asíntota horizontal (límite y, flexible)":
                ah_tipo = m_meta.get("ah_tipo")
                ah_orig_str = m_meta.get("y_simbolo")
                if ah_tipo == "Incógnita":
                    # Buscar en la solución el valor de ah_lim
                    if ah_lim in sol:
                        yh = float(sol[ah_lim])
                        fig.add_hline(
                            y=yh, line_dash="dash", line_color="#a6e3a1",
                            annotation_text=f"y={yh:.2g}", annotation_position="bottom right"
                        )
                else: # Dato (número)
                    yh = float(sp.sympify(ah_orig_str))
                    fig.add_hline(
                        y=yh, line_dash="dash", line_color="#a6e3a1",
                        annotation_text=f"y={yh:.2g}", annotation_position="bottom right"
                    )


            # Asintota oblicua (mantiene lógica anterior)
            if m_meta["tipo"] == "Asíntota oblicua (m y n flexibles)":
                # Obtenemos los valores de m y n de la solución final, basados en los metadatos originales
                m_tipo = m_meta.get("m_tipo")
                m_orig_str = m_meta.get("m_valor_o_simbolo")
                n_tipo = m_meta.get("n_tipo")
                n_orig_str = m_meta.get("n_valor_o_simbolo")

                # Resolver el valor real de m
                if m_tipo == "Incógnita":
                    # Buscar en la solución el valor de m_obl
                    if m_obl in sol:
                        mm = float(sol[m_obl])
                    else:
                        continue # No se ha resuelto esta variable aún
                else: # Dato (número)
                    mm = float(sp.sympify(m_orig_str))

                # Resolver el valor real de n
                if n_tipo == "Incógnita":
                     # Buscar en la solución el valor de n_obl
                    if n_obl in sol:
                        nn = float(sol[n_obl])
                    else:
                        continue # No se ha resuelto esta variable aún
                else: # Dato (número)
                    nn = float(sp.sympify(n_orig_str))

                # Graficar la recta y = mx + n
                yy_ob = mm * xs_plot + nn
                fig.add_trace(go.Scatter(
                    x=xs_plot, y=yy_ob, mode="lines",
                    name=f"As. oblicua y={mm:.2g}x+{nn:.2g}",
                    line=dict(color="#a6e3a1", width=1.5, dash="dash")
                ))

            if m_meta["tipo"] == "Tangencia con pendiente" and "x" in m_meta and "m" in m_meta:
                try:
                    x0_t = float(m_meta["x"])
                    m_t = float(m_meta["m"])
                    y0_t = float(m_meta.get("y", func_final.subs(x, m_meta["x"])))
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
