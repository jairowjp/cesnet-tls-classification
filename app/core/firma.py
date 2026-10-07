"""
La firma de un flujo: el elemento central de la aplicación.

Dibuja los primeros paquetes de una conexión cifrada tal como los "ve" el modelo:
  * cada barra es un paquete;
  * sube (cobalto) si va del cliente al servidor y baja (ámbar) si vuelve del servidor;
  * su altura crece con el tamaño del paquete;
  * la separación entre barras refleja el tiempo transcurrido entre paquetes.

En modo animado, los paquetes aparecen uno a uno al ritmo real del flujo (acelerado y acotado)
y al final se revela la categoría que predijo el modelo. Si el sistema pide reducir el
movimiento (prefers-reduced-motion), todo se muestra de inmediato.

Seguridad (OWASP A03, XSS)
--------------------------
st.iframe ejecuta este HTML con acceso al mismo origen de la aplicación: NO es un aislamiento.
Por eso se aplican tres reglas, verificadas en tests/test_app_seguridad.py:
  1. El HTML lo genera solo este código; nunca se pasa HTML de usuarios ni de archivos subidos.
  2. Los datos del flujo llegan como números (validados en core/seguridad.py) dentro de un JSON
     en el que <, > y & se escapan, así que ningún texto puede cerrar la etiqueta <script>.
  3. Los textos (nombres de categorías) se insertan con textContent, nunca con innerHTML: el
     navegador los trata como texto aunque contengan código.
"""

import json

import numpy as np
import streamlit as st

from core.estilo import AMBAR, COBALTO, FUENTE, LINEA, TINTA, TINTA_SUAVE


def _json_seguro(obj) -> str:
    """JSON apto para incrustar en <script>: escapa <, > y & para impedir cerrar la etiqueta."""
    return json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def construir_html(
    tamanos, direcciones, tiempos_ms, veredicto: dict | None = None, animar: bool = True, clave: str = ""
) -> str:
    """Genera el HTML de la firma de un flujo (separado de su visualización para poder probarlo).

    tamanos, direcciones, tiempos_ms: 30 valores cada uno (relleno con dirección 0).
    veredicto: {"titulo": str, "detalle": str}; se muestra al final de la animación.
    clave: cambia cuando cambia el flujo, para que la animación se repita.
    """
    d = np.asarray(direcciones, dtype=float)
    reales = d != 0
    datos = {
        "tam": np.asarray(tamanos, dtype=float)[reales].round(1).tolist(),
        "dir": d[reales].astype(int).tolist(),
        # log(1 + ms): los tiempos entre paquetes van de 0 a miles de ms; en escala logarítmica
        # la separación sigue siendo legible sin que un solo silencio largo aplaste al resto
        "gap": np.log1p(np.asarray(tiempos_ms, dtype=float)[reales]).round(3).tolist(),
        "veredicto": veredicto,
        "animar": bool(animar),
        "clave": clave,
    }
    # Sin márgenes ni desplazamiento interno: el recuadro mide exactamente lo que su contenido
    html = f"""
<style>html, body {{ margin: 0; padding: 0; overflow: hidden; }}</style>
<div id="firma" style="font-family:{FUENTE}; color:{TINTA}; margin:0; padding-bottom:6px;">
  <!-- Las etiquetas son texto HTML, no parte del dibujo: así mantienen su tamaño en cualquier ancho -->
  <div style="color:{COBALTO}; font-size:13px; font-weight:600;">cliente → servidor</div>
  <!-- El dibujo ocupa todo el ancho y su alto se ajusta en proporción (viewBox + height:auto) -->
  <svg id="lienzo" viewBox="0 0 1000 200" style="width:100%; height:auto; display:block;" role="img"
       aria-label="Firma del flujo: paquetes del cliente hacia arriba y del servidor hacia abajo">
    <line x1="0" y1="100" x2="1000" y2="100" stroke="{LINEA}" stroke-width="1.5"/>
  </svg>
  <div style="color:{AMBAR}; font-size:13px; font-weight:600;">servidor → cliente</div>
  <div id="veredicto" style="opacity:0; transition:opacity .45s ease; margin-top:6px; min-height:44px;">
    <div id="v-titulo" style="font-size:1.35rem; font-weight:700; font-stretch:112%;"></div>
    <div id="v-detalle" style="color:{TINTA_SUAVE}; font-size:.95rem; margin-top:2px;"></div>
  </div>
</div>
<script>
(function () {{
  const D = {_json_seguro(datos)};
  const svg = document.getElementById("lienzo");
  const NS = "http://www.w3.org/2000/svg";
  const n = D.tam.length;
  if (!n) return;
  // Posiciones horizontales: cada paquete avanza según el tiempo transcurrido desde el anterior.
  // Toda barra recibe una separación mínima (1,3 veces su ancho) y el resto del espacio se reparte
  // en proporción al tiempo: así dos paquetes casi simultáneos nunca se superponen.
  const W = 980;
  const ancho = Math.max(3, Math.min(22, 0.55 * W / n));
  const minimo = ancho * 1.3;
  // +0,2: cada paquete recibe también una parte igual del espacio libre, para que el dibujo
  // ocupe todo el ancho aunque todos los tiempos sean cero
  const pasos = D.gap.map((g, i) => i === 0 ? 0 : 0.2 + g);
  const total = pasos.reduce((a, b) => a + b, 0) || 1;
  const libre = Math.max(0, W - ancho - minimo * (n - 1));
  let x = 10;
  const barras = D.tam.map((t, i) => {{
    if (i > 0) x += minimo + libre * (pasos[i] / total);
    const h = Math.max(3, Math.sqrt(Math.min(t, 1500) / 1500) * 84);   // raíz: realza paquetes pequeños
    const r = document.createElementNS(NS, "rect");
    r.setAttribute("x", x.toFixed(1));
    r.setAttribute("width", ancho.toFixed(1));
    r.setAttribute("height", h.toFixed(1));
    r.setAttribute("y", D.dir[i] > 0 ? (99 - h).toFixed(1) : "101");
    r.setAttribute("rx", "1.5");
    r.setAttribute("fill", D.dir[i] > 0 ? "{COBALTO}" : "{AMBAR}");
    r.style.transformBox = "fill-box";
    r.style.transformOrigin = D.dir[i] > 0 ? "bottom" : "top";
    const tip = document.createElementNS(NS, "title");
    const sentido = D.dir[i] > 0 ? "cliente → servidor" : "servidor → cliente";
    tip.textContent = "Paquete " + (i + 1) + ": " + Math.round(t) + " bytes, " + sentido;
    r.appendChild(tip);
    svg.appendChild(r);
    return r;
  }});
  function mostrarVeredicto() {{
    if (!D.veredicto) return;
    document.getElementById("v-titulo").textContent = D.veredicto.titulo || "";
    document.getElementById("v-detalle").textContent = D.veredicto.detalle || "";
    document.getElementById("veredicto").style.opacity = 1;
  }}
  const reducir = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!D.animar || reducir) {{ mostrarVeredicto(); return; }}
  // Animación: una sola secuencia orquestada, al ritmo (acelerado y acotado) del flujo real
  barras.forEach(b => {{ b.style.transform = "scaleY(0)"; b.style.transition = "transform .18s ease-out"; }});
  let t = 250;
  barras.forEach((b, i) => {{
    t += i === 0 ? 0 : Math.min(260, 60 + D.gap[i] * 28);
    setTimeout(() => {{ b.style.transform = "scaleY(1)"; }}, t);
  }});
  setTimeout(mostrarVeredicto, t + 450);
}})();
</script>
"""
    return html


def mostrar_firma(
    tamanos,
    direcciones,
    tiempos_ms,
    veredicto: dict | None = None,
    animar: bool = True,
    alto: int = 230,
    clave: str = "",
) -> None:
    """Dibuja (y opcionalmente anima) la firma de un flujo en la página."""
    # height="content": Streamlit mide el alto real del contenido, así el marco se ajusta al ancho
    # disponible (columna completa o media columna) sin cortar el dibujo ni dejar espacio vacío.
    st.iframe(construir_html(tamanos, direcciones, tiempos_ms, veredicto, animar, clave), height="content")
