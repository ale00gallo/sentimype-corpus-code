import os
"""E1 · Gemini 2.5 Flash sin entrenar (zero-shot) como rival fuerte, en cuatro configuraciones.

    E1-a  Test-A  solo texto            (526 llamadas)
    E1-b  Test-B  solo texto            (346)
    E1-c  Test-B  texto + primera foto  (346)
    E1-d  Test-B  solo foto             (346)

Uso (parada en entrenamiento v4, después de ordenar_insumos.py):
    PY=python
    $PY python run_e1.py iniciar                 # crea la carpeta de E1 y el prompt borrador
    $PY python run_e1.py validar                 # prueba el prompt con 50 reseñas de VALIDACIÓN (nunca Test-A ni Test-B)
    $PY python run_e1.py fijar                   # congela el prompt: desde aquí no se modifica
    $PY python run_e1.py correr --config todas   # las cuatro configuraciones; se puede interrumpir y repetir: retoma donde quedó
    $PY python run_e1.py evaluar --precio-entrada X --precio-salida Y --fuente-precio URL --fecha-precio AAAA-MM-DD

La clave se lee de la variable GEMINI_API_KEY o de .env. NUNCA se imprime ni se guarda.
Reglas del protocolo experimental que cumple: un solo prompt en español con las definiciones de la guía, salida en JSON, temperatura 0,
mismo prompt en las cuatro configuraciones (solo cambia lo que se adjunta), prompt probado solo con validación y luego fijado,
un reintento si el JSON no es válido (si vuelve a fallar cuenta como error y se cuenta), tokens por llamada, identificador exacto del modelo y fecha.
"""
import argparse, base64, csv, datetime as dt, hashlib, io, json, math, os, re, socket, stat, sys, time
import urllib.error, urllib.request
import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join("sentimype_experimentos", "exportar_resultados", "datos")
IMAGENES = os.path.join("sentimype_experimentos", "solo_en_la_laptop", "imagenes_test_b")
CLASES = ("negativo", "neutro", "positivo")
CONFIGS = {"E1-a": ("A", "texto"), "E1-b": ("B", "texto"), "E1-c": ("B", "texto+foto"), "E1-d": ("B", "foto")}
SEMILLA_BOOT, SEMILLA_VALIDACION, B = 14112028, 2026, 2000
MODELO_DEFECTO = "gemini-3.5-flash-lite"

PROMPT_BORRADOR = """Eres un anotador de reseñas de negocios peruanos (restaurantes, cafeterías, bodegas y otras MYPE). Clasifica el sentimiento GENERAL del cliente hacia el negocio.

Definiciones:
- negativo: el cliente expresa una mala experiencia o un rechazo. Incluye el rechazo explícito ("no lo recomiendo", "no regresaría") y la queja predominante, aunque la reseña mencione algo positivo.
- neutro: la reseña es equilibrada (elogios y quejas de peso parecido), o es una queja leve, una sugerencia o una descripción sin una valoración clara.
- positivo: el cliente expresa una buena experiencia. Incluye el elogio con un reparo menor.

Si hay una foto adjunta, úsala como evidencia adicional. Si la reseña dice "(sin texto)", clasifica solo con la foto. Si no puedes determinarlo, elige "neutro".

Responde SOLO con un JSON de esta forma:
{"etiqueta": "negativo|neutro|positivo", "puntaje": <número de 0 a 10, donde 0 es muy negativo, 5 es neutro y 10 es muy positivo>}

Reseña:
{texto}
"""

ap = argparse.ArgumentParser()
ap.add_argument("paso", choices=["modelos", "probar", "iniciar", "validar", "fijar", "correr", "evaluar"])
ap.add_argument("--config", default="todas", help="E1-a, E1-b, E1-c, E1-d o todas")
ap.add_argument("--modelo", default=MODELO_DEFECTO)
ap.add_argument("--url-base", default="https://generativelanguage.googleapis.com")
ap.add_argument("--timeout", type=float, default=60.0, help="segundos de espera por llamada (una llamada normal tarda 1 a 5 s)")
ap.add_argument("--reintentos", type=int, default=6, help="intentos por llamada cuando Google responde saturado (429/5xx) o no responde")
ap.add_argument("--endpoint", default="auto", choices=["auto", "ai_studio", "vertex"], help="auto = como la aplicación: clave AQ. -> Vertex, las demás -> AI Studio")
ap.add_argument("--url-base-vertex", default="https://aiplatform.googleapis.com")
ap.add_argument("--esquema", action="store_true", help="enviar esquema JSON y tipo de respuesta (por defecto NO: la solicitud es mínima, como la de la aplicación)")
ap.add_argument("--modelos-probar", default="gemini-3.5-flash-lite,gemini-3.5-flash,gemini-3.1-flash-lite,gemini-flash-latest,gemini-3.8-flash,gemini-3.7-flash", help="lista separada por comas para el paso probar")
ap.add_argument("--rpm", type=float, default=8.0, help="llamadas por minuto (el plan gratuito limita; mira 'Rate limits' en AI Studio)")
ap.add_argument("--max-dia", type=int, default=0, help="tope de llamadas por ejecución (0 = sin tope); sirve para repartir en tandas")
ap.add_argument("--pensamiento", type=int, default=-1, help="presupuesto de 'razonamiento' de Gemini; -1 = omitir thinkingConfig (decisión metodológica); -1 = no enviar el ajuste (si el modelo no lo acepta)")
ap.add_argument("--limite", type=int, default=0, help="solo para pruebas: procesa N reseñas por configuración")
ap.add_argument("--n-validacion", type=int, default=50)
ap.add_argument("--precio-entrada", type=float, default=None, help="USD por millón de tokens de entrada (precio publicado por Google ese día)")
ap.add_argument("--precio-salida", type=float, default=None, help="USD por millón de tokens de salida")
ap.add_argument("--fuente-precio", default="", help="enlace de la página de precios de Google")
ap.add_argument("--fecha-precio", default="")
ap.add_argument("--rapido", action="store_true", help="solo para pruebas: 200 remuestreos")
args = ap.parse_args()
if args.rapido:
    B = 200


class CuotaDiaria(Exception):
    pass


def explicito(flag):
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv)


def sha(ruta):
    return hashlib.sha256(open(ruta, "rb").read()).hexdigest()


def clave():
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if k:
        return k
    for ruta in (os.path.expanduser(".env"), ".env"):
        if os.path.isfile(ruta):
            for linea in open(ruta, encoding="utf-8", errors="ignore"):
                if linea.strip().startswith("GEMINI_API_KEY="):
                    v = linea.split("=", 1)[1].strip().strip('"').strip("'")
                    if v:
                        return v
    print("No encuentro la clave de Gemini. Ponla en la variable de entorno (en tu terminal, NUNCA en el chat):\n    export GEMINI_API_KEY='...'\n"
          "o en .env como GEMINI_API_KEY=...")
    sys.exit(1)


# ------------------------------------------------------------------ llamada a la API
def preparar_imagen(ruta):
    from PIL import Image
    im = Image.open(ruta)
    if getattr(im, "n_frames", 1) > 1:
        im.seek(0)
    im = im.convert("RGB")
    lado = max(im.size)
    if lado > 1024:                                     # se reduce a 1024 px de lado mayor: se anota en la configuración
        f = 1024 / lado; im = im.resize((max(1, int(im.width * f)), max(1, int(im.height * f))))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def tipo_clave(k):
    return "vertex" if k.startswith("AQ.") else "ai_studio"


_EP_OK = {}          # modelo -> dirección que funcionó
ULTIMO = {"ep": ""}


def url_para(ep, modelo):
    if ep == "vertex":
        return f"{args.url_base_vertex}/v1/publishers/google/models/{modelo}:generateContent"
    return f"{args.url_base}/v1beta/models/{modelo}:generateContent"


def orden_endpoints(k, modelo):
    """auto = como la aplicación: primero la dirección que corresponde al tipo de clave (AQ. -> vertex) y, si da 403 o 404, la otra."""
    if args.endpoint != "auto":
        return [args.endpoint]
    if modelo in _EP_OK:
        return [_EP_OK[modelo]]
    pref = tipo_clave(k)
    return [pref, "ai_studio" if pref == "vertex" else "vertex"]


def _post(k, url, datos, timeout):
    """Un POST con reintentos y espera creciente si Google responde saturado (429 por minuto, 500, 502, 503, 504) o no responde."""
    intentos = 0
    while True:
        intentos += 1
        req = urllib.request.Request(url, data=datos, headers={"Content-Type": "application/json", "x-goog-api-key": k}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            cuerpo_err = e.read().decode("utf-8", errors="ignore")
            plano = re.sub(r"\s+", " ", cuerpo_err)
            if e.code == 429:
                if re.search(r"PerDay|per day|daily", cuerpo_err, re.I):
                    raise CuotaDiaria(plano[:300])
                m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', cuerpo_err)
                espera = float(m.group(1)) + 1 if m else min(60, 5 * intentos)
                if intentos >= args.reintentos:
                    raise RuntimeError("HTTP 429 repetido: " + plano[:200])
                print(f"    límite por minuto: espero {espera:.0f} s ..."); time.sleep(espera); continue
            if e.code in (500, 502, 503, 504):
                if intentos >= args.reintentos:
                    raise RuntimeError(f"HTTP {e.code} (Google saturado) tras {intentos} intentos: " + plano[:120])
                espera = min(60, 3 * 2 ** (intentos - 1)); print(f"    Google respondió {e.code} (saturado); reintento {intentos}/{args.reintentos - 1} en {espera:.0f} s ..."); time.sleep(espera); continue
            pista = ""
            if e.code in (403, 404):
                pista = "  -> Permiso denegado o modelo no disponible en esta dirección. Si tu clave empieza con AQ., prueba --endpoint ai_studio; si no, corre  run_e1.py modelos  y usa --modelo NOMBRE."
            elif e.code == 400 and re.search(r"thinking", cuerpo_err, re.I):
                pista = "  -> El modelo no acepta el ajuste de razonamiento. Repite con  --pensamiento -1  (no enviarlo) y anótalo."
            raise RuntimeError(f"HTTP {e.code}: " + plano[:300] + pista)
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as e:
            es_timeout = isinstance(e, (TimeoutError, socket.timeout)) or "timed out" in str(e).lower()
            if intentos >= args.reintentos:
                raise RuntimeError(("sin respuesta en %.0f s (timeout), %d intentos" % (timeout, intentos)) if es_timeout else "red: " + str(e))
            espera = min(60, 3 * 2 ** (intentos - 1)); print(f"    {'sin respuesta' if es_timeout else 'fallo de red'}; reintento {intentos}/{args.reintentos - 1} en {espera:.0f} s ..."); time.sleep(espera); continue


def llamar(k, prompt, imagen_b64=None, modelo=None, pensamiento=None, timeout=None):
    """Una llamada. Solicitud mínima como la de la aplicación (contents + temperatura 0). Lanza CuotaDiaria o RuntimeError."""
    partes = [{"text": prompt}]
    if imagen_b64:
        partes.append({"inline_data": {"mime_type": "image/jpeg", "data": imagen_b64}})
    gen = {"temperature": 0}
    if args.esquema:
        gen["responseMimeType"] = "application/json"
        gen["responseSchema"] = {"type": "OBJECT", "properties": {"etiqueta": {"type": "STRING", "enum": list(CLASES)}, "puntaje": {"type": "NUMBER"}}, "required": ["etiqueta", "puntaje"]}
    pens = args.pensamiento if pensamiento is None else pensamiento
    if pens >= 0:
        gen["thinkingConfig"] = {"thinkingBudget": pens}
    datos = json.dumps({"contents": [{"role": "user", "parts": partes}], "generationConfig": gen}).encode("utf-8")
    m = modelo or args.modelo
    eps = orden_endpoints(k, m); primero = None
    for i, ep in enumerate(eps):
        try:
            resp = _post(k, url_para(ep, m), datos, timeout or args.timeout)
            _EP_OK[m] = ep; ULTIMO["ep"] = ep
            return resp
        except RuntimeError as e:
            if primero is None:
                primero = e
            if i < len(eps) - 1 and re.match(r"HTTP (403|404)", str(e)):
                continue
            if i > 0:        # falló también la otra dirección: se muestra primero el error original, que es el informativo
                raise RuntimeError(str(primero) + "  | en la otra dirección: " + str(e)[:100])
            raise


def interpretar(resp):
    """Saca etiqueta y puntaje del texto de la respuesta (tolera ```json ... ``` y texto alrededor). (None, None) si no es válido."""
    try:
        partes = resp["candidates"][0]["content"]["parts"]
        txt = "".join(p.get("text", "") for p in partes if not p.get("thought"))
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip(), flags=re.I)
        m = re.search(r"\{.*\}", txt, re.S)
        j = json.loads(m.group(0))
        e = str(j["etiqueta"]).strip().lower(); p = float(j["puntaje"])
        if e in CLASES and 0 <= p <= 10:
            return e, p
    except Exception:
        pass
    return None, None


# ------------------------------------------------------------------ datos
def texto_llm(df_set, nombre):
    """Texto crudo (con sus emojis) si existe; si no, el limpio. Test-A viene de test.csv; Test-B, de test_multimodal.csv."""
    origen = "test.csv" if nombre == "A" else "test_multimodal.csv"
    ruta = os.path.join(DATOS, origen)
    t = df_set.copy()
    if os.path.isfile(ruta):
        o = pd.read_csv(ruta, encoding="utf-8", dtype=str)
        raw = o.set_index("review_id")["raw_text"] if "raw_text" in o else None
        if raw is not None:
            t["texto_llm"] = t["review_id"].map(raw).fillna("").str.strip()
            t["texto_llm"] = t["texto_llm"].where(t["texto_llm"] != "", t["texto"].fillna("").astype(str).str.strip())
            return t
    t["texto_llm"] = t["texto"].fillna("").astype(str).str.strip()
    return t


def cargar(nombre):
    d = pd.read_csv(os.path.join(DATOS, "test_a.csv" if nombre == "A" else "test_b.csv"), encoding="utf-8", dtype={"review_id": str})
    return texto_llm(d, nombre)


def prompt_final():
    ruta = os.path.join(RAIZ, "prompt_fijado.txt"); rsha = os.path.join(RAIZ, "prompt_fijado.sha256")
    if not os.path.isfile(ruta) or not os.path.isfile(rsha):
        print("El prompt todavía no está fijado. Corre 'validar' y luego 'fijar'."); sys.exit(1)
    if sha(ruta) != open(rsha).read().split()[0]:
        print("ALTO: prompt_fijado.txt fue modificado después de fijarlo. Eso rompe la regla de E1. Restáuralo."); sys.exit(1)
    return open(ruta, encoding="utf-8").read()


def armar(plantilla, texto, modo):
    t = texto if (texto and texto.strip() and modo != "foto") else ("(sin texto)" if modo == "foto" else (texto or "(sin texto)"))
    return plantilla.replace("{texto}", t)


# ------------------------------------------------------------------ pasos
if args.paso == "modelos":
    k = clave(); vistos, token = [], ""
    while True:
        url = f"{args.url_base}/v1beta/models?pageSize=200" + (f"&pageToken={token}" if token else "")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"x-goog-api-key": k}), timeout=60) as r:
                j = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: " + re.sub(r"\s+", " ", e.read().decode("utf-8", errors="ignore"))[:300]); sys.exit(1)
        vistos += [m for m in j.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
        token = j.get("nextPageToken", "")
        if not token:
            break
    print(f"Modelos que tu clave puede usar con generateContent ({len(vistos)}):\n")
    for m in sorted(vistos, key=lambda x: x["name"]):
        n = m["name"].replace("models/", "")
        if "gemini" in n:
            print(f"  {n:38} {m.get('displayName', '')}")
    print("\nElige uno de la familia Flash con entrada de imagen y úsalo así:  run_e1.py validar --modelo NOMBRE")
    print("En el paper se reporta el identificador EXACTO que devuelve la API y la fecha de la corrida.")
    sys.exit(0)

if args.paso == "probar":
    args.reintentos = min(args.reintentos, 2)          # en la prueba no se insiste: queremos ver rápido cuál responde
    k = clave(); from PIL import Image
    buf = io.BytesIO(); Image.new("RGB", (96, 96), (200, 60, 40)).save(buf, format="JPEG"); img = base64.b64encode(buf.getvalue()).decode("ascii")
    prompt = 'Clasifica el sentimiento de esta reseña de un restaurante: "La comida estuvo deliciosa y la atención fue excelente". Responde SOLO con el JSON {"etiqueta": "negativo|neutro|positivo", "puntaje": 0 a 10}.'
    modelos = [m.strip() for m in args.modelos_probar.split(",") if m.strip()]
    print(f"Clave de tipo {'Vertex (empieza con AQ.)' if tipo_clave(k) == 'vertex' else 'AI Studio'}. Dirección: {args.endpoint} " + ("(prueba primero la del tipo de clave y, si da 403 o 404, la otra)." if args.endpoint == "auto" else "(fija)."))
    print(f"Probando {len(modelos)} modelos con solicitud mínima (espera máxima {args.timeout:.0f} s, {args.reintentos} intentos). Cada modelo recibe 3 llamadas: texto, texto + imagen y texto con razonamiento = 0.\n")
    print(f"{'modelo':26} {'texto (razonamiento por defecto)':34} {'texto + imagen':26} {'razonamiento = 0':26} identificador devuelto")
    lecturas = []
    for m in modelos:
        celdas = []; toks = []; dev = ""
        for nombre, con_img, pens in (("texto", False, -1), ("texto + imagen", True, -1), ("razonamiento=0", False, 0)):
            t0 = time.time()
            try:
                resp = llamar(k, prompt, img if con_img else None, modelo=m, pensamiento=pens, timeout=args.timeout)
                e_, p_ = interpretar(resp); seg = time.time() - t0
                tk = (resp.get("usageMetadata") or {}).get("thoughtsTokenCount", 0) or 0
                toks.append(int(tk)); celdas.append(f"OK {seg:.1f}s ({e_ or 'sin JSON'}) r={int(tk)}")
                dev = resp.get("modelVersion", dev) or dev
            except CuotaDiaria:
                toks.append(None); celdas.append("cuota diaria")
            except RuntimeError as ex:
                msg = str(ex); toks.append(None); celdas.append("timeout" if "timeout" in msg else re.sub(r"^HTTP (\d+).*", r"HTTP \1", msg)[:18])
        print(f"{m:26} {celdas[0]:34} {celdas[1]:26} {celdas[2]:26} {dev} {('[' + _EP_OK[m] + ']') if m in _EP_OK else ''}", flush=True)
        t_def, t_img, t_cero = toks
        if celdas[0].startswith("OK") and celdas[1].startswith("OK") and t_cero is not None:
            if t_def == 0:
                lec = "no razona por defecto; el ajuste a 0 es aceptado." if celdas[2].startswith("OK") else "no razona por defecto (0 tokens) y NO acepta el ajuste: usar --pensamiento -1."
            else:
                lec = (f"razona por defecto ({t_def} tokens); con el ajuste en 0 baja a {t_cero}: " + ("EL AJUSTE FUNCIONA, usar --pensamiento 0." if t_cero == 0 else "el ajuste NO lo apaga del todo."))
        elif celdas[0].startswith("OK") and celdas[2].startswith("HTTP 400"):
            lec = ("no razona por defecto (0 tokens) y NO acepta el ajuste: usar --pensamiento -1." if t_def == 0 else f"razona por defecto ({t_def} tokens) y NO acepta el ajuste (400): no se puede controlar el razonamiento.")
        else:
            lec = "no respondió bien en todas las pruebas; no es candidato."
        lecturas.append((m, lec))
        time.sleep(60.0 / args.rpm)
    print("\nLECTURA (r = tokens de razonamiento de esa llamada):")
    for m, lec in lecturas:
        print(f"  {m}: {lec}")
    print("\nElige un modelo con OK en las tres columnas, rápido y con el razonamiento CONTROLADO (r=0 con el ajuste en 0, o sin razonamiento). Las cuatro configuraciones de E1 llevan el MISMO modelo.")
    sys.exit(0)

if args.paso == "iniciar":
    os.makedirs(RAIZ, exist_ok=True)
    ruta = os.path.join(RAIZ, "prompt_borrador.txt")
    if os.path.exists(ruta):
        print("Ya existe prompt_borrador.txt; no lo sobrescribo.")
    else:
        open(ruta, "w", encoding="utf-8").write(PROMPT_BORRADOR); print("Creé", ruta)
    print("\nEdítalo con las definiciones EXACTAS de la guía de anotación (el borrador usa las reglas de adjudicación de la memoria).\n"
          "Debe conservar la línea 'Reseña:' con {texto} al final. Luego corre 'validar'.")
    sys.exit(0)

if args.paso == "validar":
    k = clave()
    ruta_b = os.path.join(RAIZ, "prompt_borrador.txt")
    if not os.path.isfile(ruta_b):
        print("Corre primero 'iniciar'."); sys.exit(1)
    plantilla = open(ruta_b, encoding="utf-8").read()
    v = pd.read_csv(os.path.join(DATOS, "validation.csv"), encoding="utf-8", dtype=str)
    v["etiqueta"] = v["sentiment_label"].fillna("").str.strip().str.lower()
    v = v[v["etiqueta"].isin(CLASES)]
    ct = "raw_text" if "raw_text" in v else "clean_text"
    v["t"] = v[ct].fillna("").str.strip(); v = v[v["t"] != ""]
    muestra = v.sample(min(args.n_validacion, len(v)), random_state=SEMILLA_VALIDACION)
    print(f"Validando el prompt con {len(muestra)} reseñas de VALIDACIÓN (las etiquetas de validación son manuales, verificadas o de estrellas; sirven para pulir el prompt, no para reportar)")
    filas, pausa, tiempos = [], 60.0 / args.rpm, []
    for i, r in enumerate(muestra.itertuples(), 1):
        t0 = time.time()
        try:
            resp = llamar(k, armar(plantilla, r.t, "texto")); e, p = interpretar(resp); tiempos.append(time.time() - t0)
        except CuotaDiaria as ex:
            print("Se acabó la cuota diaria:", ex); break
        except RuntimeError as ex:
            print("\nLa llamada falló:", ex); sys.exit(1)
        u = resp.get("usageMetadata", {}) if isinstance(resp, dict) else {}
        filas.append({"review_id": r.review_id, "etiqueta": r.etiqueta, "pred": e, "puntaje": p, "tokens_entrada": u.get("promptTokenCount"), "tokens_salida": u.get("candidatesTokenCount"),
                      "tokens_pensamiento": u.get("thoughtsTokenCount"), "modelo_devuelto": resp.get("modelVersion") if isinstance(resp, dict) else None})
        print(f"  {i}/{len(muestra)}  real={r.etiqueta:9} pred={e}", end="\r"); time.sleep(pausa)
    res = pd.DataFrame(filas)
    res.to_csv(os.path.join(RAIZ, f"validacion_prompt_{dt.datetime.now():%Y%m%d_%H%M}.csv"), index=False, encoding="utf-8")
    ok = res.dropna(subset=["pred"])
    print(f"\n\nRespuestas válidas: {len(ok)} de {len(res)}")
    if tiempos:
        print(f"Tiempo por llamada: medio {np.mean(tiempos):.1f} s, máximo {np.max(tiempos):.1f} s. Dirección usada: {ULTIMO['ep'] or '(ninguna)'}")
    if len(res):
        print("Modelo que respondió (identificador exacto):", sorted(set(res["modelo_devuelto"].dropna().astype(str))) or "(la API no lo devolvió)")
        for col, nombre in (("tokens_entrada", "entrada"), ("tokens_salida", "salida"), ("tokens_pensamiento", "razonamiento")):
            v = pd.to_numeric(res[col], errors="coerce")
            print(f"  tokens de {nombre}: promedio {v.mean():.1f}" if v.notna().any() else f"  tokens de {nombre}: la API no los informó")
        pens = pd.to_numeric(res["tokens_pensamiento"], errors="coerce").fillna(0).mean()
        if pens > 0:
            print("  OJO: el modelo está usando tokens de razonamiento aunque pedimos desactivarlo. Reportar resultado: cambia el costo y puede cambiar el rendimiento del rival.")
    if len(ok):
        print("Exactitud:", round(float((ok['pred'] == ok['etiqueta']).mean()), 3))
        print(pd.crosstab(ok["etiqueta"], ok["pred"]))
    print("\nSi lo ajustas, edita prompt_borrador.txt y vuelve a validar. Cuando esté, corre 'fijar'. NUNCA ajustes mirando Test-A o Test-B.")
    sys.exit(0)

if args.paso == "fijar":
    src = os.path.join(RAIZ, "prompt_borrador.txt"); dst = os.path.join(RAIZ, "prompt_fijado.txt")
    if os.path.exists(dst):
        print("El prompt ya está fijado y no se puede cambiar:", sha(dst)); sys.exit(0)
    texto = open(src, encoding="utf-8").read()
    if "{texto}" not in texto:
        print("El prompt debe contener {texto}."); sys.exit(1)
    if not explicito("--modelo"):
        print("Al fijar tienes que indicar el modelo EXACTO que usarás en las cuatro configuraciones, por ejemplo:\n    run_e1.py fijar --modelo gemini-3.5-flash-lite --pensamiento -1"); sys.exit(1)
    open(dst, "w", encoding="utf-8").write(texto)
    open(os.path.join(RAIZ, "prompt_fijado.sha256"), "w").write(f"{sha(dst)}  prompt_fijado.txt  fijado {dt.datetime.now():%Y-%m-%d %H:%M}\n")
    os.chmod(dst, stat.S_IREAD)
    json.dump({"modelo": args.modelo, "pensamiento": args.pensamiento, "endpoint": args.endpoint, "temperatura": 0, "fijado": dt.datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(RAIZ, "modelo_fijado.json"), "w", encoding="utf-8"), indent=2)
    print(f"Modelo FIJADO: {args.modelo} | razonamiento: {'no se envía' if args.pensamiento < 0 else args.pensamiento} | temperatura 0. Se usa el mismo valor en todas las llamadas.")
    print("Prompt FIJADO. sha256:", sha(dst), "\nYa no se modifica. Guarda el archivo tal cual (va al reporte de resultados).")
    sys.exit(0)

if args.paso == "correr":
    k = clave(); plantilla = prompt_final()
    rf = os.path.join(RAIZ, "modelo_fijado.json")
    if not os.path.isfile(rf):
        print("Falta el modelo fijado. Corre:  run_e1.py fijar --modelo NOMBRE --pensamiento N"); sys.exit(1)
    fijo = json.load(open(rf, encoding="utf-8"))
    if (explicito("--modelo") and args.modelo != fijo["modelo"]) or (explicito("--pensamiento") and args.pensamiento != fijo.get("pensamiento", fijo.get("thinking_budget", -1))):
        print(f"ALTO: pediste otro modelo o razonamiento distinto del fijado ({fijo['modelo']}, razonamiento {fijo['pensamiento']}). Las cuatro configuraciones deben llevar el mismo."); sys.exit(1)
    args.modelo, args.pensamiento = fijo["modelo"], fijo.get("pensamiento", fijo.get("thinking_budget", -1))
    print(f"Modelo fijado: {args.modelo} | razonamiento: {'no se envía' if args.pensamiento < 0 else args.pensamiento}")
    elegidas = list(CONFIGS) if args.config == "todas" else [args.config]
    pausa = 60.0 / args.rpm; hechas_total = 0; fallos_seguidos = 0
    for cfg in elegidas:
        conjunto, modo = CONFIGS[cfg]
        df = cargar(conjunto)
        if args.limite:
            df = df.head(args.limite)
        ruta = os.path.join(RAIZ, f"llamadas_{cfg}.csv")
        columnas = ["config", "review_id", "id_negocio", "etiqueta_real", "pred_etiqueta", "pred_puntaje", "json_valido", "reintento", "tokens_entrada", "tokens_salida",
                    "tokens_pensamiento", "modelo_devuelto", "fecha_hora", "error"]
        ya = set(pd.read_csv(ruta, dtype=str)["review_id"]) if os.path.isfile(ruta) else set()
        if not os.path.isfile(ruta):
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(columnas)
        pendientes = df[~df["review_id"].isin(ya)]
        print(f"\n{cfg} ({modo}, Test-{conjunto}): {len(df)} reseñas | ya hechas: {len(ya)} | pendientes: {len(pendientes)}")
        for n, r in enumerate(pendientes.itertuples(), 1):
            if args.max_dia and hechas_total >= args.max_dia:
                print(f"\nLlegué al tope de {args.max_dia} llamadas de esta tanda. Vuelve a correr 'correr' para continuar."); sys.exit(0)
            img = None
            if "foto" in modo:
                p = next((os.path.join(IMAGENES, f"{r.review_id}{e}") for e in (".jpg", ".jpeg", ".png", ".webp") if os.path.exists(os.path.join(IMAGENES, f"{r.review_id}{e}"))), None)
                if not p:
                    fila = [cfg, r.review_id, r.id_negocio, r.etiqueta, "", "", 0, 0, "", "", "", "", dt.datetime.now().isoformat(timespec="seconds"), "sin imagen"]
                    csv.writer(open(ruta, "a", newline="", encoding="utf-8")).writerow(fila); continue
                img = preparar_imagen(p)
            prompt = armar(plantilla, r.texto_llm, "foto" if modo == "foto" else "texto")
            e = p_ = None; reint = 0; err = ""; resp = {}
            try:
                for intento in (0, 1):
                    resp = llamar(k, prompt, img); e, p_ = interpretar(resp)
                    if e:
                        break
                    reint = 1
                if not e:
                    err = "json no válido tras el reintento"
            except CuotaDiaria as ex:
                print("\nSe acabó la cuota DIARIA de Gemini. Todo lo hecho quedó guardado; repite 'correr' mañana (o activa la facturación).", str(ex)[:150]); sys.exit(0)
            except Exception as ex:
                # fallo de conexión o de clave: NO se guarda la fila (así se reintenta al repetir 'correr') y se corta si se repite
                fallos_seguidos += 1; msg = str(ex)[:200]
                print(f"\n    fallo de la llamada ({msg})")
                if "HTTP 400" in msg or "HTTP 401" in msg or "HTTP 403" in msg or fallos_seguidos >= 3:
                    print("Corto aquí: revisa la clave (GEMINI_API_KEY), la conexión o el nombre del modelo. Lo hecho quedó guardado; repite 'correr' cuando lo arregles."); sys.exit(1)
                continue
            fallos_seguidos = 0
            u = resp.get("usageMetadata", {}) if isinstance(resp, dict) else {}
            fila = [cfg, r.review_id, r.id_negocio, r.etiqueta, e or "", p_ if p_ is not None else "", int(bool(e)), reint, u.get("promptTokenCount", ""), u.get("candidatesTokenCount", ""),
                    u.get("thoughtsTokenCount", ""), resp.get("modelVersion", "") if isinstance(resp, dict) else "", dt.datetime.now().isoformat(timespec="seconds"), err]
            with open(ruta, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(fila)
            hechas_total += 1
            print(f"  {cfg} {n}/{len(pendientes)}  real={r.etiqueta:9} pred={e or 'ERROR':9}", end="\r"); time.sleep(pausa)
    print("\n\nListo. Corre 'evaluar' para las métricas.")
    sys.exit(0)

# ------------------------------------------------------------------ evaluar
def conf(y, p, k):
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)


def f1m(C):
    tp = np.diag(C).astype(float); fp = C.sum(0) - tp; fn = C.sum(1) - tp
    pres = (C.sum(0) + C.sum(1)) > 0; den = 2 * tp + fp + fn
    return float(np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)[pres].mean())


def mccf(C):
    t = C.sum(1).astype(float); p = C.sum(0).astype(float); c = float(np.trace(C)); s = float(C.sum())
    den = np.sqrt((s ** 2 - (p * p).sum()) * (s ** 2 - (t * t).sum()))
    return float((c * s - (t * p).sum()) / den) if den > 0 else 0.0


from scipy.stats import binomtest


def clases_desde_puntaje(s, tarea):
    s = np.asarray(s, float)
    return (s >= 5).astype(int) if tarea == "pol" else np.where(s < 3.2, 0, np.where(s < 7.3, 1, 2)).astype(int)


def preparar_eval(cfg):
    """Devuelve DataFrame por reseña con y_3, pred_3 (los errores cuentan como falla), pol y pred_pol."""
    ruta = os.path.join(RAIZ, f"llamadas_{cfg}.csv")
    if not os.path.isfile(ruta):
        return None
    d = pd.read_csv(ruta, dtype={"review_id": str})
    d["y3"] = d["etiqueta_real"].map({c: i for i, c in enumerate(CLASES)})
    d["p3"] = d["pred_etiqueta"].map({c: i for i, c in enumerate(CLASES)})
    d["p3"] = np.where(d["p3"].isna(), (d["y3"] + 1) % 3, d["p3"]).astype(int)          # un error cuenta como respuesta incorrecta
    d["ypol"] = d["etiqueta_real"].map({"negativo": 0, "positivo": 1})
    pp = pd.to_numeric(d["pred_puntaje"], errors="coerce")
    d["ppol"] = np.where(pp.isna(), 1 - d["ypol"].fillna(0), (pp >= 5).astype(int)).astype(int)
    return d


def metricas(y, p, k, rng):
    n = len(y); C = conf(y, p, k)
    boots = []
    for _ in range(B):
        i = rng.integers(0, n, n); Cb = conf(y[i], p[i], k); boots.append((f1m(Cb), mccf(Cb)))
    b = np.array(boots)
    return {"n": n, "f1": f1m(C), "f1_inf": np.percentile(b[:, 0], 2.5), "f1_sup": np.percentile(b[:, 0], 97.5), "mcc": mccf(C), "mcc_inf": np.percentile(b[:, 1], 2.5), "mcc_sup": np.percentile(b[:, 1], 97.5)}


def comparar(y, p1, p2, k, rng, delta=(0.02, 0.05)):
    """p1 = modelo nuevo, p2 = referencia, sobre las mismas reseñas. McNemar exacto, IC 95 % de la diferencia de F1 y TOST."""
    n = len(y); c1, c2 = (p1 == y), (p2 == y)
    b_, c_ = int((c2 & ~c1).sum()), int((~c2 & c1).sum())
    p = 1.0 if (b_ + c_) == 0 else float(binomtest(min(b_, c_), b_ + c_, 0.5).pvalue)
    dif = np.array([f1m(conf(y[i], p1[i], k)) - f1m(conf(y[i], p2[i], k)) for i in (rng.integers(0, n, n) for _ in range(B))])
    punto = f1m(conf(y, p1, k)) - f1m(conf(y, p2, k))
    out = {"n": n, "dif_f1": punto, "dif_ic95_inf": np.percentile(dif, 2.5), "dif_ic95_sup": np.percentile(dif, 97.5), "p_mcnemar": p,
           "ic90_inf": np.percentile(dif, 5), "ic90_sup": np.percentile(dif, 95)}
    for dl in delta:
        out[f"equivalente_delta_{dl}"] = bool(out["ic90_inf"] >= -dl and out["ic90_sup"] <= dl)
    return out


rng = np.random.default_rng(SEMILLA_BOOT)
res = {}
for cfg in CONFIGS:
    d = preparar_eval(cfg)
    if d is not None and len(d):
        res[cfg] = d
if not res:
    print("No hay llamadas guardadas todavía. Corre 'correr'."); sys.exit(1)

filas, costo = [], []
for cfg, d in res.items():
    ntot = len(d); errores = int((d["json_valido"] == 0).sum()); conteos_reintento = int(d["reintento"].sum())
    for tarea, k, y, p in (("polaridad", 2, d[d["ypol"].notna()]["ypol"].astype(int).values, d[d["ypol"].notna()]["ppol"].values), ("tres clases", 3, d["y3"].values.astype(int), d["p3"].values)):
        m = metricas(y, p, k, rng); m.update({"config": cfg, "tarea": tarea, "llamadas": ntot, "errores_json": errores, "reintentos": conteos_reintento}); filas.append(m)
    t_in = pd.to_numeric(d["tokens_entrada"], errors="coerce"); t_out = pd.to_numeric(d["tokens_salida"], errors="coerce").fillna(0) + pd.to_numeric(d["tokens_pensamiento"], errors="coerce").fillna(0)
    fila = {"config": cfg, "llamadas": ntot, "tokens_entrada_prom": float(t_in.mean()), "tokens_salida_prom": float(t_out.mean()), "tokens_entrada_total": float(t_in.sum()), "tokens_salida_total": float(t_out.sum()),
            "modelos_devueltos": ";".join(sorted(set(d["modelo_devuelto"].dropna().astype(str)) - {""}))}
    if args.precio_entrada is not None and args.precio_salida is not None:
        c = (t_in.fillna(0) * args.precio_entrada + t_out * args.precio_salida) / 1e6
        fila.update({"costo_prom_por_resena_usd": float(c.mean()), "costo_por_1000_resenas_usd": float(c.mean() * 1000)})
    costo.append(fila)
tabla = pd.DataFrame(filas)

# comparaciones pedidas
comp = []
ta = pd.read_csv(os.path.join(DATOS, "test_a.csv"), dtype={"review_id": str}); tb = pd.read_csv(os.path.join(DATOS, "test_b.csv"), dtype={"review_id": str})
mapa = {c: i for i, c in enumerate(CLASES)}


def par(cfgA, cfgB=None, ref=None, base=None):
    """Alinea por review_id las predicciones de cfgA con cfgB o con una referencia de puntajes (columna de base)."""
    a = res.get(cfgA)
    if a is None:
        return None
    if cfgB:
        b_ = res.get(cfgB)
        if b_ is None:
            return None
        m = a.merge(b_, on="review_id", suffixes=("_1", "_2"))
        return m
    m = a.merge(base[["review_id", ref]], on="review_id")
    return m


def hacer(nombre, cfgA, cfgB=None, base=None, ref=None):
    m = par(cfgA, cfgB, ref, base)
    if m is None or len(m) == 0:
        return
    for tarea, k in (("polaridad", 2), ("tres clases", 3)):
        if cfgB:
            if tarea == "polaridad":
                mm = m[m["ypol_1"].notna()]; y = mm["ypol_1"].astype(int).values; p1, p2 = mm["ppol_1"].values, mm["ppol_2"].values
            else:
                y = m["y3_1"].values.astype(int); p1, p2 = m["p3_1"].values, m["p3_2"].values
        else:
            if tarea == "polaridad":
                mm = m[m["ypol"].notna()]; y = mm["ypol"].astype(int).values; p1 = mm["ppol"].values; p2 = clases_desde_puntaje(mm[ref].values, "pol")
            else:
                y = m["y3"].values.astype(int); p1 = m["p3"].values; p2 = clases_desde_puntaje(m[ref].values, "tres")
        r = comparar(y, p1, p2, k, rng); r.update({"comparacion": nombre, "tarea": tarea}); comp.append(r)


hacer("E1-a frente al regresor de texto (Test-A)", "E1-a", base=ta, ref="s_t")
hacer("E1-b frente al regresor de texto (Test-B)", "E1-b", base=tb, ref="s_t")
hacer("E1-c frente a E1-b (¿le ayuda la foto a Gemini?)", "E1-c", cfgB="E1-b")
hacer("E1-d frente a CLIP ajustado (Test-B)", "E1-d", base=tb, ref="s_v")
comp = pd.DataFrame(comp)

carpeta = os.path.join(RAIZ, f"resultados_{dt.datetime.now():%Y%m%d_%H%M}"); os.makedirs(carpeta, exist_ok=True)
tabla.round(5).to_csv(os.path.join(carpeta, "e1_metricas.csv"), index=False, encoding="utf-8")
comp.round(5).to_csv(os.path.join(carpeta, "e1_comparaciones.csv"), index=False, encoding="utf-8")
pd.DataFrame(costo).round(6).to_csv(os.path.join(carpeta, "e1_tokens_y_costo.csv"), index=False, encoding="utf-8")
ruta_pf = os.path.join(RAIZ, "prompt_fijado.txt")
cfg_json = {"experimento": "E1 Gemini sin entrenar", "fecha_evaluacion": dt.datetime.now().isoformat(timespec="seconds"), "modelo_fijado": (json.load(open(os.path.join(RAIZ, "modelo_fijado.json"))) if os.path.isfile(os.path.join(RAIZ, "modelo_fijado.json")) else None), "modelo_solicitado": args.modelo, "endpoint": args.endpoint, "solicitud": "con esquema JSON" if args.esquema else "mínima (contents + temperatura 0)",
            "modelos_devueltos_por_la_api": sorted({m for f in costo for m in f["modelos_devueltos"].split(";") if m}), "temperatura": 0, "pensamiento_presupuesto": args.pensamiento,
            "salida": "JSON con esquema (etiqueta en {negativo, neutro, positivo}, puntaje 0 a 10)", "prompt_sha256": sha(ruta_pf) if os.path.isfile(ruta_pf) else None,
            "texto_enviado": "raw_text (con emojis) si existe; si no, clean_text", "imagen": "primera foto de la reseña, a JPEG y a un máximo de 1024 px de lado mayor",
            "errores_de_json": "tras un reintento, cuentan como respuesta incorrecta y se reportan", "semilla_bootstrap": SEMILLA_BOOT, "remuestreos": B,
            "precio": {"entrada_usd_por_millon": args.precio_entrada, "salida_usd_por_millon": args.precio_salida, "fuente": args.fuente_precio, "fecha": args.fecha_precio},
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}
json.dump(cfg_json, open(os.path.join(carpeta, "e1_configuracion.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
_ids = cfg_json["modelos_devueltos_por_la_api"]
if len(_ids) > 1:
    print("\n*** AVISO: la API devolvió MÁS DE UN identificador de modelo en las llamadas:", _ids, "\n    Las cuatro configuraciones deben correrse con el mismo modelo. Revisa llamadas_E1-*.csv (columna modelo_devuelto).")
if args.precio_entrada is None:
    print("\n(No pasaste los precios: el costo en USD queda pendiente. Los tokens sí están guardados.)")
print("\n" + "=" * 78); print("E1: resultados"); print("=" * 78)
vista = tabla.assign(**{"F1 [IC95]": tabla.apply(lambda r: f"{r.f1:.3f} [{r.f1_inf:.3f}, {r.f1_sup:.3f}]", axis=1), "MCC [IC95]": tabla.apply(lambda r: f"{r.mcc:.3f} [{r.mcc_inf:.3f}, {r.mcc_sup:.3f}]", axis=1)})
with pd.option_context("display.width", 220, "display.max_columns", 20):
    print(vista[["config", "tarea", "n", "F1 [IC95]", "MCC [IC95]", "errores_json", "reintentos"]].to_string(index=False))
    print("\nComparaciones (McNemar exacto; IC 95 % de la diferencia de F1; equivalencia con IC 90 %):")
    if len(comp):
        print(comp[["comparacion", "tarea", "n", "dif_f1", "dif_ic95_inf", "dif_ic95_sup", "p_mcnemar", "equivalente_delta_0.02", "equivalente_delta_0.05"]].round(4).to_string(index=False))
    print("\nTokens y costo:"); print(pd.DataFrame(costo).round(4).to_string(index=False))
print(f"\nGuardado en {carpeta}/")
print("RECORDATORIO: las etiquetas de prueba se preasignaron con otro modelo de lenguaje y luego se revisaron; si Gemini sale muy alto, léanlo junto con el etiquetado externo a ciegas (Cambio 7).")
