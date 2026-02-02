# Tavily Agent con LangChain

Este proyecto implementa un agente conversacional en terminal utilizando **LangChain (v0.3.x)**.  
El agente es capaz de mantener memoria de la conversación, decidir cuándo usar herramientas externas y presentar la información de forma estructurada al usuario.

El objetivo principal es demostrar el uso de **tools**, **memoria**, **branching** y **parallel chains** dentro del ecosistema LangChain.

---

## Funcionalidades principales

- Agente conversacional ejecutable desde terminal.
- Uso de **Tavily Search** como herramienta principal para acceso a información web actualizada.
- Memoria de conversación simple (historial en la sesión de terminal).
- Detección básica de intención (branching) para adaptar el estilo de respuesta.
- Postprocesado en paralelo para:
  - Formatear la respuesta final.
  - Extraer y mostrar fuentes cuando existen URLs.
- Herramientas adicionales:
  - Calculadora segura para operaciones matemáticas.
  - Consulta de hora actual del sistema (hora local).

El sistema incluye un mecanismo de **fallback** para responder usando herramientas locales o Tavily en caso de que el endpoint LLM no esté disponible.

---

## Requisitos

- Python 3.12
- Entorno virtual recomendado

Las dependencias utilizadas son **las mismas que las definidas en el `requirements.txt` del proyecto inicial**, sin modificaciones adicionales.

---

## Variables de entorno

El proyecto requiere un archivo `.env` en la raíz con las siguientes variables:

```env
BASE_URL=...
API_KEY=...
TAVILY_API_KEY=...
```

- **BASE_URL**: endpoint del proveedor LLM.
- **API_KEY**: clave de acceso al LLM.
- **TAVILY_API_KEY**: clave de la API de Tavily.

La variable **MODEL_NAME** se puede poner pero es opcional; si no se define, se utiliza el valor por defecto configurado en el código.

---

## Ejecución

Desde la raíz del proyecto y con el entorno virtual activado:

```bash
python tavily_agent_terminal.py
```
Al iniciarse, el agente mostrará un prompt interactivo en terminal.  
Para salir del programa, escribir:

**exit**

---

## Ejemplos de uso

### Cálculo y herramientas locales

Calcula 12/3 + sqrt(16) y dime la hora en Madrid.

### Uso de Tavily como herramienta de búsqueda

¿Qué es Tavily y para qué se usa en agentes de IA? Dame 3 fuentes con URL.

### Uso de memoria en la conversación

¿Y en qué se diferencia de SerpAPI?

---

## Notas

- El warning de deprecación de Tavily mostrado en consola no afecta al funcionamiento del proyecto.
- El diseño del agente sigue el enfoque moderno de LangChain (0.3.x), basado en tool-calling y cadenas explícitas, evitando APIs legacy.

