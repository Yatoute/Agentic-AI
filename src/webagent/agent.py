import json
import asyncio
from typing import Any, Dict, List, Literal, Optional, TypedDict

from playwright.async_api import async_playwright, Page
from jsonschema import validate
from jsonschema.exceptions import ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END


# -----------------------------
# 1) DSL + JSON Schema Plan (avec alternatives)
# -----------------------------
ActionType = Literal["goto", "click", "type", "press", "wait", "extract"]

PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "goal_reached": {"type": "boolean"},
        "rationale": {"type": ["string", "null"]},
        "steps": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action": {"enum": ["goto", "click", "type", "press", "wait", "extract"]},

                    # Locator principal (accessibility-first)
                    "role": {"type": "string"},
                    "name": {"type": "string"},
                    "label": {"type": "string"},

                    # Paramètres
                    "url": {"type": "string"},
                    "text": {"type": "string"},
                    "key": {"type": "string"},
                    "ms": {"type": "integer"},
                    "what": {"enum": ["body_text", "title", "url"]},

                    # Robustesse agentic
                    "alternatives": {
                        "type": "array",
                        "maxItems": 6,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "role": {"type": "string"},
                                "name": {"type": "string"},
                                "label": {"type": "string"},
                                "key": {"type": "string"},     # pour press fallback
                            },
                        },
                    },
                    "on_failure": {"enum": ["try_next", "replan"]},
                    "note": {"type": "string"},
                },
                "required": ["action"],
                "allOf": [
                    {"if": {"properties": {"action": {"const": "goto"}}}, "then": {"required": ["url"]}},
                    {"if": {"properties": {"action": {"const": "type"}}}, "then": {"required": ["text"]}},
                    {"if": {"properties": {"action": {"const": "extract"}}}, "then": {"required": ["what"]}},
                ],
            },
        },
    },
    "required": ["goal_reached", "steps"],
}


def fallback_plan(reason: str) -> Dict[str, Any]:
    return {
        "goal_reached": False,
        "rationale": reason,
        "steps": [{"action": "extract", "what": "body_text", "on_failure": "replan"}],
    }


def parse_and_validate_plan(text: str) -> Dict[str, Any]:
    try:
        plan = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return fallback_plan("Invalid JSON (no object found)")
        try:
            plan = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return fallback_plan("Invalid JSON (object parse failed)")

    try:
        validate(instance=plan, schema=PLAN_SCHEMA)
    except ValidationError as e:
        return fallback_plan(f"Schema validation failed: {e.message}")

    plan["steps"] = plan.get("steps", [])[:8]
    return plan


# -----------------------------
# 2) State LangGraph
# -----------------------------
class AgentState(TypedDict):
    goal: str
    url: str
    observation: Dict[str, Any]
    plan: Dict[str, Any]
    memory: Dict[str, Any]
    turns: int
    done: bool


# -----------------------------
# 3) Observe / Act
# -----------------------------
async def observe_page(page: Page, max_targets: int = 50) -> Dict[str, Any]:
    try:
        await page.wait_for_load_state("domcontentloaded")
    except Exception:
        pass

    body_text = (await page.inner_text("body"))[:2500]
    title = await page.title()

    targets = await page.evaluate(
        """(maxTargets) => {
          const isVisible = (el) => {
            const r = el.getBoundingClientRect();
            return r.width > 0 && r.height > 0;
          };
          const nodes = Array.from(document.querySelectorAll(
              'a, button, input, textarea, select, [role="button"], [role="link"], [role="textbox"]'
            ))
            .filter(isVisible)
            .slice(0, maxTargets);

          const clean = (s) => (s || '').replace(/\\s+/g,' ').trim().slice(0, 80);

          return nodes.map(el => {
            let role = el.getAttribute('role') || '';
            const tag = el.tagName.toLowerCase();
            if (!role) {
              if (tag === 'a') role = 'link';
              else if (tag === 'button') role = 'button';
              else if (tag === 'input' || tag === 'textarea') role = 'textbox';
            }
            const name = clean(
              el.innerText ||
              el.value ||
              el.getAttribute('aria-label') ||
              el.getAttribute('title') ||
              el.getAttribute('placeholder')
            );
            return { role, name, tag };
          });
        }""",
        max_targets,
    )

    inputs = await page.evaluate(
        """() => {
          const isVisible = (el) => {
            const r = el.getBoundingClientRect();
            return r.width > 0 && r.height > 0;
          };
          const clean = (s) => (s || '').replace(/\\s+/g,' ').trim().slice(0, 80);
          const nodes = Array.from(document.querySelectorAll('input, textarea, [role="textbox"]'))
            .filter(isVisible)
            .slice(0, 20);

          return nodes.map(el => ({
            tag: el.tagName.toLowerCase(),
            type: el.getAttribute('type') || '',
            aria_label: clean(el.getAttribute('aria-label')),
            placeholder: clean(el.getAttribute('placeholder')),
            name: clean(el.getAttribute('name')),
            id: clean(el.getAttribute('id')),
            role: el.getAttribute('role') || ''
          }));
        }"""
    )

    # petits hints textuels (utile pour "Search", "⌘K", "K")
    hints = {
        "contains_search_word": ("search" in body_text.lower()),
        "body_has_k_hint": (" k" in body_text.lower() or "\nk\n" in body_text.lower()),
    }

    return {"url": page.url, "title": title, "body_text": body_text, "targets": targets, "inputs": inputs, "hints": hints}


async def _click_variant(page: Page, variant: Dict[str, Any], timeout_ms: int = 2500) -> None:
    if variant.get("label"):
        loc = page.get_by_label(variant["label"], exact=False).first
    else:
        role = variant.get("role", "button")
        name = variant.get("name", "")
        if name:
            loc = page.get_by_role(role, name=name, exact=False).first
        else:
            loc = page.get_by_role(role).first

    await loc.wait_for(state="visible", timeout=timeout_ms)
    await loc.click(timeout=timeout_ms)


async def _type_variant(page: Page, variant: Dict[str, Any], text: str, timeout_ms: int = 2500) -> None:
    if variant.get("label"):
        loc = page.get_by_label(variant["label"], exact=False).first
    else:
        name = variant.get("name", "")
        if name:
            loc = page.get_by_role("textbox", name=name, exact=False).first
        else:
            loc = page.get_by_role("textbox").first

    await loc.wait_for(state="visible", timeout=timeout_ms)
    await loc.fill(text, timeout=timeout_ms)


async def _press_variant(page: Page, variant: Dict[str, Any]) -> None:
    key = variant.get("key", "Enter")
    await page.keyboard.press(key)


async def run_step(page: Page, step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Exécute une step avec alternatives proposées par le LLM.
    Si ça échoue:
      - try_next: tente la variante suivante
      - replan: renvoie une erreur (le planner verra l'erreur et ajustera)
    """
    print("STEP EXEC:", step)

    action: ActionType = step["action"]  # type: ignore
    on_failure = step.get("on_failure", "replan")

    try:
        if action == "goto":
            await page.goto(step["url"], wait_until="domcontentloaded")
            return None

        if action == "wait":
            await page.wait_for_timeout(int(step.get("ms", 800)))
            return None

        if action == "extract":
            what = step.get("what", "body_text")
            if what == "body_text":
                return {"body_text": (await page.inner_text("body"))[:5000], "url": page.url}
            if what == "title":
                return {"title": await page.title(), "url": page.url}
            if what == "url":
                return {"url": page.url}
            return {"note": f"unknown extract: {what}", "url": page.url}

        # --- actions avec alternatives ---
        variants: List[Dict[str, Any]] = []

        # variant principal (champ à plat)
        primary_variant = {k: step[k] for k in ("role", "name", "label", "key") if k in step}
        variants.append(primary_variant)

        # variantes additionnelles proposées par le LLM
        for v in step.get("alternatives", []) or []:
            if isinstance(v, dict):
                variants.append(v)

        last_err: Optional[Exception] = None

        for idx, v in enumerate(variants):
            try:
                if action == "click":
                    await _click_variant(page, v)
                    await page.wait_for_load_state("domcontentloaded")
                    print(f"[OK] click variant #{idx}: {v}")
                    return None

                if action == "type":
                    await _type_variant(page, v, step["text"])
                    print(f"[OK] type variant #{idx}: {v}")
                    return None

                if action == "press":
                    await _press_variant(page, v)
                    await page.wait_for_timeout(200)
                    print(f"[OK] press variant #{idx}: {v}")
                    return None

                return {"note": f"unknown action: {action}", "url": page.url}

            except Exception as e:
                last_err = e
                print(f"[FAIL] {action} variant #{idx}: {v} -> {e}")
                if on_failure == "replan":
                    break  # stop variants immediately

        # tout a échoué
        raise last_err or RuntimeError(f"{action} failed with no variants")

    except Exception as e:
        return {"error": str(e), "failed_step": step, "url": page.url}


# -----------------------------
# 4) Nodes LangGraph
# -----------------------------
def should_continue(state: AgentState) -> str:
    if state["done"]:
        return END
    if state["turns"] >= 8:
        return END
    return "observe"


async def observe_node(state: AgentState) -> AgentState:
    page: Page = state["memory"]["page"]
    obs = await observe_page(page)

    print("\n=== OBSERVE NODE ===")
    print("URL:", obs["url"])
    print("TITLE:", obs["title"])
    print("BODY (first 200 chars):", obs["body_text"][:200])
    print("INPUTS SAMPLE:", obs.get("inputs", [])[:6])
    print("TARGETS SAMPLE:", obs.get("targets", [])[:10])
    print("HINTS:", obs.get("hints"))

    state["observation"] = obs
    state["url"] = obs["url"]
    return state


async def plan_node(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-5-nano")

    system = SystemMessage(content=(
        "Tu es un agent Playwright.\n"
        "Retourne UNIQUEMENT un JSON valide.\n\n"
        "IMPORTANT ROBUSTESSE:\n"
        "- Pour toute action incertaine (ex: ouvrir un moteur de recherche), propose TOUJOURS plusieurs variantes via 'alternatives'.\n"
        "- Exemple click search:\n"
        "  primary: role=button name='Search'\n"
        "  alternatives: role=button name='Search', role=link name='Search', role=button name='Search' (exact=false), etc.\n"
        "- Tu peux aussi proposer un fallback clavier via press: key='Control+K' ou 'Meta+K'.\n"
        "- Si une étape doit tenter plusieurs variantes, mets on_failure='try_next'.\n"
        "- Si tu veux replanifier dès qu'un click échoue: on_failure='replan'.\n\n"
        "Format:\n"
        "{ goal_reached: boolean, rationale: string|null, steps: [ ... ] }\n"
        "Chaque step click/type/press peut avoir 'alternatives'.\n"
        "Ne propose pas de CSS/XPath.\n"
        "Si steps non vides -> goal_reached doit être false.\n"
    ))

    human = HumanMessage(content=json.dumps({
        "goal": state["goal"],
        "observation": state["observation"],
        "memory": {
            "last_error": state["memory"].get("last_error"),
            "failed_step": state["memory"].get("failed_step"),
        }
    }, ensure_ascii=False))

    msg = await llm.ainvoke([system, human])
    text = msg.content if isinstance(msg.content, str) else str(msg.content)

    print("\n=== RAW LLM OUTPUT ===")
    print(text)

    plan = parse_and_validate_plan(text)

    # si steps non vides => goal_reached false
    if plan.get("steps"):
        plan["goal_reached"] = False

    print("\n=== PARSED PLAN ===")
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    state["plan"] = plan
    return state


async def act_node(state: AgentState) -> AgentState:
    page: Page = state["memory"]["page"]
    plan = state["plan"]

    print("\n=== ACT NODE ===")
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    extracted_acc: List[Dict[str, Any]] = []
    last_error: Optional[str] = None
    failed_step: Optional[Dict[str, Any]] = None

    for step in plan.get("steps", []):
        out = await run_step(page, step)
        if out:
            if "error" in out:
                last_error = out["error"]
                failed_step = out.get("failed_step")
                print("\n!!! STEP FAILED => REPLAN !!!")
                print("Error:", last_error)
                print("Failed step:", failed_step)
                break
            extracted_acc.append(out)

    if extracted_acc:
        state["memory"]["extracted"] = state["memory"].get("extracted", []) + extracted_acc

    if last_error:
        state["memory"]["last_error"] = last_error
        state["memory"]["failed_step"] = failed_step
    else:
        state["memory"].pop("last_error", None)
        state["memory"].pop("failed_step", None)

    state["turns"] += 1
    state["url"] = page.url

    # stop condition simple: on s'arrête si on a extrait quelque chose et qu'il y a "get_by_role" dedans
    extracted_text = " ".join([e.get("body_text", "") for e in state["memory"].get("extracted", []) if isinstance(e, dict)]).lower()
    if "get_by_role" in extracted_text:
        state["done"] = True

    print("\n=== STATE AFTER ACT ===")
    print("done:", state["done"], "| turns:", state["turns"], "| url:", state["url"])

    return state


# -----------------------------
# 5) Build & run
# -----------------------------
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("observe", observe_node)
    g.add_node("plan", plan_node)
    g.add_node("act", act_node)

    g.set_entry_point("observe")
    g.add_edge("observe", "plan")
    g.add_edge("plan", "act")
    g.add_conditional_edges("act", should_continue)
    return g.compile()


async def final_answer(goal: str, final_url: str, extracted: List[Dict[str, Any]]) -> str:
    llm = ChatOpenAI(model="gpt-5-nano")
    extracted_tail = extracted[-2:] if len(extracted) > 2 else extracted

    system = SystemMessage(content=(
        "Tu es un assistant technique.\n"
        "Répond à l'objectif à partir des extraits.\n"
        "Sois concis.\n"
    ))

    human = HumanMessage(content=json.dumps({
        "goal": goal,
        "final_url": final_url,
        "extracted": extracted_tail
    }, ensure_ascii=False))

    msg = await llm.ainvoke([system, human])
    return msg.content if isinstance(msg.content, str) else str(msg.content)


async def run_agent(start_url: str, goal: str):
    graph = build_graph()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto(start_url, wait_until="domcontentloaded")

        init_state: AgentState = {
            "goal": goal,
            "url": page.url,
            "observation": {},
            "plan": {},
            "turns": 0,
            "done": False,
            "memory": {"page": page, "extracted": []},
        }

        final = await graph.ainvoke(init_state)

        print("\n==============================")
        print("Final URL:", final["url"])
        print("Turns:", final["turns"])

        extracted = final["memory"].get("extracted", [])
        print("Extracted items:", len(extracted))

        answer = await final_answer(goal=goal, final_url=final["url"], extracted=extracted)
        print("\nFINAL ANSWER (LLM):\n")
        print(answer)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(run_agent(
        start_url="https://playwright.dev",
        goal="Utilise le moteur de recherche de la page pour rechercher page.get_by_role()",
    ))
