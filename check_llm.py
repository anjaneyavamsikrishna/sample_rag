"""Diagnose your LLM connection in one step.   Run: python check_llm.py"""
import config
from rag import llm

print("Settings read from .env (key is masked):")
for k in ("LLM_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
    print(f"  {k:16} = {config.describe()[k]}")

if not llm.is_configured():
    print("\nNo LLM configured: the key is empty or still 'your_api_key_here', and OPENAI_BASE_URL is empty.")
    print("The assistant will run in extractive mode (no generated answers). See Step 1 of Task 7 to add one.")
    raise SystemExit(0)

ADVICE = {
    "ModuleNotFoundError": "The 'openai' package is not installed in this environment. Run: pip install openai",
    "AuthenticationError": "The provider rejected your key. Check: no quotes or spaces around it in .env; it is a key for THIS "
                           "provider (a key for one service does not work on another's OPENAI_BASE_URL); it was not revoked.",
    "APIConnectionError": "Could not reach the server. If you use Ollama: is it running (open the Ollama app)? Is OPENAI_BASE_URL "
                          "exactly http://localhost:11434/v1 ? If you use OpenAI, you should leave OPENAI_BASE_URL EMPTY. "
                          "Also check internet, VPN or firewall.",
    "NotFoundError": "The model name is not available there. Check LLM_MODEL spelling. For Ollama, run: ollama pull <model>, "
                     "and use the same name in .env.",
    "RateLimitError": "Quota or rate limit hit. Often means no credit/billing on the account (error 'insufficient_quota'), "
                      "or too many requests. Check your provider's billing/usage page.",
    "PermissionDeniedError": "Your key is valid but not allowed to use this model or region. Try another model.",
    "BadRequestError": "The provider rejected the request. Often a wrong model name or an unsupported parameter.",
    "APITimeoutError": "The server did not answer in time. Try again, or use a smaller/faster model.",
}

print("\nSending a tiny test message ...")
try:
    reply = llm.chat("Reply with exactly one word.", "Say OK")
    print(f"SUCCESS - the model answered: {reply!r}\nYour LLM setup works.")
except Exception as exc:
    name = type(exc).__name__
    text = str(exc).replace(config.OPENAI_API_KEY, "***") if config.OPENAI_API_KEY else str(exc)
    print(f"FAILED  - {name}: {text[:400]}")
    print("\nWhat it usually means:", ADVICE.get(name, "Unrecognised error. Copy the line above and send it to your mentor."))
    print("\nOther common .env mistakes: file not named exactly '.env'; file not next to config.py; "
          "a leftover OPENAI_BASE_URL line from an earlier experiment.")
