"""
Client for the CohereX local LLM server (llama.cpp, Qwen2.5-3B-Instruct) —
for turning a board-meeting transcript into what a board secretary
(أمين سر مجلس الإدارة) actually needs from it: minutes, a decisions log,
action items with owners and deadlines, conflict-of-interest flags, or an
answer to a specific question.

Task design is scoped to assets/مهام أمين سر مجلس الإدارة.docx — the real
job description that document lays out spans a lot more (scheduling,
invitations, archiving, membership tracking...), but this agent only ever
sees a transcript, so the tasks below are the subset of that job that's
actually derivable from one. See docs/LLM_DEPLOYMENT.md's "Board Secretary
task scope" section for the full mapping of what's in-scope vs. not.

The LLM server and the ASR (vLLM) server can't run warm at the same time on
this box — not enough RAM for both. Start the LLM first:

    scripts/toggle_server.sh llm      # stops coherex-vllm, starts coherex-llm
    scripts/toggle_server.sh status   # check which one is up
    scripts/toggle_server.sh asr      # switch back when done

See docs/LLM_DEPLOYMENT.md for the full picture.

Usage:
    python llm_client.py --transcript out-ar/saudi_business_03min.txt --task summary
    python llm_client.py --transcript out-ar/saudi_business_03min.txt --task minutes_draft
    python llm_client.py --transcript out-ar/saudi_business_03min.txt --task decisions_log
    python llm_client.py --transcript out-ar/saudi_business_03min.txt --task action_items
    python llm_client.py --transcript out-ar/saudi_business_03min.txt --task conflicts_of_interest
    python llm_client.py --transcript out-ar/saudi_business_03min.txt --task qa \\
        --question "Which committee approves the largest investments?"
    python llm_client.py --prompt "Translate this to English: ..."

Server URL/key come from a local .env file (COHEREX_LLM_URL / COHEREX_LLM_API_KEY
— see .env.example) or --llm_url/--llm_api_key.
"""
import argparse
import os
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent

TASK_PROMPTS = {
    "summary": (
        "أنت تساعد أمين سر مجلس الإدارة. لخّص محضر الاجتماع التالي في "
        "3-5 جمل باللغة العربية، مع تغطية أهم المواضيع التي تمت مناقشتها."
        "\n\nنص المحضر:\n{text}"
    ),
    "minutes_draft": (
        "أنت أمين سر مجلس الإدارة، وتقوم بإعداد مسودة محضر اجتماع من النص "
        "أدناه، باللغة العربية. رتّب إجابتك في الأقسام التالية بالضبط، "
        "واكتب \"غير مذكور في النص\" لأي بند لم يُذكر — لا تخترع أي "
        "تفاصيل:\n\n"
        "1. الحضور / الاعتذارات (إن وُجد)\n"
        "2. وقت بداية ونهاية الاجتماع (إن وُجد)\n"
        "3. ملخص المناقشات (حسب الموضوع، بأسلوب مهني وواضح دون الإخلال "
        "بالمضمون)\n"
        "4. القرارات والتوصيات (اقتبس الصياغة الدقيقة أو أعد صياغتها "
        "بأمانة — لا تخفف الصياغة أو تغيّر المعنى)\n"
        "5. نتائج التصويت (إن وُجدت)\n"
        "6. التحفظات / الاعتراضات المسجّلة من أي عضو (إن وُجدت)\n"
        "7. الإفصاح عن تعارض المصالح أو الامتناع عن التصويت (إن وُجد)\n\n"
        "نص المحضر:\n{text}"
    ),
    "decisions_log": (
        "استخرج كل قرار أو توصية صدرت عن المجلس في النص التالي، باللغة "
        "العربية، على شكل قائمة مرقّمة. لكل قرار، اقتبس الصياغة الدقيقة أو "
        "أعد صياغتها بأمانة كما وردت — لا تخفف الصياغة أو تعمّم أو تضف "
        "قرارات لم تُتخذ فعليًا. أدرج نتيجة التصويت إن ذُكرت لذلك القرار. "
        "إذا لم تُتخذ أي قرارات، اذكر ذلك صراحةً بدلاً من اختلاق أي شيء."
        "\n\nنص المحضر:\n{text}"
    ),
    "action_items": (
        "استخرج الإجراءات المطلوبة (Action Items) من النص التالي، باللغة "
        "العربية، على شكل قائمة. لكل إجراء اذكر بالضبط هذه الحقول الثلاثة:\n"
        "- الإجراء: ما المطلوب تنفيذه\n"
        "- المسؤول: من هو المسؤول عن التنفيذ (الاسم/الدور/المتحدث إن ذُكر، "
        "وإلا اكتب \"غير محدد\")\n"
        "- الموعد المستهدف: متى يجب إنجازه (إن ذُكر، وإلا اكتب \"غير "
        "محدد\")\n\n"
        "إذا لم يتضمن النص أي إجراءات مطلوبة، اذكر ذلك بدلاً من اختلاق أي "
        "شيء.\n\nنص المحضر:\n{text}"
    ),
    "conflicts_of_interest": (
        "راجع النص التالي بحثًا عن أي إشارة إلى تعارض مصالح، أو إفصاح أحد "
        "أعضاء المجلس عن مصلحة له، أو امتناع/استبعاد عضو من المناقشة أو "
        "التصويت على أحد البنود. اذكر كل حالة تجدها، باللغة العربية، مع "
        "تحديد من كان معنيًا وما هو الموضوع. إذا لم تُذكر أي حالة، اذكر "
        "ذلك صراحةً بدلاً من اختلاق أي شيء.\n\nنص المحضر:\n{text}"
    ),
    "qa": (
        "أجب عن هذا السؤال بالاعتماد فقط على النص أدناه، باللغة العربية. "
        "إذا لم تكن الإجابة موجودة في النص، فاذكر ذلك بدلاً من التخمين."
        "\n\nالسؤال: {question}\n\nنص المحضر:\n{text}"
    ),
}


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, no external dependency. Existing
    environment variables always win over the file."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def ask(base_url: str, api_key: str, prompt: str, max_tokens: int, temperature: float) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    with httpx.Client(timeout=180.0) as client:
        response = client.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcript", type=Path, help="path to a transcript .txt file (e.g. from main.py's api-test-out/ or the CLI's out-ar/)")
    parser.add_argument("--prompt", type=str, help="raw prompt to send instead of --transcript + --task")
    parser.add_argument("--task", choices=list(TASK_PROMPTS), default="summary", help="what to do with --transcript")
    parser.add_argument("--question", help="required when --task qa")
    parser.add_argument("--llm_url", default=os.environ.get("COHEREX_LLM_URL"), help="LLM server URL (or set COHEREX_LLM_URL / .env)")
    parser.add_argument("--llm_api_key", default=os.environ.get("COHEREX_LLM_API_KEY"), help="LLM server API key (or set COHEREX_LLM_API_KEY / .env)")
    parser.add_argument("--max_tokens", type=int, default=800, help="raise this for --task minutes_draft on longer meetings (its 7 sections can run past the default)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--output", type=Path, help="save the response text to this file (also always printed)")
    args = parser.parse_args()

    if not args.llm_url:
        parser.error("--llm_url is required (set it in .env, export COHEREX_LLM_URL, or pass --llm_url)")
    if not args.llm_api_key:
        parser.error("--llm_api_key is required (set it in .env, export COHEREX_LLM_API_KEY, or pass --llm_api_key)")

    if args.prompt:
        prompt = args.prompt
    else:
        if not args.transcript:
            parser.error("pass --transcript (with --task) or --prompt")
        if not args.transcript.is_file():
            parser.error(f"transcript not found: {args.transcript}")
        if args.task == "qa" and not args.question:
            parser.error("--task qa requires --question")
        text = args.transcript.read_text(encoding="utf-8")
        prompt = TASK_PROMPTS[args.task].format(text=text, question=args.question)

    print(f"Asking {args.llm_url} ...")
    result = ask(args.llm_url, args.llm_api_key, prompt, args.max_tokens, args.temperature)
    print("\n" + result)

    if args.output:
        args.output.write_text(result, encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
