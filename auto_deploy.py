"""一键完成心血管疾病微服务系统的版本控制与提交。

流程：写 .gitignore -> git init -> 配置 origin -> 创建/切换分支 -> add+commit -> push。
所有 git 命令通过 subprocess 执行，输出逐行实时打印。

用法（在项目根目录下）：
    python auto_deploy.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
REPO_URL = "https://github.com/hotenglish/itcast-assignment-1.git"
BRANCH = "featstudent03_01"
COMMIT_MESSAGE = "feat 完成心血管疾病微服务系统(分析/预测/语音)"

GITIGNORE_LINES = (".env", "__pycache__/", "*.model", "*.xlsx", ".DS_Store")

NOTHING_TO_COMMIT = ("nothing to commit", "nothing added to commit", "无文件要提交")


class DeployError(RuntimeError):
    """可预期的流程失败，统一在 main 中收敛为退出码 1。"""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise DeployError(message)


def decode(raw: bytes) -> str:
    for encoding in ("utf-8", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def run(args: list[str]) -> tuple[int, list[str]]:
    """执行 git 子命令，边跑边打印，返回 (退出码, 输出行)。

    stdin 指向 DEVNULL：缺少凭据时 git 会直接失败，而不是挂起等待输入。
    """
    print(f"$ git {' '.join(args)}", flush=True)
    process = subprocess.Popen(
        ["git", *args],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for raw in process.stdout:
        line = decode(raw).rstrip("\r\n")
        lines.append(line)
        print(f"  {line}", flush=True)
    return process.wait(), lines


# ---------------------------------------------------------------- 各步骤


def ensure_gitignore() -> None:
    path = REPO_ROOT / ".gitignore"
    existing = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    present = {line.strip() for line in existing}
    missing = [line for line in GITIGNORE_LINES if line not in present]
    if not missing:
        print(f"[.gitignore] 已包含全部 {len(GITIGNORE_LINES)} 条规则，跳过", flush=True)
        return
    path.write_text("\n".join([*existing, *missing]) + "\n", encoding="utf-8")
    print(f"[.gitignore] 新增规则：{', '.join(missing)}", flush=True)


def is_repo() -> bool:
    code, _ = run(["rev-parse", "--is-inside-work-tree"])
    return code == 0


def ensure_repo() -> None:
    if is_repo():
        print("[init] 已是 git 仓库，跳过初始化", flush=True)
        return
    code, _ = run(["init"])
    require(code == 0, "git init 失败")
    print("[init] 已初始化仓库", flush=True)


def current_origin() -> str | None:
    code, lines = run(["remote", "get-url", "origin"])
    return lines[0].strip() if code == 0 and lines else None


def ensure_remote(url: str) -> None:
    origin = current_origin()
    if origin is None:
        code, _ = run(["remote", "add", "origin", url])
        require(code == 0, "添加 origin 失败")
        print(f"[remote] 已添加 origin -> {url}", flush=True)
    elif origin != url:
        code, _ = run(["remote", "set-url", "origin", url])
        require(code == 0, "修改 origin 地址失败")
        print(f"[remote] origin 已更新：{origin} -> {url}", flush=True)
    else:
        print("[remote] origin 已指向目标地址，跳过", flush=True)


def ensure_identity() -> None:
    missing = [key for key in ("user.name", "user.email") if not run(["config", key])[1]]
    if not missing:
        return
    raise DeployError(
        "缺少 git 身份配置："
        + "、".join(missing)
        + '\n请先执行：\n  git config --global user.name "你的名字"\n'
        + '  git config --global user.email "你的邮箱"'
    )


def branch_exists(branch: str) -> bool:
    code, _ = run(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"])
    return code == 0


def ensure_branch(branch: str) -> None:
    if branch_exists(branch):
        code, _ = run(["switch", branch])
        require(code == 0, f"切换到 {branch} 失败")
        print(f"[branch] 已切换到已有分支 {branch}", flush=True)
        return
    code, _ = run(["switch", "-c", branch])
    require(code == 0, f"创建分支 {branch} 失败")
    print(f"[branch] 已创建并切换到 {branch}", flush=True)


def commit_all(message: str) -> None:
    code, _ = run(["add", "-A"])
    require(code == 0, "git add 失败")

    code, lines = run(["commit", "-m", message])
    if code == 0:
        print("[commit] 提交完成", flush=True)
        return
    if any(marker in line for line in lines for marker in NOTHING_TO_COMMIT):
        print("[commit] 没有新的改动，跳过提交", flush=True)
        return
    raise DeployError("git commit 失败")


def push(branch: str) -> None:
    code, _ = run(["push"])
    if code == 0:
        print("[push] 推送成功", flush=True)
        return

    print("[push] 首次推送失败，重试并设置 upstream ...", flush=True)
    code, _ = run(["push", "--set-upstream", "origin", branch])
    require(code == 0, "推送失败，请检查 GitHub 凭据（PAT）或远端分支冲突")
    print("[push] 推送成功（已设置 upstream）", flush=True)


# ---------------------------------------------------------------- 编排


def main() -> int:
    print(f"仓库根目录：{REPO_ROOT}", flush=True)
    print(f"目标远端：{REPO_URL}", flush=True)
    print(f"目标分支：{BRANCH}", flush=True)
    print("", flush=True)

    try:
        print("==> [1/6] 写入 .gitignore", flush=True)
        ensure_gitignore()
        print("==> [2/6] 检查仓库", flush=True)
        ensure_repo()
        print("==> [3/6] 配置 origin", flush=True)
        ensure_remote(REPO_URL)
        print("==> [4/6] 检查提交身份", flush=True)
        ensure_identity()
        print("==> [5/6] 切换分支并提交", flush=True)
        ensure_branch(BRANCH)
        commit_all(COMMIT_MESSAGE)
        print("==> [6/6] 推送", flush=True)
        push(BRANCH)
    except DeployError as exc:
        print(f"\n部署失败：{exc}", file=sys.stderr, flush=True)
        return 1

    print("\n部署完成。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())