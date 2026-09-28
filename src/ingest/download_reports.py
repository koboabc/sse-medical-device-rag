"""按 data/manifest.csv 批量下载 2026 年半年度报告 PDF。

覆盖的下载要求：
  1. 只下载完整半年报（manifest 已只含完整报告，脚本再断言一次）
  2. 6 线程并发
  3. 单请求超时 60 秒
  4. 失败自动重试 3 次（指数退避）
  5. 断点续传（HTTP Range，写 .part 文件，续传后校验再改名）
  6. 已存在且完整的 PDF 直接跳过
  7. 下载后校验：PDF 文件头 + 文件大小 + pymupdf 可解析
  8. 单个文件失败不中断整体（异常隔离在单任务内）
  9. 成功 -> manifest download_status=success
 10. 失败 -> manifest download_status=failed 并记录 error

关于 WAF：static.sse.com.cn 有阿里云 WAF 的 acw_sc__v2 JS 挑战，
直接请求会拿到 HTML 而不是 PDF。本脚本先"解一次挑战拿 cookie 对"，再复用到全部
下载（cookie Max-Age=3600，够跑完整批）。详见 docs/download_blocker.md。

用法：
    python src/ingest/download_reports.py
    python src/ingest/download_reports.py --workers 6 --limit 2   # 先试 2 家
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config  # noqa: E402

# ---- 常量 ----
WAF_SOLVER = config.root / "scripts" / "waf" / "sse_cookie_solver.js"
PDF_MAGIC = b"%PDF-"
MIN_PDF_BYTES = 20 * 1024          # 半年报不可能小于 20KB，小于即视为异常
CHUNK = 1 << 16
RETRIES = 3                        # 总计尝试 3 次
TIMEOUT = 60                       # 秒（单文件下载，按要求 60 秒）
COOKIE_TTL = 3000                  # 秒；WAF cookie Max-Age=3600，留余量提前重解

# static.sse.com.cn 的**新建连接**会随机卡死（实测同一 URL：25s 超时 / 18.4s /
# 1.7s / 0.4s），但连接一旦建立并复用就稳定在 0.4s 左右。所以建连阶段单独用短超时
# 多试几次，别让一次冷连接挂满 60 秒拖垮整批。
CONNECT_ATTEMPTS = 4
CONNECT_TIMEOUT = 15

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.sse.com.cn/disclosure/listedinfo/announcement/",
}

# manifest 中要新增/更新的列
EXTRA_FIELDS = ["file_size_bytes", "sha256", "downloaded_at", "attempts", "error"]


class WafBlocked(RuntimeError):
    """请求被 WAF 挑战页拦截（拿到的是 HTML 而不是 PDF）。"""


class CriticalError(RuntimeError):
    """环境级错误（如 node 不可用），重试无意义，应立即中止本任务。"""


# ============================================================
# WAF cookie 管理
# ============================================================
class WafSession:
    """维护一份可复用的 WAF cookie 对（acw_tc + acw_sc__v2）。

    挑战流程（实测）：
      1. 直接请求 PDF -> 200 但 Content-Type 是 text/html，
         Set-Cookie 下发 acw_tc / cdn_sec_tc，响应体是含 arg1 的挑战 JS
      2. 用**同一次**响应里的 arg1 跑 node 求解器算出 acw_sc__v2
      3. 带上 acw_tc + acw_sc__v2 再请求 -> 真正的 application/pdf
    注意 arg1 每次请求都变，所以不能把 cookie 写死。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cookies: dict[str, str] = {}
        self._solved_at = 0.0
        self._session = requests.Session()
        self._session.headers.update(BROWSER_HEADERS)
        # 6 线程共用这一个 Session，靠连接复用避开冷连接卡顿；
        # 默认 pool_maxsize=10 偏小，放大一些
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10, pool_maxsize=20, max_retries=0,
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._check_node()

    def _get_with_connect_retry(
        self, url: str, read_timeout: int | None = None, **kw
    ) -> requests.Response:
        """建连阶段的重试。

        冷连接会随机卡死，用短超时多试几次绕开。read_timeout 给定时用
        requests 的 (connect, read) 二元组超时：连接阶段用 CONNECT_TIMEOUT，
        读取阶段用调用方指定的值（下载用 60 秒）。
        """
        timeout = (
            (CONNECT_TIMEOUT, read_timeout) if read_timeout else CONNECT_TIMEOUT
        )
        last: Exception | None = None
        for i in range(1, CONNECT_ATTEMPTS + 1):
            try:
                return self._session.get(url, timeout=timeout, **kw)
            except (requests.Timeout, requests.ConnectionError) as e:
                last = e
                if i < CONNECT_ATTEMPTS:
                    wait = 1.5 * i
                    print(f"    建连第 {i}/{CONNECT_ATTEMPTS} 次超时，{wait:.1f}s 后重试")
                    time.sleep(wait)
        raise last if last else RuntimeError("建连失败")

    @staticmethod
    def _check_node() -> None:
        if shutil.which("node") is None:
            raise CriticalError(
                "找不到 node —— 解 WAF 挑战需要它。"
                f"求解器：{WAF_SOLVER}（依赖 jsdom，见 scripts/waf/package.json）"
            )
        if not WAF_SOLVER.exists():
            raise CriticalError(f"找不到 WAF 求解器 {WAF_SOLVER}")

    @staticmethod
    def _is_challenge(resp: requests.Response) -> bool:
        """只看 Content-Type。

        刻意不去读 resp.raw 探文件头 —— 那会把流的头几个字节吃掉，
        导致后续 iter_content 拿到残缺内容。Content-Type 缺失时按「非挑战」
        处理，交给最终 verify_pdf 兜底。
        """
        return "html" in (resp.headers.get("Content-Type") or "").lower()

    def _solve(self, probe_url: str) -> None:
        """请求一次 probe_url，从挑战页算出 cookie 对并缓存。"""
        r = self._get_with_connect_retry(probe_url, allow_redirects=True)
        if not self._is_challenge(r):
            # 没被挑战，直接用会话里已有的 cookie
            self._cookies = {c.name: c.value for c in self._session.cookies}
            self._solved_at = time.time()
            return

        acw_tc = r.cookies.get("acw_tc") or ""
        if not acw_tc:
            for h in r.raw.headers.get_all("Set-Cookie") or []:
                m = re.search(r"acw_tc=([^;]+)", h)
                if m:
                    acw_tc = m.group(1)
                    break
        if not acw_tc:
            raise WafBlocked("挑战响应里没有 acw_tc cookie，无法配对")

        with tempfile.TemporaryDirectory() as td:
            chal = Path(td) / "challenge.html"
            chal.write_bytes(r.content)
            # 必须显式指定 utf-8：中文 Windows 下 subprocess 默认用 GBK 解码，
            # node 的 stderr（jsdom 会打 "Not implemented: navigation..."）一旦
            # 含非 GBK 字节就抛 UnicodeDecodeError，而且异常发生在读取线程里，
            # 排查起来很隐蔽。
            proc = subprocess.run(
                ["node", str(WAF_SOLVER), str(chal)],
                capture_output=True, encoding="utf-8", errors="replace",
                timeout=60, cwd=str(WAF_SOLVER.parent),
            )
        cookie_str = (proc.stdout or "").strip()
        if proc.returncode != 0 or not cookie_str:
            raise WafBlocked(
                f"求解器失败 rc={proc.returncode} "
                f"stderr={proc.stderr.strip()[:200]}"
            )

        self._cookies = {"acw_tc": acw_tc}
        for part in cookie_str.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                self._cookies[k.strip()] = v.strip()
        self._solved_at = time.time()

    def _ensure_fresh(self, probe_url: str) -> None:
        with self._lock:
            if self._cookies and (time.time() - self._solved_at) < COOKIE_TTL:
                return
            self._solve(probe_url)

    def request(self, url: str, probe_url: str, **kw) -> requests.Response:
        """发请求；若被 WAF 挑战就重解 cookie 再试一次。"""
        self._ensure_fresh(probe_url)
        with self._lock:
            cookies = dict(self._cookies)
        r = self._get_with_connect_retry(
            url, read_timeout=TIMEOUT, cookies=cookies,
            stream=True, allow_redirects=True, **kw,
        )
        if self._is_challenge(r):
            r.close()
            with self._lock:
                self._cookies = {}          # 强制重解
            self._ensure_fresh(probe_url)
            with self._lock:
                cookies = dict(self._cookies)
            r = self._get_with_connect_retry(
                url, read_timeout=TIMEOUT, cookies=cookies,
                stream=True, allow_redirects=True, **kw,
            )
            if self._is_challenge(r):
                r.close()
                raise WafBlocked("重解 cookie 后仍被 WAF 挑战拦截")
        return r


# ============================================================
# 单文件下载
# ============================================================
@dataclass
class Result:
    stock_code: str
    status: str                      # success / failed / skipped
    size: int = 0
    attempts: int = 0
    error: str = ""
    sha256: str = ""
    elapsed: float = 0.0


def verify_pdf(path: Path) -> tuple[bool, str, int, str]:
    """校验 PDF：文件头 + 大小 + 能否被 pymupdf 解析。返回 (ok, err, size, sha256)。"""
    if not path.exists():
        return False, "文件不存在", 0, ""
    size = path.stat().st_size
    if size == 0:
        return False, "文件为空", 0, ""
    with path.open("rb") as f:
        head = f.read(5)
    if head[:4] != PDF_MAGIC[:4]:
        return False, f"文件头不是 PDF（前 5 字节 {head!r}）", size, ""
    if size < MIN_PDF_BYTES:
        return False, f"文件过小（{size} 字节 < {MIN_PDF_BYTES}）", size, ""
    try:
        import pymupdf

        doc = pymupdf.open(path)
        n = doc.page_count
        doc.close()
        if n <= 0:
            return False, "pymupdf 解析出 0 页", size, ""
    except Exception as e:  # noqa: BLE001
        return False, f"pymupdf 无法解析: {type(e).__name__}: {e}", size, ""

    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return True, "", size, h


def download_one(row: dict, waf: WafSession, probe_url: str) -> Result:
    """下载单个 PDF，带重试、断点续传、校验。异常全部内部消化。"""
    code = row["stock_code"]
    dest = (config.root / row["local_path"]).resolve()
    url = row["pdf_url"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    t0 = time.monotonic()

    # 要求 6：已存在且完整 -> 跳过
    if dest.exists():
        ok, err, size, sha = verify_pdf(dest)
        if ok:
            return Result(code, "skipped", size, 0, "", sha, time.monotonic() - t0)
        print(f"  [{code}] 已存在但校验不过（{err}），重新下载")
        dest.unlink()

    # 残留的 .part 若不是 PDF 开头（例如上次进程被杀时写进了 WAF 挑战页），
    # 续传只会把垃圾越续越长，必须先清掉
    if part.exists():
        with part.open("rb") as f:
            if f.read(4) != PDF_MAGIC[:4]:
                print(f"  [{code}] 发现无效的 .part 残留，清除后重新下载")
                part.unlink()

    last_err = ""
    for attempt in range(1, RETRIES + 1):
        try:
            pos = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={pos}-"} if pos else {}

            r = waf.request(url, probe_url, headers=headers)
            with r:
                if pos and r.status_code == 200:
                    # 服务端不支持续传，从头写
                    pos = 0
                elif pos and r.status_code == 206:
                    pass
                elif not pos and r.status_code == 200:
                    pass
                else:
                    raise RuntimeError(f"意外状态码 {r.status_code}")

                mode = "ab" if pos else "wb"
                with part.open(mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)

            ok, err, size, sha = verify_pdf(part)
            if not ok:
                raise RuntimeError(f"校验失败: {err}")
            part.replace(dest)          # 校验通过才改名，保证 dest 永远是完整文件
            return Result(code, "success", size, attempt, "", sha,
                          time.monotonic() - t0)

        except CriticalError:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            wait = 2 ** attempt
            print(f"  [{code}] 第 {attempt}/{RETRIES} 次失败：{last_err}"
                  + (f"，{wait}s 后重试" if attempt < RETRIES else ""))
            if attempt < RETRIES:
                time.sleep(wait)

    # 全部重试失败：清理半成品，避免下次误判为"可续传"
    if part.exists():
        try:
            part.unlink()
        except OSError:
            pass
    return Result(code, "failed", 0, RETRIES, last_err, "", time.monotonic() - t0)


# ============================================================
# 主流程
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6, help="并发线程数（要求 6）")
    ap.add_argument("--limit", type=int, default=0, help="只下前 N 家，用于试跑")
    args = ap.parse_args()

    config.ensure_dirs()
    manifest_path = config.data_dir / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())
    for c in EXTRA_FIELDS:
        if c not in fields:
            fields.append(c)
    for r in rows:
        for c in EXTRA_FIELDS:
            r.setdefault(c, "")

    # 要求 1：断言 manifest 里只有完整半年报，没有摘要
    bad = [r["stock_code"] for r in rows
           if "摘要" in r["title"] or r["bulletin_type"] != "半年报"]
    if bad:
        print(f"[FAIL] manifest 含非完整半年报: {bad}")
        return 1
    print(f"manifest: {len(rows)} 家，均为完整半年报（无摘要混入）")

    targets = rows[: args.limit] if args.limit else rows
    if args.limit:
        print(f"[--limit] 本次只处理前 {len(targets)} 家")

    print(f"并发 {args.workers} 线程 / 超时 {TIMEOUT}s / 最多 {RETRIES} 次尝试 "
          f"/ 目标目录 {config.pdf_dir}\n")

    t_start = time.monotonic()
    try:
        t_waf = time.monotonic()
        waf = WafSession()
        probe = targets[0]["pdf_url"]
        print("解 WAF 挑战（acw_sc__v2）...")
        waf._ensure_fresh(probe)
        print(f"  cookie 就绪：{sorted(waf._cookies)}"
              f"（耗时 {time.monotonic() - t_waf:.1f}s）\n")
    except CriticalError as e:
        print(f"[FAIL] {e}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] WAF 初始化失败：{type(e).__name__}: {e}")
        return 1

    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, row, waf, probe): row["stock_code"]
            for row in targets
        }
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                res = fut.result()          # 要求 8：单任务异常在此隔离
            except CriticalError as e:
                print(f"[FAIL] 环境级错误，中止：{e}")
                return 1
            except Exception as e:  # noqa: BLE001
                res = Result(code, "failed", 0, RETRIES,
                             f"{type(e).__name__}: {e}")
            results.append(res)
            mark = {"success": "✓", "skipped": "=", "failed": "✗"}[res.status]
            extra = (f"{res.size/1024/1024:.2f}MB {res.elapsed:.1f}s"
                     if res.status == "success" else
                     (f"{res.size/1024/1024:.2f}MB" if res.status == "skipped"
                      else res.error))
            print(f"  [{code}] {mark} {res.status:<8} {extra}")

    elapsed = time.monotonic() - t_start

    # ---- 回写 manifest ----
    by_code = {r.stock_code: r for r in results}
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    for r in rows:
        res = by_code.get(r["stock_code"])
        if res is None:
            continue                        # --limit 未处理的保持原状
        if res.status == "success":
            r["download_status"] = "success"
            r["error"] = ""
        elif res.status == "failed":
            r["download_status"] = "failed"
            r["error"] = res.error
        else:                               # skipped：已是完整文件，视为成功
            r["download_status"] = "success"
            r["error"] = ""
        r["file_size_bytes"] = str(res.size)
        r["sha256"] = res.sha256
        r["downloaded_at"] = now
        r["attempts"] = str(res.attempts)

    for c in EXTRA_FIELDS:
        if c not in fields:
            fields.append(c)

    # 备份 + 写临时文件 + 回读断言 + 原子替换
    backup = manifest_path.with_suffix(".csv.bak")
    shutil.copy2(manifest_path, backup)
    tmp = manifest_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with tmp.open("r", encoding="utf-8-sig", newline="") as f:
        back = list(csv.DictReader(f))
    assert len(back) == len(rows), f"回读行数不一致 {len(back)} != {len(rows)}"
    for a, b in zip(rows, back):
        assert a["stock_code"] == b["stock_code"], "行序错乱"
        assert a["download_status"] == b["download_status"], "状态未写入"
    tmp.replace(manifest_path)

    # ---- 汇总 ----
    n_ok = sum(1 for r in results if r.status == "success")
    n_skip = sum(1 for r in results if r.status == "skipped")
    n_fail = sum(1 for r in results if r.status == "failed")
    total_bytes = sum(r.size for r in results if r.status in ("success", "skipped"))

    print("\n" + "=" * 62)
    print("下载汇总")
    print("=" * 62)
    print(f"  公司总数    : {len(rows)}")
    print(f"  PDF 总数    : {len(targets)}")
    print(f"  成功        : {n_ok}")
    print(f"  失败        : {n_fail}")
    print(f"  跳过        : {n_skip}")
    print(f"  总大小      : {total_bytes/1024/1024:.2f} MB ({total_bytes} 字节)")
    print(f"  总耗时      : {elapsed:.1f} 秒")
    if n_fail:
        print("\n  失败明细：")
        for r in results:
            if r.status == "failed":
                print(f"    [{r.stock_code}] {r.error}")
    print(f"\n  manifest 已回写（备份 {backup.name}）")
    print("=" * 62)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
