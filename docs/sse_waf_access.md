# 绕过上交所 PDF 直链的 WAF 挑战（已解决）

`static.sse.com.cn` 的公告 PDF 直链受**阿里云 WAF** 保护，直接请求拿不到 PDF。
本文记录完整的现象、解决过程与结论。实现见 `src/ingest/download_reports.py`
的 `WafSession` 类与 `scripts/waf/sse_cookie_solver.js`。

## 现象

直接请求 PDF 直链：

| 观测 | 值 |
|---|---|
| HTTP 状态 | `200 OK`（**不是 403**，容易被误判为成功） |
| Content-Type | `text/html; charset=utf-8` |
| 响应体大小 | 7373 字节，全是混淆 JS |
| 响应头 | `X-Tengine-Error: denied by bot` |
| Set-Cookie | `acw_tc=…`、`cdn_sec_tc=…`（均 HttpOnly, Max-Age=3600） |

响应体解码后可读到 `acw_sc__v2=`、`;max-age=3600;path=/`、
`3000176000856006061501533003690027800375` 等特征串 —— 即 **`acw_sc__v2` JS Cookie 挑战**。

## 解法（三步）

```
1. 请求 PDF 直链
   -> 200 + text/html 挑战页；从 Set-Cookie 取 acw_tc
   -> 从响应体取 arg1（形如 arg1='8C88C7ECF140...'）

2. node 求解器算出 acw_sc__v2
   -> 脚本在 scripts/waf/sse_cookie_solver.js（依赖 jsdom）

3. 带上 acw_tc + acw_sc__v2 重新请求
   -> 200 + application/pdf  ← 真正的 PDF
```

**关键：arg1 每次请求都不同**，所以 cookie 不能写死，必须用同一次响应里的 arg1
配同一次下发的 `acw_tc`。

cookie 的 `Max-Age=3600`，**解一次可复用到整批下载**（实测：同一对 cookie 拉下
多个不同 PDF 均成功）。`WafSession` 因此只解一次，并把 cookie 缓存 3000 秒。

## 排除掉的方案（都试过，无效）

| 方案 | 结果 |
|---|---|
| 加 `Referer: https://www.sse.com.cn/` | 无效 |
| 加全套浏览器请求头（Accept / Accept-Language / …） | 无效 |
| `curl_cffi` 伪装 Chrome TLS 指纹（JA3） | 无效（chrome/chrome124/safari17_0/edge101 全试过） |
| 先访问 `www.sse.com.cn` 主站建会话取 cookie | 主站不下发任何 cookie，此路不通 |

说明 WAF 判定依据既不是请求头也不是 TLS 指纹，而是 **JS 挑战本身**。

## 求解器的三个坑

写 `scripts/waf/sse_cookie_solver.js` 时踩到的，记录备查：

1. **不能用 `vm.createContext` 沙箱**：挑战代码用 `Function('return this')()` 取真实
   全局对象，会绕过沙箱 —— 所以改用 jsdom，让代码跑在真实 window 里。
   早期手写 DOM 桩卡在 `Cannot read properties of undefined (reading 'split')`，
   补桩是打地鼠，直接上 jsdom 才干净。
2. **响应体是 `<html><script>…</script></html>`**，整份当 JS 跑会挂在 `<html` 上，
   必须先剥出 script 标签。
3. **`subprocess` 必须显式 `encoding="utf-8"`**：中文 Windows 下默认用 GBK 解码，
   node 的 stderr（jsdom 会打 `Not implemented: navigation to another Document`）
   一旦含非 GBK 字节就抛 `UnicodeDecodeError`，且异常在读取线程里，非常隐蔽。

## 另一个独立的坑：冷连接随机卡死

`static.sse.com.cn` 的**新建连接**会随机卡死。实测同一 URL 连续四次：
`25.0s 超时` / `18.4s 成功` / `1.7s 成功` / `0.4s 成功`。

**连接一旦建立并复用就稳定在 0.4s 左右。** 因此下载器：

- 全部请求共用一个 `requests.Session`，靠连接复用避开冷连接
- 建连阶段单独用 15 秒短超时重试 4 次（`CONNECT_ATTEMPTS`）
- 读取阶段仍用要求的 60 秒（`timeout=(15, 60)` 二元组）
- `pool_maxsize` 从默认 10 放大到 20 以容纳 6 线程

不处理这一点的话，一次冷连接卡顿就能吃掉 20-60 秒，甚至直接超时失败。
