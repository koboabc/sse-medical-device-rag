// 解上交所 static 域的阿里云 WAF 挑战（acw_sc__v2），输出可用 Cookie 头。
//
// 为什么用 jsdom 而不是手写 DOM 桩：挑战 JS 是混淆过的，需要 document / navigator /
// location / atob / Date 等一整套浏览器对象，手写桩会陷入打地鼠（此前卡在
// "Cannot read properties of undefined (reading 'split')"）。jsdom 提供真实 DOM。
//
// 另外两个坑（都已处理）：
//   1. 挑战代码用 Function('return this')() 取全局对象，会绕过 vm.createContext
//      的沙箱 —— 所以不能用 vm，必须让代码跑在 jsdom 的 window 里。
//   2. 响应体是 <html><script>…</script></html>，整份当 JS 跑会挂在 '<html' 上。
//
// 用法：node sse_cookie_solver.js <challenge.html>
// 输出：stdout 打印可直接用于 Cookie 头的字符串

const fs = require('fs');
const { JSDOM, CookieJar } = require('jsdom');

const html = fs.readFileSync(process.argv[2], 'utf8');

const dom = new JSDOM(html, {
  url: 'https://www.sse.com.cn/disclosure/listedinfo/announcement/',
  referrer: 'https://www.sse.com.cn/',
  runScripts: 'dangerously',   // 关键：执行内联脚本
  pretendToBeVisual: true,
  cookieJar: new CookieJar(),
});

// 挑战脚本是同步内联的，解析完即已执行；给微任务一点机会
setTimeout(() => {
  const cookie = dom.window.document.cookie || '';
  process.stdout.write(cookie);
}, 50);
