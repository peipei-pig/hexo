'use strict';

// HTML articles keep their original files under source/. Their Hexo post supplies
// metadata and points to the public URL with `link` and `html_source`.
const fs = require('fs');
const path = require('path');
const { stripHTML } = require('hexo-util');

function articleFile(post) {
  const relative = String(post.html_source || '');
  const base = path.resolve(hexo.source_dir);
  const file = path.resolve(base, relative);
  if (!relative || !file.startsWith(base + path.sep) || !relative.endsWith('.html')) {
    throw new Error(`Invalid html_source: ${relative}`);
  }
  return file;
}
function readArticle(post) {
  return fs.readFileSync(articleFile(post), 'utf8');
}
function publicUrl(post) {
  const link = String(post.link || '');
  if (!/^\/[a-zA-Z0-9/_-]+\/$/.test(link)) throw new Error(`Invalid HTML article link: ${link}`);
  return hexo.config.root.replace(/\/$/, '') + link;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
}
hexo.extend.helper.register('html_article_symbols', function(post) {
  if (!post.html_source) return null;
  const html = readArticle(post);
  const main = html.match(/<main\b[^>]*>([\s\S]*?)<\/main>/i);
  const content = (main ? main[1] : html).replace(/<(script|style)\b[^>]*>[\s\S]*?<\/\1>/gi, '');
  return stripHTML(content).length;
});
hexo.extend.filter.register('after_generate', function() {
  hexo.locals.get('posts').forEach(post => {
    if (!post.html_source) return;
    const url = publicUrl(post);
    const route = String(post.link).replace(/^\//, '') + 'index.html';
    const original = readArticle(post);
    if (!/<\/head>/i.test(original) || !/<body\b[^>]*>/i.test(original)) {
      throw new Error(`HTML article needs head and body: ${post.html_source}`);
    }
    const root = hexo.config.root;
    const injected = original
      .replace(/<\/head>/i, `<link rel="canonical" href="${escapeHtml(hexo.config.url.replace(/\/$/, '') + url)}"><link rel="stylesheet" href="${escapeHtml(root)}css/html-article.css"></head>`)
      .replace(/<body\b[^>]*>/i, match => `${match}<div id="hexo-html-article" data-home="${escapeHtml(root)}" data-root="${escapeHtml(root)}"></div><script defer src="${escapeHtml(root)}js/html-article.js"></script>`);
    hexo.route.set(route, injected);
    // Keep the generated Hexo post URL working for old links and search results.
    if (post.path !== route) {
      const target = escapeHtml(url);
      hexo.route.set(post.path, `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="robots" content="noindex"><link rel="canonical" href="${target}"><meta http-equiv="refresh" content="0; url=${target}"><title>页面跳转</title></head><body><p>页面已迁移：<a href="${target}">打开文章</a></p></body></html>`);
    }
  });
});
