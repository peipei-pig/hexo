# 发布独立 HTML 文章

1. 将完整 HTML 文件放在 `source/papers/<文章名>/index.html`。文件需包含 `head` 和 `body`；建议用 `main` 包住正文，以便准确估算阅读时间。页面中的相对资源需一并放在对应目录。
2. 在 `source/_posts/` 创建 Markdown 入口，填写元数据，例如：

   ```yaml
   ---
   title: 文章标题
   date: 2026-11-07 12:00:00
   categories:
     - 推理
   tags:
     - 论文阅读
   description: 文章简介
   link: /papers/example/
   html_source: papers/example/index.html
   ---

   文章简介，显示在首页。
   ```

   `link` 对应发布地址，`html_source` 是相对于 `source/` 的 HTML 文件路径。两者的文章目录应保持一致，不要与其他页面使用相同地址。每篇文章应使用唯一的目录。
3. 执行 `npm run build`，用 `npm run server` 本地预览。首页、归档、分类和标签的标题直接进入完整 HTML；原 Hexo 文章地址保留为兼容跳转。
4. 按站点原有流程部署。发布前应检查图示、相对资源和手机端效果。

构建时会为独立页面加入返回博客、阅读时间和 Vercount 浏览量。阅读时间按正文可见文本长度估算，并非实际停留时间；浏览量依赖 Vercount 和访客浏览器加载脚本。HTML 原始文件不会被改写。对已有 HTML 的修改可直接重新构建；若新增或修改 Hexo 配置后结果未刷新，执行 `npm run clean` 后再构建。
