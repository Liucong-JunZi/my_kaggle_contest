"""解析 Kaggle discussion topic 的 JSON 输出"""
import json, re, html

with open('/tmp/topic_699853.txt') as f:
    raw = f.read()

# 找每一行的 JSON
lines = raw.strip().split('\n')
# 跳过 header 行
for line in lines:
    if not line.strip() or line.startswith('---') or line.startswith('     id'):
        continue
    # 尝试匹配 JSON-like content
    # kaggle CLI 输出用双空格分隔列
    # 最后一列是 content (HTML)
    # 找到第一个 <p 或 < 的位置
    idx = line.find(' <')
    if idx < 0:
        continue
    content = line[idx:].strip()
    # 解码
    content = html.unescape(content)
    # 替换 HTML
    content = content.replace('<p>', '\n').replace('</p>', '\n')
    content = content.replace('<pre><code>', '\n```\n').replace('</code></pre>', '\n```\n')
    content = content.replace('<br>', '\n').replace('<br/>', '\n')
    content = content.replace('<ul>', '').replace('</ul>', '')
    content = content.replace('<li>', '  • ').replace('</li>', '\n')
    content = content.replace('<strong>', '**').replace('</strong>', '**')
    content = content.replace('<em>', '*').replace('</em>', '*')
    content = content.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
    # 移除 IMG
    content = re.sub(r'<img[^>]*>', '[IMG]', content)
    # 移除其他 HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    # 清理多余空行
    content = re.sub(r'\n{3,}', '\n\n', content)
    if content.strip():
        print(content)
        print('\n' + '='*70 + '\n')
