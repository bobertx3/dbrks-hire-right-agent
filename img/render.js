const puppeteer = require('/opt/homebrew/lib/node_modules/@mermaid-js/mermaid-cli/node_modules/puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  const page = await browser.newPage();

  const htmlPath = path.resolve('/Users/robert.leach/dev/vibe/dbrks-hire-right-agent/img', 'hire-right-render.html');
  await page.goto('file://' + htmlPath, { waitUntil: 'networkidle0' });

  // Get canvas dimensions
  const dims = await page.evaluate(() => {
    const el = document.querySelector('.canvas');
    const rect = el.getBoundingClientRect();
    return { width: Math.ceil(rect.width), height: Math.ceil(rect.height) };
  });

  await page.setViewport({ width: dims.width + 4, height: dims.height + 4, deviceScaleFactor: 2 });
  await page.reload({ waitUntil: 'networkidle0' });

  const canvas = await page.$('.canvas');
  const outPath = '/Users/robert.leach/dev/vibe/dbrks-hire-right-agent/img/hire-right-architecture.png';
  await canvas.screenshot({ path: outPath, type: 'png' });

  console.log('Done: ' + dims.width + 'x' + dims.height + ' (2x)');
  await browser.close();
})();
