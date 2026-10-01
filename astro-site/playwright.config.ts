import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  use: { baseURL: 'http://127.0.0.1:14474', headless: true },
  webServer: {
    command: 'helia-ui-serve-dist --dist dist --base /helia-rt --port 14474',
    url: 'http://127.0.0.1:14474/helia-rt/',
    reuseExistingServer: false,
  },
});
