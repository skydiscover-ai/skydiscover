import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <img src="/img/lockup-light-bg.svg" alt="SkyDiscover" style={{ height: 28 }} />
      ),
      url: '/docs',
    },
    links: [
      { text: 'Optimize', url: '/docs/optimize' },
      { text: 'Synthesize', url: '/docs/synthesize' },
      { text: 'Blog', url: 'https://skydiscover-ai.github.io/blogs.html', external: true },
    ],
    githubUrl: 'https://github.com/skydiscover-ai/skydiscover',
  };
}
