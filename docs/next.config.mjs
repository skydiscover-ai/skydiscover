import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  async redirects() {
    return [
      {
        source: '/',
        destination: '/docs',
        permanent: true,
      },
      // Pre-restructure URLs.
      { source: '/docs/getting-started', destination: '/docs/installation', permanent: true },
      { source: '/docs/getting-started/installation', destination: '/docs/installation', permanent: true },
      { source: '/docs/getting-started/quick-start', destination: '/docs/optimize/quick-start', permanent: true },
      { source: '/docs/getting-started/configuration', destination: '/docs/optimize/configuration', permanent: true },
      { source: '/docs/guides/synthesize', destination: '/docs/synthesize', permanent: true },
      { source: '/docs/guides/extending', destination: '/docs/optimize', permanent: true },
      { source: '/docs/guides/algorithms', destination: '/docs/optimize/configuration', permanent: true },
      { source: '/docs/guides/:slug', destination: '/docs/optimize/:slug', permanent: true },
      { source: '/docs/optimize/extending', destination: '/docs/optimize', permanent: true },
      { source: '/docs/optimize/algorithms', destination: '/docs/optimize/configuration', permanent: true },
      { source: '/docs/synthesize/coding-agents', destination: '/docs/synthesize/quick-start', permanent: true },
      { source: '/docs/synthesize/add-a-domain', destination: '/docs/synthesize/examples', permanent: true },
    ];
  },
};

export default withMDX(config);
