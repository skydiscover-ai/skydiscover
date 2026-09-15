import { RootProvider } from 'fumadocs-ui/provider/next';
import './global.css';
import { Inter } from 'next/font/google';

const inter = Inter({
  subsets: ['latin'],
});

export const metadata = {
  title: { default: 'SkyDiscover', template: '%s | SkyDiscover' },
  icons: { icon: '/img/lockup-light-bg.svg' },
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={`${inter.className} light`} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen bg-white">
        {/* Light only: the architecture figures have white backgrounds. */}
        <RootProvider theme={{ enabled: false }}>{children}</RootProvider>
      </body>
    </html>
  );
}
