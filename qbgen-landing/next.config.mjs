/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8080/:path*',
      },
      {
        source: '/static/:path*',
        destination: 'http://localhost:8080/static/:path*',
      },
    ]
  },
  async redirects() {
    return [
      {
        source: '/tools',
        destination: 'http://localhost:8080/static/',
        permanent: false,
      },
    ]
  },
}

export default nextConfig
