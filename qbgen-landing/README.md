# QBGen - Next.js Integration

This is the frontend for QBGen, built with Next.js and designed to be hosted separately from the Flask backend.

## Architecture

- **Landing Page**: Professional marketing page with smooth animations
- **Unique Clues Tool**: `/unique-clues` - Generate unique clues for any answerline
- **Set Carding Tool**: `/set-carding` - Generate clues from specific quiz bowl sets
- **About Page**: `/about` - Information about the tool and how to use it
- **Backend**: Flask API hosted separately, locally on port `8080` or in production on Cloud Run
- **Integration**: Client-side fetches call the backend using `NEXT_PUBLIC_API_BASE_URL`

## Features

- **Professional Landing Page**: Modern design with Framer Motion animations
- **Persistent Navigation**: Shared navigation bar across all pages
- **Dark/Light Mode**: Toggle between themes (persistent across pages)
- **Responsive Design**: Works on all devices
- **API Integration**: Seamlessly connects to your Flask backend
- **shadcn/ui Components**: Modern, accessible UI components
- **TypeScript**: Full type safety

## Development

### Prerequisites
- Node.js 18+
- pnpm (recommended) or npm
- Your Flask backend running on port `8080`

### Setup
1. Install dependencies:
   ```bash
   pnpm install
   ```

2. Configure the backend URL:
   ```bash
   cp .env.example .env.local
   ```

3. Start the development server:
   ```bash
   pnpm dev
   ```

4. Make sure your Flask backend is running on port `8080`

### Building for Production
```bash
pnpm build
pnpm start
```

## Pages

- **Landing Page**: `/` - Marketing page with navigation to tools
- **Unique Clues**: `/unique-clues` - Generate unique clues with semantic filtering
- **Set Carding**: `/set-carding` - Generate clues from specific quiz bowl sets
- **About**: `/about` - Information about the tool and usage instructions

## Navigation

The application includes a persistent navigation bar with:
- **qbgen** logo (links to home)
- **Unique Clues** link
- **Set Carding** link  
- **About** link
- **Dark/Light Mode** toggle (persistent across pages)

## API Integration

The frontend calls the backend directly using `NEXT_PUBLIC_API_BASE_URL`.

For local development:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8080
```

For production on Vercel, set `NEXT_PUBLIC_API_BASE_URL` to your public Cloud Run backend URL.

## Testing

Run the test script to verify everything is working:
```bash
pnpm test-setup
```

## Deployment

- Host `qbgen-landing` on Vercel
- Set `NEXT_PUBLIC_API_BASE_URL` in Vercel to your Cloud Run backend URL
- Deploy the Flask backend separately to Cloud Run

## Styling

- **Consistent Theme**: All pages use the same color scheme and design system
- **Dark Mode Support**: Full dark mode support across all components
- **Modern UI**: Clean, professional appearance with smooth animations
- **Accessibility**: Proper contrast ratios and keyboard navigation

All functionality from your original React components has been preserved and enhanced with the new design system. 