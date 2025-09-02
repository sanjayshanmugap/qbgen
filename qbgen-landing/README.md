# QBGen - Next.js Integration

This is the complete QBGen application with a professional landing page and integrated tools, built with Next.js and connected to your existing Flask backend.

## Architecture

- **Landing Page**: Professional marketing page with smooth animations
- **Unique Clues Tool**: `/unique-clues` - Generate unique clues for any answerline
- **Set Carding Tool**: `/set-carding` - Generate clues from specific quiz bowl sets
- **About Page**: `/about` - Information about the tool and how to use it
- **Backend**: Your existing Flask API (port 8080)
- **Integration**: Next.js pages call your Flask API endpoints

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
- Your Flask backend running on port 8080

### Setup
1. Install dependencies:
   ```bash
   pnpm install
   ```

2. Start the development server:
   ```bash
   pnpm dev
   ```

3. Make sure your Flask backend is running on port 8080

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

The Next.js app proxies API calls to your Flask backend:
- `/api/process_clues` → `http://localhost:8080/process_clues`
- `/api/process_set_clues` → `http://localhost:8080/process_set_clues`
- `/api/generate_apkg` → `http://localhost:8080/generate_apkg`
- `/api/get_sets` → `http://localhost:8080/get_sets`

## Testing

Run the test script to verify everything is working:
```bash
pnpm test-setup
```

## Deployment

See `DEPLOYMENT.md` for detailed deployment instructions.

## Styling

- **Consistent Theme**: All pages use the same color scheme and design system
- **Dark Mode Support**: Full dark mode support across all components
- **Modern UI**: Clean, professional appearance with smooth animations
- **Accessibility**: Proper contrast ratios and keyboard navigation

All functionality from your original React components has been preserved and enhanced with the new design system. 