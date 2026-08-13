#!/usr/bin/env node

const http = require('http');

console.log('🧪 Testing QBGen Next.js Integration...\n');

// Test 1: Check if landing page is accessible
console.log('1. Testing landing page (port 3000)...');
testEndpoint('http://localhost:3000', 'Landing page');

// Test 2: Check if Unique Clues page is accessible
console.log('\n2. Testing Unique Clues page (port 3000/unique-clues)...');
testEndpoint('http://localhost:3000/unique-clues', 'Unique Clues page');

// Test 3: Check if Set Carding page is accessible
console.log('\n3. Testing Set Carding page (port 3000/set-carding)...');
testEndpoint('http://localhost:3000/set-carding', 'Set Carding page');

// Test 4: Check if About page is accessible
console.log('\n4. Testing About page (port 3000/about)...');
testEndpoint('http://localhost:3000/about', 'About page');

// Test 5: Check if Bonus Frequency page is accessible
console.log('\n5. Testing Bonus Frequency page (port 3000/bonus-frequency)...');
testEndpoint('http://localhost:3000/bonus-frequency', 'Bonus Frequency page');

// Test 6: Check if backend API is accessible
console.log('\n6. Testing backend API (port 8080)...');
testEndpoint('http://localhost:8080/api/get_sets', 'Backend API');

function testEndpoint(url, description) {
  const req = http.get(url, (res) => {
    if (res.statusCode >= 200 && res.statusCode < 300) {
      console.log(`   ✅ ${description}: ${res.statusCode} ${res.statusMessage}`);
      console.log(`   📍 URL: ${url}`);
    } else {
      console.log(`   ❌ ${description}: ${res.statusCode} ${res.statusMessage}`);
      console.log(`   📍 URL: ${url}`);
      process.exitCode = 1;
    }
  });

  req.on('error', (err) => {
    console.log(`   ❌ ${description}: ${err.message}`);
    console.log(`   📍 URL: ${url}`);
  });

  req.setTimeout(5000, () => {
    console.log(`   ⏰ ${description}: Timeout`);
    req.destroy();
  });
}

console.log('\n🎯 Setup complete! Check the results above.');
console.log('\n📋 Next steps:');
console.log('   1. If all tests pass, your Next.js integration is working!');
console.log('   2. Visit http://localhost:3000 to see the landing page');
console.log('   3. Click navigation links to test the new pages');
console.log('   4. Test the dark/light mode toggle in the navigation');
console.log('   5. If any tests fail, check that both Next.js and Flask are running');
console.log('   6. Make sure your Flask backend is running on port 8080'); 