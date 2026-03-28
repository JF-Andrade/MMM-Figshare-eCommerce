const fs = require('fs');

async function uploadDiagram() {
  const mermaidDiagram = fs.readFileSync('d:/Projects/MMM-Figshare-eCommerce/architecture.mermaid', 'utf8');
  
  try {
    const response = await fetch('http://localhost:3000/api/elements/from-mermaid', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ mermaidDiagram })
    });
    
    const result = await response.json();
    console.log(result);
  } catch (error) {
    console.error('Error:', error);
  }
}

uploadDiagram();
