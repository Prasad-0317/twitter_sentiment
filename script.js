document.getElementById('iframeLink').addEventListener('click', function(event) {
    event.preventDefault();
  
    const newTab = window.open('', '_blank');
  
    newTab.document.write(`
      <html>
        <head>
          <title>Dashboard</title>
        </head>
        <body>
            <iframe title="ISRO" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=f68bbc0b-0cff-47fe-addc-2c5e0eaa2f25&autoAuth=true&ctid=cca3f0fe-586f-4426-a8bd-b8146307e738" frameborder="0" allowFullScreen="true"></iframe>
        </body>
      </html>
    `);
  });
  

  document.getElementById('tweets').addEventListener('click', function(event) {
    event.preventDefault();
  
    const newTab = window.open('', '_blank');
  
    newTab.document.write(`
      <html>
        <head>
          <title>Tweets Sentiment Analysis</title>
        </head>
        <body>
            <iframe src="http://localhost:8502/" width="100%" height="600px" frameborder="0"></iframe>
        </body>
      </html>
    `);
  });
// python -m http.server 3000