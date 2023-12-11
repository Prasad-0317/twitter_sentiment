document.getElementById("iframeLink").addEventListener("click", function() {
    var iframeContainer = document.getElementById("iframeContainer");
    iframeContainer.classList.remove("hidden"); // Show the iframe container
});

// python -m http.server 3000