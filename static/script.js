fetch("/api/images")
  .then((response) => response.json())
  .then((imageFilenames) => {
    const imageContainer = document.getElementById("image-container");
    imageFilenames.forEach((filename) => {
      const img = document.createElement("img");
      img.src = `/static/images/${filename}`;
      img.style.width = "100%";
      imageContainer.appendChild(img);
    });
  });

function showVisuals() {
  const imageContainer = document.getElementById("image-container");
  imageContainer.style.display = "block";
}
