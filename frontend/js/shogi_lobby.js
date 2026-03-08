document.getElementById("start-game").addEventListener("click", () => {
  const params = new URLSearchParams({
    whitePlayer: document.getElementById("white-player").value,
    blackPlayer: document.getElementById("black-player").value,
  });

  const url = `shogi_game.html?${params.toString()}`;
  window.open(url, "_blank");
});
