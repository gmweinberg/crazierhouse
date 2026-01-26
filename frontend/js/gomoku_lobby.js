document.getElementById("start-game").addEventListener("click", () => {
  const params = new URLSearchParams({
    white: document.getElementById("white-player").value,
    black: document.getElementById("black-player").value,
    size: document.getElementById("size").value,
    dims: document.getElementById("dims").value,
    connect: document.getElementById("connect").value,
    wrap: document.getElementById("wrap").checked ? "1" : "0",
    style: document.querySelector("input[name='style']:checked").value,
  });

  const url = `gomoku_game.html?${params.toString()}`;
  window.open(url, "_blank");
});
