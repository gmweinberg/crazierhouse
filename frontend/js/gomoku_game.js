function getParams() {
  const params = new URLSearchParams(window.location.search);
  return {
    white: params.get("white"),
    black: params.get("black"),
    size: parseInt(params.get("size"), 10),
    dims: parseInt(params.get("dims"), 10),
    connect: parseInt(params.get("connect"), 10),
    wrap: params.get("wrap") === "1",
    style: params.get("style"),
  };
}

document.addEventListener("DOMContentLoaded", () => {
  const cfg = getParams();
	window.gameConfig = cfg;

  // Populate info panel
  document.getElementById("white-player").textContent = cfg.white;
  document.getElementById("black-player").textContent = cfg.black;
  document.getElementById("size").textContent = cfg.size;
  document.getElementById("dims").textContent = cfg.dims;
  document.getElementById("connect").textContent = cfg.connect;
  document.getElementById("wrap").textContent = cfg.wrap ? "yes" : "no";
  document.getElementById("style").textContent = cfg.style;

  // Style hook
  const board = document.getElementById("board");
  // board.classList.add(cfg.style === "tictactoe" ? "tictactoe" : "gomoku");

  renderAllBoards();

  //document.getElementById("status").textContent =
  //  "Game page loaded. (No server connection yet)";
	 openConnection();
});


function handleCoordClick(coord) {
	console.log("coord clicked" + coord);
}

function indexToCoord(index, size, dims) {
  const coord = new Array(dims);
  for (let d = dims - 1; d >= 0; d--) {
    coord[d] = index % size;
    index = Math.floor(index / size);
  }
  return coord;
}

function indexToCoordND(index, size, dims) {
  const coord = new Array(dims);
  for (let d = dims - 1; d >= 0; d--) {
    coord[d] = index % size;
    index = Math.floor(index / size);
  }
  return coord;
}

function renderAllBoards() {
  const { dims, size } = window.gameConfig;
	const container = document.getElementById("board-container");
  container.classList.add(
    window.gameConfig.style === "tictactoe" ? "tictactoe" : "gomoku"
  );
  container.innerHTML = "";

  if (dims === 2) {
    container.appendChild(renderBoard2D([]));
  } else if (dims === 3) {
		container.classList.add("board-stack");
    container.classList.remove("board-grid");
    for (let z = 0; z < size; z++) {
      container.appendChild(renderBoard2D([z]));
    }
  } else if (dims === 4) {
    container.classList.add("board-grid");
    container.classList.remove("board-stack");
    container.style.gridTemplateColumns = `repeat(${size}, auto)`;
    for (let w = 0; w < size; w++) {
      for (let z = 0; z < size; z++) {
        container.appendChild(renderBoard2D([z, w]));
      }
    }
  }
}

function renderBoard2D(sliceCoords) {
  const { size, dims } = window.gameConfig;

  const wrapper = document.createElement("div");
  wrapper.className = "slice";
  if (sliceCoords.length > 0) {
    const label = document.createElement("div");
    label.className = "slice-label";
    label.textContent =
      dims === 3
        ? `z = ${sliceCoords[0]}`
        : `z = ${sliceCoords[0]}, w = ${sliceCoords[1]}`;
    wrapper.appendChild(label);
  }

  const board = document.createElement("div");
  board.className = "board";
  board.style.gridTemplateColumns = `repeat(${size}, 32px)`;

  const numCells = Math.pow(size, dims);

  for (let index = 0; index < numCells; index++) {
    const coord = indexToCoordND(index, size, dims);

    // Check slice match
    if (dims >= 3 && coord[2] !== sliceCoords[0]) continue;
    if (dims === 4 && coord[3] !== sliceCoords[1]) continue;

    const cell = document.createElement("div");
    cell.className = "cell";
    cell.dataset.coord = JSON.stringify(coord);

    cell.addEventListener("click", () => {
      handleCoordClick(coord);
    });

    board.appendChild(cell);
  }

  wrapper.appendChild(board);
  return wrapper;
}

function markCell(index) {
  const board = document.getElementById("board");
  const cell = board.children[index];

  if (!cell || cell.textContent !== "") {
    return; // already marked
  }

  const style = window.gameConfig.style;

  if (style === "tictactoe") {
    cell.textContent = "X";
    cell.style.fontSize = "24px";
    cell.style.textAlign = "center";
    cell.style.lineHeight = "32px";
  } else {
    cell.textContent = "⚫";
    cell.style.fontSize = "24px";
    cell.style.textAlign = "center";
    cell.style.lineHeight = "32px";
  }
}

function openConnection() {
  ws = new WebSocket("ws://127.0.0.1:8000/ws");

  ws.onopen = () => {
    ws.send(JSON.stringify({
      cmd: "newgame",
      "game": "gomoku",
      whitePlayer: window.gameConfig.white,
      blackPlayer: window.gameConfig.black,
      //startpos: gameStartpos,
			dims:window.gameConfig.dims,
      size: window.gameConfig.size,
      connect:window.gameConfig.connect,
      wrap:window.gameConfig.wrap
    }));
  };

  ws.onmessage = evt => {
    const msg = JSON.parse(evt.data);
    if (msg.type === "state") {
			//console.log("rendering boards")
      //renderAllBoards(msg.board);
  }
  if (msg.type === "terminal") {
    let text;
    if (msg.result === "white_win") text = "White wins!";
    else if (msg.result === "black_win") text = "Black wins!";
    else text = "Draw.";

    document.getElementById("status").textContent = text;
    gameOver = true;
    console.log("gameOverMan");
  }
  };
}
