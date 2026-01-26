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
	// console.log("coord clicked" + coord);
	// markCoord(coord);
	ws.send(JSON.stringify({
       cmd: "move",
       coord: coord
  }));


}

function makeStrides(size, dims) {
  const strides = new Array(dims);
  let s = 1;
  for (let d = dims - 1; d >= 0; d--) {
    strides[d] = s;
    s *= size;
  }
  return strides; // strides[d] = size^(dims-1-d)? (actually this yields least-sig at end)
}

function indexToCoordND(index, size, dims) {
  const strides = makeStrides(size, dims);
  const coord = new Array(dims);
  for (let d = 0; d < dims; d++) {
    coord[d] = Math.floor(index / strides[d]);
    index = index % strides[d];
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
		const y = coord[0];
    const x = coord[1];

    if (x === 0) cell.classList.add("left-edge");
    if (x === size - 1) cell.classList.add("right-edge");
    if (y === 0) cell.classList.add("top-edge");
    if (y === size - 1) cell.classList.add("bottom-edge");
    cell.dataset.coord = JSON.stringify(coord);

    cell.addEventListener("click", () => {
      handleCoordClick(coord);
    });

    board.appendChild(cell);
  }

  wrapper.appendChild(board);
  return wrapper;
}


function coordsEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function markCoord(coord) {
  const cells = document.querySelectorAll(".cell");

  for (const cell of cells) {
    const cellCoord = JSON.parse(cell.dataset.coord);

    if (coordsEqual(cellCoord, coord)) {
			//console.log("found cell");
      renderMark(cell);
      return;
    }
  }
}

function renderMark(cell, color) {

  const style = window.gameConfig.style;

  if (style === "tictactoe") {
    cell.textContent = color === "b" ? "X" : "O";
    cell.style.fontSize = "24px";
    cell.style.textAlign = "center";
    cell.style.lineHeight = "32px";
  } else {
    cell.textContent = color === "b" ? "⚫" : "⚪";
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
