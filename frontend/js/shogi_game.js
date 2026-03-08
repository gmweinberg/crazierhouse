//-------------------------------------------------------------
// Parse URL query parameters from lobby.html → game.html
//-------------------------------------------------------------
//
const urlParams = new URLSearchParams(window.location.search);
const whitePlayer = urlParams.get("whitePlayer") || "human";
const blackPlayer = urlParams.get("blackPlayer") || "bot";
//const gameSide     = urlParams.get("side")     || "white";
const gameSide = 'white';
const gameStartpos = urlParams.get("startpos") || "standard";

let ws = null;

let selected = null;

const boardDiv = document.getElementById("board");


const PIECES = {
  P: "歩",
  L: "香",
  N: "桂",
  S: "銀",
  G: "金",
  B: "角",
  R: "飛",
  K: "玉",

  "+P": "と",
  "+L": "成香",
  "+N": "成桂",
  "+S": "成銀",
  "+B": "馬",
  "+R": "龍"
};


window.addEventListener("load", () => {
  createBoard();
  openConnection();
});

function decodePiece(cell) {
  if (!cell || cell === "") return null;

  let promoted = false;

  // Handle "+P" style promoted pieces
  if (cell[0] === "+") {
    promoted = true;
    cell = cell[1];
  }

  const color = (cell === cell.toUpperCase()) ? "black" : "white";
  const type = cell.toUpperCase();

  return { type, color, promoted };
}

function createBoard() {

  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {

      const sq = document.createElement("div");

      sq.classList.add("square");

      if ((r + c) % 2 === 0)
        sq.classList.add("light");
      else
        sq.classList.add("light"); // keep it all light

      sq.dataset.row = r;
      sq.dataset.col = c;

      sq.addEventListener("click", onSquareClick);
      boardDiv.appendChild(sq);
    }
  }
}

function onSquareClick(e) {

  const sq = e.currentTarget;
  const r = sq.dataset.row;
  const c = sq.dataset.col;

  if (selected === null) {

    selected = { r, c };
    sq.classList.add("selected");
    return;

  }

  sendMove(selected.r, selected.c, r, c);

  clearSelection();
}

function clearSelection() {

  selected = null;

  document
    .querySelectorAll(".selected")
    .forEach(s => s.classList.remove("selected"));
}

function sendMove(r1, c1, r2, c2) {

  const move = {
    from: [r1, c1],
    to: [r2, c2]
  };

  console.log("move", move);

  // eventually call backend here
}

function drawPiece(square, piece, color) {

  const el = document.createElement("span");

  el.classList.add("piece");

  if (color === "white")
    el.classList.add("white");

  el.textContent = PIECES[piece];

  square.appendChild(el);
}

function drawBoard(boardState) {

  const squares = document.querySelectorAll(".square");
  // clear out the old crap
  squares.forEach(s => s.innerHTML = "");

  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {

      const piece = boardState[r][c];

      if (!piece)
        continue;

      const sq = squares[r * 9 + c];

      drawPiece(sq, piece.type, piece.color);
    }
  }
}

function drawHand(containerId, pocket) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  for (const piece in pocket) {
    const count = pocket[piece];
    for (let i = 0; i < count; i++) {
      const el = document.createElement("span");
      el.classList.add("piece");
      el.textContent = PIECES[piece];
      container.appendChild(el);
    }
  }
}

function openConnection() {
  ws = new WebSocket("ws://127.0.0.1:8000/ws");

  ws.onopen = () => {
    ws.send(JSON.stringify({
      cmd: "newgame",
      "game": "shogi",
      //side: gameSide,
      whitePlayer: whitePlayer,
      blackPlayer: blackPlayer,
      // fen:fen
    }));
  };

  ws.onmessage = evt => {
    const msg = JSON.parse(evt.data);
    if (msg.type === "state") {
			console.log("got state")
      drawBoard(msg.board);
      const { white, black } = msg.pockets;
			drawHand("white-stand", white);
			drawHand("black-stand", black);
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
	}
}


