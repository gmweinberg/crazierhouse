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
let selectedHand = null;
let pendingMove = null;

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

function onHandClick(pieceType) {
  selected = null;            // cancel board selection
  selectedHand = pieceType;
}

window.addEventListener("load", () => {
  createBoard();
  openConnection();
});


document.getElementById("promoteYes").onclick = () => {
  ws.send(JSON.stringify({
    cmd: "move",
    move: pendingMove + "+"
  }));
  hidePromotionDialog();
};

document.getElementById("promoteNo").onclick = () => {
  ws.send(JSON.stringify({
    cmd: "move",
    move: pendingMove
  }));
  hidePromotionDialog();
};

function decodePiece(cell) {
  if (!cell || cell === "") return null;

  let promoted = false;

  // Handle "+P" style promoted pieces
  if (cell[0] === "+") {
    promoted = true;
  }

  const color = (cell === cell.toUpperCase()) ? "black" : "white";
  const type = cell.toUpperCase();
	if (promoted){
		console.log("promoted " + type);
	}

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

function rankChar(rank) {
	console.log("rank", rank);
	const offset = 'a'.charCodeAt(0); // 97
  return String.fromCharCode(offset + parseInt(rank)); // 0→'a', 1→'b', etc.
}

function onSquareClick(e) {
  const sq = e.currentTarget;
  const r = parseInt(sq.dataset.row);
  const c = parseInt(sq.dataset.col);
	// console.log("clicked " + r + " " + c);
  // DROP MOVE
  if (selectedHand !== null) {

    const move =
      selectedHand + "*" +
      (9 - c) +
      rankChar(r);

    console.log("drop", move);

    ws.send(JSON.stringify({
      cmd: "move",
      move: move
    }));

    selectedHand = null;
    return;
  }
	// TODO handle drop moves

  if (selected === null) {
    selected = { r, c };
    sq.classList.add("selected");
    return;
  }
	if (selected.r == r && selected.c == c) {
    clearSelection();
    return;
  }

	const move = String(9 - selected.c) + rankChar(selected.r) +  String(9 - c) + rankChar(r);
  console.log("move", move);
	  ws.send(JSON.stringify({
       cmd: "promote",
       move: move
  }));
  clearSelection();
}

function clearSelection() {
  selected = null;
	selectedHand = null;
  document
    .querySelectorAll(".selected")
    .forEach(s => s.classList.remove("selected"));
}

function decodePiece(cell) {
  if (!cell || cell === "") return null;

  let promoted = false;

  // Handle "+P" style promoted pieces
  if (cell[0] === "+") {
    promoted = true;
  }

  const color = (cell === cell.toUpperCase()) ? "black" : "white";
  const type = cell.toUpperCase();

  return { type, color, promoted };
}

function drawPiece(square, piece, color, promoted) {

  const el = document.createElement("span");

  el.classList.add("piece");

  if (color === "white")
    el.classList.add("white");
	if (promoted){
		el.classList.add("promoted");
	}

  el.textContent = PIECES[piece];

  square.appendChild(el);
}

function drawBoard(boardState) {

  const squares = document.querySelectorAll(".square");
  // clear out the old crap
  squares.forEach(s => s.innerHTML = "");

  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const piece = decodePiece(boardState[r][c]);

      if (!piece)
        continue;

      const sq = squares[r * 9 + c];

      drawPiece(sq, piece.type, piece.color, piece.promoted);
    }
  }
}

function drawHand(containerId, hand) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  for (const piece in hand) {
    const count = hand[piece];
    for (let i = 0; i < count; i++) {
      const el = document.createElement("span");
      el.classList.add("piece");
      el.textContent = PIECES[piece];
			el.onclick = () => onHandClick(piece);
      container.appendChild(el);
    }
  }
}

function showPromotionDialog() {
  document.getElementById("promotionDialog").style.display = "block";
}


function hidePromotionDialog() {
  document.getElementById("promotionDialog").style.display = "none";
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
		console.log(msg)
    if (msg.type === "state") {
			console.log("got state " + msg.board)
      drawBoard(msg.board);
      const { white, black } = msg.hands;
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

		if (msg.type == 'promote') {
			if (msg.promote == 'y') {
						ws.send(JSON.stringify({
						cmd: "move",
						"move": msg.move + '+'
					 }));
			} else if  (msg.promote == 'n'){
						ws.send(JSON.stringify({
						cmd: "move",
						"move": msg.move
					 }));
			} else if (msg.promote == 'm'){
				pendingMove = msg.move;
				showPromotionDialog();
			} else {
				console.log("illegal move");
			}
		}
	}
}
