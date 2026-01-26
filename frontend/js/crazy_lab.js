
const gameSide = "white";
//-------------------------------------------------------------
// Global variables
//-------------------------------------------------------------


let ws = null; 
let selectedPiece = null;
let boardState = [];   // 8x8 array
//let selectedPocketPiece = null;
let gameOver = false;

function addPieceButton(piece, col, row) {
  const pieces = document.getElementById("pieces");
  const div = document.createElement("div");
  div.classList.add("square");
  div.classList.add("light");
  div.dataset.row = row;
  div.dataset.col = col;
  div.addEventListener("click", onPieceButtonClick);
  div.textContent = piece;
  pieces.appendChild(div);
}

function openConnection() {
  ws = new WebSocket("ws://127.0.0.1:8000/ws");

  ws.onopen = () => {
    ws.send(JSON.stringify({
      cmd: "reset_position",
			game: "crazyhouse"
    }));
  };

  ws.onmessage = evt => {
    const msg = JSON.parse(evt.data);
    if (msg.type === "state") {
      drawBoard(msg.board);
      const { white, black } = msg.pockets;
      //console.log("white" + white);
      if (gameSide === "white") {
          renderPocket("human-pocket", white);
          renderPocket("comp-pocket", black);
      } else {
          renderPocket("human-pocket", black);
          renderPocket("comp-pocket", white);
      }

    } else if (msg.type === "fen") {
			launchGame(msg.fen);
			/*  ws.send(JSON.stringify({
        cmd: "round_trip_fen",
        "fen":msg.fen}
      ))  */
		}
	};
}


window.addEventListener("load", () => {
  const boardDiv = document.getElementById("board");

  // Store globally for other functions
  window.boardDiv = boardDiv;

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const div = document.createElement("div");
      div.classList.add("square");
      div.classList.add((r + c) % 2 === 0 ? "light" : "dark");
      div.dataset.row = r;
      div.dataset.col = c;
      div.addEventListener("click", onSquareClick);
      boardDiv.appendChild(div);
    }
  }
  const launch = document.getElementById("launch");
	launch.addEventListener("click", onLaunchClick);

  // draw the pieces buttons
  let row = 0;
  let col = 0;
  addPieceButton("", col, row);
  col++;
  addPieceButton("♙", col, row);
  col++;
  addPieceButton("♘", col, row);
  col++;
  addPieceButton("♗", col, row);
  col++;
  addPieceButton("♖", col, row);
  col++;
  addPieceButton("♕", col, row);
  col++;
  addPieceButton("♔", col, row);
	col++
  addPieceButton("H", col, row);
  col++;
  addPieceButton("A", col, row);
  col++;
  addPieceButton("C", col, row);
  col++;
  addPieceButton("E", col, row);
  col++;
  col = 0
  row = 1;
  addPieceButton("", col, row);
  col++;
  addPieceButton("♟", col, row);
  col++;
  addPieceButton("♞", col, row);
  col++;
  addPieceButton("♝", col, row);
  col++;
  addPieceButton("♜", col, row);
  col++;
  addPieceButton("♛", col, row);
  col++;
  addPieceButton("♚", col, row);
	col++;
  addPieceButton("h", col, row);
  col++;
  addPieceButton("a", col, row);
  col++;
  addPieceButton("c", col, row);
  col++;
  addPieceButton("e", col, row);
  col++;
  // Now that DOM is ready, open WebSocket
  openConnection();
});


function drawBoard(board) {
  boardState = board;
  const squares = boardDiv.children;

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const idx = r * 8 + c;
      squares[idx].textContent = board[r][c];
    }
  }
}

function renderPocket(which, stuff){
}

function onPieceButtonClick(evt){
	if (!ws) return;
	const text = evt.currentTarget.textContent;
  console.log(text);
  selectedPiece = text
}



function onSquareClick(evt) {
  if (!ws) return;

  const row = parseInt(evt.currentTarget.dataset.row);
  const col = parseInt(evt.currentTarget.dataset.col);
  ws.send(JSON.stringify({
    cmd: "set_square",
		"x":col,
		"y":row,
		"piece":selectedPiece}
	))
}

function onLaunchClick(evt){
  ws.send(JSON.stringify({
    cmd: "get_fen"}
	))
}

function launchGame(fen){
	const forced_fen = document.getElementById('force-fen').value.trim();
	let encodedFen;
	if (forced_fen) {
		encodedFen = encodeURIComponent(forced_fen);
	} else {
		encodedFen = encodeURIComponent(fen);
	}
	const game = "crazyhouse";
	const koth = document.getElementById('koth').checked ? 'true': '';
	const sticky = document.getElementById('sticky').checked ? 'true': '';
	const whitePlayer = document.getElementById('white-player').value;
	const blackPlayer = document.getElementById('black-player').value;
	const insanity = document.getElementById('insanity').value;
	const startpos = document.getElementById('startpos').value;
	const params = new URLSearchParams({
		game,
    whitePlayer,
    blackPlayer,
    startpos,
    insanity,
    koth,
		sticky,
    fen: encodedFen
  });

  const url = `crazy_game.html?${params.toString()}`;
  const win = window.open(url, "_blank");
  if (win) {
    win.focus();
  }



}
