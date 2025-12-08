//-------------------------------------------------------------
// Parse URL query parameters from lobby.html → game.html
//-------------------------------------------------------------
const urlParams = new URLSearchParams(window.location.search);
const gameSide     = urlParams.get("side")     || "white";
const gameStartpos = urlParams.get("startpos") || "standard";
const gameInsanity = urlParams.get("insanity") || "1";

//-------------------------------------------------------------
// Global variables
//-------------------------------------------------------------
let ws = null;
let selectedFrom = null;
let pendingPromotion = null;
let boardState = [];   // 8x8 array

//-------------------------------------------------------------
// Build the chessboard DOM once, after page loads
//-------------------------------------------------------------
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

  // Now that DOM is ready, open WebSocket
  openConnection();
});

//-------------------------------------------------------------
// Open WebSocket exactly once
//-------------------------------------------------------------
function openConnection() {
  ws = new WebSocket("ws://127.0.0.1:8000/ws");

  ws.onopen = () => {
    ws.send(JSON.stringify({
      cmd: "newgame",
      side: gameSide,
      startpos: gameStartpos,
      insanity: gameInsanity
    }));
  };

  ws.onmessage = evt => {
    const msg = JSON.parse(evt.data);
    if (msg.type === "state") {
      drawBoard(msg.board);
    }
  };
}

//-------------------------------------------------------------
// Draw the board from backend data
// (Do NOT clear selection here — that caused our bug)
//-------------------------------------------------------------
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

//-------------------------------------------------------------
// Square click handling
//-------------------------------------------------------------
function onSquareClick(evt) {
  if (!ws) return;

  const row = parseInt(evt.currentTarget.dataset.row);
  const col = parseInt(evt.currentTarget.dataset.col);
  const clickedPiece = boardState[row][col];
  console.log("first click");

  // FIRST CLICK — select a piece
  if (!selectedFrom) {
    if (clickedPiece !== "") {
      clearSelection();
      highlightSquare(row, col);
      selectedFrom = { row, col };
    }
    return;
  }
  console.log("second click");

  // SECOND CLICK — attempt move
  const fromPiece = boardState[selectedFrom.row][selectedFrom.col];
  const toPiece   = clickedPiece;

  // Castling attempt: king → rook
  if (isKing(fromPiece) && isRook(toPiece)) {
    performCastling(selectedFrom, { row, col });
    return;
  }

  // Normal move
  attemptMove({ from: selectedFrom, to: { row, col } });
}

//-------------------------------------------------------------
// Attempt a normal move (with promotion check)
//-------------------------------------------------------------
function attemptMove(move) {
  const { from, to } = move;
  const piece = boardState[from.row][from.col];

  // Pawn promotion?
  if ((piece === "♙" && to.row === 0) ||
      (piece === "♟" && to.row === 7)) {

    pendingPromotion = move;
    showPromotionPopup(event);
    return;
  }

  sendMoveUCI(move, null);
}

//-------------------------------------------------------------
// Promotion popup
//-------------------------------------------------------------
function showPromotionPopup(evt) {
  const dlg = document.getElementById("promotionDialog");
  dlg.style.left = (evt.pageX + 10) + "px";
  dlg.style.top = (evt.pageY + 10) + "px";
  dlg.style.display = "block";
}

function finishPromotion(letter) {
  const dlg = document.getElementById("promotionDialog");
  dlg.style.display = "none";

  if (!pendingPromotion) return;

  sendMoveUCI(pendingPromotion, letter);
  pendingPromotion = null;
}

//-------------------------------------------------------------
// Castling: user clicked king → rook
//-------------------------------------------------------------
function performCastling(kingSq, rookSq) {
  const kRow = kingSq.row;
  const kCol = kingSq.col;
  const rCol = rookSq.col;

  const isKingside = rCol > kCol;
  const kingDestCol = isKingside ? 6 : 2;

  sendMoveUCI({
    from: kingSq,
    to: { row: kRow, col: kingDestCol }
  }, null);
}

//-------------------------------------------------------------
// Convert move to UCI and send to backend
//-------------------------------------------------------------
function sendMoveUCI(move, promoLetter) {
  const uci =
    coordsToUCI(move.from) +
    coordsToUCI(move.to) +
    (promoLetter || "");

  ws.send(JSON.stringify({
    cmd: "move",
    uci: uci
  }));

  clearSelection();
}

//-------------------------------------------------------------
// Helpers
//-------------------------------------------------------------
function coordsToUCI(sq) {
  const files = "abcdefgh";
  return files[sq.col] + (8 - sq.row);
}

function isKing(piece) {
  return piece === "♔" || piece === "♚";
}

function isRook(piece) {
  return piece === "♖" || piece === "♜";
}

function highlightSquare(r, c) {
  const idx = r * 8 + c;
  boardDiv.children[idx].classList.add("selected");
}

function clearSelection() {
  console.log("clearing selection");
  selectedFrom = null;
  Array.from(boardDiv.children).forEach(sq =>
    sq.classList.remove("selected")
  );
}
