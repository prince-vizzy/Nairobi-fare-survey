// ─────────────────────────────────────────────────────────────────────────────
// Nairobi Fare Survey — Google Apps Script backend
//
// HOW TO SET UP (one-time, ~5 minutes):
//
// 1. Go to https://sheets.google.com  →  create a new blank spreadsheet
//    and name it "Nairobi Fare Responses"
//
// 2. Open Extensions → Apps Script
//    Delete any existing code, paste THIS entire file, then click Save (Ctrl+S)
//
// 3. Click "Deploy" → "New deployment"
//    - Type: Web app
//    - Execute as: Me
//    - Who has access: Anyone
//    Click "Deploy", then Authorize when prompted.
//
// 4. Copy the Web App URL that appears (looks like:
//    https://script.google.com/macros/s/ABC123.../exec)
//
// 5. Open fare_survey.html in a text editor, find this line near the top:
//      const SHEET_URL = "YOUR_APPS_SCRIPT_URL_HERE";
//    Replace the placeholder with your URL (keep the quotes), save the file.
//
// 6. Open fare_survey.html in a browser — responses now land in your Sheet!
//    To share: upload the HTML to GitHub Pages (free) or just email/WhatsApp
//    the file to friends (they open it locally, it still POSTs to your Sheet).
// ─────────────────────────────────────────────────────────────────────────────

const SHEET_NAME = "Responses";

const COLUMNS = [
  "Timestamp",
  "Route ID",
  "Route",
  "Fare Paid (KSh)",
  "Reference Fare (KSh)",
  "Direction"
];

function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);
    const ss    = SpreadsheetApp.getActiveSpreadsheet();
    let   sheet = ss.getSheetByName(SHEET_NAME);

    // Create sheet + header row on first run
    if (!sheet) {
      sheet = ss.insertSheet(SHEET_NAME);
      sheet.appendRow(COLUMNS);
      sheet.getRange(1, 1, 1, COLUMNS.length)
           .setFontWeight("bold")
           .setBackground("#1a5276")
           .setFontColor("#ffffff");
      sheet.setFrozenRows(1);
    }

    sheet.appendRow([
      data.timestamp,
      data.routeId,
      data.route,
      data.farePaid,
      data.refFare,
      data.direction
    ]);

    return ContentService
      .createTextOutput(JSON.stringify({ status: "ok" }))
      .setMimeType(ContentService.MimeType.JSON);

  } catch (err) {
    return ContentService
      .createTextOutput(JSON.stringify({ status: "error", message: err.message }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

// Lets you test the endpoint in a browser (GET request)
function doGet() {
  return ContentService
    .createTextOutput("Fare Survey endpoint is live.")
    .setMimeType(ContentService.MimeType.TEXT);
}
