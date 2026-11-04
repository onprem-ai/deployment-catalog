# 2026-08-16 - PDF Validation

**Goal:** Validate multi-page PDF OCR through the authenticated application and ensure the Playground PDF sample creates the correct request.

## Application-path test

Used the bundled two-page German CV sample at:

```text
app/frontend/public/playground-samples/cv_de.pdf
```

The PDF was base64-encoded and submitted to the API gateway's passthrough path `POST /pp-ocrv6/ocr`, `fileType: 0`, and visualization disabled.

The request returned HTTP 200 in 0.523 seconds with:

- `errorCode: 0`
- Two entries in `result.ocrResults`, one for each PDF page
- 49 recognized text regions on page 1
- 20 recognized text regions on page 2
- Correct German text including `LEBENSLAUF`, employment history, education, language skills, and `Seite 1 von 2` / `Seite 2 von 2`

This proves that base64 PDF input and multi-page output work through authentication, Playground backend proxying, AI Gateway routing, and the deployed PaddleX `/ocr` endpoint.

## Playground sample behavior

The Playground already ships and displays `cv_de.pdf` in its document sample picker. The proxy panel accepted PDF files but retained the catalog image example's `fileType: 1`, which mislabeled selected PDFs.

The attachment path now sets PaddleX's explicit file type from the selected MIME type:

- `application/pdf` -> `fileType: 0`
- `image/*` -> `fileType: 1`
- Any other MIME type fails visibly instead of guessing

The same path is used by manual file attachment and bundled sample selection, so the existing German CV PDF is now a working PP-OCRv6 Playground example.

## Outcome

PP-OCRv6 PDF support is validated for a two-page PDF. The Playground's bundled PDF example now automatically generates the required request field instead of relying on the user to edit it manually.
