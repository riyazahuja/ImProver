def main : IO Unit := do
  -- let test <- IO.FS.readFile "lakefile.lean"
  -- IO.println test
  IO.println "HELLO!"
  let out <- IO.Process.run {cmd := "./models/test.sh"}
  IO.println out
