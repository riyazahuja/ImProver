local syntax "haveIDummy" haveDecl : term

macro_rules
  | `(assert! haveIDummy $hd:haveDecl; $body) => `(haveI $hd:haveDecl; $body)


/--
`haveI'` behaves like `have`, but inlines the value instead of producing a `let_fun` term.

(This is the do-notation version of the term-mode `haveI`.)
-/
macro "haveI' " hd:haveDecl : doElem =>
  `(doElem| assert! haveIDummy $hd:haveDecl)


local syntax "letIDummy" haveDecl : term

macro_rules
  | `(assert! letIDummy $hd:haveDecl; $body) => `(letI $hd:haveDecl; $body)


/--
`letI` behaves like `let`, but inlines the value instead of producing a `let_fun` term.

(This is the do-notation version of the term-mode `haveI`.)
-/
macro "letI' " hd:haveDecl : doElem =>
  `(doElem| assert! letIDummy $hd:haveDecl)


