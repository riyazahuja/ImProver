/-- `extend_docs <declName> before <prefix_string> after <suffix_string>` extends the
docs of `<declName>` by adding `<prefix_string>` before and `<suffix_string>` after. -/
syntax "extend_docs" ident (colGt &"before" str)? (colGt &"after" str)? : command


open Lean Elab Command in
elab_rules : command
  | `(command| extend_docs $na:ident $[before $bef:str]? $[after $aft:str]?) => do
    if bef.isNone && aft.isNone then throwError "expected at least one of 'before' or 'after'"
    let declName ← liftCoreM <| Elab.realizeGlobalConstNoOverloadWithInfo na
    let bef := if bef.isNone then "" else (bef.get!).getString ++ "\n\n"
    let aft := if aft.isNone then "" else "\n\n" ++ (aft.get!).getString
    let oldDoc := (← findDocString? (← getEnv) declName).getD ""
    addDocString declName <| bef ++ oldDoc ++ aft


