/-
Copyright (c) 2021 Microsoft Corporation. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Leonardo de Moura
-/
prelude
import Lean.Util.CollectLevelParams
import Lean.Util.CollectAxioms
import Lean.Meta.Reduce
import Lean.Elab.DeclarationRange
import Lean.Elab.Eval
import Lean.Elab.Command
import Lean.Elab.Open
import Lean.Elab.SetOption
import Init.System.Platform

import Lean.Parser.Term
import Lean.Parser.Do



namespace Lean
namespace Parser
namespace Command

@[command_parser] def «where_with_end»        := leading_parser
  "#where_with_end"

end Command
end Parser
end Lean


namespace Lean.Elab.Command


def whereWithEndCore : CommandElabM ((MessageData) × (MessageData)) := do
  let scope ← getScope
  let mut msg : Array MessageData := #[]

  let mut endMsg : Array MessageData := #[]
  -- Noncomputable
  if scope.isNoncomputable then
    msg := msg.push <| ← `(command| noncomputable section)
    endMsg := endMsg.push <| ← `(command| end)
  -- Namespace
  if !scope.currNamespace.isAnonymous then
    msg := msg.push <| ← `(command| namespace $(mkIdent scope.currNamespace))
    endMsg := endMsg.push <| ← `(command| end $(mkIdent scope.currNamespace))
  -- Open namespaces
  if let some openMsg ← describeOpenDecls scope.openDecls.reverse then
    msg := msg.push openMsg
    -- "Open" does not need to be closed
  -- Universe levels
  if !scope.levelNames.isEmpty then
    let levels := scope.levelNames.reverse.map mkIdent
    msg := msg.push <| ← `(command| universe $levels.toArray*)
    -- levels do not need to be closed

  -- Variables
  if !scope.varDecls.isEmpty then
    let varDecls : Array (TSyntax `Lean.Parser.Term.bracketedBinder) := scope.varDecls.map (⟨·.raw.unsetTrailing⟩)
    msg := msg.push <| ← `(command| variable $varDecls*)
    -- variables do not need to be closed

  -- Included variables
  if !scope.includedVars.isEmpty then
    msg := msg.push <| ← `(command| include $(scope.includedVars.toArray.map (mkIdent ·.eraseMacroScopes))*)
    -- inclusions do not need to be closed
  -- Options
  if let some optionsMsg ← describeOptions scope.opts then
    msg := msg.push optionsMsg
    -- options do not need to be closed

  let (prescopes, postscopes) ← if msg.isEmpty then
    let prescopes := m!"-- In root namespace with initial scope"
    logInfo prescopes
    pure (prescopes, m!"")
  else
    let prescopes ← addMessageContext <| MessageData.joinSep msg.toList "\n\n"
    let postscopes ← addMessageContext <| MessageData.joinSep endMsg.toList "\n\n"
    logInfo prescopes
    logInfo postscopes
    pure (prescopes, postscopes)

  return (prescopes, postscopes)
where
  /--
  'Delaborate' open declarations.
  Current limitations:
  - does not check whether or not successive namespaces need `_root_`
  - does not combine commands with `renaming` clauses into a single command
  -/
  describeOpenDecls (ds : List OpenDecl) : CommandElabM (Option MessageData) := do
    let mut lines : Array MessageData := #[]
    let mut simple : Array Name := #[]
    let flush (lines : Array MessageData) (simple : Array Name) : CommandElabM (Array MessageData × Array Name) := do
      if simple.isEmpty then
        return (lines, simple)
      else
        return (lines.push <| ← `(command| open $(simple.map mkIdent)*), #[])
    for d in ds do
      match d with
      | .explicit id decl =>
        (lines, simple) ← flush lines simple
        let ns := decl.getPrefix
        let «from» := Name.mkSimple decl.getString!
        lines := lines.push <| ← `(command| open $(mkIdent ns) renaming $(mkIdent «from») → $(mkIdent id))
      | .simple ns ex =>
        if ex == [] then
          simple := simple.push ns
        else
          (lines, simple) ← flush lines simple
          lines := lines.push <| ← `(command| open $(mkIdent ns) hiding $[$(ex.toArray.map mkIdent)]*)
    (lines, _) ← flush lines simple
    return if lines.isEmpty then none else MessageData.joinSep lines.toList "\n"

  describeOptions (opts : Options) : CommandElabM (Option MessageData) := do
    let mut lines : Array MessageData := #[]
    let decls ← getOptionDecls
    for (name, val) in opts do
      -- `#guard_msgs` sets this option internally, we don't want it to end up in its output
      if name == `Elab.async then
        continue
      let (isSet, isUnknown) :=
        match decls.find? name with
        | some decl => (decl.defValue != val, false)
        | none      => (true, true)
      if isSet then
        let cmd : TSyntax `command ←
          match val with
          | .ofBool true  => `(set_option $(mkIdent name) true)
          | .ofBool false => `(set_option $(mkIdent name) false)
          | .ofString str => `(set_option $(mkIdent name) $(Syntax.mkStrLit str))
          | .ofNat n      => `(set_option $(mkIdent name) $(Syntax.mkNatLit n))
          | _             => `(set_option $(mkIdent name) 0 /- unrepresentable value -/)
        if isUnknown then
          lines := lines.push m!"-- {cmd} -- unknown option"
        else
          lines := lines.push cmd
    return if lines.isEmpty then none else MessageData.joinSep lines.toList "\n"



@[command_elab Parser.Command.where_with_end] def elabWhereWithEnd : CommandElab := fun _ => do
  let _ ← whereWithEndCore



variable {f : Type}
open Lean.Parser.Command

/-
namespace Lean.Elab.Command

open Lean.Parser.Command

variable {f : Type}
-/
/-
end Lean.Elab.Command
-/
#where_with_end


def CommandElabM.run (x : CommandElabM α) (ctx : Context) (s : State) : EIO Exception (α × State) :=
  (x ctx).run s


def CommandElabM.toIO (x : CommandElabM α) (ctx : Context) (s : State) : IO (α × State) := do
  match (← (x.run ctx s).toIO') with
  | Except.error (Exception.error _ msg)   => throw <| IO.userError (← msg.toString)
  | Except.error (Exception.internal id _) => throw <| IO.userError <| "internal exception #" ++ toString id.idx
  | Except.ok a                            => return a
end Lean.Elab.Command
