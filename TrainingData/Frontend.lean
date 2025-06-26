/-
Copyright (c) 2023 Scott Morrison. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Scott Morrison
-/
import Lean.Elab.Frontend
import Lean.Util.Paths
import Batteries.Data.MLList.Basic

/-!
# Compiling Lean sources to obtain `Environment`, `Message`s and `InfoTree`s.

The main entry point is

```
def processInput (input : String) (env? : Option Environment := none)
    (opts : Options := {}) (fileName : Option String := none) (info : Bool := true) :
    IO (Environment × List Message × List InfoTree) :=
  ...
```

which attempts to compile Lean source code, returning an `Environment`,
along with any generated `Message`s and `InfoTree`s.

The optional argument `env?` allows specifying an existing `Environment`, for partial compilation.
If this is non-empty, then the source code may not contain any `import` statements.

You may suppress the generation of `InfoTree`s using `info := false`.

For finer-grained control of compilation, we define a `CompilationStep` structure
which contains information about the results of each command.

You can use `processInput'` to obtain a monadic lazy list of `CompilationStep`s.

The functions `compileModule : Name → IO (List CompilationStep)` and
`moduleInfoTrees : Name → IO (List InfoTree)` are useful for compiling single modules from source.
-/

set_option autoImplicit true

open Lean Elab Frontend

namespace Lean.PersistentArray

/--
Drop the first `n` elements of a `PersistentArray`, returning the results as a `List`.
-/
-- We can't remove the `[Inhabited α]` hypotheses here until
-- `PersistentArray`'s `GetElem` instance also does.
def drop [Inhabited α] (t : PersistentArray α) (n : Nat) : List α :=
  List.range (t.size - n) |>.map fun i => t.get! (n + i)

end Lean.PersistentArray

namespace MLList

/-- Run a lazy list in a `ReaderT` monad on some fixed state. -/
partial def runReaderT [Monad m] (L : MLList (ReaderT.{u, u} ρ m) α) (r : ρ) : MLList m α :=
  squash fun _ =>
    return match ← (uncons L).run r with
    | none => nil
    | some (a, L') => cons a (L'.runReaderT r)

/-- Run a lazy list in a `StateRefT'` monad on some initial state. -/
partial def runStateRefT [Monad m] [MonadLiftT (ST ω) m] (L : MLList (StateRefT' ω σ m) α) (s : σ) :
    MLList m α :=
  squash fun _ =>
    return match ← (uncons L).run s with
    | (none, _) => nil
    | (some (a, L'), s') => cons a (L'.runStateRefT s')

end MLList

namespace Lean.Elab.IO

/--
Results from processing a command.

Contains the `Environment` before and after,
the `src : Substring` and `stx : Syntax` of the command,
and any `Message`s and `InfoTree`s produced while processing.
-/
structure CompilationStep where
  src : Substring
  stx : Syntax
  before : Environment
  after : Environment
  msgs : List Message
  trees : List InfoTree
  parserStateBefore : Parser.ModuleParserState
  commandStateBefore : Command.State


namespace CompilationStep

/--
Process one command, returning a `CompilationStep` and
`done : Bool`, indicating whether this was the last command.
-/
def one : FrontendM (CompilationStep × Bool) := do
  let s := (← get).commandState
  let parserStateBefore := (← get).parserState
  let before := s.env
  let done ← processCommand
  let stx := (← get).commands.back!
  let src := ⟨(← read).inputCtx.input, (← get).cmdPos, (← get).parserState.pos⟩
  let s' := (← get).commandState
  let after := s'.env
  let msgs := s'.messages.unreported.drop s.messages.unreported.size
  let trees := s'.infoState.trees.drop s.infoState.trees.size
  return ({ src, stx, before, after, msgs, trees, parserStateBefore, commandStateBefore := s }, done)

/-- Process all commands in the input. -/
partial def all : FrontendM (List CompilationStep) := do
  let (cmd, done) ← CompilationStep.one
  if done then
    return [cmd]
  else
    return cmd :: (← all)

/-- Return all new `ConstantInfo`s added during the processed command. -/
def diff (cmd : CompilationStep) : List ConstantInfo :=
  cmd.after.constants.map₂.toList.filterMap
    fun (c, i) => if cmd.before.constants.map₂.contains c then none else some i

end CompilationStep

/--
Returns a monadic lazy list of `CompilationStep`s.
This needs to be provided with initial state, see `compilationSteps`.
-/
partial def compilationSteps_aux :  MLList FrontendM CompilationStep :=
  .squash fun _ => aux
where
  /-- Implementation of `compilationSteps_aux`.  -/
  aux := do
    let (cmd, done) ← CompilationStep.one
    if done then
      return .ofList [cmd]
    else
      return .cons cmd (← aux)

/-- Return the the `CompilationStep`s, as a monadic lazy list in `IO`. -/
def compilationSteps (inputCtx : Parser.InputContext) (parserState : Parser.ModuleParserState)
    (commandState : Command.State) : MLList IO CompilationStep :=
  compilationSteps_aux.runReaderT { inputCtx }
    |>.runStateRefT { commandState, parserState, cmdPos := parserState.pos }

/--
Process some text input, with or without an existing environment.
If there is no existing environment, we parse the input for headers (e.g. import statements),
and create a new environment.
Otherwise, we add to the existing environment.
Returns a list containing data about each processed command.

Be aware that Lean does not support compiling multiple files in the same sessions.
Often it works, but if the compiled files do anything complicated with initializers then
nothing is gauranteed.
-/
def processInput' (input : String) (env? : Option Environment := none)
    (opts : Options := {}) (fileName : Option String := none) (info : Bool := true) :
    MLList IO CompilationStep := unsafe do
  let fileName   := fileName.getD "<input>"
  let inputCtx   := Parser.mkInputContext input fileName
  let (parserState, commandState) ← match env? with
  | none => do
    enableInitializersExecution
    let (header, parserState, messages) ← Parser.parseHeader inputCtx
    let (env, messages) ← processHeader header opts messages inputCtx
    pure (parserState, (Command.mkState env messages opts))
  | some env => do
    pure ({ : Parser.ModuleParserState }, Command.mkState env {} opts)
  compilationSteps inputCtx parserState { commandState with infoState.enabled := info }

/--
Process some text input, with or without an existing environment.
If there is no existing environment, we parse the input for headers (e.g. import statements),
and create a new environment.
Otherwise, we add to the existing environment.
Returns the resulting environment, along with a list of messages and info trees.
-/
def processInput (input : String) (env? : Option Environment := none)
    (opts : Options := {}) (fileName : Option String := none) (info : Bool := true) :
    IO (Environment × List Message × List InfoTree) := do
  let steps ← processInput' input env? opts fileName info |>.force
  match steps.getLast? with
  | none => throw <| IO.userError "No commands found in input."
  | some { after, .. } =>
    return (after, steps.flatMap CompilationStep.msgs, steps.flatMap CompilationStep.trees)

open System

/-- Parallel to compile_time_search_path% -/
elab "compile_time_src_search_path%" : term =>
  return toExpr (← initSrcSearchPath)

def findLean (mod : Name) : IO FilePath := do
  let srcSearchPath : SearchPath := compile_time_src_search_path%
  if let some fname ← srcSearchPath.findModuleWithExt "lean" mod then
    return fname
  else
    let fname := FilePath.mk ((← findOLean mod).toString.replace ".lake/build/lib/" "") |>.withExtension "lean"
    if !(← fname.pathExists) then
      throw <| IO.userError s!"Path to {mod} not found"
    return fname

/-- Like `findLean` but produces the version of the file in `Examples/WithImports`. This only supports Lean versions at least
    as recent as Lean v4.3. -/
def findLeanWithImports (mod : Name) (repoName : String) (withImportsDir : String) : IO FilePath := do
  let withImportsPathPrefix := withImportsDir ++ "/"
  let path := (← findOLean mod).toString
  let path := path.replace "./" ""
  let path := path.replace "/" "."
  let path := path.replace s!".lake.packages.{repoName}..lake.build.lib." withImportsPathPrefix
  return FilePath.mk path |>.withExtension "lean"

/-- Given `mod`, the name of the repository `mod` is from, and the `Examples` directory containing relevant JSON files,
    returns the JSON file corresponding to `mod` within `jsonDir`. -/
def findJSONFile (mod : Name) (repoName : String) (jsonDir : String) : IO FilePath := do
  let jsonDirPrefix := jsonDir ++ "/"
  let path := (← findOLean mod).toString
  let path := path.replace "./" ""
  let path := path.replace "/" "."
  let path := path.replace s!".lake.packages.{repoName}..lake.build.lib." jsonDirPrefix
  return FilePath.mk path |>.withExtension "jsonl"

/-- Implementation of `moduleSource`, which is the cached version of this function. -/
def moduleSource' (mod : Name) : IO String := do
  IO.FS.readFile (← findLean mod)

/-- Like `moduleSource'` but uses the version of the module that appears in the `Examples/WithImports` directory -/
def moduleSourceWithImports' (mod : Name) (repoName : String) (withImportsDir : String) : IO String := do
  IO.FS.readFile (← findLeanWithImports mod repoName withImportsDir)

initialize sourceCache : IO.Ref <| Std.HashMap Name String ←
  IO.mkRef .empty

/-- Read the source code of the named module. The results are cached. -/
def moduleSource (mod : Name) : IO String := do
  let m ← sourceCache.get
  match m.get? mod with
  | some r => return r
  | none => do
    let v ← moduleSource' mod
    sourceCache.set (m.insert mod v)
    return v

/-- Like `moduleSource` but uses the version of the module that appears in the `Examples/WithImports` directory -/
def moduleSourceWithImports (mod : Name) (repoName : String) (withImportsDir : String) : IO String := do
  let m ← sourceCache.get
  match m.get? mod with
  | some r => return r
  | none => do
    let v ← moduleSourceWithImports' mod repoName withImportsDir
    sourceCache.set (m.insert mod v)
    return v

/-- Implementation of `compileModule`, which is the cached version of this function. -/
def compileModule' (mod : Name) : MLList IO CompilationStep := do
  Lean.Elab.IO.processInput' (← moduleSource mod) none {} (← findLean mod).toString

/-- Like `compileModule'` but compiles the version of the module that appears in the `Examples/WithImports` directory -/
def compileModuleWithImports' (mod : Name) (repoName : String) (withImportsDir : String) : MLList IO CompilationStep := do
  let modSource ← moduleSourceWithImports mod repoName withImportsDir
  Lean.Elab.IO.processInput' modSource none {} (← findLeanWithImports mod repoName withImportsDir).toString

initialize compilationCache : IO.Ref <| Std.HashMap Name (List CompilationStep) ←
  IO.mkRef .empty

/--
Compile the source file for the named module, returning the
resulting environment, any generated messages, and all info trees.

The results are cached, although be aware that compiling multiple files in the same session
is unsupported, and may lead to exciting results:
you should check all compiled files for error messages if attempting this.
-/
def compileModule (mod : Name) : IO (List CompilationStep) := do
  let m ← compilationCache.get
  match m.get? mod with
  | some r => return r
  | none => do
    let v ← compileModule' mod |>.force
    compilationCache.set (m.insert mod v)
    return v

/-- Compile the source file for the named module, returning all info trees. -/
def moduleInfoTrees (mod : Name) : IO (List InfoTree) := do
  let steps ← compileModule mod
  return steps.flatMap (fun c => c.trees)
