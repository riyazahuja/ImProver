import Lean.Exception

open Lean

/-- `SimpAllHint` is used to tag names in `hammer` recommendations to indicate whether the name in correction should be passed to `hammer`'s
    `simp_all` preprocessing call, and if so, whether it should be modified with `←`, `-`, `↑`, or `↓`. Note that the `↑` and `↓` modifiers can
    be present alone or in conjunction with the `←` modifer, so there are different hints corresponding to each different possible combination. -/
inductive SimpAllHint where
| notInSimpAll -- Don't pass the current hint to the `simp_all` preprocessing step
| unmodified -- Pass the current hint to `simp_all` without modification
| simpErase -- Pass the current hint to `simp_all` with the `-` modifier
| simpPreOnly -- Pass the current hint to `simp_all` with just the `↓` modifier
| simpPostOnly -- Pass the current hint to `simp_all` with just the `↑` modifier
| backwardOnly -- Pass the current hint to `simp_all` with just the `←` modifier
| simpPreAndBackward -- Pass the current hint to `simp_all` with both the `↓` and `←` modifiers
| simpPostAndBackward -- Pass the current hint to `simp_all` with both the `↑` and `←` modifier
deriving Inhabited, BEq

open SimpAllHint

def simpAllHintToString : SimpAllHint → String
  | notInSimpAll => "notInSimpAll"
  | unmodified => "unmodified"
  | simpErase => "simpErase"
  | simpPreOnly => "simpPreOnly"
  | simpPostOnly => "simpPostOnly"
  | backwardOnly => "backwardOnly"
  | simpPreAndBackward => "simpPreAndBackward"
  | simpPostAndBackward => "simpPostAndBackward"

instance : ToString SimpAllHint := ⟨simpAllHintToString⟩

def parseSimpAllHint [Monad m] [MonadError m] (s : String) : m SimpAllHint := do
  match s with
  | "notInSimpAll" => return notInSimpAll
  | "unmodified" => return unmodified
  | "simpErase" => return simpErase
  | "simpPreOnly" => return simpPreOnly
  | "simpPostOnly" => return simpPostOnly
  | "backwardOnly" => return backwardOnly
  | "simpPreAndBackward" => return simpPreAndBackward
  | "simpPostAndBackward" => return simpPostAndBackward
  | _ => throwError "Unable to parse {s} as a SimpAllHint"
