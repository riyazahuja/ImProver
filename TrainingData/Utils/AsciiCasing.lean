import Batteries.Data.Char
import Batteries.Tactic.Basic

namespace Char


abbrev caseFoldAsciiOnly := Char.toLower

def beqCaseInsensitiveAsciiOnly (c₁ c₂ : Char) : Bool :=
  c₁.caseFoldAsciiOnly == c₂.caseFoldAsciiOnly


end Char


namespace String

abbrev caseFoldAsciiOnly (s : String) := s.map Char.caseFoldAsciiOnly


private partial def beqCaseInsensitiveAsciiOnlyImpl (s₁ s₂ : String) : Bool :=
  s₁.length == s₂.length && loop (ToStream.toStream s₁) (ToStream.toStream s₂)
where
  loop i₁ i₂ := match Stream.next? i₁, Stream.next? i₂ with
    | some (c₁, i₁), some (c₂, i₂) => c₁.beqCaseInsensitiveAsciiOnly c₂ && loop i₁ i₂
    | none, none => true
    | _, _ => false


@[implemented_by beqCaseInsensitiveAsciiOnlyImpl]
def beqCaseInsensitiveAsciiOnly (s₁ s₂ : String) : Bool :=
  s₁.caseFoldAsciiOnly == s₂.caseFoldAsciiOnly
