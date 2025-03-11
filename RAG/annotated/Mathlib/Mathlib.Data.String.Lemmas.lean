lemma congr_append : ∀ (a b : String), a ++ b = String.mk (a.data ++ b.data)
  | ⟨_⟩, ⟨_⟩ => rfl


@[simp] lemma length_replicate (n : ℕ) (c : Char) : (replicate n c).length = n := by
  /-
    n : Nat
    c : Char
    ⊢ Eq (String.replicate n c).length n
  -/
  simp only [String.length, String.replicate, List.length_replicate]
  /-
    🎉 no goals
  -/


lemma length_eq_list_length (l : List Char) : (String.mk l).length = l.length := by
  /-
    l : List Char
    ⊢ Eq { data := l }.length l.length
  -/
  simp only [String.length]
  /-
    🎉 no goals
  -/


/-- The length of the String returned by `String.leftpad n a c` is equal
  to the larger of `n` and `s.length` -/
@[simp] lemma leftpad_length (n : ℕ) (c : Char) :
    ∀ (s : String), (leftpad n c s).length = max n s.length
              /-
                n : Nat
                c : Char
                s : List Char
                ⊢ Eq (String.leftpad n c { data := s }).length (Max.max n { data := s }.length)
              -/
  | ⟨s⟩ => by simp only [leftpad, String.length, List.leftpad_length]
              /-
                🎉 no goals
              -/


lemma leftpad_prefix (n : ℕ) (c : Char) : ∀ s, IsPrefix (replicate (n - length s) c) (leftpad n c s)
              /-
                n : Nat
                c : Char
                l : List Char
                ⊢ (String.replicate (HSub.hSub n { data := l }.length) c).IsPrefix (String.lef …
              -/
  | ⟨l⟩ => by simp only [IsPrefix, replicate, leftpad, String.length, List.leftpad_prefix]
              /-
                🎉 no goals
              -/


lemma leftpad_suffix (n : ℕ) (c : Char) : ∀ s, IsSuffix s (leftpad n c s)
              /-
                n : Nat
                c : Char
                l : List Char
                ⊢ { data := l }.IsSuffix (String.leftpad n c { data := l })
              -/
  | ⟨l⟩ => by simp only [IsSuffix, replicate, leftpad, String.length, List.leftpad_suffix]
              /-
                🎉 no goals
              -/


