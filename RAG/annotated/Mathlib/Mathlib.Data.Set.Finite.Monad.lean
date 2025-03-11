/-- If `s : Set α` is a set with `Fintype` instance and `f : α → Set β` is a function such that
each `f a`, `a ∈ s`, has a `Fintype` structure, then `s >>= f` has a `Fintype` structure. -/
def fintypeBind {α β} [DecidableEq β] (s : Set α) [Fintype s] (f : α → Set β)
    (H : ∀ a ∈ s, Fintype (f a)) : Fintype (s >>= f) :=
  Set.fintypeBiUnion s f H


instance fintypeBind' {α β} [DecidableEq β] (s : Set α) [Fintype s] (f : α → Set β)
    [∀ a, Fintype (f a)] : Fintype (s >>= f) :=
  Set.fintypeBiUnion' s f


instance fintypePure : ∀ a : α, Fintype (pure a : Set α) :=
  Set.fintypeSingleton


instance fintypeSeq [DecidableEq β] (f : Set (α → β)) (s : Set α) [Fintype f] [Fintype s] :
    Fintype (f.seq s) := by
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    inst✝² : DecidableEq β
    f : Set (α → β)
    s : Set α
    inst✝¹ : Fintype ↑f
    inst✝ : Fintype ↑s
    ⊢ Fintype ↑(f.seq s)
  -/
  rw [seq_def]
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    inst✝² : DecidableEq β
    f : Set (α → β)
    s : Set α
    inst✝¹ : Fintype ↑f
    inst✝ : Fintype ↑s
    ⊢ Fintype ↑(Set.iUnion fun f_1 => Set.iUnion fun h => Set.image f_1 s)
  -/
  apply Set.fintypeBiUnion'
  /-
    🎉 no goals
  -/


instance fintypeSeq' {α β : Type u} [DecidableEq β] (f : Set (α → β)) (s : Set α) [Fintype f]
    [Fintype s] : Fintype (f <*> s) :=
  Set.fintypeSeq f s


theorem finite_pure (a : α) : (pure a : Set α).Finite :=
  toFinite _


instance finite_seq (f : Set (α → β)) (s : Set α) [Finite f] [Finite s] : Finite (f.seq s) := by
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    f : Set (α → β)
    s : Set α
    inst✝¹ : Finite ↑f
    inst✝ : Finite ↑s
    ⊢ Finite ↑(f.seq s)
  -/
  rw [seq_def]
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    f : Set (α → β)
    s : Set α
    inst✝¹ : Finite ↑f
    inst✝ : Finite ↑s
    ⊢ Finite ↑(Set.iUnion fun f_1 => Set.iUnion fun h => Set.image f_1 s)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem Finite.bind {α β} {s : Set α} {f : α → Set β} (h : s.Finite) (hf : ∀ a ∈ s, (f a).Finite) :
    (s >>= f).Finite :=
  h.biUnion hf


theorem Finite.seq {f : Set (α → β)} {s : Set α} (hf : f.Finite) (hs : s.Finite) :
    (f.seq s).Finite :=
  hf.image2 _ hs


theorem Finite.seq' {α β : Type u} {f : Set (α → β)} {s : Set α} (hf : f.Finite) (hs : s.Finite) :
    (f <*> s).Finite :=
  hf.seq hs


