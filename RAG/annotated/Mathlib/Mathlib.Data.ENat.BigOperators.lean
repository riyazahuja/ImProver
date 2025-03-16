lemma sum_iSup {α ι : Type*} {s : Finset α} {f : α → ι → ℕ∞}
    (hf : ∀ i j, ∃ k, ∀ a, f a i ≤ f a k ∧ f a j ≤ f a k) :
    ∑ a ∈ s, ⨆ i, f a i = ⨆ i, ∑ a ∈ s, f a i := by
  /-
    α : Type u_1
    ι : Type u_2
    s : Finset α
    f : α → ι → ENat
    hf : ∀ (i j : ι), Exists fun k => ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.l …
    ⊢ Eq (s.sum fun a => iSup fun i => f a i) (iSup fun i => s.sum fun a => f a i)
  -/
  induction' s using Finset.cons_induction with a s ha ihs
    /-
      case empty
      α : Type u_1
      ι : Type u_2
      f : α → ι → ENat
      hf : ∀ (i j : ι), Exists fun k => ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.l …
      ⊢ Eq (EmptyCollection.emptyCollection.sum fun a => iSup fun i => f a i) (iSup  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    ι : Type u_2
    f : α → ι → ENat
    hf : ∀ (i j : ι), Exists fun k => ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.l …
    a : α
    s : Finset α
    ha : Not (Membership.mem s a)
    ihs : Eq (s.sum fun a => iSup fun i => f a i) (iSup fun i => s.sum fun a => f  …
    ⊢ Eq ((Finset.cons a s ha).sum fun a => iSup fun i => f a i) (iSup fun i => (F …
  -/
  simp_rw [Finset.sum_cons, ihs]
  /-
    case cons
    α : Type u_1
    ι : Type u_2
    f : α → ι → ENat
    hf : ∀ (i j : ι), Exists fun k => ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.l …
    a : α
    s : Finset α
    ha : Not (Membership.mem s a)
    ihs : Eq (s.sum fun a => iSup fun i => f a i) (iSup fun i => s.sum fun a => f  …
    ⊢ Eq (HAdd.hAdd (iSup fun i => f a i) (iSup fun i => s.sum fun a => f a i)) (i …
  -/
  refine iSup_add_iSup fun i j ↦ (hf i j).imp fun k hk ↦ ?_
  /-
    case cons
    α : Type u_1
    ι : Type u_2
    f : α → ι → ENat
    hf : ∀ (i j : ι), Exists fun k => ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.l …
    a : α
    s : Finset α
    ha : Not (Membership.mem s a)
    ihs : Eq (s.sum fun a => iSup fun i => f a i) (iSup fun i => s.sum fun a => f  …
    i j k : ι
    hk : ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.le (f a j) (f a k))
    ⊢ LE.le (HAdd.hAdd (f a i) (s.sum fun a => f a j)) (HAdd.hAdd (f a k) (s.sum f …
  -/
  gcongr
  /-
    case cons.h₁
    α : Type u_1
    ι : Type u_2
    f : α → ι → ENat
    hf : ∀ (i j : ι), Exists fun k => ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.l …
    a : α
    s : Finset α
    ha : Not (Membership.mem s a)
    ihs : Eq (s.sum fun a => iSup fun i => f a i) (iSup fun i => s.sum fun a => f  …
    i j k : ι
    hk : ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.le (f a j) (f a k))
    ⊢ LE.le (f a i) (f a k)
  -/
  exacts [(hk a).1, (hk _).2]
  /-
    🎉 no goals
  -/


lemma sum_iSup_of_monotone {α ι : Type*} [Preorder ι] [IsDirected ι (· ≤ ·)] {s : Finset α}
    {f : α → ι → ℕ∞} (hf : ∀ a, Monotone (f a)) : (∑ a ∈ s, iSup (f a)) = ⨆ n, ∑ a ∈ s, f a n :=
  sum_iSup fun i j ↦ (exists_ge_ge i j).imp fun _k ⟨hi, hj⟩ a ↦ ⟨hf a hi, hf a hj⟩


