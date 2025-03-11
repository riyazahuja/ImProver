/-- *Bubble sort induction*: Prove that the sorted version of `f` has some property `P`
if `f` satisfies `P` and `P` is preserved on permutations of `f` when swapping two
antitone values. -/
theorem bubble_sort_induction' {n : ℕ} {α : Type*} [LinearOrder α] {f : Fin n → α}
    {P : (Fin n → α) → Prop} (hf : P f)
    (h : ∀ (σ : Equiv.Perm (Fin n)) (i j : Fin n),
      i < j → (f ∘ σ) j < (f ∘ σ) i → P (f ∘ σ) → P (f ∘ σ ∘ Equiv.swap i j)) :
    P (f ∘ sort f) := by
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    P : (Fin n → α) → Prop
    hf : P f
    h : ∀ (σ : Equiv.Perm (Fin n)) (i j : Fin n), LT.lt i j → LT.lt (Function.comp …
    ⊢ P (Function.comp f ⇑(Tuple.sort f))
  -/
  letI := @Preorder.lift _ (Lex (Fin n → α)) _ fun σ : Equiv.Perm (Fin n) => toLex (f ∘ σ)
  refine
    @WellFounded.induction_bot' _ _ _ (IsWellFounded.wf : WellFounded (· < ·))
      (Equiv.refl _) (sort f) P (fun σ => f ∘ σ) (fun σ hσ hfσ => ?_) hf
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    P : (Fin n → α) → Prop
    hf : P f
    h : ∀ (σ : Equiv.Perm (Fin n)) (i j : Fin n), LT.lt i j → LT.lt (Function.comp …
    this : Preorder (Equiv.Perm (Fin n)) := Preorder.lift fun σ => toLex (Function …
    σ : Equiv (Fin n) (Fin n)
    hσ : Ne ((fun σ => Function.comp f ⇑σ) σ) ((fun σ => Function.comp f ⇑σ) (Tupl …
    hfσ : P ((fun σ => Function.comp f ⇑σ) σ)
    ⊢ Exists fun c => And (LT.lt c σ) (P ((fun σ => Function.comp f ⇑σ) c))
  -/
  obtain ⟨i, j, hij₁, hij₂⟩ := antitone_pair_of_not_sorted' hσ
  /-
    case intro.intro.intro
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    P : (Fin n → α) → Prop
    hf : P f
    h : ∀ (σ : Equiv.Perm (Fin n)) (i j : Fin n), LT.lt i j → LT.lt (Function.comp …
    this : Preorder (Equiv.Perm (Fin n)) := Preorder.lift fun σ => toLex (Function …
    σ : Equiv (Fin n) (Fin n)
    hσ : Ne ((fun σ => Function.comp f ⇑σ) σ) ((fun σ => Function.comp f ⇑σ) (Tupl …
    hfσ : P ((fun σ => Function.comp f ⇑σ) σ)
    i j : Fin n
    hij₁ : LT.lt i j
    hij₂ : LT.lt (Function.comp f (⇑σ) j) (Function.comp f (⇑σ) i)
    ⊢ Exists fun c => And (LT.lt c σ) (P ((fun σ => Function.comp f ⇑σ) c))
  -/
  exact ⟨σ * Equiv.swap i j, Pi.lex_desc hij₁.le hij₂, h σ i j hij₁ hij₂ hfσ⟩
  /-
    🎉 no goals
  -/


/-- *Bubble sort induction*: Prove that the sorted version of `f` has some property `P`
if `f` satisfies `P` and `P` is preserved when swapping two antitone values. -/
theorem bubble_sort_induction {n : ℕ} {α : Type*} [LinearOrder α] {f : Fin n → α}
    {P : (Fin n → α) → Prop} (hf : P f)
    (h : ∀ (g : Fin n → α) (i j : Fin n), i < j → g j < g i → P g → P (g ∘ Equiv.swap i j)) :
    P (f ∘ sort f) :=
  bubble_sort_induction' hf fun _ => h _


