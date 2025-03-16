/-- Intuitively, a fixed point operator `fix` is lawful if it satisfies `fix f = f (fix f)` for all
`f`, but this is inconsistent / uninteresting in most cases due to the existence of "exotic"
functions `f`, such as the function that is defined iff its argument is not, familiar from the
halting problem. Instead, this requirement is limited to only functions that are `Continuous` in the
sense of `ω`-complete partial orders, which excludes the example because it is not monotone
(making the input argument less defined can make `f` more defined). -/
class LawfulFix (α : Type*) [OmegaCompletePartialOrder α] extends Fix α where
  fix_eq : ∀ {f : α → α}, ωScottContinuous f → Fix.fix f = f (Fix.fix f)


@[deprecated LawfulFix.fix_eq (since := "2024-08-26")]
theorem LawfulFix.fix_eq' {α} [OmegaCompletePartialOrder α] [LawfulFix α] {f : α → α}
    (hf : ωScottContinuous f) : Fix.fix f = f (Fix.fix f) :=
  LawfulFix.fix_eq hf


theorem approx_mono' {i : ℕ} : Fix.approx f i ≤ Fix.approx f (succ i) := by
  induction i with
  | zero => dsimp [approx]; apply @bot_le _ _ _ (f ⊥)
  | succ _ i_ih => intro; apply f.monotone; apply i_ih


theorem approx_mono ⦃i j : ℕ⦄ (hij : i ≤ j) : approx f i ≤ approx f j := by
  /-
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    i j : Nat
    hij : LE.le i j
    ⊢ LE.le (Part.Fix.approx (⇑f) i) (Part.Fix.approx (⇑f) j)
  -/
  induction' j with j ih
    /-
      case zero
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      i : Nat
      hij : LE.le i 0
      ⊢ LE.le (Part.Fix.approx (⇑f) i) (Part.Fix.approx (⇑f) 0)
    -/
  · cases hij
    /-
      case zero.refl
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      ⊢ LE.le (Part.Fix.approx (⇑f) 0) (Part.Fix.approx (⇑f) 0)
    -/
    exact le_rfl
    /-
      🎉 no goals
    -/
  /-
    case succ
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    i j : Nat
    ih : LE.le i j → LE.le (Part.Fix.approx (⇑f) i) (Part.Fix.approx (⇑f) j)
    hij : LE.le i (HAdd.hAdd j 1)
    ⊢ LE.le (Part.Fix.approx (⇑f) i) (Part.Fix.approx (⇑f) (HAdd.hAdd j 1))
  -/
  cases hij; · exact le_rfl
               /-
                 🎉 no goals
               -/
  /-
    case succ.step
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    i j : Nat
    ih : LE.le i j → LE.le (Part.Fix.approx (⇑f) i) (Part.Fix.approx (⇑f) j)
    a✝ : i.le j
    ⊢ LE.le (Part.Fix.approx (⇑f) i) (Part.Fix.approx (⇑f) (HAdd.hAdd j 1))
  -/
  exact le_trans (ih ‹_›) (approx_mono' f)
  /-
    🎉 no goals
  -/


theorem mem_iff (a : α) (b : β a) : b ∈ Part.fix f a ↔ ∃ i, b ∈ approx f i a := by
  /-
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    a : α
    b : β a
    ⊢ Iff (Membership.mem (Part.fix (⇑f) a) b) (Exists fun i => Membership.mem (Pa …
  -/
  by_cases h₀ : ∃ i : ℕ, (approx f i a).Dom
    /-
      case pos
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      a : α
      b : β a
      h₀ : Exists fun i => (Part.Fix.approx (⇑f) i a).Dom
      ⊢ Iff (Membership.mem (Part.fix (⇑f) a) b) (Exists fun i => Membership.mem (Pa …
    -/
  · simp only [Part.fix_def f h₀]
    /-
      case pos
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      a : α
      b : β a
      h₀ : Exists fun i => (Part.Fix.approx (⇑f) i a).Dom
      ⊢ Iff (Membership.mem (Part.Fix.approx (⇑f) (Nat.find h₀).succ a) b) (Exists f …
    -/
    constructor <;> intro hh
      /-
        case pos.mp
        α : Type u_1
        β : α → Type u_2
        f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
        a : α
        b : β a
        h₀ : Exists fun i => (Part.Fix.approx (⇑f) i a).Dom
        hh : Membership.mem (Part.Fix.approx (⇑f) (Nat.find h₀).succ a) b
        ⊢ Exists fun i => Membership.mem (Part.Fix.approx (⇑f) i a) b
      -/
    · exact ⟨_, hh⟩
      /-
        🎉 no goals
      -/
    /-
      case pos.mpr
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      a : α
      b : β a
      h₀ : Exists fun i => (Part.Fix.approx (⇑f) i a).Dom
      hh : Exists fun i => Membership.mem (Part.Fix.approx (⇑f) i a) b
      ⊢ Membership.mem (Part.Fix.approx (⇑f) (Nat.find h₀).succ a) b
    -/
    have h₁ := Nat.find_spec h₀
    /-
      case pos.mpr
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      a : α
      b : β a
      h₀ : Exists fun i => (Part.Fix.approx (⇑f) i a).Dom
      hh : Exists fun i => Membership.mem (Part.Fix.approx (⇑f) i a) b
      h₁ : (Part.Fix.approx (⇑f) (Nat.find h₀) a).Dom
      ⊢ Membership.mem (Part.Fix.approx (⇑f) (Nat.find h₀).succ a) b
    -/
    rw [dom_iff_mem] at h₁
    /-
      case pos.mpr
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      a : α
      b : β a
      h₀ : Exists fun i => (Part.Fix.approx (⇑f) i a).Dom
      hh : Exists fun i => Membership.mem (Part.Fix.approx (⇑f) i a) b
      h₁ : Exists fun y => Membership.mem (Part.Fix.approx (⇑f) (Nat.find h₀) a) y
      ⊢ Membership.mem (Part.Fix.approx (⇑f) (Nat.find h₀).succ a) b
    -/
    cases' h₁ with y h₁
    /-
      case pos.mpr.intro
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      a : α
      b : β a
      h₀ : Exists fun i => (Part.Fix.approx (⇑f) i a).Dom
      hh : Exists fun i => Membership.mem (Part.Fix.approx (⇑f) i a) b
      y : β a
      h₁ : Membership.mem (Part.Fix.approx (⇑f) (Nat.find h₀) a) y
      ⊢ Membership.mem (Part.Fix.approx (⇑f) (Nat.find h₀).succ a) b
    -/
    replace h₁ := approx_mono' f _ _ h₁
    suffices y = b by
      subst this
      exact h₁
    /-
      case pos.mpr.intro
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      a : α
      b : β a
      h₀ : Exists fun i => (Part.Fix.approx (⇑f) i a).Dom
      hh : Exists fun i => Membership.mem (Part.Fix.approx (⇑f) i a) b
      y : β a
      h₁ : Membership.mem (Part.Fix.approx (⇑f) (Nat.find h₀).succ a) y
      ⊢ Eq y b
    -/
    cases' hh with i hh
    /-
      case pos.mpr.intro.intro
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      a : α
      b : β a
      h₀ : Exists fun i => (Part.Fix.approx (⇑f) i a).Dom
      y : β a
      h₁ : Membership.mem (Part.Fix.approx (⇑f) (Nat.find h₀).succ a) y
      i : Nat
      hh : Membership.mem (Part.Fix.approx (⇑f) i a) b
      ⊢ Eq y b
    -/
    revert h₁; generalize succ (Nat.find h₀) = j; intro h₁
    /-
      case pos.mpr.intro.intro
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      a : α
      b : β a
      h₀ : Exists fun i => (Part.Fix.approx (⇑f) i a).Dom
      y : β a
      i : Nat
      hh : Membership.mem (Part.Fix.approx (⇑f) i a) b
      j : Nat
      h₁ : Membership.mem (Part.Fix.approx (⇑f) j a) y
      ⊢ Eq y b
    -/
    wlog case : i ≤ j
      /-
        case pos.mpr.intro.intro.inr
        α : Type u_1
        β : α → Type u_2
        f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
        a : α
        b : β a
        h₀ : Exists fun i => (Part.Fix.approx (⇑f) i a).Dom
        y : β a
        i : Nat
        hh : Membership.mem (Part.Fix.approx (⇑f) i a) b
        j : Nat
        h₁ : Membership.mem (Part.Fix.approx (⇑f) j a) y
        this : ∀ {α : Type u_1} {β : α → Type u_2} (f : OrderHom ((a : α) → Part (β a) …
        case : Not (LE.le i j)
        ⊢ Eq y b
      -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
    · rcases le_total i j with H | H <;> [skip; symm] <;> apply_assumption <;> assumption
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
    /-
      α✝ : Type u_1
      β✝ : α✝ → Type u_2
      f✝ : OrderHom ((a : α✝) → Part (β✝ a)) ((a : α✝) → Part (β✝ a))
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      a : α
      b : β a
      h₀ : Exists fun i => (Part.Fix.approx (⇑f) i a).Dom
      y : β a
      i : Nat
      hh : Membership.mem (Part.Fix.approx (⇑f) i a) b
      j : Nat
      h₁ : Membership.mem (Part.Fix.approx (⇑f) j a) y
      case : LE.le i j
      ⊢ Eq y b
    -/
    replace hh := approx_mono f case _ _ hh
    /-
      α✝ : Type u_1
      β✝ : α✝ → Type u_2
      f✝ : OrderHom ((a : α✝) → Part (β✝ a)) ((a : α✝) → Part (β✝ a))
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      a : α
      b : β a
      h₀ : Exists fun i => (Part.Fix.approx (⇑f) i a).Dom
      y : β a
      i j : Nat
      h₁ : Membership.mem (Part.Fix.approx (⇑f) j a) y
      case : LE.le i j
      hh : Membership.mem (Part.Fix.approx (⇑f) j a) b
      ⊢ Eq y b
    -/
    apply Part.mem_unique h₁ hh
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      a : α
      b : β a
      h₀ : Not (Exists fun i => (Part.Fix.approx (⇑f) i a).Dom)
      ⊢ Iff (Membership.mem (Part.fix (⇑f) a) b) (Exists fun i => Membership.mem (Pa …
    -/
  · simp only [fix_def' (⇑f) h₀, not_exists, false_iff, not_mem_none]
    /-
      case neg
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      a : α
      b : β a
      h₀ : Not (Exists fun i => (Part.Fix.approx (⇑f) i a).Dom)
      ⊢ ∀ (x : Nat), Not (Membership.mem (Part.Fix.approx (⇑f) x a) b)
    -/
    simp only [dom_iff_mem, not_exists] at h₀
    /-
      case neg
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      a : α
      b : β a
      h₀ : ∀ (x : Nat) (x_1 : β a), Not (Membership.mem (Part.Fix.approx (⇑f) x a) x …
      ⊢ ∀ (x : Nat), Not (Membership.mem (Part.Fix.approx (⇑f) x a) b)
    -/
    intro; apply h₀
           /-
             🎉 no goals
           -/


theorem approx_le_fix (i : ℕ) : approx f i ≤ Part.fix f := fun a b hh ↦ by
  /-
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    i : Nat
    a : α
    b : β a
    hh : Membership.mem (Part.Fix.approx (⇑f) i a) b
    ⊢ Membership.mem (Part.fix (⇑f) a) b
  -/
  rw [mem_iff f]
  /-
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    i : Nat
    a : α
    b : β a
    hh : Membership.mem (Part.Fix.approx (⇑f) i a) b
    ⊢ Exists fun i => Membership.mem (Part.Fix.approx (⇑f) i a) b
  -/
  exact ⟨_, hh⟩
  /-
    🎉 no goals
  -/


theorem exists_fix_le_approx (x : α) : ∃ i, Part.fix f x ≤ approx f i x := by
  /-
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    x : α
    ⊢ Exists fun i => LE.le (Part.fix (⇑f) x) (Part.Fix.approx (⇑f) i x)
  -/
  by_cases hh : ∃ i b, b ∈ approx f i x
    /-
      case pos
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      hh : Exists fun i => Exists fun b => Membership.mem (Part.Fix.approx (⇑f) i x) b
      ⊢ Exists fun i => LE.le (Part.fix (⇑f) x) (Part.Fix.approx (⇑f) i x)
    -/
  · rcases hh with ⟨i, b, hb⟩
    /-
      case pos.intro.intro
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      i : Nat
      b : β x
      hb : Membership.mem (Part.Fix.approx (⇑f) i x) b
      ⊢ Exists fun i => LE.le (Part.fix (⇑f) x) (Part.Fix.approx (⇑f) i x)
    -/
    exists i
    /-
      case pos.intro.intro
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      i : Nat
      b : β x
      hb : Membership.mem (Part.Fix.approx (⇑f) i x) b
      ⊢ LE.le (Part.fix (⇑f) x) (Part.Fix.approx (⇑f) i x)
    -/
    intro b' h'
    /-
      case pos.intro.intro
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      i : Nat
      b : β x
      hb : Membership.mem (Part.Fix.approx (⇑f) i x) b
      b' : β x
      h' : Membership.mem (Part.fix (⇑f) x) b'
      ⊢ Membership.mem (Part.Fix.approx (⇑f) i x) b'
    -/
    have hb' := approx_le_fix f i _ _ hb
    /-
      case pos.intro.intro
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      i : Nat
      b : β x
      hb : Membership.mem (Part.Fix.approx (⇑f) i x) b
      b' : β x
      h' : Membership.mem (Part.fix (⇑f) x) b'
      hb' : Membership.mem (Part.fix (⇑f) x) b
      ⊢ Membership.mem (Part.Fix.approx (⇑f) i x) b'
    -/
    obtain rfl := Part.mem_unique h' hb'
    /-
      case pos.intro.intro
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      i : Nat
      b' : β x
      h' : Membership.mem (Part.fix (⇑f) x) b'
      hb : Membership.mem (Part.Fix.approx (⇑f) i x) b'
      hb' : Membership.mem (Part.fix (⇑f) x) b'
      ⊢ Membership.mem (Part.Fix.approx (⇑f) i x) b'
    -/
    exact hb
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      hh : Not (Exists fun i => Exists fun b => Membership.mem (Part.Fix.approx (⇑f) …
      ⊢ Exists fun i => LE.le (Part.fix (⇑f) x) (Part.Fix.approx (⇑f) i x)
    -/
  · simp only [not_exists] at hh
    /-
      case neg
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      hh : ∀ (x_1 : Nat) (x_2 : β x), Not (Membership.mem (Part.Fix.approx (⇑f) x_1  …
      ⊢ Exists fun i => LE.le (Part.fix (⇑f) x) (Part.Fix.approx (⇑f) i x)
    -/
    exists 0
    /-
      case neg
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      hh : ∀ (x_1 : Nat) (x_2 : β x), Not (Membership.mem (Part.Fix.approx (⇑f) x_1  …
      ⊢ LE.le (Part.fix (⇑f) x) (Part.Fix.approx (⇑f) 0 x)
    -/
    intro b' h'
    /-
      case neg
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      hh : ∀ (x_1 : Nat) (x_2 : β x), Not (Membership.mem (Part.Fix.approx (⇑f) x_1  …
      b' : β x
      h' : Membership.mem (Part.fix (⇑f) x) b'
      ⊢ Membership.mem (Part.Fix.approx (⇑f) 0 x) b'
    -/
    simp only [mem_iff f] at h'
    /-
      case neg
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      hh : ∀ (x_1 : Nat) (x_2 : β x), Not (Membership.mem (Part.Fix.approx (⇑f) x_1  …
      b' : β x
      h' : Exists fun i => Membership.mem (Part.Fix.approx (⇑f) i x) b'
      ⊢ Membership.mem (Part.Fix.approx (⇑f) 0 x) b'
    -/
    cases' h' with i h'
    /-
      case neg.intro
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      hh : ∀ (x_1 : Nat) (x_2 : β x), Not (Membership.mem (Part.Fix.approx (⇑f) x_1  …
      b' : β x
      i : Nat
      h' : Membership.mem (Part.Fix.approx (⇑f) i x) b'
      ⊢ Membership.mem (Part.Fix.approx (⇑f) 0 x) b'
    -/
    cases hh _ _ h'
    /-
      🎉 no goals
    -/


/-- The series of approximations of `fix f` (see `approx`) as a `Chain` -/
def approxChain : Chain ((a : _) → Part <| β a) :=
  ⟨approx f, approx_mono f⟩


theorem le_f_of_mem_approx {x} : x ∈ approxChain f → x ≤ f x := by
  /-
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    x : (a : α) → Part (β a)
    ⊢ Membership.mem (Part.Fix.approxChain f) x → LE.le x (f x)
  -/
  simp only [Membership.mem, forall_exists_index]
  /-
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    x : (a : α) → Part (β a)
    ⊢ ∀ (x_1 : Nat), Eq x ((Part.Fix.approxChain f) x_1) → LE.le x (f x)
  -/
  rintro i rfl
  /-
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    i : Nat
    ⊢ LE.le ((Part.Fix.approxChain f) i) (f ((Part.Fix.approxChain f) i))
  -/
  apply approx_mono'
  /-
    🎉 no goals
  -/


theorem approx_mem_approxChain {i} : approx f i ∈ approxChain f :=
  Stream'.mem_of_get_eq rfl


theorem fix_eq_ωSup : Part.fix f = ωSup (approxChain f) := by
  /-
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    ⊢ Eq (Part.fix ⇑f) (OmegaCompletePartialOrder.ωSup (Part.Fix.approxChain f))
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      ⊢ LE.le (Part.fix ⇑f) (OmegaCompletePartialOrder.ωSup (Part.Fix.approxChain f))
    -/
  · intro x
    /-
      case a
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      ⊢ LE.le (Part.fix (⇑f) x) (OmegaCompletePartialOrder.ωSup (Part.Fix.approxChai …
    -/
    cases' exists_fix_le_approx f x with i hx
    /-
      case a.intro
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      i : Nat
      hx : LE.le (Part.fix (⇑f) x) (Part.Fix.approx (⇑f) i x)
      ⊢ LE.le (Part.fix (⇑f) x) (OmegaCompletePartialOrder.ωSup (Part.Fix.approxChai …
    -/
    trans approx f i.succ x
      /-
        α : Type u_1
        β : α → Type u_2
        f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
        x : α
        i : Nat
        hx : LE.le (Part.fix (⇑f) x) (Part.Fix.approx (⇑f) i x)
        ⊢ LE.le (Part.fix (⇑f) x) (Part.Fix.approx (⇑f) i.succ x)
      -/
    · trans
        /-
          α : Type u_1
          β : α → Type u_2
          f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
          x : α
          i : Nat
          hx : LE.le (Part.fix (⇑f) x) (Part.Fix.approx (⇑f) i x)
          ⊢ LE.le (Part.fix (⇑f) x) ?m.8501
        -/
      · apply hx
        /-
          🎉 no goals
        -/
        /-
          α : Type u_1
          β : α → Type u_2
          f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
          x : α
          i : Nat
          hx : LE.le (Part.fix (⇑f) x) (Part.Fix.approx (⇑f) i x)
          ⊢ LE.le (Part.Fix.approx (⇑f) i x) (Part.Fix.approx (⇑f) i.succ x)
        -/
      · apply approx_mono' f
        /-
          🎉 no goals
        -/
    /-
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      i : Nat
      hx : LE.le (Part.fix (⇑f) x) (Part.Fix.approx (⇑f) i x)
      ⊢ LE.le (Part.Fix.approx (⇑f) i.succ x) (OmegaCompletePartialOrder.ωSup (Part. …
    -/
    apply le_ωSup_of_le i.succ
    /-
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      i : Nat
      hx : LE.le (Part.fix (⇑f) x) (Part.Fix.approx (⇑f) i x)
      ⊢ LE.le (Part.Fix.approx (⇑f) i.succ x) (((Part.Fix.approxChain f).map (Pi.eva …
    -/
    dsimp [approx]
    /-
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      x : α
      i : Nat
      hx : LE.le (Part.fix (⇑f) x) (Part.Fix.approx (⇑f) i x)
      ⊢ LE.le (f (Part.Fix.approx (⇑f) i) x) ((Part.Fix.approxChain f) (HAdd.hAdd i  …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      ⊢ LE.le (OmegaCompletePartialOrder.ωSup (Part.Fix.approxChain f)) (Part.fix ⇑f)
    -/
  · apply ωSup_le _ _ _
    /-
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      ⊢ ∀ (i : Nat), LE.le ((Part.Fix.approxChain f) i) (Part.fix ⇑f)
    -/
    simp only [Fix.approxChain, OrderHom.coe_mk]
    /-
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      ⊢ ∀ (i : Nat), LE.le ({ toFun := Part.Fix.approx ⇑f, monotone' := ⋯ } i) (Part …
    -/
    intro y x
    /-
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      y : Nat
      x : α
      ⊢ LE.le ({ toFun := Part.Fix.approx ⇑f, monotone' := ⋯ } y x) (Part.fix (⇑f) x)
    -/
    apply approx_le_fix f
    /-
      🎉 no goals
    -/


theorem fix_le {X : (a : _) → Part <| β a} (hX : f X ≤ X) : Part.fix f ≤ X := by
  /-
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    X : (a : α) → Part (β a)
    hX : LE.le (f X) X
    ⊢ LE.le (Part.fix ⇑f) X
  -/
  rw [fix_eq_ωSup f]
  /-
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    X : (a : α) → Part (β a)
    hX : LE.le (f X) X
    ⊢ LE.le (OmegaCompletePartialOrder.ωSup (Part.Fix.approxChain f)) X
  -/
  apply ωSup_le _ _ _
  /-
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    X : (a : α) → Part (β a)
    hX : LE.le (f X) X
    ⊢ ∀ (i : Nat), LE.le ((Part.Fix.approxChain f) i) X
  -/
  simp only [Fix.approxChain, OrderHom.coe_mk]
  /-
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    X : (a : α) → Part (β a)
    hX : LE.le (f X) X
    ⊢ ∀ (i : Nat), LE.le ({ toFun := Part.Fix.approx ⇑f, monotone' := ⋯ } i) X
  -/
  intro i
  induction i with
  | zero => dsimp [Fix.approx]; apply bot_le
  | succ _ i_ih =>
    trans f X
    · apply f.monotone i_ih
    · apply hX


theorem fix_eq_ωSup_of_ωScottContinuous (hc : ωScottContinuous g) : Part.fix g =
    ωSup (approxChain (⟨g,hc.monotone⟩ : ((a : _) → Part <| β a) →o (a : _) → Part <| β a)) := by
  /-
    α : Type u_1
    β : α → Type u_2
    g : ((a : α) → Part (β a)) → (a : α) → Part (β a)
    hc : OmegaCompletePartialOrder.ωScottContinuous g
    ⊢ Eq (Part.fix g) (OmegaCompletePartialOrder.ωSup (Part.Fix.approxChain { toFu …
  -/
  rw [← fix_eq_ωSup]
  /-
    α : Type u_1
    β : α → Type u_2
    g : ((a : α) → Part (β a)) → (a : α) → Part (β a)
    hc : OmegaCompletePartialOrder.ωScottContinuous g
    ⊢ Eq (Part.fix g) (Part.fix ⇑{ toFun := g, monotone' := ⋯ })
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem fix_eq_of_ωScottContinuous (hc : ωScottContinuous g) :
    Part.fix g = g (Part.fix g) := by
  /-
    α : Type u_1
    β : α → Type u_2
    g : ((a : α) → Part (β a)) → (a : α) → Part (β a)
    hc : OmegaCompletePartialOrder.ωScottContinuous g
    ⊢ Eq (Part.fix g) (g (Part.fix g))
  -/
  rw [fix_eq_ωSup_of_ωScottContinuous hc, hc.map_ωSup]
  /-
    α : Type u_1
    β : α → Type u_2
    g : ((a : α) → Part (β a)) → (a : α) → Part (β a)
    hc : OmegaCompletePartialOrder.ωScottContinuous g
    ⊢ Eq (OmegaCompletePartialOrder.ωSup (Part.Fix.approxChain { toFun := g, monot …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      β : α → Type u_2
      g : ((a : α) → Part (β a)) → (a : α) → Part (β a)
      hc : OmegaCompletePartialOrder.ωScottContinuous g
      ⊢ LE.le (OmegaCompletePartialOrder.ωSup (Part.Fix.approxChain { toFun := g, mo …
    -/
  · apply ωSup_le_ωSup_of_le _
    /-
      α : Type u_1
      β : α → Type u_2
      g : ((a : α) → Part (β a)) → (a : α) → Part (β a)
      hc : OmegaCompletePartialOrder.ωScottContinuous g
      ⊢ LE.le (Part.Fix.approxChain { toFun := g, monotone' := ⋯ }) ((Part.Fix.appro …
    -/
    intro i
    /-
      α : Type u_1
      β : α → Type u_2
      g : ((a : α) → Part (β a)) → (a : α) → Part (β a)
      hc : OmegaCompletePartialOrder.ωScottContinuous g
      i : Nat
      ⊢ Exists fun j => LE.le ((Part.Fix.approxChain { toFun := g, monotone' := ⋯ }) …
    -/
    exists i
    /-
      α : Type u_1
      β : α → Type u_2
      g : ((a : α) → Part (β a)) → (a : α) → Part (β a)
      hc : OmegaCompletePartialOrder.ωScottContinuous g
      i : Nat
      ⊢ LE.le ((Part.Fix.approxChain { toFun := g, monotone' := ⋯ }) i) (((Part.Fix. …
    -/
    intro x
    /-
      α : Type u_1
      β : α → Type u_2
      g : ((a : α) → Part (β a)) → (a : α) → Part (β a)
      hc : OmegaCompletePartialOrder.ωScottContinuous g
      i : Nat
      x : α
      ⊢ LE.le ((Part.Fix.approxChain { toFun := g, monotone' := ⋯ }) i x) (((Part.Fi …
    -/
    apply le_f_of_mem_approx _ ⟨i, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      β : α → Type u_2
      g : ((a : α) → Part (β a)) → (a : α) → Part (β a)
      hc : OmegaCompletePartialOrder.ωScottContinuous g
      ⊢ LE.le (OmegaCompletePartialOrder.ωSup ((Part.Fix.approxChain { toFun := g, m …
    -/
  · apply ωSup_le_ωSup_of_le _
    /-
      α : Type u_1
      β : α → Type u_2
      g : ((a : α) → Part (β a)) → (a : α) → Part (β a)
      hc : OmegaCompletePartialOrder.ωScottContinuous g
      ⊢ LE.le ((Part.Fix.approxChain { toFun := g, monotone' := ⋯ }).map { toFun :=  …
    -/
    intro i
    /-
      α : Type u_1
      β : α → Type u_2
      g : ((a : α) → Part (β a)) → (a : α) → Part (β a)
      hc : OmegaCompletePartialOrder.ωScottContinuous g
      i : Nat
      ⊢ Exists fun j => LE.le (((Part.Fix.approxChain { toFun := g, monotone' := ⋯ } …
    -/
    exists i.succ
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
@[deprecated fix_eq_of_ωScottContinuous (since := "2024-08-26")]
theorem fix_eq (hc : Continuous f) : Part.fix f = f (Part.fix f) := by
  /-
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    hc : OmegaCompletePartialOrder.Continuous f
    ⊢ Eq (Part.fix ⇑f) (f (Part.fix ⇑f))
  -/
  rw [fix_eq_ωSup f, hc]
  /-
    α : Type u_1
    β : α → Type u_2
    f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
    hc : OmegaCompletePartialOrder.Continuous f
    ⊢ Eq (OmegaCompletePartialOrder.ωSup (Part.Fix.approxChain f)) (OmegaCompleteP …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      hc : OmegaCompletePartialOrder.Continuous f
      ⊢ LE.le (OmegaCompletePartialOrder.ωSup (Part.Fix.approxChain f)) (OmegaComple …
    -/
  · apply ωSup_le_ωSup_of_le _
    /-
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      hc : OmegaCompletePartialOrder.Continuous f
      ⊢ LE.le (Part.Fix.approxChain f) ((Part.Fix.approxChain f).map f)
    -/
    intro i
    /-
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      hc : OmegaCompletePartialOrder.Continuous f
      i : Nat
      ⊢ Exists fun j => LE.le ((Part.Fix.approxChain f) i) (((Part.Fix.approxChain f …
    -/
    exists i
    /-
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      hc : OmegaCompletePartialOrder.Continuous f
      i : Nat
      ⊢ LE.le ((Part.Fix.approxChain f) i) (((Part.Fix.approxChain f).map f) i)
    -/
    intro x
    -- intros x y hx,
    /-
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      hc : OmegaCompletePartialOrder.Continuous f
      i : Nat
      x : α
      ⊢ LE.le ((Part.Fix.approxChain f) i x) (((Part.Fix.approxChain f).map f) i x)
    -/
    apply le_f_of_mem_approx _ ⟨i, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      hc : OmegaCompletePartialOrder.Continuous f
      ⊢ LE.le (OmegaCompletePartialOrder.ωSup ((Part.Fix.approxChain f).map f)) (Ome …
    -/
  · apply ωSup_le_ωSup_of_le _
    /-
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      hc : OmegaCompletePartialOrder.Continuous f
      ⊢ LE.le ((Part.Fix.approxChain f).map f) (Part.Fix.approxChain f)
    -/
    intro i
    /-
      α : Type u_1
      β : α → Type u_2
      f : OrderHom ((a : α) → Part (β a)) ((a : α) → Part (β a))
      hc : OmegaCompletePartialOrder.Continuous f
      i : Nat
      ⊢ Exists fun j => LE.le (((Part.Fix.approxChain f).map f) i) ((Part.Fix.approx …
    -/
    exists i.succ
    /-
      🎉 no goals
    -/


/-- `toUnit` as a monotone function -/
@[simps]
def toUnitMono (f : Part α →o Part α) : (Unit → Part α) →o Unit → Part α where
  toFun x u := f (x u)
  monotone' x y (h : x ≤ y) u := f.monotone <| h u


theorem ωScottContinuous_toUnitMono (f : Part α → Part α) (hc : ωScottContinuous f) :
    ωScottContinuous (toUnitMono ⟨f,hc.monotone⟩) := .of_map_ωSup_of_orderHom fun _ => by
  /-
    α : Type u_1
    f : Part α → Part α
    hc : OmegaCompletePartialOrder.ωScottContinuous f
    x✝ : OmegaCompletePartialOrder.Chain (Unit → Part α)
    ⊢ Eq ((Part.toUnitMono { toFun := f, monotone' := ⋯ }) (OmegaCompletePartialOr …
  -/
  ext ⟨⟩ : 1
  /-
    case h.unit
    α : Type u_1
    f : Part α → Part α
    hc : OmegaCompletePartialOrder.ωScottContinuous f
    x✝ : OmegaCompletePartialOrder.Chain (Unit → Part α)
    ⊢ Eq ((Part.toUnitMono { toFun := f, monotone' := ⋯ }) (OmegaCompletePartialOr …
  -/
  dsimp [OmegaCompletePartialOrder.ωSup]
  /-
    case h.unit
    α : Type u_1
    f : Part α → Part α
    hc : OmegaCompletePartialOrder.ωScottContinuous f
    x✝ : OmegaCompletePartialOrder.Chain (Unit → Part α)
    ⊢ Eq (f (Part.ωSup (x✝.map (Pi.evalOrderHom PUnit.unit)))) (Part.ωSup ((x✝.map …
  -/
  erw [hc.map_ωSup, Chain.map_comp]; rfl
                                     /-
                                       🎉 no goals
                                     -/


set_option linter.deprecated false in
@[deprecated ωScottContinuous_toUnitMono (since := "2024-08-26")]
theorem to_unit_cont (f : Part α →o Part α) (hc : Continuous f) : Continuous (toUnitMono f)
  | _ => by
    /-
      α : Type u_1
      f : OrderHom (Part α) (Part α)
      hc : OmegaCompletePartialOrder.Continuous f
      x✝ : OmegaCompletePartialOrder.Chain (Unit → Part α)
      ⊢ Eq ((Part.toUnitMono f) (OmegaCompletePartialOrder.ωSup x✝)) (OmegaCompleteP …
    -/
    ext ⟨⟩ : 1
    /-
      case h.unit
      α : Type u_1
      f : OrderHom (Part α) (Part α)
      hc : OmegaCompletePartialOrder.Continuous f
      x✝ : OmegaCompletePartialOrder.Chain (Unit → Part α)
      ⊢ Eq ((Part.toUnitMono f) (OmegaCompletePartialOrder.ωSup x✝) PUnit.unit) (Ome …
    -/
    dsimp [OmegaCompletePartialOrder.ωSup]
    /-
      case h.unit
      α : Type u_1
      f : OrderHom (Part α) (Part α)
      hc : OmegaCompletePartialOrder.Continuous f
      x✝ : OmegaCompletePartialOrder.Chain (Unit → Part α)
      ⊢ Eq (f (Part.ωSup (x✝.map (Pi.evalOrderHom PUnit.unit)))) (Part.ωSup ((x✝.map …
    -/
    erw [hc, Chain.map_comp]; rfl
                              /-
                                🎉 no goals
                              -/


instance lawfulFix : LawfulFix (Part α) :=
  ⟨fun {f : Part α → Part α} hc ↦ show Part.fix (toUnitMono ⟨f,hc.monotone⟩) () = _ by
    /-
      α : Type u_1
      β : α → Type u_2
      f : Part α → Part α
      hc : OmegaCompletePartialOrder.ωScottContinuous f
      ⊢ Eq (Part.fix (⇑(Part.toUnitMono { toFun := f, monotone' := ⋯ })) Unit.unit)  …
    -/
    rw [Part.fix_eq_of_ωScottContinuous (ωScottContinuous_toUnitMono f hc)]; rfl⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


instance lawfulFix {β} : LawfulFix (α → Part β) :=
  ⟨fun {_f} ↦ Part.fix_eq_of_ωScottContinuous⟩


/-- `Sigma.curry` as a monotone function. -/
@[simps]
def monotoneCurry [(x y : _) → Preorder <| γ x y] :
    (∀ x : Σa, β a, γ x.1 x.2) →o ∀ (a) (b : β a), γ a b where
  toFun := curry
  monotone' _x _y h a b := h ⟨a, b⟩


/-- `Sigma.uncurry` as a monotone function. -/
@[simps]
def monotoneUncurry [(x y : _) → Preorder <| γ x y] :
    (∀ (a) (b : β a), γ a b) →o ∀ x : Σa, β a, γ x.1 x.2 where
  toFun := uncurry
  monotone' _x _y h a := h a.1 a.2


theorem ωScottContinuous_curry :
    ωScottContinuous (monotoneCurry α β γ) :=
  ωScottContinuous.of_map_ωSup_of_orderHom fun c ↦ by
    /-
      α : Type u_1
      β : α → Type u_2
      γ : (a : α) → β a → Type u_3
      inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
      c : OmegaCompletePartialOrder.Chain ((x : Sigma fun a => β a) → γ x.fst x.snd)
      ⊢ Eq ((Pi.monotoneCurry α β γ) (OmegaCompletePartialOrder.ωSup c)) (OmegaCompl …
    -/
    ext x y
    /-
      case h.h
      α : Type u_1
      β : α → Type u_2
      γ : (a : α) → β a → Type u_3
      inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
      c : OmegaCompletePartialOrder.Chain ((x : Sigma fun a => β a) → γ x.fst x.snd)
      x : α
      y : β x
      ⊢ Eq ((Pi.monotoneCurry α β γ) (OmegaCompletePartialOrder.ωSup c) x y) (OmegaC …
    -/
    dsimp [curry, ωSup]
    /-
      case h.h
      α : Type u_1
      β : α → Type u_2
      γ : (a : α) → β a → Type u_3
      inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
      c : OmegaCompletePartialOrder.Chain ((x : Sigma fun a => β a) → γ x.fst x.snd)
      x : α
      y : β x
      ⊢ Eq (OmegaCompletePartialOrder.ωSup (c.map (Pi.evalOrderHom ⟨x, y⟩))) (OmegaC …
    -/
    rw [map_comp, map_comp]
    /-
      case h.h
      α : Type u_1
      β : α → Type u_2
      γ : (a : α) → β a → Type u_3
      inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
      c : OmegaCompletePartialOrder.Chain ((x : Sigma fun a => β a) → γ x.fst x.snd)
      x : α
      y : β x
      ⊢ Eq (OmegaCompletePartialOrder.ωSup (c.map (Pi.evalOrderHom ⟨x, y⟩))) (OmegaC …
    -/
    rfl
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
@[deprecated ωScottContinuous_curry (since := "2024-08-26")]
theorem continuous_curry : Continuous <| monotoneCurry α β γ := fun c ↦ by
  /-
    α : Type u_1
    β : α → Type u_2
    γ : (a : α) → β a → Type u_3
    inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
    c : OmegaCompletePartialOrder.Chain ((x : Sigma fun a => β a) → γ x.fst x.snd)
    ⊢ Eq ((Pi.monotoneCurry α β γ) (OmegaCompletePartialOrder.ωSup c)) (OmegaCompl …
  -/
  ext x y
  /-
    case h.h
    α : Type u_1
    β : α → Type u_2
    γ : (a : α) → β a → Type u_3
    inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
    c : OmegaCompletePartialOrder.Chain ((x : Sigma fun a => β a) → γ x.fst x.snd)
    x : α
    y : β x
    ⊢ Eq ((Pi.monotoneCurry α β γ) (OmegaCompletePartialOrder.ωSup c) x y) (OmegaC …
  -/
  dsimp [curry, ωSup]
  /-
    case h.h
    α : Type u_1
    β : α → Type u_2
    γ : (a : α) → β a → Type u_3
    inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
    c : OmegaCompletePartialOrder.Chain ((x : Sigma fun a => β a) → γ x.fst x.snd)
    x : α
    y : β x
    ⊢ Eq (OmegaCompletePartialOrder.ωSup (c.map (Pi.evalOrderHom ⟨x, y⟩))) (OmegaC …
  -/
  rw [map_comp, map_comp]
  /-
    case h.h
    α : Type u_1
    β : α → Type u_2
    γ : (a : α) → β a → Type u_3
    inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
    c : OmegaCompletePartialOrder.Chain ((x : Sigma fun a => β a) → γ x.fst x.snd)
    x : α
    y : β x
    ⊢ Eq (OmegaCompletePartialOrder.ωSup (c.map (Pi.evalOrderHom ⟨x, y⟩))) (OmegaC …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem ωScottContinuous_uncurry :
    ωScottContinuous (monotoneUncurry α β γ) :=
    .of_map_ωSup_of_orderHom fun c ↦ by
  /-
    α : Type u_1
    β : α → Type u_2
    γ : (a : α) → β a → Type u_3
    inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
    c : OmegaCompletePartialOrder.Chain ((a : α) → (b : β a) → γ a b)
    ⊢ Eq ((Pi.monotoneUncurry α β γ) (OmegaCompletePartialOrder.ωSup c)) (OmegaCom …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : α → Type u_2
    γ : (a : α) → β a → Type u_3
    inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
    c : OmegaCompletePartialOrder.Chain ((a : α) → (b : β a) → γ a b)
    x : α
    y : β x
    ⊢ Eq ((Pi.monotoneUncurry α β γ) (OmegaCompletePartialOrder.ωSup c) ⟨x, y⟩) (O …
  -/
  dsimp [uncurry, ωSup]
  /-
    case h.mk
    α : Type u_1
    β : α → Type u_2
    γ : (a : α) → β a → Type u_3
    inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
    c : OmegaCompletePartialOrder.Chain ((a : α) → (b : β a) → γ a b)
    x : α
    y : β x
    ⊢ Eq (OmegaCompletePartialOrder.ωSup ((c.map (Pi.evalOrderHom x)).map (Pi.eval …
  -/
  rw [map_comp, map_comp]
  /-
    case h.mk
    α : Type u_1
    β : α → Type u_2
    γ : (a : α) → β a → Type u_3
    inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
    c : OmegaCompletePartialOrder.Chain ((a : α) → (b : β a) → γ a b)
    x : α
    y : β x
    ⊢ Eq (OmegaCompletePartialOrder.ωSup (c.map ((Pi.evalOrderHom y).comp (Pi.eval …
  -/
  rfl
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated ωScottContinuous_uncurry  (since := "2024-08-26")]
theorem continuous_uncurry : Continuous <| monotoneUncurry α β γ := fun c ↦ by
  /-
    α : Type u_1
    β : α → Type u_2
    γ : (a : α) → β a → Type u_3
    inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
    c : OmegaCompletePartialOrder.Chain ((a : α) → (b : β a) → γ a b)
    ⊢ Eq ((Pi.monotoneUncurry α β γ) (OmegaCompletePartialOrder.ωSup c)) (OmegaCom …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : α → Type u_2
    γ : (a : α) → β a → Type u_3
    inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
    c : OmegaCompletePartialOrder.Chain ((a : α) → (b : β a) → γ a b)
    x : α
    y : β x
    ⊢ Eq ((Pi.monotoneUncurry α β γ) (OmegaCompletePartialOrder.ωSup c) ⟨x, y⟩) (O …
  -/
  dsimp [uncurry, ωSup]
  /-
    case h.mk
    α : Type u_1
    β : α → Type u_2
    γ : (a : α) → β a → Type u_3
    inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
    c : OmegaCompletePartialOrder.Chain ((a : α) → (b : β a) → γ a b)
    x : α
    y : β x
    ⊢ Eq (OmegaCompletePartialOrder.ωSup ((c.map (Pi.evalOrderHom x)).map (Pi.eval …
  -/
  rw [map_comp, map_comp]
  /-
    case h.mk
    α : Type u_1
    β : α → Type u_2
    γ : (a : α) → β a → Type u_3
    inst✝ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
    c : OmegaCompletePartialOrder.Chain ((a : α) → (b : β a) → γ a b)
    x : α
    y : β x
    ⊢ Eq (OmegaCompletePartialOrder.ωSup (c.map ((Pi.evalOrderHom y).comp (Pi.eval …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance hasFix [Fix <| (x : Sigma β) → γ x.1 x.2] : Fix ((x : _) → (y : β x) → γ x y) :=
  ⟨fun f ↦ curry (fix <| uncurry ∘ f ∘ curry)⟩


theorem uncurry_curry_ωScottContinuous (hc : ωScottContinuous f) :
    ωScottContinuous <| (monotoneUncurry α β γ).comp <|
      (⟨f,hc.monotone⟩ : ((x : _) → (y : β x) → γ x y) →o (x : _) → (y : β x) → γ x y).comp <|
      monotoneCurry α β γ :=
  (ωScottContinuous_uncurry _ _ _).comp (hc.comp (ωScottContinuous_curry _ _ _))


set_option linter.deprecated false in
@[deprecated uncurry_curry_ωScottContinuous  (since := "2024-08-26")]
theorem uncurry_curry_continuous {f : ((x : _) → (y : β x) → γ x y) →o (x : _) → (y : β x) → γ x y}
    (hc : Continuous f) :
    Continuous <| (monotoneUncurry α β γ).comp <| f.comp <| monotoneCurry α β γ :=
  continuous_comp _ _ (continuous_comp _ _ (continuous_curry _ _ _) hc) (continuous_uncurry _ _ _)


instance lawfulFix' [LawfulFix <| (x : Sigma β) → γ x.1 x.2] :
    LawfulFix ((x y : _) → γ x y) where
  fix_eq {_f} hc := by
    /-
      α : Type u_1
      β : α → Type u_2
      γ : (a : α) → β a → Type u_3
      inst✝¹ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
      inst✝ : LawfulFix ((x : Sigma β) → γ x.fst x.snd)
      _f : ((x : α) → (y : β x) → γ x y) → (x : α) → (y : β x) → γ x y
      hc : OmegaCompletePartialOrder.ωScottContinuous _f
      ⊢ Eq (Fix.fix _f) (_f (Fix.fix _f))
    -/
    dsimp [fix]
    /-
      α : Type u_1
      β : α → Type u_2
      γ : (a : α) → β a → Type u_3
      inst✝¹ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
      inst✝ : LawfulFix ((x : Sigma β) → γ x.fst x.snd)
      _f : ((x : α) → (y : β x) → γ x y) → (x : α) → (y : β x) → γ x y
      hc : OmegaCompletePartialOrder.ωScottContinuous _f
      ⊢ Eq (Sigma.curry (Fix.fix (Function.comp Sigma.uncurry (Function.comp _f Sigm …
    -/
    conv_lhs => erw [LawfulFix.fix_eq (uncurry_curry_ωScottContinuous hc)]
    /-
      α : Type u_1
      β : α → Type u_2
      γ : (a : α) → β a → Type u_3
      inst✝¹ : (x : α) → (y : β x) → OmegaCompletePartialOrder (γ x y)
      inst✝ : LawfulFix ((x : Sigma β) → γ x.fst x.snd)
      _f : ((x : α) → (y : β x) → γ x y) → (x : α) → (y : β x) → γ x y
      hc : OmegaCompletePartialOrder.ωScottContinuous _f
      ⊢ Eq (Sigma.curry (((Pi.monotoneUncurry α β γ).comp ({ toFun := _f, monotone'  …
    -/
    rfl
    /-
      🎉 no goals
    -/


