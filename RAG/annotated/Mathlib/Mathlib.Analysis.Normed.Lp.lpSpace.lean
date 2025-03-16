/-- The property that `f : ∀ i : α, E i`
* is finitely supported, if `p = 0`, or
* admits an upper bound for `Set.range (fun i ↦ ‖f i‖)`, if `p = ∞`, or
* has the series `∑' i, ‖f i‖ ^ p` be summable, if `0 < p < ∞`. -/
def Memℓp (f : ∀ i, E i) (p : ℝ≥0∞) : Prop :=
  if p = 0 then Set.Finite { i | f i ≠ 0 }
  else if p = ∞ then BddAbove (Set.range fun i => ‖f i‖)
  else Summable fun i => ‖f i‖ ^ p.toReal


theorem memℓp_zero_iff {f : ∀ i, E i} : Memℓp f 0 ↔ Set.Finite { i | f i ≠ 0 } := by
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f : (i : α) → E i
    ⊢ Iff (Memℓp f 0) (setOf fun i => Ne (f i) 0).Finite
  -/
  dsimp [Memℓp]
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f : (i : α) → E i
    ⊢ Iff (ite (Eq 0 0) (setOf fun i => Not (Eq (f i) 0)).Finite (Summable fun i = …
  -/
  rw [if_pos rfl]
  /-
    🎉 no goals
  -/


theorem memℓp_zero {f : ∀ i, E i} (hf : Set.Finite { i | f i ≠ 0 }) : Memℓp f 0 :=
  memℓp_zero_iff.2 hf


theorem memℓp_infty_iff {f : ∀ i, E i} : Memℓp f ∞ ↔ BddAbove (Set.range fun i => ‖f i‖) := by
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f : (i : α) → E i
    ⊢ Iff (Memℓp f Top.top) (BddAbove (Set.range fun i => Norm.norm (f i)))
  -/
  simp [Memℓp]
  /-
    🎉 no goals
  -/


theorem memℓp_infty {f : ∀ i, E i} (hf : BddAbove (Set.range fun i => ‖f i‖)) : Memℓp f ∞ :=
  memℓp_infty_iff.2 hf


theorem memℓp_gen_iff (hp : 0 < p.toReal) {f : ∀ i, E i} :
    Memℓp f p ↔ Summable fun i => ‖f i‖ ^ p.toReal := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : LT.lt 0 p.toReal
    f : (i : α) → E i
    ⊢ Iff (Memℓp f p) (Summable fun i => HPow.hPow (Norm.norm (f i)) p.toReal)
  -/
  rw [ENNReal.toReal_pos_iff] at hp
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : And (LT.lt 0 p) (LT.lt p Top.top)
    f : (i : α) → E i
    ⊢ Iff (Memℓp f p) (Summable fun i => HPow.hPow (Norm.norm (f i)) p.toReal)
  -/
  dsimp [Memℓp]
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : And (LT.lt 0 p) (LT.lt p Top.top)
    f : (i : α) → E i
    ⊢ Iff (ite (Eq p 0) (setOf fun i => Not (Eq (f i) 0)).Finite (ite (Eq p Top.to …
  -/
  rw [if_neg hp.1.ne', if_neg hp.2.ne]
  /-
    🎉 no goals
  -/


theorem memℓp_gen {f : ∀ i, E i} (hf : Summable fun i => ‖f i‖ ^ p.toReal) : Memℓp f p := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f : (i : α) → E i
    hf : Summable fun i => HPow.hPow (Norm.norm (f i)) p.toReal
    ⊢ Memℓp f p
  -/
  rcases p.trichotomy with (rfl | rfl | hp)
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hf : Summable fun i => HPow.hPow (Norm.norm (f i)) (ENNReal.toReal 0)
      ⊢ Memℓp f 0
    -/
  · apply memℓp_zero
    /-
      case inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hf : Summable fun i => HPow.hPow (Norm.norm (f i)) (ENNReal.toReal 0)
      ⊢ (setOf fun i => Ne (f i) 0).Finite
    -/
    have H : Summable fun _ : α => (1 : ℝ) := by simpa using hf
    /-
      case inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hf : Summable fun i => HPow.hPow (Norm.norm (f i)) (ENNReal.toReal 0)
      H : Summable fun x => 1
      ⊢ (setOf fun i => Ne (f i) 0).Finite
    -/
    exact (Set.Finite.of_summable_const (by norm_num) H).subset (Set.subset_univ _)
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hf : Summable fun i => HPow.hPow (Norm.norm (f i)) Top.top.toReal
      ⊢ Memℓp f Top.top
    -/
  · apply memℓp_infty
    /-
      case inr.inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hf : Summable fun i => HPow.hPow (Norm.norm (f i)) Top.top.toReal
      ⊢ BddAbove (Set.range fun i => Norm.norm (f i))
    -/
    have H : Summable fun _ : α => (1 : ℝ) := by simpa using hf
    /-
      case inr.inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hf : Summable fun i => HPow.hPow (Norm.norm (f i)) Top.top.toReal
      H : Summable fun x => 1
      ⊢ BddAbove (Set.range fun i => Norm.norm (f i))
    -/
    simpa using ((Set.Finite.of_summable_const (by norm_num) H).image fun i => ‖f i‖).bddAbove
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f : (i : α) → E i
    hf : Summable fun i => HPow.hPow (Norm.norm (f i)) p.toReal
    hp : LT.lt 0 p.toReal
    ⊢ Memℓp f p
  -/
  exact (memℓp_gen_iff hp).2 hf
  /-
    🎉 no goals
  -/


theorem memℓp_gen' {C : ℝ} {f : ∀ i, E i} (hf : ∀ s : Finset α, ∑ i ∈ s, ‖f i‖ ^ p.toReal ≤ C) :
    Memℓp f p := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    C : Real
    f : (i : α) → E i
    hf : ∀ (s : Finset α), LE.le (s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toR …
    ⊢ Memℓp f p
  -/
  apply memℓp_gen
  /-
    case hf
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    C : Real
    f : (i : α) → E i
    hf : ∀ (s : Finset α), LE.le (s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toR …
    ⊢ Summable fun i => HPow.hPow (Norm.norm (f i)) p.toReal
  -/
  use ⨆ s : Finset α, ∑ i ∈ s, ‖f i‖ ^ p.toReal
  /-
    case h
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    C : Real
    f : (i : α) → E i
    hf : ∀ (s : Finset α), LE.le (s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toR …
    ⊢ HasSum (fun i => HPow.hPow (Norm.norm (f i)) p.toReal) (iSup fun s => s.sum  …
  -/
  apply hasSum_of_isLUB_of_nonneg
    /-
      case h.h
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      C : Real
      f : (i : α) → E i
      hf : ∀ (s : Finset α), LE.le (s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toR …
      ⊢ ∀ (i : α), LE.le 0 (HPow.hPow (Norm.norm (f i)) p.toReal)
    -/
  · intro b
    /-
      case h.h
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      C : Real
      f : (i : α) → E i
      hf : ∀ (s : Finset α), LE.le (s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toR …
      b : α
      ⊢ LE.le 0 (HPow.hPow (Norm.norm (f b)) p.toReal)
    -/
    exact Real.rpow_nonneg (norm_nonneg _) _
    /-
      🎉 no goals
    -/
  /-
    case h.hf
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    C : Real
    f : (i : α) → E i
    hf : ∀ (s : Finset α), LE.le (s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toR …
    ⊢ IsLUB (Set.range fun s => s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toRea …
  -/
  apply isLUB_ciSup
  /-
    case h.hf.H
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    C : Real
    f : (i : α) → E i
    hf : ∀ (s : Finset α), LE.le (s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toR …
    ⊢ BddAbove (Set.range fun s => s.sum fun i => HPow.hPow (Norm.norm (f i)) p.to …
  -/
  use C
  /-
    case h
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    C : Real
    f : (i : α) → E i
    hf : ∀ (s : Finset α), LE.le (s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toR …
    ⊢ Membership.mem (upperBounds (Set.range fun s => s.sum fun i => HPow.hPow (No …
  -/
  rintro - ⟨s, rfl⟩
  /-
    case h.intro
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    C : Real
    f : (i : α) → E i
    hf : ∀ (s : Finset α), LE.le (s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toR …
    s : Finset α
    ⊢ LE.le ((fun s => s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toReal) s) C
  -/
  exact hf s
  /-
    🎉 no goals
  -/


theorem zero_memℓp : Memℓp (0 : ∀ i, E i) p := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    ⊢ Memℓp 0 p
  -/
  rcases p.trichotomy with (rfl | rfl | hp)
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      ⊢ Memℓp 0 0
    -/
  · apply memℓp_zero
    /-
      case inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      ⊢ (setOf fun i => Ne (0 i) 0).Finite
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      ⊢ Memℓp 0 Top.top
    -/
  · apply memℓp_infty
    /-
      case inr.inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      ⊢ BddAbove (Set.range fun i => Norm.norm (0 i))
    -/
    simp only [norm_zero, Pi.zero_apply]
    /-
      case inr.inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      ⊢ BddAbove (Set.range fun i => 0)
    -/
    exact bddAbove_singleton.mono Set.range_const_subset
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      hp : LT.lt 0 p.toReal
      ⊢ Memℓp 0 p
    -/
  · apply memℓp_gen
    /-
      case inr.inr.hf
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      hp : LT.lt 0 p.toReal
      ⊢ Summable fun i => HPow.hPow (Norm.norm (0 i)) p.toReal
    -/
    simp [Real.zero_rpow hp.ne', summable_zero]
    /-
      🎉 no goals
    -/


theorem zero_mem_ℓp' : Memℓp (fun i : α => (0 : E i)) p :=
  zero_memℓp


theorem finite_dsupport {f : ∀ i, E i} (hf : Memℓp f 0) : Set.Finite { i | f i ≠ 0 } :=
  memℓp_zero_iff.1 hf


theorem bddAbove {f : ∀ i, E i} (hf : Memℓp f ∞) : BddAbove (Set.range fun i => ‖f i‖) :=
  memℓp_infty_iff.1 hf


theorem summable (hp : 0 < p.toReal) {f : ∀ i, E i} (hf : Memℓp f p) :
    Summable fun i => ‖f i‖ ^ p.toReal :=
  (memℓp_gen_iff hp).1 hf


theorem neg {f : ∀ i, E i} (hf : Memℓp f p) : Memℓp (-f) p := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f : (i : α) → E i
    hf : Memℓp f p
    ⊢ Memℓp (Neg.neg f) p
  -/
  rcases p.trichotomy with (rfl | rfl | hp)
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hf : Memℓp f 0
      ⊢ Memℓp (Neg.neg f) 0
    -/
  · apply memℓp_zero
    /-
      case inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hf : Memℓp f 0
      ⊢ (setOf fun i => Ne (Neg.neg f i) 0).Finite
    -/
    simp [hf.finite_dsupport]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hf : Memℓp f Top.top
      ⊢ Memℓp (Neg.neg f) Top.top
    -/
  · apply memℓp_infty
    /-
      case inr.inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hf : Memℓp f Top.top
      ⊢ BddAbove (Set.range fun i => Norm.norm (Neg.neg f i))
    -/
    simpa using hf.bddAbove
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hf : Memℓp f p
      hp : LT.lt 0 p.toReal
      ⊢ Memℓp (Neg.neg f) p
    -/
  · apply memℓp_gen
    /-
      case inr.inr.hf
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hf : Memℓp f p
      hp : LT.lt 0 p.toReal
      ⊢ Summable fun i => HPow.hPow (Norm.norm (Neg.neg f i)) p.toReal
    -/
    simpa using hf.summable hp
    /-
      🎉 no goals
    -/


@[simp]
theorem neg_iff {f : ∀ i, E i} : Memℓp (-f) p ↔ Memℓp f p :=
  ⟨fun h => neg_neg f ▸ h.neg, Memℓp.neg⟩


theorem of_exponent_ge {p q : ℝ≥0∞} {f : ∀ i, E i} (hfq : Memℓp f q) (hpq : q ≤ p) : Memℓp f p := by
  rcases ENNReal.trichotomy₂ hpq with
    (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, hp⟩ | ⟨rfl, rfl⟩ | ⟨hq, rfl⟩ | ⟨hq, _, hpq'⟩)
    /-
      case inl.intro
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hfq : Memℓp f 0
      hpq : LE.le 0 0
      ⊢ Memℓp f 0
    -/
  · exact hfq
    /-
      🎉 no goals
    -/
    /-
      case inr.inl.intro
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hfq : Memℓp f 0
      hpq : LE.le 0 Top.top
      ⊢ Memℓp f Top.top
    -/
  · apply memℓp_infty
    /-
      case inr.inl.intro.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hfq : Memℓp f 0
      hpq : LE.le 0 Top.top
      ⊢ BddAbove (Set.range fun i => Norm.norm (f i))
    -/
    obtain ⟨C, hC⟩ := (hfq.finite_dsupport.image fun i => ‖f i‖).bddAbove
    /-
      case inr.inl.intro.hf.intro
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hfq : Memℓp f 0
      hpq : LE.le 0 Top.top
      C : Real
      hC : Membership.mem (upperBounds (Set.image (fun i => Norm.norm (f i)) (setOf  …
      ⊢ BddAbove (Set.range fun i => Norm.norm (f i))
    -/
    use max 0 C
    /-
      case h
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hfq : Memℓp f 0
      hpq : LE.le 0 Top.top
      C : Real
      hC : Membership.mem (upperBounds (Set.image (fun i => Norm.norm (f i)) (setOf  …
      ⊢ Membership.mem (upperBounds (Set.range fun i => Norm.norm (f i))) (Max.max 0 …
    -/
    rintro x ⟨i, rfl⟩
    /-
      case h.intro
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hfq : Memℓp f 0
      hpq : LE.le 0 Top.top
      C : Real
      hC : Membership.mem (upperBounds (Set.image (fun i => Norm.norm (f i)) (setOf  …
      i : α
      ⊢ LE.le ((fun i => Norm.norm (f i)) i) (Max.max 0 C)
    -/
    by_cases hi : f i = 0
      /-
        case pos
        α : Type u_1
        E : α → Type u_2
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        f : (i : α) → E i
        hfq : Memℓp f 0
        hpq : LE.le 0 Top.top
        C : Real
        hC : Membership.mem (upperBounds (Set.image (fun i => Norm.norm (f i)) (setOf  …
        i : α
        hi : Eq (f i) 0
        ⊢ LE.le ((fun i => Norm.norm (f i)) i) (Max.max 0 C)
      -/
    · simp [hi]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        E : α → Type u_2
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        f : (i : α) → E i
        hfq : Memℓp f 0
        hpq : LE.le 0 Top.top
        C : Real
        hC : Membership.mem (upperBounds (Set.image (fun i => Norm.norm (f i)) (setOf  …
        i : α
        hi : Not (Eq (f i) 0)
        ⊢ LE.le ((fun i => Norm.norm (f i)) i) (Max.max 0 C)
      -/
    · exact (hC ⟨i, hi, rfl⟩).trans (le_max_right _ _)
      /-
        🎉 no goals
      -/
    /-
      case inr.inr.inl.intro
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      p : ENNReal
      f : (i : α) → E i
      hp : LT.lt 0 p.toReal
      hfq : Memℓp f 0
      hpq : LE.le 0 p
      ⊢ Memℓp f p
    -/
  · apply memℓp_gen
    have : ∀ i ∉ hfq.finite_dsupport.toFinset, ‖f i‖ ^ p.toReal = 0 := by
      intro i hi
      have : f i = 0 := by simpa using hi
      simp [this, Real.zero_rpow hp.ne']
    /-
      case inr.inr.inl.intro.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      p : ENNReal
      f : (i : α) → E i
      hp : LT.lt 0 p.toReal
      hfq : Memℓp f 0
      hpq : LE.le 0 p
      this : ∀ (i : α), Not (Membership.mem ⋯.toFinset i) → Eq (HPow.hPow (Norm.norm …
      ⊢ Summable fun i => HPow.hPow (Norm.norm (f i)) p.toReal
    -/
    exact summable_of_ne_finset_zero this
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.inr.inl.intro
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : (i : α) → E i
      hfq : Memℓp f Top.top
      hpq : LE.le Top.top Top.top
      ⊢ Memℓp f Top.top
    -/
  · exact hfq
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.inr.inr.inl.intro
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      q : ENNReal
      f : (i : α) → E i
      hfq : Memℓp f q
      hq : LT.lt 0 q.toReal
      hpq : LE.le q Top.top
      ⊢ Memℓp f Top.top
    -/
  · apply memℓp_infty
    /-
      case inr.inr.inr.inr.inl.intro.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      q : ENNReal
      f : (i : α) → E i
      hfq : Memℓp f q
      hq : LT.lt 0 q.toReal
      hpq : LE.le q Top.top
      ⊢ BddAbove (Set.range fun i => Norm.norm (f i))
    -/
    obtain ⟨A, hA⟩ := (hfq.summable hq).tendsto_cofinite_zero.bddAbove_range_of_cofinite
    /-
      case inr.inr.inr.inr.inl.intro.hf.intro
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      q : ENNReal
      f : (i : α) → E i
      hfq : Memℓp f q
      hq : LT.lt 0 q.toReal
      hpq : LE.le q Top.top
      A : Real
      hA : Membership.mem (upperBounds (Set.range fun i => HPow.hPow (Norm.norm (f i …
      ⊢ BddAbove (Set.range fun i => Norm.norm (f i))
    -/
    use A ^ q.toReal⁻¹
    /-
      case h
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      q : ENNReal
      f : (i : α) → E i
      hfq : Memℓp f q
      hq : LT.lt 0 q.toReal
      hpq : LE.le q Top.top
      A : Real
      hA : Membership.mem (upperBounds (Set.range fun i => HPow.hPow (Norm.norm (f i …
      ⊢ Membership.mem (upperBounds (Set.range fun i => Norm.norm (f i))) (HPow.hPow …
    -/
    rintro x ⟨i, rfl⟩
    /-
      case h.intro
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      q : ENNReal
      f : (i : α) → E i
      hfq : Memℓp f q
      hq : LT.lt 0 q.toReal
      hpq : LE.le q Top.top
      A : Real
      hA : Membership.mem (upperBounds (Set.range fun i => HPow.hPow (Norm.norm (f i …
      i : α
      ⊢ LE.le ((fun i => Norm.norm (f i)) i) (HPow.hPow A (Inv.inv q.toReal))
    -/
    have : 0 ≤ ‖f i‖ ^ q.toReal := by positivity
    simpa [← Real.rpow_mul, mul_inv_cancel₀ hq.ne'] using
      Real.rpow_le_rpow this (hA ⟨i, rfl⟩) (inv_nonneg.mpr hq.le)
    /-
      case inr.inr.inr.inr.inr.intro.intro
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      p q : ENNReal
      f : (i : α) → E i
      hfq : Memℓp f q
      hpq : LE.le q p
      hq : LT.lt 0 q.toReal
      left✝ : LT.lt 0 p.toReal
      hpq' : LE.le q.toReal p.toReal
      ⊢ Memℓp f p
    -/
  · apply memℓp_gen
    /-
      case inr.inr.inr.inr.inr.intro.intro.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      p q : ENNReal
      f : (i : α) → E i
      hfq : Memℓp f q
      hpq : LE.le q p
      hq : LT.lt 0 q.toReal
      left✝ : LT.lt 0 p.toReal
      hpq' : LE.le q.toReal p.toReal
      ⊢ Summable fun i => HPow.hPow (Norm.norm (f i)) p.toReal
    -/
    have hf' := hfq.summable hq
    /-
      case inr.inr.inr.inr.inr.intro.intro.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      p q : ENNReal
      f : (i : α) → E i
      hfq : Memℓp f q
      hpq : LE.le q p
      hq : LT.lt 0 q.toReal
      left✝ : LT.lt 0 p.toReal
      hpq' : LE.le q.toReal p.toReal
      hf' : Summable fun i => HPow.hPow (Norm.norm (f i)) q.toReal
      ⊢ Summable fun i => HPow.hPow (Norm.norm (f i)) p.toReal
    -/
    refine .of_norm_bounded_eventually _ hf' (@Set.Finite.subset _ { i | 1 ≤ ‖f i‖ } ?_ _ ?_)
    · have H : { x : α | 1 ≤ ‖f x‖ ^ q.toReal }.Finite := by
        simpa using hf'.tendsto_cofinite_zero.eventually_lt_const (by norm_num)
      /-
        case inr.inr.inr.inr.inr.intro.intro.hf.refine_1
        α : Type u_1
        E : α → Type u_2
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        p q : ENNReal
        f : (i : α) → E i
        hfq : Memℓp f q
        hpq : LE.le q p
        hq : LT.lt 0 q.toReal
        left✝ : LT.lt 0 p.toReal
        hpq' : LE.le q.toReal p.toReal
        hf' : Summable fun i => HPow.hPow (Norm.norm (f i)) q.toReal
        H : (setOf fun x => LE.le 1 (HPow.hPow (Norm.norm (f x)) q.toReal)).Finite
        ⊢ (setOf fun i => LE.le 1 (Norm.norm (f i))).Finite
      -/
      exact H.subset fun i hi => Real.one_le_rpow hi hq.le
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.inr.inr.inr.intro.intro.hf.refine_2
        α : Type u_1
        E : α → Type u_2
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        p q : ENNReal
        f : (i : α) → E i
        hfq : Memℓp f q
        hpq : LE.le q p
        hq : LT.lt 0 q.toReal
        left✝ : LT.lt 0 p.toReal
        hpq' : LE.le q.toReal p.toReal
        hf' : Summable fun i => HPow.hPow (Norm.norm (f i)) q.toReal
        ⊢ HasSubset.Subset (HasCompl.compl (setOf fun x => (fun i => LE.le (Norm.norm  …
      -/
    · show ∀ i, ¬|‖f i‖ ^ p.toReal| ≤ ‖f i‖ ^ q.toReal → 1 ≤ ‖f i‖
      /-
        case inr.inr.inr.inr.inr.intro.intro.hf.refine_2
        α : Type u_1
        E : α → Type u_2
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        p q : ENNReal
        f : (i : α) → E i
        hfq : Memℓp f q
        hpq : LE.le q p
        hq : LT.lt 0 q.toReal
        left✝ : LT.lt 0 p.toReal
        hpq' : LE.le q.toReal p.toReal
        hf' : Summable fun i => HPow.hPow (Norm.norm (f i)) q.toReal
        ⊢ ∀ (i : α), Not (LE.le (abs (HPow.hPow (Norm.norm (f i)) p.toReal)) (HPow.hPo …
      -/
      intro i hi
      /-
        case inr.inr.inr.inr.inr.intro.intro.hf.refine_2
        α : Type u_1
        E : α → Type u_2
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        p q : ENNReal
        f : (i : α) → E i
        hfq : Memℓp f q
        hpq : LE.le q p
        hq : LT.lt 0 q.toReal
        left✝ : LT.lt 0 p.toReal
        hpq' : LE.le q.toReal p.toReal
        hf' : Summable fun i => HPow.hPow (Norm.norm (f i)) q.toReal
        i : α
        hi : Not (LE.le (abs (HPow.hPow (Norm.norm (f i)) p.toReal)) (HPow.hPow (Norm. …
        ⊢ LE.le 1 (Norm.norm (f i))
      -/
      have : 0 ≤ ‖f i‖ ^ p.toReal := Real.rpow_nonneg (norm_nonneg _) p.toReal
      /-
        case inr.inr.inr.inr.inr.intro.intro.hf.refine_2
        α : Type u_1
        E : α → Type u_2
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        p q : ENNReal
        f : (i : α) → E i
        hfq : Memℓp f q
        hpq : LE.le q p
        hq : LT.lt 0 q.toReal
        left✝ : LT.lt 0 p.toReal
        hpq' : LE.le q.toReal p.toReal
        hf' : Summable fun i => HPow.hPow (Norm.norm (f i)) q.toReal
        i : α
        hi : Not (LE.le (abs (HPow.hPow (Norm.norm (f i)) p.toReal)) (HPow.hPow (Norm. …
        this : LE.le 0 (HPow.hPow (Norm.norm (f i)) p.toReal)
        ⊢ LE.le 1 (Norm.norm (f i))
      -/
      simp only [abs_of_nonneg, this] at hi
      /-
        case inr.inr.inr.inr.inr.intro.intro.hf.refine_2
        α : Type u_1
        E : α → Type u_2
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        p q : ENNReal
        f : (i : α) → E i
        hfq : Memℓp f q
        hpq : LE.le q p
        hq : LT.lt 0 q.toReal
        left✝ : LT.lt 0 p.toReal
        hpq' : LE.le q.toReal p.toReal
        hf' : Summable fun i => HPow.hPow (Norm.norm (f i)) q.toReal
        i : α
        this : LE.le 0 (HPow.hPow (Norm.norm (f i)) p.toReal)
        hi : Not (LE.le (HPow.hPow (Norm.norm (f i)) p.toReal) (HPow.hPow (Norm.norm ( …
        ⊢ LE.le 1 (Norm.norm (f i))
      -/
      contrapose! hi
      /-
        case inr.inr.inr.inr.inr.intro.intro.hf.refine_2
        α : Type u_1
        E : α → Type u_2
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        p q : ENNReal
        f : (i : α) → E i
        hfq : Memℓp f q
        hpq : LE.le q p
        hq : LT.lt 0 q.toReal
        left✝ : LT.lt 0 p.toReal
        hpq' : LE.le q.toReal p.toReal
        hf' : Summable fun i => HPow.hPow (Norm.norm (f i)) q.toReal
        i : α
        this : LE.le 0 (HPow.hPow (Norm.norm (f i)) p.toReal)
        hi : LT.lt (Norm.norm (f i)) 1
        ⊢ LE.le (HPow.hPow (Norm.norm (f i)) p.toReal) (HPow.hPow (Norm.norm (f i)) q. …
      -/
      exact Real.rpow_le_rpow_of_exponent_ge' (norm_nonneg _) hi.le hq.le hpq'
      /-
        🎉 no goals
      -/


theorem add {f g : ∀ i, E i} (hf : Memℓp f p) (hg : Memℓp g p) : Memℓp (f + g) p := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f g : (i : α) → E i
    hf : Memℓp f p
    hg : Memℓp g p
    ⊢ Memℓp (HAdd.hAdd f g) p
  -/
  rcases p.trichotomy with (rfl | rfl | hp)
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f 0
      hg : Memℓp g 0
      ⊢ Memℓp (HAdd.hAdd f g) 0
    -/
  · apply memℓp_zero
    /-
      case inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f 0
      hg : Memℓp g 0
      ⊢ (setOf fun i => Ne (HAdd.hAdd f g i) 0).Finite
    -/
    refine (hf.finite_dsupport.union hg.finite_dsupport).subset fun i => ?_
    /-
      case inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f 0
      hg : Memℓp g 0
      i : α
      ⊢ Membership.mem (setOf fun i => Ne (HAdd.hAdd f g i) 0) i → Membership.mem (U …
    -/
    simp only [Pi.add_apply, Ne, Set.mem_union, Set.mem_setOf_eq]
    /-
      case inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f 0
      hg : Memℓp g 0
      i : α
      ⊢ Not (Eq (HAdd.hAdd (f i) (g i)) 0) → Or (Not (Eq (f i) 0)) (Not (Eq (g i) 0))
    -/
    contrapose!
    /-
      case inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f 0
      hg : Memℓp g 0
      i : α
      ⊢ And (Eq (f i) 0) (Eq (g i) 0) → Eq (HAdd.hAdd (f i) (g i)) 0
    -/
    rintro ⟨hf', hg'⟩
    /-
      case inl.hf.intro
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f 0
      hg : Memℓp g 0
      i : α
      hf' : Eq (f i) 0
      hg' : Eq (g i) 0
      ⊢ Eq (HAdd.hAdd (f i) (g i)) 0
    -/
    simp [hf', hg']
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f Top.top
      hg : Memℓp g Top.top
      ⊢ Memℓp (HAdd.hAdd f g) Top.top
    -/
  · apply memℓp_infty
    /-
      case inr.inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f Top.top
      hg : Memℓp g Top.top
      ⊢ BddAbove (Set.range fun i => Norm.norm (HAdd.hAdd f g i))
    -/
    obtain ⟨A, hA⟩ := hf.bddAbove
    /-
      case inr.inl.hf.intro
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f Top.top
      hg : Memℓp g Top.top
      A : Real
      hA : Membership.mem (upperBounds (Set.range fun i => Norm.norm (f i))) A
      ⊢ BddAbove (Set.range fun i => Norm.norm (HAdd.hAdd f g i))
    -/
    obtain ⟨B, hB⟩ := hg.bddAbove
    /-
      case inr.inl.hf.intro.intro
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f Top.top
      hg : Memℓp g Top.top
      A : Real
      hA : Membership.mem (upperBounds (Set.range fun i => Norm.norm (f i))) A
      B : Real
      hB : Membership.mem (upperBounds (Set.range fun i => Norm.norm (g i))) B
      ⊢ BddAbove (Set.range fun i => Norm.norm (HAdd.hAdd f g i))
    -/
    refine ⟨A + B, ?_⟩
    /-
      case inr.inl.hf.intro.intro
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f Top.top
      hg : Memℓp g Top.top
      A : Real
      hA : Membership.mem (upperBounds (Set.range fun i => Norm.norm (f i))) A
      B : Real
      hB : Membership.mem (upperBounds (Set.range fun i => Norm.norm (g i))) B
      ⊢ Membership.mem (upperBounds (Set.range fun i => Norm.norm (HAdd.hAdd f g i)) …
    -/
    rintro a ⟨i, rfl⟩
    /-
      case inr.inl.hf.intro.intro.intro
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f Top.top
      hg : Memℓp g Top.top
      A : Real
      hA : Membership.mem (upperBounds (Set.range fun i => Norm.norm (f i))) A
      B : Real
      hB : Membership.mem (upperBounds (Set.range fun i => Norm.norm (g i))) B
      i : α
      ⊢ LE.le ((fun i => Norm.norm (HAdd.hAdd f g i)) i) (HAdd.hAdd A B)
    -/
    exact le_trans (norm_add_le _ _) (add_le_add (hA ⟨i, rfl⟩) (hB ⟨i, rfl⟩))
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f g : (i : α) → E i
    hf : Memℓp f p
    hg : Memℓp g p
    hp : LT.lt 0 p.toReal
    ⊢ Memℓp (HAdd.hAdd f g) p
  -/
  apply memℓp_gen
  /-
    case inr.inr.hf
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f g : (i : α) → E i
    hf : Memℓp f p
    hg : Memℓp g p
    hp : LT.lt 0 p.toReal
    ⊢ Summable fun i => HPow.hPow (Norm.norm (HAdd.hAdd f g i)) p.toReal
  -/
  let C : ℝ := if p.toReal < 1 then 1 else (2 : ℝ) ^ (p.toReal - 1)
  /-
    case inr.inr.hf
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f g : (i : α) → E i
    hf : Memℓp f p
    hg : Memℓp g p
    hp : LT.lt 0 p.toReal
    C : Real := ite (LT.lt p.toReal 1) 1 (HPow.hPow 2 (HSub.hSub p.toReal 1))
    ⊢ Summable fun i => HPow.hPow (Norm.norm (HAdd.hAdd f g i)) p.toReal
  -/
  refine .of_nonneg_of_le ?_ (fun i => ?_) (((hf.summable hp).add (hg.summable hp)).mul_left C)
    /-
      case inr.inr.hf.refine_1
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f p
      hg : Memℓp g p
      hp : LT.lt 0 p.toReal
      C : Real := ite (LT.lt p.toReal 1) 1 (HPow.hPow 2 (HSub.hSub p.toReal 1))
      ⊢ ∀ (b : α), LE.le 0 (HPow.hPow (Norm.norm (HAdd.hAdd f g b)) p.toReal)
    -/
  · intro; positivity
           /-
             🎉 no goals
           -/
    /-
      case inr.inr.hf.refine_2
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f p
      hg : Memℓp g p
      hp : LT.lt 0 p.toReal
      C : Real := ite (LT.lt p.toReal 1) 1 (HPow.hPow 2 (HSub.hSub p.toReal 1))
      i : α
      ⊢ LE.le (HPow.hPow (Norm.norm (HAdd.hAdd f g i)) p.toReal) (HMul.hMul C (HAdd. …
    -/
  · refine (Real.rpow_le_rpow (norm_nonneg _) (norm_add_le _ _) hp.le).trans ?_
    /-
      case inr.inr.hf.refine_2
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f p
      hg : Memℓp g p
      hp : LT.lt 0 p.toReal
      C : Real := ite (LT.lt p.toReal 1) 1 (HPow.hPow 2 (HSub.hSub p.toReal 1))
      i : α
      ⊢ LE.le (HPow.hPow (HAdd.hAdd (Norm.norm (f i)) (Norm.norm (g i))) p.toReal) ( …
    -/
    dsimp only [C]
    /-
      case inr.inr.hf.refine_2
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f g : (i : α) → E i
      hf : Memℓp f p
      hg : Memℓp g p
      hp : LT.lt 0 p.toReal
      C : Real := ite (LT.lt p.toReal 1) 1 (HPow.hPow 2 (HSub.hSub p.toReal 1))
      i : α
      ⊢ LE.le (HPow.hPow (HAdd.hAdd (Norm.norm (f i)) (Norm.norm (g i))) p.toReal) ( …
    -/
    split_ifs with h
      /-
        case pos
        α : Type u_1
        E : α → Type u_2
        p : ENNReal
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        f g : (i : α) → E i
        hf : Memℓp f p
        hg : Memℓp g p
        hp : LT.lt 0 p.toReal
        C : Real := ite (LT.lt p.toReal 1) 1 (HPow.hPow 2 (HSub.hSub p.toReal 1))
        i : α
        h : LT.lt p.toReal 1
        ⊢ LE.le (HPow.hPow (HAdd.hAdd (Norm.norm (f i)) (Norm.norm (g i))) p.toReal) ( …
      -/
    · simpa using NNReal.coe_le_coe.2 (NNReal.rpow_add_le_add_rpow ‖f i‖₊ ‖g i‖₊ hp.le h.le)
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        E : α → Type u_2
        p : ENNReal
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        f g : (i : α) → E i
        hf : Memℓp f p
        hg : Memℓp g p
        hp : LT.lt 0 p.toReal
        C : Real := ite (LT.lt p.toReal 1) 1 (HPow.hPow 2 (HSub.hSub p.toReal 1))
        i : α
        h : Not (LT.lt p.toReal 1)
        ⊢ LE.le (HPow.hPow (HAdd.hAdd (Norm.norm (f i)) (Norm.norm (g i))) p.toReal) ( …
      -/
    · let F : Fin 2 → ℝ≥0 := ![‖f i‖₊, ‖g i‖₊]
      /-
        case neg
        α : Type u_1
        E : α → Type u_2
        p : ENNReal
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        f g : (i : α) → E i
        hf : Memℓp f p
        hg : Memℓp g p
        hp : LT.lt 0 p.toReal
        C : Real := ite (LT.lt p.toReal 1) 1 (HPow.hPow 2 (HSub.hSub p.toReal 1))
        i : α
        h : Not (LT.lt p.toReal 1)
        F : Fin 2 → NNReal := Matrix.vecCons (NNNorm.nnnorm (f i)) (Matrix.vecCons (NN …
        ⊢ LE.le (HPow.hPow (HAdd.hAdd (Norm.norm (f i)) (Norm.norm (g i))) p.toReal) ( …
      -/
      simp only [not_lt] at h
      simpa [Fin.sum_univ_succ] using
        Real.rpow_sum_le_const_mul_sum_rpow_of_nonneg Finset.univ h fun i _ => (F i).coe_nonneg


theorem sub {f g : ∀ i, E i} (hf : Memℓp f p) (hg : Memℓp g p) : Memℓp (f - g) p := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f g : (i : α) → E i
    hf : Memℓp f p
    hg : Memℓp g p
    ⊢ Memℓp (HSub.hSub f g) p
  -/
  rw [sub_eq_add_neg]; exact hf.add hg.neg
                       /-
                         🎉 no goals
                       -/


theorem finset_sum {ι} (s : Finset ι) {f : ι → ∀ i, E i} (hf : ∀ i ∈ s, Memℓp (f i) p) :
    Memℓp (fun a => ∑ i ∈ s, f i a) p := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    s : Finset ι
    f : ι → (i : α) → E i
    hf : ∀ (i : ι), Membership.mem s i → Memℓp (f i) p
    ⊢ Memℓp (fun a => s.sum fun i => f i a) p
  -/
  haveI : DecidableEq ι := Classical.decEq _
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    s : Finset ι
    f : ι → (i : α) → E i
    hf : ∀ (i : ι), Membership.mem s i → Memℓp (f i) p
    this : DecidableEq ι
    ⊢ Memℓp (fun a => s.sum fun i => f i a) p
  -/
  revert hf
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    s : Finset ι
    f : ι → (i : α) → E i
    this : DecidableEq ι
    ⊢ (∀ (i : ι), Membership.mem s i → Memℓp (f i) p) → Memℓp (fun a => s.sum fun  …
  -/
  refine Finset.induction_on s ?_ ?_
    /-
      case refine_1
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      s : Finset ι
      f : ι → (i : α) → E i
      this : DecidableEq ι
      ⊢ (∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i → Memℓp (f i) p …
    -/
  · simp only [zero_mem_ℓp', Finset.sum_empty, imp_true_iff]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      s : Finset ι
      f : ι → (i : α) → E i
      this : DecidableEq ι
      ⊢ ∀ ⦃a : ι⦄ {s : Finset ι}, Not (Membership.mem s a) → ((∀ (i : ι), Membership …
    -/
  · intro i s his ih hf
    /-
      case refine_2
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      s✝ : Finset ι
      f : ι → (i : α) → E i
      this : DecidableEq ι
      i : ι
      s : Finset ι
      his : Not (Membership.mem s i)
      ih : (∀ (i : ι), Membership.mem s i → Memℓp (f i) p) → Memℓp (fun a => s.sum f …
      hf : ∀ (i_1 : ι), Membership.mem (Insert.insert i s) i_1 → Memℓp (f i_1) p
      ⊢ Memℓp (fun a => (Insert.insert i s).sum fun i => f i a) p
    -/
    simp only [his, Finset.sum_insert, not_false_iff]
    /-
      case refine_2
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      s✝ : Finset ι
      f : ι → (i : α) → E i
      this : DecidableEq ι
      i : ι
      s : Finset ι
      his : Not (Membership.mem s i)
      ih : (∀ (i : ι), Membership.mem s i → Memℓp (f i) p) → Memℓp (fun a => s.sum f …
      hf : ∀ (i_1 : ι), Membership.mem (Insert.insert i s) i_1 → Memℓp (f i_1) p
      ⊢ Memℓp (fun a => HAdd.hAdd (f i a) (s.sum fun i => f i a)) p
    -/
    exact (hf i (s.mem_insert_self i)).add (ih fun j hj => hf j (Finset.mem_insert_of_mem hj))
    /-
      🎉 no goals
    -/


theorem const_smul {f : ∀ i, E i} (hf : Memℓp f p) (c : 𝕜) : Memℓp (c • f) p := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝³ : (i : α) → NormedAddCommGroup (E i)
    𝕜 : Type u_3
    inst✝² : NormedRing 𝕜
    inst✝¹ : (i : α) → Module 𝕜 (E i)
    inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
    f : (i : α) → E i
    hf : Memℓp f p
    c : 𝕜
    ⊢ Memℓp (HSMul.hSMul c f) p
  -/
  rcases p.trichotomy with (rfl | rfl | hp)
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      c : 𝕜
      hf : Memℓp f 0
      ⊢ Memℓp (HSMul.hSMul c f) 0
    -/
  · apply memℓp_zero
    /-
      case inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      c : 𝕜
      hf : Memℓp f 0
      ⊢ (setOf fun i => Ne (HSMul.hSMul c f i) 0).Finite
    -/
    refine hf.finite_dsupport.subset fun i => (?_ : ¬c • f i = 0 → ¬f i = 0)
    /-
      case inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      c : 𝕜
      hf : Memℓp f 0
      i : α
      ⊢ Not (Eq (HSMul.hSMul c (f i)) 0) → Not (Eq (f i) 0)
    -/
    exact not_imp_not.mpr fun hf' => hf'.symm ▸ smul_zero c
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      c : 𝕜
      hf : Memℓp f Top.top
      ⊢ Memℓp (HSMul.hSMul c f) Top.top
    -/
  · obtain ⟨A, hA⟩ := hf.bddAbove
    /-
      case inr.inl.intro
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      c : 𝕜
      hf : Memℓp f Top.top
      A : Real
      hA : Membership.mem (upperBounds (Set.range fun i => Norm.norm (f i))) A
      ⊢ Memℓp (HSMul.hSMul c f) Top.top
    -/
    refine memℓp_infty ⟨‖c‖ * A, ?_⟩
    /-
      case inr.inl.intro
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      c : 𝕜
      hf : Memℓp f Top.top
      A : Real
      hA : Membership.mem (upperBounds (Set.range fun i => Norm.norm (f i))) A
      ⊢ Membership.mem (upperBounds (Set.range fun i => Norm.norm (HSMul.hSMul c f i …
    -/
    rintro a ⟨i, rfl⟩
    /-
      case inr.inl.intro.intro
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      c : 𝕜
      hf : Memℓp f Top.top
      A : Real
      hA : Membership.mem (upperBounds (Set.range fun i => Norm.norm (f i))) A
      i : α
      ⊢ LE.le ((fun i => Norm.norm (HSMul.hSMul c f i)) i) (HMul.hMul (Norm.norm c) A)
    -/
    dsimp only [Pi.smul_apply]
    /-
      case inr.inl.intro.intro
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      c : 𝕜
      hf : Memℓp f Top.top
      A : Real
      hA : Membership.mem (upperBounds (Set.range fun i => Norm.norm (f i))) A
      i : α
      ⊢ LE.le (Norm.norm (HSMul.hSMul c (f i))) (HMul.hMul (Norm.norm c) A)
    -/
    refine (norm_smul_le _ _).trans ?_
    /-
      case inr.inl.intro.intro
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      c : 𝕜
      hf : Memℓp f Top.top
      A : Real
      hA : Membership.mem (upperBounds (Set.range fun i => Norm.norm (f i))) A
      i : α
      ⊢ LE.le (HMul.hMul (Norm.norm c) (Norm.norm (f i))) (HMul.hMul (Norm.norm c) A)
    -/
    gcongr
    /-
      case inr.inl.intro.intro.h
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      c : 𝕜
      hf : Memℓp f Top.top
      A : Real
      hA : Membership.mem (upperBounds (Set.range fun i => Norm.norm (f i))) A
      i : α
      ⊢ LE.le (Norm.norm (f i)) A
    -/
    exact hA ⟨i, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      hf : Memℓp f p
      c : 𝕜
      hp : LT.lt 0 p.toReal
      ⊢ Memℓp (HSMul.hSMul c f) p
    -/
  · apply memℓp_gen
    /-
      case inr.inr.hf
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      hf : Memℓp f p
      c : 𝕜
      hp : LT.lt 0 p.toReal
      ⊢ Summable fun i => HPow.hPow (Norm.norm (HSMul.hSMul c f i)) p.toReal
    -/
    dsimp only [Pi.smul_apply]
    /-
      case inr.inr.hf
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      hf : Memℓp f p
      c : 𝕜
      hp : LT.lt 0 p.toReal
      ⊢ Summable fun i => HPow.hPow (Norm.norm (HSMul.hSMul c (f i))) p.toReal
    -/
    have := (hf.summable hp).mul_left (↑(‖c‖₊ ^ p.toReal) : ℝ)
    simp_rw [← coe_nnnorm, ← NNReal.coe_rpow, ← NNReal.coe_mul, NNReal.summable_coe,
      ← NNReal.mul_rpow] at this ⊢
    /-
      case inr.inr.hf
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      hf : Memℓp f p
      c : 𝕜
      hp : LT.lt 0 p.toReal
      this : Summable fun a => HPow.hPow (HMul.hMul (NNNorm.nnnorm c) (NNNorm.nnnorm …
      ⊢ Summable fun a => HPow.hPow (NNNorm.nnnorm (HSMul.hSMul c (f a))) p.toReal
    -/
    refine NNReal.summable_of_le ?_ this
    /-
      case inr.inr.hf
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      hf : Memℓp f p
      c : 𝕜
      hp : LT.lt 0 p.toReal
      this : Summable fun a => HPow.hPow (HMul.hMul (NNNorm.nnnorm c) (NNNorm.nnnorm …
      ⊢ ∀ (b : α), LE.le (HPow.hPow (NNNorm.nnnorm (HSMul.hSMul c (f b))) p.toReal)  …
    -/
    intro i
    /-
      case inr.inr.hf
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      hf : Memℓp f p
      c : 𝕜
      hp : LT.lt 0 p.toReal
      this : Summable fun a => HPow.hPow (HMul.hMul (NNNorm.nnnorm c) (NNNorm.nnnorm …
      i : α
      ⊢ LE.le (HPow.hPow (NNNorm.nnnorm (HSMul.hSMul c (f i))) p.toReal) (HPow.hPow  …
    -/
    gcongr
    /-
      case inr.inr.hf.h₁
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      f : (i : α) → E i
      hf : Memℓp f p
      c : 𝕜
      hp : LT.lt 0 p.toReal
      this : Summable fun a => HPow.hPow (HMul.hMul (NNNorm.nnnorm c) (NNNorm.nnnorm …
      i : α
      ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul c (f i))) (HMul.hMul (NNNorm.nnnorm c) (NN …
    -/
    apply nnnorm_smul_le
    /-
      🎉 no goals
    -/


theorem const_mul {f : α → 𝕜} (hf : Memℓp f p) (c : 𝕜) : Memℓp (fun x => c * f x) p :=
                                                          /-
                                                            α : Type u_1
                                                            p : ENNReal
                                                            𝕜 : Type u_3
                                                            inst✝ : NormedRing 𝕜
                                                            f : α → 𝕜
                                                            hf : Memℓp f p
                                                            c : 𝕜
                                                            i : α
                                                            ⊢ BoundedSMul 𝕜 ((fun x => 𝕜) i)
                                                          -/
  @Memℓp.const_smul α (fun _ => 𝕜) _ _ 𝕜 _ _ (fun i => by infer_instance) _ hf c
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- We define `PreLp E` to be a type synonym for `∀ i, E i` which, importantly, does not inherit
the `pi` topology on `∀ i, E i` (otherwise this topology would descend to `lp E p` and conflict
with the normed group topology we will later equip it with.)

We choose to deal with this issue by making a type synonym for `∀ i, E i` rather than for the `lp`
subgroup itself, because this allows all the spaces `lp E p` (for varying `p`) to be subgroups of
the same ambient group, which permits lemma statements like `lp.monotone` (below). -/
@[nolint unusedArguments]
def PreLp (E : α → Type*) [∀ i, NormedAddCommGroup (E i)] : Type _ :=
  ∀ i, E i --deriving AddCommGroup


                                        /-
                                          α : Type u_1
                                          E : α → Type u_2
                                          p q : ENNReal
                                          inst✝ : (i : α) → NormedAddCommGroup (E i)
                                          ⊢ AddCommGroup (PreLp E)
                                        -/
instance : AddCommGroup (PreLp E) := by unfold PreLp; infer_instance
                                                      /-
                                                        🎉 no goals
                                                      -/


instance PreLp.unique [IsEmpty α] : Unique (PreLp E) :=
  Pi.uniqueOfIsEmpty E


/-- lp space -/
def lp (E : α → Type*) [∀ i, NormedAddCommGroup (E i)] (p : ℝ≥0∞) : AddSubgroup (PreLp E) where
  carrier := { f | Memℓp f p }
  zero_mem' := zero_memℓp
  add_mem' := Memℓp.add
  neg_mem' := Memℓp.neg


@[inherit_doc] scoped[lp] notation "ℓ^∞(" ι ", " E ")" => lp (fun i : ι => E) ∞

@[inherit_doc] scoped[lp] notation "ℓ^∞(" ι ")" => lp (fun i : ι => ℝ) ∞


instance : CoeOut (lp E p) (∀ i, E i) :=
  ⟨Subtype.val (α := ∀ i, E i)⟩ -- Porting note: Originally `coeSubtype`


instance coeFun : CoeFun (lp E p) fun _ => ∀ i, E i :=
  ⟨fun f => (f : ∀ i, E i)⟩


@[ext]
theorem ext {f g : lp E p} (h : (f : ∀ i, E i) = g) : f = g :=
  Subtype.ext h


theorem eq_zero' [IsEmpty α] (f : lp E p) : f = 0 :=
  Subsingleton.elim f 0


protected theorem monotone {p q : ℝ≥0∞} (hpq : q ≤ p) : lp E q ≤ lp E p :=
  fun _ hf => Memℓp.of_exponent_ge hf hpq


protected theorem memℓp (f : lp E p) : Memℓp f p :=
  f.prop


@[simp]
theorem coeFn_zero : ⇑(0 : lp E p) = 0 :=
  rfl


@[simp]
theorem coeFn_neg (f : lp E p) : ⇑(-f) = -f :=
  rfl


@[simp]
theorem coeFn_add (f g : lp E p) : ⇑(f + g) = f + g :=
  rfl


theorem coeFn_sum {ι : Type*} (f : ι → lp E p) (s : Finset ι) :
    ⇑(∑ i ∈ s, f i) = ∑ i ∈ s, ⇑(f i) := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    f : ι → Subtype fun x => Membership.mem (lp E p) x
    s : Finset ι
    ⊢ Eq (↑(s.sum fun i => f i)) (s.sum fun i => ↑(f i))
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem coeFn_sub (f g : lp E p) : ⇑(f - g) = f - g :=
  rfl


instance : Norm (lp E p) where
  norm f :=
    if hp : p = 0 then by
      /-
        α : Type u_1
        E : α → Type u_2
        p q : ENNReal
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        f : Subtype fun x => Membership.mem (lp E p) x
        hp : Eq p 0
        ⊢ Real
      -/
      subst hp
      /-
        α : Type u_1
        E : α → Type u_2
        q : ENNReal
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        f : Subtype fun x => Membership.mem (lp E 0) x
        ⊢ Real
      -/
      exact ((lp.memℓp f).finite_dsupport.toFinset.card : ℝ)
      /-
        🎉 no goals
      -/
    else if p = ∞ then ⨆ i, ‖f i‖ else (∑' i, ‖f i‖ ^ p.toReal) ^ (1 / p.toReal)


theorem norm_eq_card_dsupport (f : lp E 0) : ‖f‖ = (lp.memℓp f).finite_dsupport.toFinset.card :=
  dif_pos rfl


theorem norm_eq_ciSup (f : lp E ∞) : ‖f‖ = ⨆ i, ‖f i‖ := rfl


theorem isLUB_norm [Nonempty α] (f : lp E ∞) : IsLUB (Set.range fun i => ‖f i‖) ‖f‖ := by
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : Nonempty α
    f : Subtype fun x => Membership.mem (lp E Top.top) x
    ⊢ IsLUB (Set.range fun i => Norm.norm (↑f i)) (Norm.norm f)
  -/
  rw [lp.norm_eq_ciSup]
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : Nonempty α
    f : Subtype fun x => Membership.mem (lp E Top.top) x
    ⊢ IsLUB (Set.range fun i => Norm.norm (↑f i)) (iSup fun i => Norm.norm (↑f i))
  -/
  exact isLUB_ciSup (lp.memℓp f)
  /-
    🎉 no goals
  -/


theorem norm_eq_tsum_rpow (hp : 0 < p.toReal) (f : lp E p) :
    ‖f‖ = (∑' i, ‖f i‖ ^ p.toReal) ^ (1 / p.toReal) := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ Eq (Norm.norm f) (HPow.hPow (tsum fun i => HPow.hPow (Norm.norm (↑f i)) p.to …
  -/
  dsimp [norm]
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ Eq (dite (Eq p 0) (fun hp => Eq.rec (motive := fun x x_1 => (Subtype fun x_2 …
  -/
  rw [ENNReal.toReal_pos_iff] at hp
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : And (LT.lt 0 p) (LT.lt p Top.top)
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ Eq (dite (Eq p 0) (fun hp => Eq.rec (motive := fun x x_1 => (Subtype fun x_2 …
  -/
  rw [dif_neg hp.1.ne', if_neg hp.2.ne]
  /-
    🎉 no goals
  -/


theorem norm_rpow_eq_tsum (hp : 0 < p.toReal) (f : lp E p) :
    ‖f‖ ^ p.toReal = ∑' i, ‖f i‖ ^ p.toReal := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ Eq (HPow.hPow (Norm.norm f) p.toReal) (tsum fun i => HPow.hPow (Norm.norm (↑ …
  -/
  rw [norm_eq_tsum_rpow hp, ← Real.rpow_mul]
    /-
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      hp : LT.lt 0 p.toReal
      f : Subtype fun x => Membership.mem (lp E p) x
      ⊢ Eq (HPow.hPow (tsum fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HMul.hM …
    -/
  · field_simp
    /-
      🎉 no goals
    -/
  /-
    case hx
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ LE.le 0 (tsum fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal)
  -/
  apply tsum_nonneg
  /-
    case hx.h
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ ∀ (i : α), LE.le 0 (HPow.hPow (Norm.norm (↑f i)) p.toReal)
  -/
  intro i
  calc
    (0 : ℝ) = (0 : ℝ) ^ p.toReal := by rw [Real.zero_rpow hp.ne']
    _ ≤ _ := by gcongr; apply norm_nonneg


theorem hasSum_norm (hp : 0 < p.toReal) (f : lp E p) :
    HasSum (fun i => ‖f i‖ ^ p.toReal) (‖f‖ ^ p.toReal) := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ HasSum (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow (Norm.nor …
  -/
  rw [norm_rpow_eq_tsum hp]
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ HasSum (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (tsum fun i => HPow. …
  -/
  exact ((lp.memℓp f).summable hp).hasSum
  /-
    🎉 no goals
  -/


theorem norm_nonneg' (f : lp E p) : 0 ≤ ‖f‖ := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ LE.le 0 (Norm.norm f)
  -/
  rcases p.trichotomy with (rfl | rfl | hp)
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E 0) x
      ⊢ LE.le 0 (Norm.norm f)
    -/
  · simp [lp.norm_eq_card_dsupport f]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      ⊢ LE.le 0 (Norm.norm f)
    -/
  · cases' isEmpty_or_nonempty α with _i _i
      /-
        case inr.inl.inl
        α : Type u_1
        E : α → Type u_2
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        f : Subtype fun x => Membership.mem (lp E Top.top) x
        _i : IsEmpty α
        ⊢ LE.le 0 (Norm.norm f)
      -/
    · rw [lp.norm_eq_ciSup]
      /-
        case inr.inl.inl
        α : Type u_1
        E : α → Type u_2
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        f : Subtype fun x => Membership.mem (lp E Top.top) x
        _i : IsEmpty α
        ⊢ LE.le 0 (iSup fun i => Norm.norm (↑f i))
      -/
      simp [Real.iSup_of_isEmpty]
      /-
        🎉 no goals
      -/
    /-
      case inr.inl.inr
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      _i : Nonempty α
      ⊢ LE.le 0 (Norm.norm f)
    -/
    inhabit α
    /-
      case inr.inl.inr
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      _i : Nonempty α
      inhabited_h : Inhabited α
      ⊢ LE.le 0 (Norm.norm f)
    -/
    exact (norm_nonneg (f default)).trans ((lp.isLUB_norm f).1 ⟨default, rfl⟩)
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      ⊢ LE.le 0 (Norm.norm f)
    -/
  · rw [lp.norm_eq_tsum_rpow hp f]
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      ⊢ LE.le 0 (HPow.hPow (tsum fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HD …
    -/
    refine Real.rpow_nonneg (tsum_nonneg ?_) _
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      ⊢ ∀ (i : α), LE.le 0 (HPow.hPow (Norm.norm (↑f i)) p.toReal)
    -/
    exact fun i => Real.rpow_nonneg (norm_nonneg _) _
    /-
      🎉 no goals
    -/


@[simp]
theorem norm_zero : ‖(0 : lp E p)‖ = 0 := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    ⊢ Eq (Norm.norm 0) 0
  -/
  rcases p.trichotomy with (rfl | rfl | hp)
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      ⊢ Eq (Norm.norm 0) 0
    -/
  · simp [lp.norm_eq_card_dsupport]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      ⊢ Eq (Norm.norm 0) 0
    -/
  · simp [lp.norm_eq_ciSup]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      hp : LT.lt 0 p.toReal
      ⊢ Eq (Norm.norm 0) 0
    -/
  · rw [lp.norm_eq_tsum_rpow hp]
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      hp : LT.lt 0 p.toReal
      ⊢ Eq (HPow.hPow (tsum fun i => HPow.hPow (Norm.norm (↑0 i)) p.toReal) (HDiv.hD …
    -/
    have hp' : 1 / p.toReal ≠ 0 := one_div_ne_zero hp.ne'
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      hp : LT.lt 0 p.toReal
      hp' : Ne (HDiv.hDiv 1 p.toReal) 0
      ⊢ Eq (HPow.hPow (tsum fun i => HPow.hPow (Norm.norm (↑0 i)) p.toReal) (HDiv.hD …
    -/
    simpa [Real.zero_rpow hp.ne'] using Real.zero_rpow hp'
    /-
      🎉 no goals
    -/


theorem norm_eq_zero_iff {f : lp E p} : ‖f‖ = 0 ↔ f = 0 := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ Iff (Eq (Norm.norm f) 0) (Eq f 0)
  -/
  refine ⟨fun h => ?_, by rintro rfl; exact norm_zero⟩
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f : Subtype fun x => Membership.mem (lp E p) x
    h : Eq (Norm.norm f) 0
    ⊢ Eq f 0
  -/
  rcases p.trichotomy with (rfl | rfl | hp)
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E 0) x
      h : Eq (Norm.norm f) 0
      ⊢ Eq f 0
    -/
  · ext i
    /-
      case inl.h.h
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E 0) x
      h : Eq (Norm.norm f) 0
      i : α
      ⊢ Eq (↑f i) (↑0 i)
    -/
    have : { i : α | ¬f i = 0 } = ∅ := by simpa [lp.norm_eq_card_dsupport f] using h
    /-
      case inl.h.h
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E 0) x
      h : Eq (Norm.norm f) 0
      i : α
      this : Eq (setOf fun i => Not (Eq (↑f i) 0)) EmptyCollection.emptyCollection
      ⊢ Eq (↑f i) (↑0 i)
    -/
    have : (¬f i = 0) = False := congr_fun this i
    /-
      case inl.h.h
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E 0) x
      h : Eq (Norm.norm f) 0
      i : α
      this✝ : Eq (setOf fun i => Not (Eq (↑f i) 0)) EmptyCollection.emptyCollection
      this : Eq (Not (Eq (↑f i) 0)) False
      ⊢ Eq (↑f i) (↑0 i)
    -/
    tauto
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      h : Eq (Norm.norm f) 0
      ⊢ Eq f 0
    -/
  · cases' isEmpty_or_nonempty α with _i _i
      /-
        case inr.inl.inl
        α : Type u_1
        E : α → Type u_2
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        f : Subtype fun x => Membership.mem (lp E Top.top) x
        h : Eq (Norm.norm f) 0
        _i : IsEmpty α
        ⊢ Eq f 0
      -/
    · simp [eq_iff_true_of_subsingleton]
      /-
        🎉 no goals
      -/
    /-
      case inr.inl.inr
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      h : Eq (Norm.norm f) 0
      _i : Nonempty α
      ⊢ Eq f 0
    -/
    have H : IsLUB (Set.range fun i => ‖f i‖) 0 := by simpa [h] using lp.isLUB_norm f
    /-
      case inr.inl.inr
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      h : Eq (Norm.norm f) 0
      _i : Nonempty α
      H : IsLUB (Set.range fun i => Norm.norm (↑f i)) 0
      ⊢ Eq f 0
    -/
    ext i
    /-
      case inr.inl.inr.h.h
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      h : Eq (Norm.norm f) 0
      _i : Nonempty α
      H : IsLUB (Set.range fun i => Norm.norm (↑f i)) 0
      i : α
      ⊢ Eq (↑f i) (↑0 i)
    -/
    have : ‖f i‖ = 0 := le_antisymm (H.1 ⟨i, rfl⟩) (norm_nonneg _)
    /-
      case inr.inl.inr.h.h
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      h : Eq (Norm.norm f) 0
      _i : Nonempty α
      H : IsLUB (Set.range fun i => Norm.norm (↑f i)) 0
      i : α
      this : Eq (Norm.norm (↑f i)) 0
      ⊢ Eq (↑f i) (↑0 i)
    -/
    simpa using this
    /-
      🎉 no goals
    -/
  · have hf : HasSum (fun i : α => ‖f i‖ ^ p.toReal) 0 := by
      have := lp.hasSum_norm hp f
      rwa [h, Real.zero_rpow hp.ne'] at this
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E p) x
      h : Eq (Norm.norm f) 0
      hp : LT.lt 0 p.toReal
      hf : HasSum (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) 0
      ⊢ Eq f 0
    -/
    have : ∀ i, 0 ≤ ‖f i‖ ^ p.toReal := fun i => Real.rpow_nonneg (norm_nonneg _) _
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E p) x
      h : Eq (Norm.norm f) 0
      hp : LT.lt 0 p.toReal
      hf : HasSum (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) 0
      this : ∀ (i : α), LE.le 0 (HPow.hPow (Norm.norm (↑f i)) p.toReal)
      ⊢ Eq f 0
    -/
    rw [hasSum_zero_iff_of_nonneg this] at hf
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E p) x
      h : Eq (Norm.norm f) 0
      hp : LT.lt 0 p.toReal
      hf : Eq (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) 0
      this : ∀ (i : α), LE.le 0 (HPow.hPow (Norm.norm (↑f i)) p.toReal)
      ⊢ Eq f 0
    -/
    ext i
    have : f i = 0 ∧ p.toReal ≠ 0 := by
      simpa [Real.rpow_eq_zero_iff_of_nonneg (norm_nonneg (f i))] using congr_fun hf i
    /-
      case inr.inr.h.h
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E p) x
      h : Eq (Norm.norm f) 0
      hp : LT.lt 0 p.toReal
      hf : Eq (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) 0
      this✝ : ∀ (i : α), LE.le 0 (HPow.hPow (Norm.norm (↑f i)) p.toReal)
      i : α
      this : And (Eq (↑f i) 0) (Ne p.toReal 0)
      ⊢ Eq (↑f i) (↑0 i)
    -/
    exact this.1
    /-
      🎉 no goals
    -/


theorem eq_zero_iff_coeFn_eq_zero {f : lp E p} : f = 0 ↔ ⇑f = 0 := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ Iff (Eq f 0) (Eq (↑f) 0)
  -/
  rw [lp.ext_iff, coeFn_zero]
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11083): this was very slow, so I squeezed the `simp` calls

@[simp]
theorem norm_neg ⦃f : lp E p⦄ : ‖-f‖ = ‖f‖ := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ Eq (Norm.norm (Neg.neg f)) (Norm.norm f)
  -/
  rcases p.trichotomy with (rfl | rfl | hp)
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E 0) x
      ⊢ Eq (Norm.norm (Neg.neg f)) (Norm.norm f)
    -/
  · simp only [norm_eq_card_dsupport, coeFn_neg, Pi.neg_apply, ne_eq, neg_eq_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      ⊢ Eq (Norm.norm (Neg.neg f)) (Norm.norm f)
    -/
  · cases isEmpty_or_nonempty α
      /-
        case inr.inl.inl
        α : Type u_1
        E : α → Type u_2
        inst✝ : (i : α) → NormedAddCommGroup (E i)
        f : Subtype fun x => Membership.mem (lp E Top.top) x
        h✝ : IsEmpty α
        ⊢ Eq (Norm.norm (Neg.neg f)) (Norm.norm f)
      -/
    · simp only [lp.eq_zero' f, neg_zero, norm_zero]
      /-
        🎉 no goals
      -/
    /-
      case inr.inl.inr
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      h✝ : Nonempty α
      ⊢ Eq (Norm.norm (Neg.neg f)) (Norm.norm f)
    -/
    apply (lp.isLUB_norm (-f)).unique
    /-
      case inr.inl.inr
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      h✝ : Nonempty α
      ⊢ IsLUB (Set.range fun i => Norm.norm (↑(Neg.neg f) i)) (Norm.norm f)
    -/
    simpa only [coeFn_neg, Pi.neg_apply, norm_neg] using lp.isLUB_norm f
    /-
      🎉 no goals
    -/
  · suffices ‖-f‖ ^ p.toReal = ‖f‖ ^ p.toReal by
      exact Real.rpow_left_injOn hp.ne' (norm_nonneg' _) (norm_nonneg' _) this
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      ⊢ Eq (HPow.hPow (Norm.norm (Neg.neg f)) p.toReal) (HPow.hPow (Norm.norm f) p.t …
    -/
    apply (lp.hasSum_norm hp (-f)).unique
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      ⊢ HasSum (fun i => HPow.hPow (Norm.norm (↑(Neg.neg f) i)) p.toReal) (HPow.hPow …
    -/
    simpa only [coeFn_neg, Pi.neg_apply, _root_.norm_neg] using lp.hasSum_norm hp f
    /-
      🎉 no goals
    -/


instance normedAddCommGroup [hp : Fact (1 ≤ p)] : NormedAddCommGroup (lp E p) :=
  AddGroupNorm.toNormedAddCommGroup
    { toFun := norm
      map_zero' := norm_zero
      neg' := norm_neg
      add_le' := fun f g => by
        /-
          α : Type u_1
          E : α → Type u_2
          p q : ENNReal
          inst✝ : (i : α) → NormedAddCommGroup (E i)
          hp : Fact (LE.le 1 p)
          f g : Subtype fun x => Membership.mem (lp E p) x
          ⊢ LE.le (Norm.norm (HAdd.hAdd f g)) (HAdd.hAdd (Norm.norm f) (Norm.norm g))
        -/
        rcases p.dichotomy with (rfl | hp')
          /-
            case inl
            α : Type u_1
            E : α → Type u_2
            q : ENNReal
            inst✝ : (i : α) → NormedAddCommGroup (E i)
            hp : Fact (LE.le 1 Top.top)
            f g : Subtype fun x => Membership.mem (lp E Top.top) x
            ⊢ LE.le (Norm.norm (HAdd.hAdd f g)) (HAdd.hAdd (Norm.norm f) (Norm.norm g))
          -/
        · cases isEmpty_or_nonempty α
            /-
              case inl.inl
              α : Type u_1
              E : α → Type u_2
              q : ENNReal
              inst✝ : (i : α) → NormedAddCommGroup (E i)
              hp : Fact (LE.le 1 Top.top)
              f g : Subtype fun x => Membership.mem (lp E Top.top) x
              h✝ : IsEmpty α
              ⊢ LE.le (Norm.norm (HAdd.hAdd f g)) (HAdd.hAdd (Norm.norm f) (Norm.norm g))
            -/
          · simp only [lp.eq_zero' f, zero_add, norm_zero, le_refl]
            /-
              🎉 no goals
            -/
          /-
            case inl.inr
            α : Type u_1
            E : α → Type u_2
            q : ENNReal
            inst✝ : (i : α) → NormedAddCommGroup (E i)
            hp : Fact (LE.le 1 Top.top)
            f g : Subtype fun x => Membership.mem (lp E Top.top) x
            h✝ : Nonempty α
            ⊢ LE.le (Norm.norm (HAdd.hAdd f g)) (HAdd.hAdd (Norm.norm f) (Norm.norm g))
          -/
          refine (lp.isLUB_norm (f + g)).2 ?_
          /-
            case inl.inr
            α : Type u_1
            E : α → Type u_2
            q : ENNReal
            inst✝ : (i : α) → NormedAddCommGroup (E i)
            hp : Fact (LE.le 1 Top.top)
            f g : Subtype fun x => Membership.mem (lp E Top.top) x
            h✝ : Nonempty α
            ⊢ Membership.mem (upperBounds (Set.range fun i => Norm.norm (↑(HAdd.hAdd f g)  …
          -/
          rintro x ⟨i, rfl⟩
          refine le_trans ?_ (add_mem_upperBounds_add
            (lp.isLUB_norm f).1 (lp.isLUB_norm g).1 ⟨_, ⟨i, rfl⟩, _, ⟨i, rfl⟩, rfl⟩)
          /-
            case inl.inr.intro
            α : Type u_1
            E : α → Type u_2
            q : ENNReal
            inst✝ : (i : α) → NormedAddCommGroup (E i)
            hp : Fact (LE.le 1 Top.top)
            f g : Subtype fun x => Membership.mem (lp E Top.top) x
            h✝ : Nonempty α
            i : α
            ⊢ LE.le ((fun i => Norm.norm (↑(HAdd.hAdd f g) i)) i) ((fun x1 x2 => HAdd.hAdd …
          -/
          exact norm_add_le (f i) (g i)
          /-
            🎉 no goals
          -/
          /-
            case inr
            α : Type u_1
            E : α → Type u_2
            p q : ENNReal
            inst✝ : (i : α) → NormedAddCommGroup (E i)
            hp : Fact (LE.le 1 p)
            f g : Subtype fun x => Membership.mem (lp E p) x
            hp' : LE.le 1 p.toReal
            ⊢ LE.le (Norm.norm (HAdd.hAdd f g)) (HAdd.hAdd (Norm.norm f) (Norm.norm g))
          -/
        · have hp'' : 0 < p.toReal := zero_lt_one.trans_le hp'
          /-
            case inr
            α : Type u_1
            E : α → Type u_2
            p q : ENNReal
            inst✝ : (i : α) → NormedAddCommGroup (E i)
            hp : Fact (LE.le 1 p)
            f g : Subtype fun x => Membership.mem (lp E p) x
            hp' : LE.le 1 p.toReal
            hp'' : LT.lt 0 p.toReal
            ⊢ LE.le (Norm.norm (HAdd.hAdd f g)) (HAdd.hAdd (Norm.norm f) (Norm.norm g))
          -/
          have hf₁ : ∀ i, 0 ≤ ‖f i‖ := fun i => norm_nonneg _
          /-
            case inr
            α : Type u_1
            E : α → Type u_2
            p q : ENNReal
            inst✝ : (i : α) → NormedAddCommGroup (E i)
            hp : Fact (LE.le 1 p)
            f g : Subtype fun x => Membership.mem (lp E p) x
            hp' : LE.le 1 p.toReal
            hp'' : LT.lt 0 p.toReal
            hf₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑f i))
            ⊢ LE.le (Norm.norm (HAdd.hAdd f g)) (HAdd.hAdd (Norm.norm f) (Norm.norm g))
          -/
          have hg₁ : ∀ i, 0 ≤ ‖g i‖ := fun i => norm_nonneg _
          /-
            case inr
            α : Type u_1
            E : α → Type u_2
            p q : ENNReal
            inst✝ : (i : α) → NormedAddCommGroup (E i)
            hp : Fact (LE.le 1 p)
            f g : Subtype fun x => Membership.mem (lp E p) x
            hp' : LE.le 1 p.toReal
            hp'' : LT.lt 0 p.toReal
            hf₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑f i))
            hg₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑g i))
            ⊢ LE.le (Norm.norm (HAdd.hAdd f g)) (HAdd.hAdd (Norm.norm f) (Norm.norm g))
          -/
          have hf₂ := lp.hasSum_norm hp'' f
          /-
            case inr
            α : Type u_1
            E : α → Type u_2
            p q : ENNReal
            inst✝ : (i : α) → NormedAddCommGroup (E i)
            hp : Fact (LE.le 1 p)
            f g : Subtype fun x => Membership.mem (lp E p) x
            hp' : LE.le 1 p.toReal
            hp'' : LT.lt 0 p.toReal
            hf₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑f i))
            hg₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑g i))
            hf₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow (Norm …
            ⊢ LE.le (Norm.norm (HAdd.hAdd f g)) (HAdd.hAdd (Norm.norm f) (Norm.norm g))
          -/
          have hg₂ := lp.hasSum_norm hp'' g
          -- apply Minkowski's inequality
          obtain ⟨C, hC₁, hC₂, hCfg⟩ :=
            Real.Lp_add_le_hasSum_of_nonneg hp' hf₁ hg₁ (norm_nonneg' _) (norm_nonneg' _) hf₂ hg₂
          /-
            case inr.intro.intro.intro
            α : Type u_1
            E : α → Type u_2
            p q : ENNReal
            inst✝ : (i : α) → NormedAddCommGroup (E i)
            hp : Fact (LE.le 1 p)
            f g : Subtype fun x => Membership.mem (lp E p) x
            hp' : LE.le 1 p.toReal
            hp'' : LT.lt 0 p.toReal
            hf₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑f i))
            hg₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑g i))
            hf₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow (Norm …
            hg₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑g i)) p.toReal) (HPow.hPow (Norm …
            C : Real
            hC₁ : LE.le 0 C
            hC₂ : LE.le C (HAdd.hAdd (Norm.norm f) (Norm.norm g))
            hCfg : HasSum (fun i => HPow.hPow (HAdd.hAdd (Norm.norm (↑f i)) (Norm.norm (↑g …
            ⊢ LE.le (Norm.norm (HAdd.hAdd f g)) (HAdd.hAdd (Norm.norm f) (Norm.norm g))
          -/
          refine le_trans ?_ hC₂
          /-
            case inr.intro.intro.intro
            α : Type u_1
            E : α → Type u_2
            p q : ENNReal
            inst✝ : (i : α) → NormedAddCommGroup (E i)
            hp : Fact (LE.le 1 p)
            f g : Subtype fun x => Membership.mem (lp E p) x
            hp' : LE.le 1 p.toReal
            hp'' : LT.lt 0 p.toReal
            hf₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑f i))
            hg₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑g i))
            hf₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow (Norm …
            hg₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑g i)) p.toReal) (HPow.hPow (Norm …
            C : Real
            hC₁ : LE.le 0 C
            hC₂ : LE.le C (HAdd.hAdd (Norm.norm f) (Norm.norm g))
            hCfg : HasSum (fun i => HPow.hPow (HAdd.hAdd (Norm.norm (↑f i)) (Norm.norm (↑g …
            ⊢ LE.le (Norm.norm (HAdd.hAdd f g)) C
          -/
          rw [← Real.rpow_le_rpow_iff (norm_nonneg' (f + g)) hC₁ hp'']
          /-
            case inr.intro.intro.intro
            α : Type u_1
            E : α → Type u_2
            p q : ENNReal
            inst✝ : (i : α) → NormedAddCommGroup (E i)
            hp : Fact (LE.le 1 p)
            f g : Subtype fun x => Membership.mem (lp E p) x
            hp' : LE.le 1 p.toReal
            hp'' : LT.lt 0 p.toReal
            hf₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑f i))
            hg₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑g i))
            hf₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow (Norm …
            hg₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑g i)) p.toReal) (HPow.hPow (Norm …
            C : Real
            hC₁ : LE.le 0 C
            hC₂ : LE.le C (HAdd.hAdd (Norm.norm f) (Norm.norm g))
            hCfg : HasSum (fun i => HPow.hPow (HAdd.hAdd (Norm.norm (↑f i)) (Norm.norm (↑g …
            ⊢ LE.le (HPow.hPow (Norm.norm (HAdd.hAdd f g)) p.toReal) (HPow.hPow C p.toReal)
          -/
          refine hasSum_le ?_ (lp.hasSum_norm hp'' (f + g)) hCfg
          /-
            case inr.intro.intro.intro
            α : Type u_1
            E : α → Type u_2
            p q : ENNReal
            inst✝ : (i : α) → NormedAddCommGroup (E i)
            hp : Fact (LE.le 1 p)
            f g : Subtype fun x => Membership.mem (lp E p) x
            hp' : LE.le 1 p.toReal
            hp'' : LT.lt 0 p.toReal
            hf₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑f i))
            hg₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑g i))
            hf₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow (Norm …
            hg₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑g i)) p.toReal) (HPow.hPow (Norm …
            C : Real
            hC₁ : LE.le 0 C
            hC₂ : LE.le C (HAdd.hAdd (Norm.norm f) (Norm.norm g))
            hCfg : HasSum (fun i => HPow.hPow (HAdd.hAdd (Norm.norm (↑f i)) (Norm.norm (↑g …
            ⊢ ∀ (i : α), LE.le (HPow.hPow (Norm.norm (↑(HAdd.hAdd f g) i)) p.toReal) (HPow …
          -/
          intro i
          /-
            case inr.intro.intro.intro
            α : Type u_1
            E : α → Type u_2
            p q : ENNReal
            inst✝ : (i : α) → NormedAddCommGroup (E i)
            hp : Fact (LE.le 1 p)
            f g : Subtype fun x => Membership.mem (lp E p) x
            hp' : LE.le 1 p.toReal
            hp'' : LT.lt 0 p.toReal
            hf₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑f i))
            hg₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑g i))
            hf₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow (Norm …
            hg₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑g i)) p.toReal) (HPow.hPow (Norm …
            C : Real
            hC₁ : LE.le 0 C
            hC₂ : LE.le C (HAdd.hAdd (Norm.norm f) (Norm.norm g))
            hCfg : HasSum (fun i => HPow.hPow (HAdd.hAdd (Norm.norm (↑f i)) (Norm.norm (↑g …
            i : α
            ⊢ LE.le (HPow.hPow (Norm.norm (↑(HAdd.hAdd f g) i)) p.toReal) (HPow.hPow (HAdd …
          -/
          gcongr
          /-
            case inr.intro.intro.intro.h₁
            α : Type u_1
            E : α → Type u_2
            p q : ENNReal
            inst✝ : (i : α) → NormedAddCommGroup (E i)
            hp : Fact (LE.le 1 p)
            f g : Subtype fun x => Membership.mem (lp E p) x
            hp' : LE.le 1 p.toReal
            hp'' : LT.lt 0 p.toReal
            hf₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑f i))
            hg₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑g i))
            hf₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow (Norm …
            hg₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑g i)) p.toReal) (HPow.hPow (Norm …
            C : Real
            hC₁ : LE.le 0 C
            hC₂ : LE.le C (HAdd.hAdd (Norm.norm f) (Norm.norm g))
            hCfg : HasSum (fun i => HPow.hPow (HAdd.hAdd (Norm.norm (↑f i)) (Norm.norm (↑g …
            i : α
            ⊢ LE.le (Norm.norm (↑(HAdd.hAdd f g) i)) (HAdd.hAdd (Norm.norm (↑f i)) (Norm.n …
          -/
          apply norm_add_le
          /-
            🎉 no goals
          -/
      eq_zero_of_map_eq_zero' := fun _ => norm_eq_zero_iff.1 }

-- TODO: define an `ENNReal` version of `IsConjExponent`, and then express this inequality
-- in a better version which also covers the case `p = 1, q = ∞`.

/-- Hölder inequality -/
protected theorem tsum_mul_le_mul_norm {p q : ℝ≥0∞} (hpq : p.toReal.IsConjExponent q.toReal)
    (f : lp E p) (g : lp E q) :
    (Summable fun i => ‖f i‖ * ‖g i‖) ∧ ∑' i, ‖f i‖ * ‖g i‖ ≤ ‖f‖ * ‖g‖ := by
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    p q : ENNReal
    hpq : p.toReal.IsConjExponent q.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    g : Subtype fun x => Membership.mem (lp E q) x
    ⊢ And (Summable fun i => HMul.hMul (Norm.norm (↑f i)) (Norm.norm (↑g i))) (LE. …
  -/
  have hf₁ : ∀ i, 0 ≤ ‖f i‖ := fun i => norm_nonneg _
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    p q : ENNReal
    hpq : p.toReal.IsConjExponent q.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    g : Subtype fun x => Membership.mem (lp E q) x
    hf₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑f i))
    ⊢ And (Summable fun i => HMul.hMul (Norm.norm (↑f i)) (Norm.norm (↑g i))) (LE. …
  -/
  have hg₁ : ∀ i, 0 ≤ ‖g i‖ := fun i => norm_nonneg _
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    p q : ENNReal
    hpq : p.toReal.IsConjExponent q.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    g : Subtype fun x => Membership.mem (lp E q) x
    hf₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑f i))
    hg₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑g i))
    ⊢ And (Summable fun i => HMul.hMul (Norm.norm (↑f i)) (Norm.norm (↑g i))) (LE. …
  -/
  have hf₂ := lp.hasSum_norm hpq.pos f
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    p q : ENNReal
    hpq : p.toReal.IsConjExponent q.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    g : Subtype fun x => Membership.mem (lp E q) x
    hf₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑f i))
    hg₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑g i))
    hf₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow (Norm …
    ⊢ And (Summable fun i => HMul.hMul (Norm.norm (↑f i)) (Norm.norm (↑g i))) (LE. …
  -/
  have hg₂ := lp.hasSum_norm hpq.symm.pos g
  obtain ⟨C, -, hC', hC⟩ :=
    Real.inner_le_Lp_mul_Lq_hasSum_of_nonneg hpq (norm_nonneg' _) (norm_nonneg' _) hf₁ hg₁ hf₂ hg₂
  /-
    case intro.intro.intro
    α : Type u_1
    E : α → Type u_2
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    p q : ENNReal
    hpq : p.toReal.IsConjExponent q.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    g : Subtype fun x => Membership.mem (lp E q) x
    hf₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑f i))
    hg₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑g i))
    hf₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow (Norm …
    hg₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑g i)) q.toReal) (HPow.hPow (Norm …
    C : Real
    hC' : LE.le C (HMul.hMul (Norm.norm f) (Norm.norm g))
    hC : HasSum (fun i => HMul.hMul (Norm.norm (↑f i)) (Norm.norm (↑g i))) C
    ⊢ And (Summable fun i => HMul.hMul (Norm.norm (↑f i)) (Norm.norm (↑g i))) (LE. …
  -/
  rw [← hC.tsum_eq] at hC'
  /-
    case intro.intro.intro
    α : Type u_1
    E : α → Type u_2
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    p q : ENNReal
    hpq : p.toReal.IsConjExponent q.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    g : Subtype fun x => Membership.mem (lp E q) x
    hf₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑f i))
    hg₁ : ∀ (i : α), LE.le 0 (Norm.norm (↑g i))
    hf₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow (Norm …
    hg₂ : HasSum (fun i => HPow.hPow (Norm.norm (↑g i)) q.toReal) (HPow.hPow (Norm …
    C : Real
    hC' : LE.le (tsum fun b => HMul.hMul (Norm.norm (↑f b)) (Norm.norm (↑g b))) (H …
    hC : HasSum (fun i => HMul.hMul (Norm.norm (↑f i)) (Norm.norm (↑g i))) C
    ⊢ And (Summable fun i => HMul.hMul (Norm.norm (↑f i)) (Norm.norm (↑g i))) (LE. …
  -/
  exact ⟨hC.summable, hC'⟩
  /-
    🎉 no goals
  -/


protected theorem summable_mul {p q : ℝ≥0∞} (hpq : p.toReal.IsConjExponent q.toReal)
    (f : lp E p) (g : lp E q) : Summable fun i => ‖f i‖ * ‖g i‖ :=
  (lp.tsum_mul_le_mul_norm hpq f g).1


protected theorem tsum_mul_le_mul_norm' {p q : ℝ≥0∞} (hpq : p.toReal.IsConjExponent q.toReal)
    (f : lp E p) (g : lp E q) : ∑' i, ‖f i‖ * ‖g i‖ ≤ ‖f‖ * ‖g‖ :=
  (lp.tsum_mul_le_mul_norm hpq f g).2


theorem norm_apply_le_norm (hp : p ≠ 0) (f : lp E p) (i : α) : ‖f i‖ ≤ ‖f‖ := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : Ne p 0
    f : Subtype fun x => Membership.mem (lp E p) x
    i : α
    ⊢ LE.le (Norm.norm (↑f i)) (Norm.norm f)
  -/
  rcases eq_or_ne p ∞ with (rfl | hp')
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      i : α
      hp : Ne Top.top 0
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      ⊢ LE.le (Norm.norm (↑f i)) (Norm.norm f)
    -/
  · haveI : Nonempty α := ⟨i⟩
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      i : α
      hp : Ne Top.top 0
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      this : Nonempty α
      ⊢ LE.le (Norm.norm (↑f i)) (Norm.norm f)
    -/
    exact (isLUB_norm f).1 ⟨i, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : Ne p 0
    f : Subtype fun x => Membership.mem (lp E p) x
    i : α
    hp' : Ne p Top.top
    ⊢ LE.le (Norm.norm (↑f i)) (Norm.norm f)
  -/
  have hp'' : 0 < p.toReal := ENNReal.toReal_pos hp hp'
  /-
    case inr
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : Ne p 0
    f : Subtype fun x => Membership.mem (lp E p) x
    i : α
    hp' : Ne p Top.top
    hp'' : LT.lt 0 p.toReal
    ⊢ LE.le (Norm.norm (↑f i)) (Norm.norm f)
  -/
  have : ∀ i, 0 ≤ ‖f i‖ ^ p.toReal := fun i => Real.rpow_nonneg (norm_nonneg _) _
  /-
    case inr
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : Ne p 0
    f : Subtype fun x => Membership.mem (lp E p) x
    i : α
    hp' : Ne p Top.top
    hp'' : LT.lt 0 p.toReal
    this : ∀ (i : α), LE.le 0 (HPow.hPow (Norm.norm (↑f i)) p.toReal)
    ⊢ LE.le (Norm.norm (↑f i)) (Norm.norm f)
  -/
  rw [← Real.rpow_le_rpow_iff (norm_nonneg _) (norm_nonneg' _) hp'']
  /-
    case inr
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : Ne p 0
    f : Subtype fun x => Membership.mem (lp E p) x
    i : α
    hp' : Ne p Top.top
    hp'' : LT.lt 0 p.toReal
    this : ∀ (i : α), LE.le 0 (HPow.hPow (Norm.norm (↑f i)) p.toReal)
    ⊢ LE.le (HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow (Norm.norm f) p.toR …
  -/
  convert le_hasSum (hasSum_norm hp'' f) i fun i _ => this i
  /-
    🎉 no goals
  -/


theorem sum_rpow_le_norm_rpow (hp : 0 < p.toReal) (f : lp E p) (s : Finset α) :
    ∑ i ∈ s, ‖f i‖ ^ p.toReal ≤ ‖f‖ ^ p.toReal := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    s : Finset α
    ⊢ LE.le (s.sum fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow (Nor …
  -/
  rw [lp.norm_rpow_eq_tsum hp f]
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    s : Finset α
    ⊢ LE.le (s.sum fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (tsum fun i =>  …
  -/
  have : ∀ i, 0 ≤ ‖f i‖ ^ p.toReal := fun i => Real.rpow_nonneg (norm_nonneg _) _
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    s : Finset α
    this : ∀ (i : α), LE.le 0 (HPow.hPow (Norm.norm (↑f i)) p.toReal)
    ⊢ LE.le (s.sum fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (tsum fun i =>  …
  -/
  refine sum_le_tsum _ (fun i _ => this i) ?_
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    s : Finset α
    this : ∀ (i : α), LE.le 0 (HPow.hPow (Norm.norm (↑f i)) p.toReal)
    ⊢ Summable fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal
  -/
  exact (lp.memℓp f).summable hp
  /-
    🎉 no goals
  -/


theorem norm_le_of_forall_le' [Nonempty α] {f : lp E ∞} (C : ℝ) (hCf : ∀ i, ‖f i‖ ≤ C) :
    ‖f‖ ≤ C := by
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : Nonempty α
    f : Subtype fun x => Membership.mem (lp E Top.top) x
    C : Real
    hCf : ∀ (i : α), LE.le (Norm.norm (↑f i)) C
    ⊢ LE.le (Norm.norm f) C
  -/
  refine (isLUB_norm f).2 ?_
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : Nonempty α
    f : Subtype fun x => Membership.mem (lp E Top.top) x
    C : Real
    hCf : ∀ (i : α), LE.le (Norm.norm (↑f i)) C
    ⊢ Membership.mem (upperBounds (Set.range fun i => Norm.norm (↑f i))) C
  -/
  rintro - ⟨i, rfl⟩
  /-
    case intro
    α : Type u_1
    E : α → Type u_2
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : Nonempty α
    f : Subtype fun x => Membership.mem (lp E Top.top) x
    C : Real
    hCf : ∀ (i : α), LE.le (Norm.norm (↑f i)) C
    i : α
    ⊢ LE.le ((fun i => Norm.norm (↑f i)) i) C
  -/
  exact hCf i
  /-
    🎉 no goals
  -/


theorem norm_le_of_forall_le {f : lp E ∞} {C : ℝ} (hC : 0 ≤ C) (hCf : ∀ i, ‖f i‖ ≤ C) :
    ‖f‖ ≤ C := by
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    f : Subtype fun x => Membership.mem (lp E Top.top) x
    C : Real
    hC : LE.le 0 C
    hCf : ∀ (i : α), LE.le (Norm.norm (↑f i)) C
    ⊢ LE.le (Norm.norm f) C
  -/
  cases isEmpty_or_nonempty α
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      C : Real
      hC : LE.le 0 C
      hCf : ∀ (i : α), LE.le (Norm.norm (↑f i)) C
      h✝ : IsEmpty α
      ⊢ LE.le (Norm.norm f) C
    -/
  · simpa [eq_zero' f] using hC
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      E : α → Type u_2
      inst✝ : (i : α) → NormedAddCommGroup (E i)
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      C : Real
      hC : LE.le 0 C
      hCf : ∀ (i : α), LE.le (Norm.norm (↑f i)) C
      h✝ : Nonempty α
      ⊢ LE.le (Norm.norm f) C
    -/
  · exact norm_le_of_forall_le' C hCf
    /-
      🎉 no goals
    -/


theorem norm_le_of_tsum_le (hp : 0 < p.toReal) {C : ℝ} (hC : 0 ≤ C) {f : lp E p}
    (hf : ∑' i, ‖f i‖ ^ p.toReal ≤ C ^ p.toReal) : ‖f‖ ≤ C := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : LT.lt 0 p.toReal
    C : Real
    hC : LE.le 0 C
    f : Subtype fun x => Membership.mem (lp E p) x
    hf : LE.le (tsum fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow C  …
    ⊢ LE.le (Norm.norm f) C
  -/
  rw [← Real.rpow_le_rpow_iff (norm_nonneg' _) hC hp, norm_rpow_eq_tsum hp]
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    hp : LT.lt 0 p.toReal
    C : Real
    hC : LE.le 0 C
    f : Subtype fun x => Membership.mem (lp E p) x
    hf : LE.le (tsum fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow C  …
    ⊢ LE.le (tsum fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow C p.t …
  -/
  exact hf
  /-
    🎉 no goals
  -/


theorem norm_le_of_forall_sum_le (hp : 0 < p.toReal) {C : ℝ} (hC : 0 ≤ C) {f : lp E p}
    (hf : ∀ s : Finset α, ∑ i ∈ s, ‖f i‖ ^ p.toReal ≤ C ^ p.toReal) : ‖f‖ ≤ C :=
  norm_le_of_tsum_le hp hC (tsum_le_of_sum_le ((lp.memℓp f).summable hp) hf)


instance : Module 𝕜 (PreLp E) :=
  Pi.module α E 𝕜


instance [∀ i, SMulCommClass 𝕜' 𝕜 (E i)] : SMulCommClass 𝕜' 𝕜 (PreLp E) :=
  Pi.smulCommClass


instance [SMul 𝕜' 𝕜] [∀ i, IsScalarTower 𝕜' 𝕜 (E i)] : IsScalarTower 𝕜' 𝕜 (PreLp E) :=
  Pi.isScalarTower


instance [∀ i, Module 𝕜ᵐᵒᵖ (E i)] [∀ i, IsCentralScalar 𝕜 (E i)] : IsCentralScalar 𝕜 (PreLp E) :=
  Pi.isCentralScalar


theorem mem_lp_const_smul (c : 𝕜) (f : lp E p) : c • (f : PreLp E) ∈ lp E p :=
  (lp.memℓp f).const_smul c


/-- The `𝕜`-submodule of elements of `∀ i : α, E i` whose `lp` norm is finite. This is `lp E p`,
with extra structure. -/
def _root_.lpSubmodule : Submodule 𝕜 (PreLp E) :=
                                              /-
                                                α : Type u_1
                                                E : α → Type u_2
                                                p q : ENNReal
                                                inst✝⁶ : (i : α) → NormedAddCommGroup (E i)
                                                𝕜 : Type u_3
                                                𝕜' : Type u_4
                                                inst✝⁵ : NormedRing 𝕜
                                                inst✝⁴ : NormedRing 𝕜'
                                                inst✝³ : (i : α) → Module 𝕜 (E i)
                                                inst✝² : (i : α) → Module 𝕜' (E i)
                                                inst✝¹ : ∀ (i : α), BoundedSMul 𝕜 (E i)
                                                inst✝ : ∀ (i : α), BoundedSMul 𝕜' (E i)
                                                c : 𝕜
                                                f : PreLp E
                                                hf : Membership.mem __src✝.carrier f
                                                ⊢ Membership.mem __src✝.carrier (HSMul.hSMul c f)
                                              -/
  { lp E p with smul_mem' := fun c f hf => by simpa using mem_lp_const_smul c ⟨f, hf⟩ }
                                              /-
                                                🎉 no goals
                                              -/


theorem coe_lpSubmodule : (lpSubmodule E p 𝕜).toAddSubgroup = lp E p :=
  rfl


instance : Module 𝕜 (lp E p) :=
  { (lpSubmodule E p 𝕜).module with }


@[simp]
theorem coeFn_smul (c : 𝕜) (f : lp E p) : ⇑(c • f) = c • ⇑f :=
  rfl


instance [∀ i, SMulCommClass 𝕜' 𝕜 (E i)] : SMulCommClass 𝕜' 𝕜 (lp E p) :=
  ⟨fun _ _ _ => Subtype.ext <| smul_comm _ _ _⟩


instance [SMul 𝕜' 𝕜] [∀ i, IsScalarTower 𝕜' 𝕜 (E i)] : IsScalarTower 𝕜' 𝕜 (lp E p) :=
  ⟨fun _ _ _ => Subtype.ext <| smul_assoc _ _ _⟩


instance [∀ i, Module 𝕜ᵐᵒᵖ (E i)] [∀ i, IsCentralScalar 𝕜 (E i)] : IsCentralScalar 𝕜 (lp E p) :=
  ⟨fun _ _ => Subtype.ext <| op_smul_eq_smul _ _⟩


theorem norm_const_smul_le (hp : p ≠ 0) (c : 𝕜) (f : lp E p) : ‖c • f‖ ≤ ‖c‖ * ‖f‖ := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝³ : (i : α) → NormedAddCommGroup (E i)
    𝕜 : Type u_3
    inst✝² : NormedRing 𝕜
    inst✝¹ : (i : α) → Module 𝕜 (E i)
    inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
    hp : Ne p 0
    c : 𝕜
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ LE.le (Norm.norm (HSMul.hSMul c f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
  -/
  rcases p.trichotomy with (rfl | rfl | hp)
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      c : 𝕜
      hp : Ne 0 0
      f : Subtype fun x => Membership.mem (lp E 0) x
      ⊢ LE.le (Norm.norm (HSMul.hSMul c f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
    -/
  · exact absurd rfl hp
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      c : 𝕜
      hp : Ne Top.top 0
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      ⊢ LE.le (Norm.norm (HSMul.hSMul c f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
    -/
  · cases isEmpty_or_nonempty α
      /-
        case inr.inl.inl
        α : Type u_1
        E : α → Type u_2
        inst✝³ : (i : α) → NormedAddCommGroup (E i)
        𝕜 : Type u_3
        inst✝² : NormedRing 𝕜
        inst✝¹ : (i : α) → Module 𝕜 (E i)
        inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
        c : 𝕜
        hp : Ne Top.top 0
        f : Subtype fun x => Membership.mem (lp E Top.top) x
        h✝ : IsEmpty α
        ⊢ LE.le (Norm.norm (HSMul.hSMul c f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
      -/
    · simp [lp.eq_zero' f]
      /-
        🎉 no goals
      -/
    /-
      case inr.inl.inr
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      c : 𝕜
      hp : Ne Top.top 0
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      h✝ : Nonempty α
      ⊢ LE.le (Norm.norm (HSMul.hSMul c f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
    -/
    have hcf := lp.isLUB_norm (c • f)
    /-
      case inr.inl.inr
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      c : 𝕜
      hp : Ne Top.top 0
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      h✝ : Nonempty α
      hcf : IsLUB (Set.range fun i => Norm.norm (↑(HSMul.hSMul c f) i)) (Norm.norm ( …
      ⊢ LE.le (Norm.norm (HSMul.hSMul c f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
    -/
    have hfc := (lp.isLUB_norm f).mul_left (norm_nonneg c)
    /-
      case inr.inl.inr
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      c : 𝕜
      hp : Ne Top.top 0
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      h✝ : Nonempty α
      hcf : IsLUB (Set.range fun i => Norm.norm (↑(HSMul.hSMul c f) i)) (Norm.norm ( …
      hfc : IsLUB (Set.image (fun b => HMul.hMul (Norm.norm c) b) (Set.range fun i = …
      ⊢ LE.le (Norm.norm (HSMul.hSMul c f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
    -/
    simp_rw [← Set.range_comp, Function.comp_def] at hfc
    -- TODO: some `IsLUB` API should make it a one-liner from here.
    /-
      case inr.inl.inr
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      c : 𝕜
      hp : Ne Top.top 0
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      h✝ : Nonempty α
      hcf : IsLUB (Set.range fun i => Norm.norm (↑(HSMul.hSMul c f) i)) (Norm.norm ( …
      hfc : IsLUB (Set.range fun x => HMul.hMul (Norm.norm c) (Norm.norm (↑f x))) (H …
      ⊢ LE.le (Norm.norm (HSMul.hSMul c f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
    -/
    refine hcf.right ?_
    /-
      case inr.inl.inr
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      c : 𝕜
      hp : Ne Top.top 0
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      h✝ : Nonempty α
      hcf : IsLUB (Set.range fun i => Norm.norm (↑(HSMul.hSMul c f) i)) (Norm.norm ( …
      hfc : IsLUB (Set.range fun x => HMul.hMul (Norm.norm c) (Norm.norm (↑f x))) (H …
      ⊢ Membership.mem (upperBounds (Set.range fun i => Norm.norm (↑(HSMul.hSMul c f …
    -/
    have := hfc.left
    simp_rw [mem_upperBounds, Set.mem_range,
      forall_exists_index, forall_apply_eq_imp_iff] at this ⊢
    /-
      case inr.inl.inr
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      c : 𝕜
      hp : Ne Top.top 0
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      h✝ : Nonempty α
      hcf : IsLUB (Set.range fun i => Norm.norm (↑(HSMul.hSMul c f) i)) (Norm.norm ( …
      hfc : IsLUB (Set.range fun x => HMul.hMul (Norm.norm c) (Norm.norm (↑f x))) (H …
      this : ∀ (a : α), LE.le (HMul.hMul (Norm.norm c) (Norm.norm (↑f a))) (HMul.hMu …
      ⊢ ∀ (a : α), LE.le (Norm.norm (↑(HSMul.hSMul c f) a)) (HMul.hMul (Norm.norm c) …
    -/
    intro a
    /-
      case inr.inl.inr
      α : Type u_1
      E : α → Type u_2
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      c : 𝕜
      hp : Ne Top.top 0
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      h✝ : Nonempty α
      hcf : IsLUB (Set.range fun i => Norm.norm (↑(HSMul.hSMul c f) i)) (Norm.norm ( …
      hfc : IsLUB (Set.range fun x => HMul.hMul (Norm.norm c) (Norm.norm (↑f x))) (H …
      this : ∀ (a : α), LE.le (HMul.hMul (Norm.norm c) (Norm.norm (↑f a))) (HMul.hMu …
      a : α
      ⊢ LE.le (Norm.norm (↑(HSMul.hSMul c f) a)) (HMul.hMul (Norm.norm c) (Norm.norm …
    -/
    exact (norm_smul_le _ _).trans (this a)
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      hp✝ : Ne p 0
      c : 𝕜
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      ⊢ LE.le (Norm.norm (HSMul.hSMul c f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
    -/
  · letI inst : NNNorm (lp E p) := ⟨fun f => ⟨‖f‖, norm_nonneg' _⟩⟩
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      hp✝ : Ne p 0
      c : 𝕜
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      inst : NNNorm (Subtype fun x => Membership.mem (lp E p) x) := { nnnorm := fun  …
      ⊢ LE.le (Norm.norm (HSMul.hSMul c f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
    -/
    have coe_nnnorm : ∀ f : lp E p, ↑‖f‖₊ = ‖f‖ := fun _ => rfl
    suffices ‖c • f‖₊ ^ p.toReal ≤ (‖c‖₊ * ‖f‖₊) ^ p.toReal by
      rwa [NNReal.rpow_le_rpow_iff hp] at this
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      hp✝ : Ne p 0
      c : 𝕜
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      inst : NNNorm (Subtype fun x => Membership.mem (lp E p) x) := { nnnorm := fun  …
      coe_nnnorm : ∀ (f : Subtype fun x => Membership.mem (lp E p) x), Eq (↑(NNNorm. …
      ⊢ LE.le (HPow.hPow (NNNorm.nnnorm (HSMul.hSMul c f)) p.toReal) (HPow.hPow (HMu …
    -/
    clear_value inst
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      hp✝ : Ne p 0
      c : 𝕜
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      inst : NNNorm (Subtype fun x => Membership.mem (lp E p) x)
      coe_nnnorm : ∀ (f : Subtype fun x => Membership.mem (lp E p) x), Eq (↑(NNNorm. …
      ⊢ LE.le (HPow.hPow (NNNorm.nnnorm (HSMul.hSMul c f)) p.toReal) (HPow.hPow (HMu …
    -/
    rw [NNReal.mul_rpow]
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      hp✝ : Ne p 0
      c : 𝕜
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      inst : NNNorm (Subtype fun x => Membership.mem (lp E p) x)
      coe_nnnorm : ∀ (f : Subtype fun x => Membership.mem (lp E p) x), Eq (↑(NNNorm. …
      ⊢ LE.le (HPow.hPow (NNNorm.nnnorm (HSMul.hSMul c f)) p.toReal) (HMul.hMul (HPo …
    -/
    have hLHS := lp.hasSum_norm hp (c • f)
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      hp✝ : Ne p 0
      c : 𝕜
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      inst : NNNorm (Subtype fun x => Membership.mem (lp E p) x)
      coe_nnnorm : ∀ (f : Subtype fun x => Membership.mem (lp E p) x), Eq (↑(NNNorm. …
      hLHS : HasSum (fun i => HPow.hPow (Norm.norm (↑(HSMul.hSMul c f) i)) p.toReal) …
      ⊢ LE.le (HPow.hPow (NNNorm.nnnorm (HSMul.hSMul c f)) p.toReal) (HMul.hMul (HPo …
    -/
    have hRHS := (lp.hasSum_norm hp f).mul_left (‖c‖ ^ p.toReal)
    simp_rw [← coe_nnnorm, ← _root_.coe_nnnorm, ← NNReal.coe_rpow, ← NNReal.coe_mul,
      NNReal.hasSum_coe] at hRHS hLHS
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      hp✝ : Ne p 0
      c : 𝕜
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      inst : NNNorm (Subtype fun x => Membership.mem (lp E p) x)
      coe_nnnorm : ∀ (f : Subtype fun x => Membership.mem (lp E p) x), Eq (↑(NNNorm. …
      hRHS : HasSum (fun a => HMul.hMul (HPow.hPow (NNNorm.nnnorm c) p.toReal) (HPow …
      hLHS : HasSum (fun a => HPow.hPow (NNNorm.nnnorm (↑(HSMul.hSMul c f) a)) p.toR …
      ⊢ LE.le (HPow.hPow (NNNorm.nnnorm (HSMul.hSMul c f)) p.toReal) (HMul.hMul (HPo …
    -/
    refine hasSum_mono hLHS hRHS fun i => ?_
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      hp✝ : Ne p 0
      c : 𝕜
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      inst : NNNorm (Subtype fun x => Membership.mem (lp E p) x)
      coe_nnnorm : ∀ (f : Subtype fun x => Membership.mem (lp E p) x), Eq (↑(NNNorm. …
      hRHS : HasSum (fun a => HMul.hMul (HPow.hPow (NNNorm.nnnorm c) p.toReal) (HPow …
      hLHS : HasSum (fun a => HPow.hPow (NNNorm.nnnorm (↑(HSMul.hSMul c f) a)) p.toR …
      i : α
      ⊢ LE.le (HPow.hPow (NNNorm.nnnorm (↑(HSMul.hSMul c f) i)) p.toReal) (HMul.hMul …
    -/
    dsimp only
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      hp✝ : Ne p 0
      c : 𝕜
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      inst : NNNorm (Subtype fun x => Membership.mem (lp E p) x)
      coe_nnnorm : ∀ (f : Subtype fun x => Membership.mem (lp E p) x), Eq (↑(NNNorm. …
      hRHS : HasSum (fun a => HMul.hMul (HPow.hPow (NNNorm.nnnorm c) p.toReal) (HPow …
      hLHS : HasSum (fun a => HPow.hPow (NNNorm.nnnorm (↑(HSMul.hSMul c f) a)) p.toR …
      i : α
      ⊢ LE.le (HPow.hPow (NNNorm.nnnorm (↑(HSMul.hSMul c f) i)) p.toReal) (HMul.hMul …
    -/
    rw [← NNReal.mul_rpow]
    -- Porting note: added
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      hp✝ : Ne p 0
      c : 𝕜
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      inst : NNNorm (Subtype fun x => Membership.mem (lp E p) x)
      coe_nnnorm : ∀ (f : Subtype fun x => Membership.mem (lp E p) x), Eq (↑(NNNorm. …
      hRHS : HasSum (fun a => HMul.hMul (HPow.hPow (NNNorm.nnnorm c) p.toReal) (HPow …
      hLHS : HasSum (fun a => HPow.hPow (NNNorm.nnnorm (↑(HSMul.hSMul c f) a)) p.toR …
      i : α
      ⊢ LE.le (HPow.hPow (NNNorm.nnnorm (↑(HSMul.hSMul c f) i)) p.toReal) (HPow.hPow …
    -/
    rw [lp.coeFn_smul, Pi.smul_apply]
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      hp✝ : Ne p 0
      c : 𝕜
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      inst : NNNorm (Subtype fun x => Membership.mem (lp E p) x)
      coe_nnnorm : ∀ (f : Subtype fun x => Membership.mem (lp E p) x), Eq (↑(NNNorm. …
      hRHS : HasSum (fun a => HMul.hMul (HPow.hPow (NNNorm.nnnorm c) p.toReal) (HPow …
      hLHS : HasSum (fun a => HPow.hPow (NNNorm.nnnorm (↑(HSMul.hSMul c f) a)) p.toR …
      i : α
      ⊢ LE.le (HPow.hPow (NNNorm.nnnorm (HSMul.hSMul c (↑f i))) p.toReal) (HPow.hPow …
    -/
    gcongr
    /-
      case inr.inr.h₁
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      hp✝ : Ne p 0
      c : 𝕜
      f : Subtype fun x => Membership.mem (lp E p) x
      hp : LT.lt 0 p.toReal
      inst : NNNorm (Subtype fun x => Membership.mem (lp E p) x)
      coe_nnnorm : ∀ (f : Subtype fun x => Membership.mem (lp E p) x), Eq (↑(NNNorm. …
      hRHS : HasSum (fun a => HMul.hMul (HPow.hPow (NNNorm.nnnorm c) p.toReal) (HPow …
      hLHS : HasSum (fun a => HPow.hPow (NNNorm.nnnorm (↑(HSMul.hSMul c f) a)) p.toR …
      i : α
      ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul c (↑f i))) (HMul.hMul (NNNorm.nnnorm c) (N …
    -/
    apply nnnorm_smul_le
    /-
      🎉 no goals
    -/


instance [Fact (1 ≤ p)] : BoundedSMul 𝕜 (lp E p) :=
  BoundedSMul.of_norm_smul_le <| norm_const_smul_le (zero_lt_one.trans_le <| Fact.out).ne'


theorem norm_const_smul (hp : p ≠ 0) {c : 𝕜} (f : lp E p) : ‖c • f‖ = ‖c‖ * ‖f‖ := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝³ : (i : α) → NormedAddCommGroup (E i)
    𝕜 : Type u_3
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : (i : α) → Module 𝕜 (E i)
    inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
    hp : Ne p 0
    c : 𝕜
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ Eq (Norm.norm (HSMul.hSMul c f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
  -/
  obtain rfl | hc := eq_or_ne c 0
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝³ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝² : NormedDivisionRing 𝕜
      inst✝¹ : (i : α) → Module 𝕜 (E i)
      inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      hp : Ne p 0
      f : Subtype fun x => Membership.mem (lp E p) x
      ⊢ Eq (Norm.norm (HSMul.hSMul 0 f)) (HMul.hMul (Norm.norm 0) (Norm.norm f))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝³ : (i : α) → NormedAddCommGroup (E i)
    𝕜 : Type u_3
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : (i : α) → Module 𝕜 (E i)
    inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
    hp : Ne p 0
    c : 𝕜
    f : Subtype fun x => Membership.mem (lp E p) x
    hc : Ne c 0
    ⊢ Eq (Norm.norm (HSMul.hSMul c f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
  -/
  refine le_antisymm (norm_const_smul_le hp c f) ?_
  /-
    case inr
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝³ : (i : α) → NormedAddCommGroup (E i)
    𝕜 : Type u_3
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : (i : α) → Module 𝕜 (E i)
    inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
    hp : Ne p 0
    c : 𝕜
    f : Subtype fun x => Membership.mem (lp E p) x
    hc : Ne c 0
    ⊢ LE.le (HMul.hMul (Norm.norm c) (Norm.norm f)) (Norm.norm (HSMul.hSMul c f))
  -/
  have := mul_le_mul_of_nonneg_left (norm_const_smul_le hp c⁻¹ (c • f)) (norm_nonneg c)
  /-
    case inr
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝³ : (i : α) → NormedAddCommGroup (E i)
    𝕜 : Type u_3
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : (i : α) → Module 𝕜 (E i)
    inst✝ : ∀ (i : α), BoundedSMul 𝕜 (E i)
    hp : Ne p 0
    c : 𝕜
    f : Subtype fun x => Membership.mem (lp E p) x
    hc : Ne c 0
    this : LE.le (HMul.hMul (Norm.norm c) (Norm.norm (HSMul.hSMul (Inv.inv c) (HSM …
    ⊢ LE.le (HMul.hMul (Norm.norm c) (Norm.norm f)) (Norm.norm (HSMul.hSMul c f))
  -/
  rwa [inv_smul_smul₀ hc, norm_inv, mul_inv_cancel_left₀ (norm_ne_zero_iff.mpr hc)] at this
  /-
    🎉 no goals
  -/


instance instNormedSpace [Fact (1 ≤ p)] : NormedSpace 𝕜 (lp E p) where
  norm_smul_le c f := norm_smul_le c f


theorem _root_.Memℓp.star_mem {f : ∀ i, E i} (hf : Memℓp f p) : Memℓp (star f) p := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝² : (i : α) → NormedAddCommGroup (E i)
    inst✝¹ : (i : α) → StarAddMonoid (E i)
    inst✝ : ∀ (i : α), NormedStarGroup (E i)
    f : (i : α) → E i
    hf : Memℓp f p
    ⊢ Memℓp (Star.star f) p
  -/
  rcases p.trichotomy with (rfl | rfl | hp)
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝² : (i : α) → NormedAddCommGroup (E i)
      inst✝¹ : (i : α) → StarAddMonoid (E i)
      inst✝ : ∀ (i : α), NormedStarGroup (E i)
      f : (i : α) → E i
      hf : Memℓp f 0
      ⊢ Memℓp (Star.star f) 0
    -/
  · apply memℓp_zero
    /-
      case inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝² : (i : α) → NormedAddCommGroup (E i)
      inst✝¹ : (i : α) → StarAddMonoid (E i)
      inst✝ : ∀ (i : α), NormedStarGroup (E i)
      f : (i : α) → E i
      hf : Memℓp f 0
      ⊢ (setOf fun i => Ne (Star.star f i) 0).Finite
    -/
    simp [hf.finite_dsupport]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      E : α → Type u_2
      inst✝² : (i : α) → NormedAddCommGroup (E i)
      inst✝¹ : (i : α) → StarAddMonoid (E i)
      inst✝ : ∀ (i : α), NormedStarGroup (E i)
      f : (i : α) → E i
      hf : Memℓp f Top.top
      ⊢ Memℓp (Star.star f) Top.top
    -/
  · apply memℓp_infty
    /-
      case inr.inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝² : (i : α) → NormedAddCommGroup (E i)
      inst✝¹ : (i : α) → StarAddMonoid (E i)
      inst✝ : ∀ (i : α), NormedStarGroup (E i)
      f : (i : α) → E i
      hf : Memℓp f Top.top
      ⊢ BddAbove (Set.range fun i => Norm.norm (Star.star f i))
    -/
    simpa using hf.bddAbove
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝² : (i : α) → NormedAddCommGroup (E i)
      inst✝¹ : (i : α) → StarAddMonoid (E i)
      inst✝ : ∀ (i : α), NormedStarGroup (E i)
      f : (i : α) → E i
      hf : Memℓp f p
      hp : LT.lt 0 p.toReal
      ⊢ Memℓp (Star.star f) p
    -/
  · apply memℓp_gen
    /-
      case inr.inr.hf
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝² : (i : α) → NormedAddCommGroup (E i)
      inst✝¹ : (i : α) → StarAddMonoid (E i)
      inst✝ : ∀ (i : α), NormedStarGroup (E i)
      f : (i : α) → E i
      hf : Memℓp f p
      hp : LT.lt 0 p.toReal
      ⊢ Summable fun i => HPow.hPow (Norm.norm (Star.star f i)) p.toReal
    -/
    simpa using hf.summable hp
    /-
      🎉 no goals
    -/


@[simp]
theorem _root_.Memℓp.star_iff {f : ∀ i, E i} : Memℓp (star f) p ↔ Memℓp f p :=
  ⟨fun h => star_star f ▸ Memℓp.star_mem h, Memℓp.star_mem⟩


instance : Star (lp E p) where
  star f := ⟨(star f : ∀ i, E i), f.property.star_mem⟩


@[simp]
theorem coeFn_star (f : lp E p) : ⇑(star f) = star (⇑f) :=
  rfl


@[simp]
protected theorem star_apply (f : lp E p) (i : α) : star f i = star (f i) :=
  rfl


instance instInvolutiveStar : InvolutiveStar (lp E p) where
                          /-
                            α : Type u_1
                            E : α → Type u_2
                            p q : ENNReal
                            inst✝² : (i : α) → NormedAddCommGroup (E i)
                            inst✝¹ : (i : α) → StarAddMonoid (E i)
                            inst✝ : ∀ (i : α), NormedStarGroup (E i)
                            x : Subtype fun x => Membership.mem (lp E p) x
                            ⊢ Eq (Star.star (Star.star x)) x
                          -/
  star_involutive x := by simp [star]
                          /-
                            🎉 no goals
                          -/


instance instStarAddMonoid : StarAddMonoid (lp E p) where
  star_add _f _g := ext <| star_add (R := ∀ i, E i) _ _


instance [hp : Fact (1 ≤ p)] : NormedStarGroup (lp E p) where
  norm_star f := by
    /-
      α : Type u_1
      E : α → Type u_2
      p q : ENNReal
      inst✝² : (i : α) → NormedAddCommGroup (E i)
      inst✝¹ : (i : α) → StarAddMonoid (E i)
      inst✝ : ∀ (i : α), NormedStarGroup (E i)
      hp : Fact (LE.le 1 p)
      f : Subtype fun x => Membership.mem (lp E p) x
      ⊢ Eq (Norm.norm (Star.star f)) (Norm.norm f)
    -/
    rcases p.trichotomy with (rfl | rfl | h)
      /-
        case inl
        α : Type u_1
        E : α → Type u_2
        q : ENNReal
        inst✝² : (i : α) → NormedAddCommGroup (E i)
        inst✝¹ : (i : α) → StarAddMonoid (E i)
        inst✝ : ∀ (i : α), NormedStarGroup (E i)
        hp : Fact (LE.le 1 0)
        f : Subtype fun x => Membership.mem (lp E 0) x
        ⊢ Eq (Norm.norm (Star.star f)) (Norm.norm f)
      -/
    · exfalso
      /-
        case inl
        α : Type u_1
        E : α → Type u_2
        q : ENNReal
        inst✝² : (i : α) → NormedAddCommGroup (E i)
        inst✝¹ : (i : α) → StarAddMonoid (E i)
        inst✝ : ∀ (i : α), NormedStarGroup (E i)
        hp : Fact (LE.le 1 0)
        f : Subtype fun x => Membership.mem (lp E 0) x
        ⊢ False
      -/
      have := ENNReal.toReal_mono ENNReal.zero_ne_top hp.elim
      /-
        case inl
        α : Type u_1
        E : α → Type u_2
        q : ENNReal
        inst✝² : (i : α) → NormedAddCommGroup (E i)
        inst✝¹ : (i : α) → StarAddMonoid (E i)
        inst✝ : ∀ (i : α), NormedStarGroup (E i)
        hp : Fact (LE.le 1 0)
        f : Subtype fun x => Membership.mem (lp E 0) x
        this : LE.le (ENNReal.toReal 1) (ENNReal.toReal 0)
        ⊢ False
      -/
      norm_num at this
      /-
        🎉 no goals
      -/
      /-
        case inr.inl
        α : Type u_1
        E : α → Type u_2
        q : ENNReal
        inst✝² : (i : α) → NormedAddCommGroup (E i)
        inst✝¹ : (i : α) → StarAddMonoid (E i)
        inst✝ : ∀ (i : α), NormedStarGroup (E i)
        hp : Fact (LE.le 1 Top.top)
        f : Subtype fun x => Membership.mem (lp E Top.top) x
        ⊢ Eq (Norm.norm (Star.star f)) (Norm.norm f)
      -/
    · simp only [lp.norm_eq_ciSup, lp.star_apply, norm_star]
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        α : Type u_1
        E : α → Type u_2
        p q : ENNReal
        inst✝² : (i : α) → NormedAddCommGroup (E i)
        inst✝¹ : (i : α) → StarAddMonoid (E i)
        inst✝ : ∀ (i : α), NormedStarGroup (E i)
        hp : Fact (LE.le 1 p)
        f : Subtype fun x => Membership.mem (lp E p) x
        h : LT.lt 0 p.toReal
        ⊢ Eq (Norm.norm (Star.star f)) (Norm.norm f)
      -/
    · simp only [lp.norm_eq_tsum_rpow h, lp.star_apply, norm_star]
      /-
        🎉 no goals
      -/


instance : StarModule 𝕜 (lp E p) where
  star_smul _r _f := ext <| star_smul (A := ∀ i, E i) _ _


theorem _root_.Memℓp.infty_mul {f g : ∀ i, B i} (hf : Memℓp f ∞) (hg : Memℓp g ∞) :
    Memℓp (f * g) ∞ := by
  /-
    I : Type u_3
    B : I → Type u_4
    inst✝ : (i : I) → NonUnitalNormedRing (B i)
    f g : (i : I) → B i
    hf : Memℓp f Top.top
    hg : Memℓp g Top.top
    ⊢ Memℓp (HMul.hMul f g) Top.top
  -/
  rw [memℓp_infty_iff]
  /-
    I : Type u_3
    B : I → Type u_4
    inst✝ : (i : I) → NonUnitalNormedRing (B i)
    f g : (i : I) → B i
    hf : Memℓp f Top.top
    hg : Memℓp g Top.top
    ⊢ BddAbove (Set.range fun i => Norm.norm (HMul.hMul f g i))
  -/
  obtain ⟨⟨Cf, hCf⟩, ⟨Cg, hCg⟩⟩ := hf.bddAbove, hg.bddAbove
  /-
    case intro.intro
    I : Type u_3
    B : I → Type u_4
    inst✝ : (i : I) → NonUnitalNormedRing (B i)
    f g : (i : I) → B i
    hf : Memℓp f Top.top
    hg : Memℓp g Top.top
    Cf : Real
    hCf : Membership.mem (upperBounds (Set.range fun i => Norm.norm (f i))) Cf
    Cg : Real
    hCg : Membership.mem (upperBounds (Set.range fun i => Norm.norm (g i))) Cg
    ⊢ BddAbove (Set.range fun i => Norm.norm (HMul.hMul f g i))
  -/
  refine ⟨Cf * Cg, ?_⟩
  /-
    case intro.intro
    I : Type u_3
    B : I → Type u_4
    inst✝ : (i : I) → NonUnitalNormedRing (B i)
    f g : (i : I) → B i
    hf : Memℓp f Top.top
    hg : Memℓp g Top.top
    Cf : Real
    hCf : Membership.mem (upperBounds (Set.range fun i => Norm.norm (f i))) Cf
    Cg : Real
    hCg : Membership.mem (upperBounds (Set.range fun i => Norm.norm (g i))) Cg
    ⊢ Membership.mem (upperBounds (Set.range fun i => Norm.norm (HMul.hMul f g i)) …
  -/
  rintro _ ⟨i, rfl⟩
  calc
    ‖(f * g) i‖ ≤ ‖f i‖ * ‖g i‖ := norm_mul_le (f i) (g i)
    _ ≤ Cf * Cg :=
      mul_le_mul (hCf ⟨i, rfl⟩) (hCg ⟨i, rfl⟩) (norm_nonneg _)
        ((norm_nonneg _).trans (hCf ⟨i, rfl⟩))


instance : Mul (lp B ∞) where
  mul f g := ⟨HMul.hMul (α := ∀ i, B i) _ _ , f.property.infty_mul g.property⟩


@[simp]
theorem infty_coeFn_mul (f g : lp B ∞) : ⇑(f * g) = ⇑f * ⇑g :=
  rfl


instance nonUnitalRing : NonUnitalRing (lp B ∞) :=
  Function.Injective.nonUnitalRing lp.coeFun.coe Subtype.coe_injective (lp.coeFn_zero B ∞)
    lp.coeFn_add infty_coeFn_mul lp.coeFn_neg lp.coeFn_sub (fun _ _ => rfl) fun _ _ => rfl


instance nonUnitalNormedRing : NonUnitalNormedRing (lp B ∞) :=
  { lp.normedAddCommGroup, lp.nonUnitalRing with
    norm_mul := fun f g =>
      lp.norm_le_of_forall_le (mul_nonneg (norm_nonneg f) (norm_nonneg g)) fun i =>
        calc
          ‖(f * g) i‖ ≤ ‖f i‖ * ‖g i‖ := norm_mul_le _ _
          _ ≤ ‖f‖ * ‖g‖ :=
            mul_le_mul (lp.norm_apply_le_norm ENNReal.top_ne_zero f i)
              (lp.norm_apply_le_norm ENNReal.top_ne_zero g i) (norm_nonneg _) (norm_nonneg _) }


instance nonUnitalNormedCommRing {B : I → Type*} [∀ i, NonUnitalNormedCommRing (B i)] :
    NonUnitalNormedCommRing (lp B ∞) where
  mul_comm _ _ := ext <| mul_comm ..

-- we also want a `NonUnitalNormedCommRing` instance, but this has to wait for https://github.com/leanprover-community/mathlib3/pull/13719

instance infty_isScalarTower {𝕜} [NormedRing 𝕜] [∀ i, Module 𝕜 (B i)] [∀ i, BoundedSMul 𝕜 (B i)]
    [∀ i, IsScalarTower 𝕜 (B i) (B i)] : IsScalarTower 𝕜 (lp B ∞) (lp B ∞) :=
  ⟨fun r f g => lp.ext <| smul_assoc (N := ∀ i, B i) (α := ∀ i, B i) r (⇑f) (⇑g)⟩


instance infty_smulCommClass {𝕜} [NormedRing 𝕜] [∀ i, Module 𝕜 (B i)] [∀ i, BoundedSMul 𝕜 (B i)]
    [∀ i, SMulCommClass 𝕜 (B i) (B i)] : SMulCommClass 𝕜 (lp B ∞) (lp B ∞) :=
  ⟨fun r f g => lp.ext <| smul_comm (N := ∀ i, B i) (α := ∀ i, B i) r (⇑f) (⇑g)⟩


instance inftyStarRing : StarRing (lp B ∞) :=
  { lp.instStarAddMonoid with
    star_mul := fun _f _g => ext <| star_mul (R := ∀ i, B i) _ _ }


instance inftyCStarRing [∀ i, CStarRing (B i)] : CStarRing (lp B ∞) where
  norm_mul_self_le f := by
    /-
      α : Type u_1
      E : α → Type u_2
      p q : ENNReal
      inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
      I : Type u_3
      B : I → Type u_4
      inst✝³ : (i : I) → NonUnitalNormedRing (B i)
      inst✝² : (i : I) → StarRing (B i)
      inst✝¹ : ∀ (i : I), NormedStarGroup (B i)
      inst✝ : ∀ (i : I), CStarRing (B i)
      f : Subtype fun x => Membership.mem (lp B Top.top) x
      ⊢ LE.le (HMul.hMul (Norm.norm f) (Norm.norm f)) (Norm.norm (HMul.hMul (Star.st …
    -/
    rw [← sq, ← Real.le_sqrt (norm_nonneg _) (norm_nonneg _)]
    /-
      α : Type u_1
      E : α → Type u_2
      p q : ENNReal
      inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
      I : Type u_3
      B : I → Type u_4
      inst✝³ : (i : I) → NonUnitalNormedRing (B i)
      inst✝² : (i : I) → StarRing (B i)
      inst✝¹ : ∀ (i : I), NormedStarGroup (B i)
      inst✝ : ∀ (i : I), CStarRing (B i)
      f : Subtype fun x => Membership.mem (lp B Top.top) x
      ⊢ LE.le (Norm.norm f) (Norm.norm (HMul.hMul (Star.star f) f)).sqrt
    -/
    refine lp.norm_le_of_forall_le ‖star f * f‖.sqrt_nonneg fun i => ?_
    /-
      α : Type u_1
      E : α → Type u_2
      p q : ENNReal
      inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
      I : Type u_3
      B : I → Type u_4
      inst✝³ : (i : I) → NonUnitalNormedRing (B i)
      inst✝² : (i : I) → StarRing (B i)
      inst✝¹ : ∀ (i : I), NormedStarGroup (B i)
      inst✝ : ∀ (i : I), CStarRing (B i)
      f : Subtype fun x => Membership.mem (lp B Top.top) x
      i : I
      ⊢ LE.le (Norm.norm (↑f i)) (Norm.norm (HMul.hMul (Star.star f) f)).sqrt
    -/
    rw [Real.le_sqrt (norm_nonneg _) (norm_nonneg _), sq, ← CStarRing.norm_star_mul_self]
    /-
      α : Type u_1
      E : α → Type u_2
      p q : ENNReal
      inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
      I : Type u_3
      B : I → Type u_4
      inst✝³ : (i : I) → NonUnitalNormedRing (B i)
      inst✝² : (i : I) → StarRing (B i)
      inst✝¹ : ∀ (i : I), NormedStarGroup (B i)
      inst✝ : ∀ (i : I), CStarRing (B i)
      f : Subtype fun x => Membership.mem (lp B Top.top) x
      i : I
      ⊢ LE.le (Norm.norm (HMul.hMul (Star.star (↑f i)) (↑f i))) (Norm.norm (HMul.hMu …
    -/
    exact lp.norm_apply_le_norm ENNReal.top_ne_zero (star f * f) i
    /-
      🎉 no goals
    -/


instance _root_.PreLp.ring : Ring (PreLp B) :=
  Pi.ring


theorem _root_.one_memℓp_infty : Memℓp (1 : ∀ i, B i) ∞ :=
         /-
           I : Type u_3
           B : I → Type u_4
           inst✝¹ : (i : I) → NormedRing (B i)
           inst✝ : ∀ (i : I), NormOneClass (B i)
           ⊢ Membership.mem (upperBounds (Set.range fun i => Norm.norm (1 i))) 1
         -/
  ⟨1, by rintro i ⟨i, rfl⟩; exact norm_one.le⟩
                            /-
                              🎉 no goals
                            -/


/-- The `𝕜`-subring of elements of `∀ i : α, B i` whose `lp` norm is finite. This is `lp E ∞`,
with extra structure. -/
def _root_.lpInftySubring : Subring (PreLp B) :=
  { lp B ∞ with
    carrier := { f | Memℓp f ∞ }
    one_mem' := one_memℓp_infty
    mul_mem' := Memℓp.infty_mul }


instance inftyRing : Ring (lp B ∞) :=
  (lpInftySubring B).toRing


theorem _root_.Memℓp.infty_pow {f : ∀ i, B i} (hf : Memℓp f ∞) (n : ℕ) : Memℓp (f ^ n) ∞ :=
  (lpInftySubring B).pow_mem hf n


theorem _root_.natCast_memℓp_infty (n : ℕ) : Memℓp (n : ∀ i, B i) ∞ :=
  natCast_mem (lpInftySubring B) n


@[deprecated (since := "2024-04-17")]
alias _root_.nat_cast_memℓp_infty := _root_.natCast_memℓp_infty


theorem _root_.intCast_memℓp_infty (z : ℤ) : Memℓp (z : ∀ i, B i) ∞ :=
  intCast_mem (lpInftySubring B) z


@[deprecated (since := "2024-04-17")]
alias _root_.int_cast_memℓp_infty := _root_.intCast_memℓp_infty


@[simp]
theorem infty_coeFn_one : ⇑(1 : lp B ∞) = 1 :=
  rfl


@[simp]
theorem infty_coeFn_pow (f : lp B ∞) (n : ℕ) : ⇑(f ^ n) = (⇑f) ^ n :=
  rfl


@[simp]
theorem infty_coeFn_natCast (n : ℕ) : ⇑(n : lp B ∞) = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias infty_coeFn_nat_cast := infty_coeFn_natCast


@[simp]
theorem infty_coeFn_intCast (z : ℤ) : ⇑(z : lp B ∞) = z :=
  rfl


@[deprecated (since := "2024-04-17")]
alias infty_coeFn_int_cast := infty_coeFn_intCast


instance [Nonempty I] : NormOneClass (lp B ∞) where
                 /-
                   α : Type u_1
                   E : α → Type u_2
                   p q : ENNReal
                   inst✝³ : (i : α) → NormedAddCommGroup (E i)
                   I : Type u_3
                   B : I → Type u_4
                   inst✝² : (i : I) → NormedRing (B i)
                   inst✝¹ : ∀ (i : I), NormOneClass (B i)
                   inst✝ : Nonempty I
                   ⊢ Eq (Norm.norm 1) 1
                 -/
  norm_one := by simp_rw [lp.norm_eq_ciSup, infty_coeFn_one, Pi.one_apply, norm_one, ciSup_const]
                 /-
                   🎉 no goals
                 -/


instance inftyNormedRing : NormedRing (lp B ∞) :=
  { lp.inftyRing, lp.nonUnitalNormedRing with }


instance inftyNormedCommRing : NormedCommRing (lp B ∞) where
  mul_comm := mul_comm


/-- A variant of `Pi.algebra` that lean can't find otherwise. -/
instance _root_.Pi.algebraOfNormedAlgebra : Algebra 𝕜 (∀ i, B i) :=
  @Pi.algebra I 𝕜 B _ _ fun _ => NormedAlgebra.toAlgebra


instance _root_.PreLp.algebra : Algebra 𝕜 (PreLp B) :=
  Pi.algebraOfNormedAlgebra


theorem _root_.algebraMap_memℓp_infty (k : 𝕜) : Memℓp (algebraMap 𝕜 (∀ i, B i) k) ∞ := by
  /-
    I : Type u_3
    𝕜 : Type u_4
    B : I → Type u_5
    inst✝³ : NormedField 𝕜
    inst✝² : (i : I) → NormedRing (B i)
    inst✝¹ : (i : I) → NormedAlgebra 𝕜 (B i)
    inst✝ : ∀ (i : I), NormOneClass (B i)
    k : 𝕜
    ⊢ Memℓp ((algebraMap 𝕜 ((i : I) → B i)) k) Top.top
  -/
  rw [Algebra.algebraMap_eq_smul_one]
  /-
    I : Type u_3
    𝕜 : Type u_4
    B : I → Type u_5
    inst✝³ : NormedField 𝕜
    inst✝² : (i : I) → NormedRing (B i)
    inst✝¹ : (i : I) → NormedAlgebra 𝕜 (B i)
    inst✝ : ∀ (i : I), NormOneClass (B i)
    k : 𝕜
    ⊢ Memℓp (HSMul.hSMul k 1) Top.top
  -/
  exact (one_memℓp_infty.const_smul k : Memℓp (k • (1 : ∀ i, B i)) ∞)
  /-
    🎉 no goals
  -/


/-- The `𝕜`-subalgebra of elements of `∀ i : α, B i` whose `lp` norm is finite. This is `lp E ∞`,
with extra structure. -/
def _root_.lpInftySubalgebra : Subalgebra 𝕜 (PreLp B) :=
  { lpInftySubring B with
    carrier := { f | Memℓp f ∞ }
    algebraMap_mem' := algebraMap_memℓp_infty }


instance inftyNormedAlgebra : NormedAlgebra 𝕜 (lp B ∞) :=
  { (lpInftySubalgebra 𝕜 B).algebra, (lp.instNormedSpace : NormedSpace 𝕜 (lp B ∞)) with }


/-- The element of `lp E p` which is `a : E i` at the index `i`, and zero elsewhere. -/
protected def single (p) (i : α) (a : E i) : lp E p :=
  ⟨fun j => if h : j = i then Eq.ndrec a h.symm else 0, by
    /-
      α : Type u_1
      E : α → Type u_2
      p✝ q : ENNReal
      inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝³ : NormedRing 𝕜
      inst✝² : (i : α) → Module 𝕜 (E i)
      inst✝¹ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      inst✝ : DecidableEq α
      p : ENNReal
      i : α
      a : E i
      ⊢ Membership.mem (lp E p) fun j => dite (Eq j i) (fun h => Eq.ndrec a ⋯) fun h …
    -/
    refine (memℓp_zero ?_).of_exponent_ge (zero_le p)
    /-
      α : Type u_1
      E : α → Type u_2
      p✝ q : ENNReal
      inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝³ : NormedRing 𝕜
      inst✝² : (i : α) → Module 𝕜 (E i)
      inst✝¹ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      inst✝ : DecidableEq α
      p : ENNReal
      i : α
      a : E i
      ⊢ (setOf fun i_1 => Ne (dite (Eq i_1 i) (fun h => Eq.ndrec a ⋯) fun h => 0) 0) …
    -/
    refine (Set.finite_singleton i).subset ?_
    /-
      α : Type u_1
      E : α → Type u_2
      p✝ q : ENNReal
      inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝³ : NormedRing 𝕜
      inst✝² : (i : α) → Module 𝕜 (E i)
      inst✝¹ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      inst✝ : DecidableEq α
      p : ENNReal
      i : α
      a : E i
      ⊢ HasSubset.Subset (setOf fun i_1 => Ne (dite (Eq i_1 i) (fun h => Eq.ndrec a  …
    -/
    intro j
    simp only [forall_exists_index, Set.mem_singleton_iff, Ne, dite_eq_right_iff,
      Set.mem_setOf_eq, not_forall]
    /-
      α : Type u_1
      E : α → Type u_2
      p✝ q : ENNReal
      inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝³ : NormedRing 𝕜
      inst✝² : (i : α) → Module 𝕜 (E i)
      inst✝¹ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      inst✝ : DecidableEq α
      p : ENNReal
      i : α
      a : E i
      j : α
      ⊢ ∀ (x : Eq j i), Not (Eq (Eq.rec a ⋯) 0) → Eq j i
    -/
    rintro rfl
    /-
      α : Type u_1
      E : α → Type u_2
      p✝ q : ENNReal
      inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝³ : NormedRing 𝕜
      inst✝² : (i : α) → Module 𝕜 (E i)
      inst✝¹ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      inst✝ : DecidableEq α
      p : ENNReal
      j : α
      a : E j
      ⊢ Not (Eq (Eq.rec a ⋯) 0) → Eq j j
    -/
    simp⟩
    /-
      🎉 no goals
    -/


protected theorem single_apply (p) (i : α) (a : E i) (j : α) :
    lp.single p i a j = if h : j = i then Eq.ndrec a h.symm else 0 :=
  rfl


protected theorem single_apply_self (p) (i : α) (a : E i) : lp.single p i a i = a := by
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : DecidableEq α
    p : ENNReal
    i : α
    a : E i
    ⊢ Eq (↑(lp.single p i a) i) a
  -/
  rw [lp.single_apply, dif_pos rfl]
  /-
    🎉 no goals
  -/


protected theorem single_apply_ne (p) (i : α) (a : E i) {j : α} (hij : j ≠ i) :
    lp.single p i a j = 0 := by
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : DecidableEq α
    p : ENNReal
    i : α
    a : E i
    j : α
    hij : Ne j i
    ⊢ Eq (↑(lp.single p i a) j) 0
  -/
  rw [lp.single_apply, dif_neg hij]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem single_neg (p) (i : α) (a : E i) : lp.single p i (-a) = -lp.single p i a := by
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : DecidableEq α
    p : ENNReal
    i : α
    a : E i
    ⊢ Eq (lp.single p i (Neg.neg a)) (Neg.neg (lp.single p i a))
  -/
  refine ext (funext (fun (j : α) => ?_))
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : DecidableEq α
    p : ENNReal
    i : α
    a : E i
    j : α
    ⊢ Eq (↑(lp.single p i (Neg.neg a)) j) (↑(Neg.neg (lp.single p i a)) j)
  -/
  by_cases hi : j = i
    /-
      case pos
      α : Type u_1
      E : α → Type u_2
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      inst✝ : DecidableEq α
      p : ENNReal
      i : α
      a : E i
      j : α
      hi : Eq j i
      ⊢ Eq (↑(lp.single p i (Neg.neg a)) j) (↑(Neg.neg (lp.single p i a)) j)
    -/
  · subst hi
    /-
      case pos
      α : Type u_1
      E : α → Type u_2
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      inst✝ : DecidableEq α
      p : ENNReal
      j : α
      a : E j
      ⊢ Eq (↑(lp.single p j (Neg.neg a)) j) (↑(Neg.neg (lp.single p j a)) j)
    -/
    simp [lp.single_apply_self]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      E : α → Type u_2
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      inst✝ : DecidableEq α
      p : ENNReal
      i : α
      a : E i
      j : α
      hi : Not (Eq j i)
      ⊢ Eq (↑(lp.single p i (Neg.neg a)) j) (↑(Neg.neg (lp.single p i a)) j)
    -/
  · simp [lp.single_apply_ne p i _ hi]
    /-
      🎉 no goals
    -/


@[simp]
protected theorem single_smul (p) (i : α) (a : E i) (c : 𝕜) :
    lp.single p i (c • a) = c • lp.single p i a := by
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
    𝕜 : Type u_3
    inst✝³ : NormedRing 𝕜
    inst✝² : (i : α) → Module 𝕜 (E i)
    inst✝¹ : ∀ (i : α), BoundedSMul 𝕜 (E i)
    inst✝ : DecidableEq α
    p : ENNReal
    i : α
    a : E i
    c : 𝕜
    ⊢ Eq (lp.single p i (HSMul.hSMul c a)) (HSMul.hSMul c (lp.single p i a))
  -/
  refine ext (funext (fun (j : α) => ?_))
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
    𝕜 : Type u_3
    inst✝³ : NormedRing 𝕜
    inst✝² : (i : α) → Module 𝕜 (E i)
    inst✝¹ : ∀ (i : α), BoundedSMul 𝕜 (E i)
    inst✝ : DecidableEq α
    p : ENNReal
    i : α
    a : E i
    c : 𝕜
    j : α
    ⊢ Eq (↑(lp.single p i (HSMul.hSMul c a)) j) (↑(HSMul.hSMul c (lp.single p i a) …
  -/
  by_cases hi : j = i
    /-
      case pos
      α : Type u_1
      E : α → Type u_2
      inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝³ : NormedRing 𝕜
      inst✝² : (i : α) → Module 𝕜 (E i)
      inst✝¹ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      inst✝ : DecidableEq α
      p : ENNReal
      i : α
      a : E i
      c : 𝕜
      j : α
      hi : Eq j i
      ⊢ Eq (↑(lp.single p i (HSMul.hSMul c a)) j) (↑(HSMul.hSMul c (lp.single p i a) …
    -/
  · subst hi
    /-
      case pos
      α : Type u_1
      E : α → Type u_2
      inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝³ : NormedRing 𝕜
      inst✝² : (i : α) → Module 𝕜 (E i)
      inst✝¹ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      inst✝ : DecidableEq α
      p : ENNReal
      c : 𝕜
      j : α
      a : E j
      ⊢ Eq (↑(lp.single p j (HSMul.hSMul c a)) j) (↑(HSMul.hSMul c (lp.single p j a) …
    -/
    dsimp
    /-
      case pos
      α : Type u_1
      E : α → Type u_2
      inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝³ : NormedRing 𝕜
      inst✝² : (i : α) → Module 𝕜 (E i)
      inst✝¹ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      inst✝ : DecidableEq α
      p : ENNReal
      c : 𝕜
      j : α
      a : E j
      ⊢ Eq (↑(lp.single p j (HSMul.hSMul c a)) j) (HSMul.hSMul c (↑(lp.single p j a) …
    -/
    simp [lp.single_apply_self]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      E : α → Type u_2
      inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝³ : NormedRing 𝕜
      inst✝² : (i : α) → Module 𝕜 (E i)
      inst✝¹ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      inst✝ : DecidableEq α
      p : ENNReal
      i : α
      a : E i
      c : 𝕜
      j : α
      hi : Not (Eq j i)
      ⊢ Eq (↑(lp.single p i (HSMul.hSMul c a)) j) (↑(HSMul.hSMul c (lp.single p i a) …
    -/
  · dsimp
    /-
      case neg
      α : Type u_1
      E : α → Type u_2
      inst✝⁴ : (i : α) → NormedAddCommGroup (E i)
      𝕜 : Type u_3
      inst✝³ : NormedRing 𝕜
      inst✝² : (i : α) → Module 𝕜 (E i)
      inst✝¹ : ∀ (i : α), BoundedSMul 𝕜 (E i)
      inst✝ : DecidableEq α
      p : ENNReal
      i : α
      a : E i
      c : 𝕜
      j : α
      hi : Not (Eq j i)
      ⊢ Eq (↑(lp.single p i (HSMul.hSMul c a)) j) (HSMul.hSMul c (↑(lp.single p i a) …
    -/
    simp [lp.single_apply_ne p i _ hi]
    /-
      🎉 no goals
    -/


protected theorem norm_sum_single (hp : 0 < p.toReal) (f : ∀ i, E i) (s : Finset α) :
    ‖∑ i ∈ s, lp.single p i (f i)‖ ^ p.toReal = ∑ i ∈ s, ‖f i‖ ^ p.toReal := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : DecidableEq α
    hp : LT.lt 0 p.toReal
    f : (i : α) → E i
    s : Finset α
    ⊢ Eq (HPow.hPow (Norm.norm (s.sum fun i => lp.single p i (f i))) p.toReal) (s. …
  -/
  refine (hasSum_norm hp (∑ i ∈ s, lp.single p i (f i))).unique ?_
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : DecidableEq α
    hp : LT.lt 0 p.toReal
    f : (i : α) → E i
    s : Finset α
    ⊢ HasSum (fun i => HPow.hPow (Norm.norm (↑(s.sum fun i => lp.single p i (f i)) …
  -/
  simp only [lp.single_apply, coeFn_sum, Finset.sum_apply, Finset.sum_dite_eq]
  have h : ∀ i ∉ s, ‖ite (i ∈ s) (f i) 0‖ ^ p.toReal = 0 := fun i hi ↦ by
    simp [if_neg hi, Real.zero_rpow hp.ne']
  have h' : ∀ i ∈ s, ‖f i‖ ^ p.toReal = ‖ite (i ∈ s) (f i) 0‖ ^ p.toReal := by
    intro i hi
    rw [if_pos hi]
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : DecidableEq α
    hp : LT.lt 0 p.toReal
    f : (i : α) → E i
    s : Finset α
    h : ∀ (i : α), Not (Membership.mem s i) → Eq (HPow.hPow (Norm.norm (ite (Membe …
    h' : ∀ (i : α), Membership.mem s i → Eq (HPow.hPow (Norm.norm (f i)) p.toReal) …
    ⊢ HasSum (fun i => HPow.hPow (Norm.norm (ite (Membership.mem s i) (f i) 0)) p. …
  -/
  simpa [Finset.sum_congr rfl h'] using hasSum_sum_of_ne_finset_zero h
  /-
    🎉 no goals
  -/


protected theorem norm_single (hp : 0 < p.toReal) (f : ∀ i, E i) (i : α) :
    ‖lp.single p i (f i)‖ = ‖f i‖ := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : DecidableEq α
    hp : LT.lt 0 p.toReal
    f : (i : α) → E i
    i : α
    ⊢ Eq (Norm.norm (lp.single p i (f i))) (Norm.norm (f i))
  -/
  refine Real.rpow_left_injOn hp.ne' (norm_nonneg' _) (norm_nonneg _) ?_
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : DecidableEq α
    hp : LT.lt 0 p.toReal
    f : (i : α) → E i
    i : α
    ⊢ Eq ((fun y => HPow.hPow y p.toReal) (Norm.norm (lp.single p i (f i)))) ((fun …
  -/
  simpa using lp.norm_sum_single hp f {i}
  /-
    🎉 no goals
  -/


protected theorem norm_sub_norm_compl_sub_single (hp : 0 < p.toReal) (f : lp E p) (s : Finset α) :
    ‖f‖ ^ p.toReal - ‖f - ∑ i ∈ s, lp.single p i (f i)‖ ^ p.toReal =
      ∑ i ∈ s, ‖f i‖ ^ p.toReal := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : DecidableEq α
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    s : Finset α
    ⊢ Eq (HSub.hSub (HPow.hPow (Norm.norm f) p.toReal) (HPow.hPow (Norm.norm (HSub …
  -/
  refine ((hasSum_norm hp f).sub (hasSum_norm hp (f - ∑ i ∈ s, lp.single p i (f i)))).unique ?_
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : DecidableEq α
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    s : Finset α
    ⊢ HasSum (fun b => HSub.hSub (HPow.hPow (Norm.norm (↑f b)) p.toReal) (HPow.hPo …
  -/
  let F : α → ℝ := fun i => ‖f i‖ ^ p.toReal - ‖(f - ∑ i ∈ s, lp.single p i (f i)) i‖ ^ p.toReal
  have hF : ∀ i ∉ s, F i = 0 := by
    intro i hi
    suffices ‖f i‖ ^ p.toReal - ‖f i - ite (i ∈ s) (f i) 0‖ ^ p.toReal = 0 by
      simpa only [F, coeFn_sum, lp.single_apply, coeFn_sub, Pi.sub_apply, Finset.sum_apply,
        Finset.sum_dite_eq] using this
    simp only [if_neg hi, sub_zero, sub_self]
  have hF' : ∀ i ∈ s, F i = ‖f i‖ ^ p.toReal := by
    intro i hi
    simp only [F, coeFn_sum, lp.single_apply, if_pos hi, sub_self, eq_self_iff_true, coeFn_sub,
      Pi.sub_apply, Finset.sum_apply, Finset.sum_dite_eq, sub_eq_self]
    simp [Real.zero_rpow hp.ne']
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : DecidableEq α
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    s : Finset α
    F : α → Real := fun i => HSub.hSub (HPow.hPow (Norm.norm (↑f i)) p.toReal) (HP …
    hF : ∀ (i : α), Not (Membership.mem s i) → Eq (F i) 0
    hF' : ∀ (i : α), Membership.mem s i → Eq (F i) (HPow.hPow (Norm.norm (↑f i)) p …
    ⊢ HasSum (fun b => HSub.hSub (HPow.hPow (Norm.norm (↑f b)) p.toReal) (HPow.hPo …
  -/
  have : HasSum F (∑ i ∈ s, F i) := hasSum_sum_of_ne_finset_zero hF
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : DecidableEq α
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    s : Finset α
    F : α → Real := fun i => HSub.hSub (HPow.hPow (Norm.norm (↑f i)) p.toReal) (HP …
    hF : ∀ (i : α), Not (Membership.mem s i) → Eq (F i) 0
    hF' : ∀ (i : α), Membership.mem s i → Eq (F i) (HPow.hPow (Norm.norm (↑f i)) p …
    this : HasSum F (s.sum fun i => F i)
    ⊢ HasSum (fun b => HSub.hSub (HPow.hPow (Norm.norm (↑f b)) p.toReal) (HPow.hPo …
  -/
  rwa [Finset.sum_congr rfl hF'] at this
  /-
    🎉 no goals
  -/


protected theorem norm_compl_sum_single (hp : 0 < p.toReal) (f : lp E p) (s : Finset α) :
    ‖f - ∑ i ∈ s, lp.single p i (f i)‖ ^ p.toReal = ‖f‖ ^ p.toReal - ∑ i ∈ s, ‖f i‖ ^ p.toReal := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    inst✝ : DecidableEq α
    hp : LT.lt 0 p.toReal
    f : Subtype fun x => Membership.mem (lp E p) x
    s : Finset α
    ⊢ Eq (HPow.hPow (Norm.norm (HSub.hSub f (s.sum fun i => lp.single p i (↑f i))) …
  -/
  linarith [lp.norm_sub_norm_compl_sub_single hp f s]
  /-
    🎉 no goals
  -/


/-- The canonical finitely-supported approximations to an element `f` of `lp` converge to it, in the
`lp` topology. -/
protected theorem hasSum_single [Fact (1 ≤ p)] (hp : p ≠ ⊤) (f : lp E p) :
    HasSum (fun i : α => lp.single p i (f i : E i)) f := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝² : (i : α) → NormedAddCommGroup (E i)
    inst✝¹ : DecidableEq α
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ HasSum (fun i => lp.single p i (↑f i)) f
  -/
  have hp₀ : 0 < p := zero_lt_one.trans_le Fact.out
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝² : (i : α) → NormedAddCommGroup (E i)
    inst✝¹ : DecidableEq α
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    f : Subtype fun x => Membership.mem (lp E p) x
    hp₀ : LT.lt 0 p
    ⊢ HasSum (fun i => lp.single p i (↑f i)) f
  -/
  have hp' : 0 < p.toReal := ENNReal.toReal_pos hp₀.ne' hp
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝² : (i : α) → NormedAddCommGroup (E i)
    inst✝¹ : DecidableEq α
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    f : Subtype fun x => Membership.mem (lp E p) x
    hp₀ : LT.lt 0 p
    hp' : LT.lt 0 p.toReal
    ⊢ HasSum (fun i => lp.single p i (↑f i)) f
  -/
  have := lp.hasSum_norm hp' f
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝² : (i : α) → NormedAddCommGroup (E i)
    inst✝¹ : DecidableEq α
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    f : Subtype fun x => Membership.mem (lp E p) x
    hp₀ : LT.lt 0 p
    hp' : LT.lt 0 p.toReal
    this : HasSum (fun i => HPow.hPow (Norm.norm (↑f i)) p.toReal) (HPow.hPow (Nor …
    ⊢ HasSum (fun i => lp.single p i (↑f i)) f
  -/
  rw [HasSum, Metric.tendsto_nhds] at this ⊢
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝² : (i : α) → NormedAddCommGroup (E i)
    inst✝¹ : DecidableEq α
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    f : Subtype fun x => Membership.mem (lp E p) x
    hp₀ : LT.lt 0 p
    hp' : LT.lt 0 p.toReal
    this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist  …
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist (x.su …
  -/
  intro ε hε
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝² : (i : α) → NormedAddCommGroup (E i)
    inst✝¹ : DecidableEq α
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    f : Subtype fun x => Membership.mem (lp E p) x
    hp₀ : LT.lt 0 p
    hp' : LT.lt 0 p.toReal
    this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist  …
    ε : Real
    hε : GT.gt ε 0
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (x.sum fun b => lp.single p b ( …
  -/
  refine (this _ (Real.rpow_pos_of_pos hε p.toReal)).mono ?_
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝² : (i : α) → NormedAddCommGroup (E i)
    inst✝¹ : DecidableEq α
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    f : Subtype fun x => Membership.mem (lp E p) x
    hp₀ : LT.lt 0 p
    hp' : LT.lt 0 p.toReal
    this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist  …
    ε : Real
    hε : GT.gt ε 0
    ⊢ ∀ (x : Finset α), LT.lt (Dist.dist (x.sum fun b => HPow.hPow (Norm.norm (↑f  …
  -/
  intro s hs
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝² : (i : α) → NormedAddCommGroup (E i)
    inst✝¹ : DecidableEq α
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    f : Subtype fun x => Membership.mem (lp E p) x
    hp₀ : LT.lt 0 p
    hp' : LT.lt 0 p.toReal
    this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist  …
    ε : Real
    hε : GT.gt ε 0
    s : Finset α
    hs : LT.lt (Dist.dist (s.sum fun b => HPow.hPow (Norm.norm (↑f b)) p.toReal) ( …
    ⊢ LT.lt (Dist.dist (s.sum fun b => lp.single p b (↑f b)) f) ε
  -/
  rw [← Real.rpow_lt_rpow_iff dist_nonneg (le_of_lt hε) hp']
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝² : (i : α) → NormedAddCommGroup (E i)
    inst✝¹ : DecidableEq α
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    f : Subtype fun x => Membership.mem (lp E p) x
    hp₀ : LT.lt 0 p
    hp' : LT.lt 0 p.toReal
    this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist  …
    ε : Real
    hε : GT.gt ε 0
    s : Finset α
    hs : LT.lt (Dist.dist (s.sum fun b => HPow.hPow (Norm.norm (↑f b)) p.toReal) ( …
    ⊢ LT.lt (HPow.hPow (Dist.dist (s.sum fun b => lp.single p b (↑f b)) f) p.toRea …
  -/
  rw [dist_comm] at hs
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝² : (i : α) → NormedAddCommGroup (E i)
    inst✝¹ : DecidableEq α
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    f : Subtype fun x => Membership.mem (lp E p) x
    hp₀ : LT.lt 0 p
    hp' : LT.lt 0 p.toReal
    this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist  …
    ε : Real
    hε : GT.gt ε 0
    s : Finset α
    hs : LT.lt (Dist.dist (HPow.hPow (Norm.norm f) p.toReal) (s.sum fun b => HPow. …
    ⊢ LT.lt (HPow.hPow (Dist.dist (s.sum fun b => lp.single p b (↑f b)) f) p.toRea …
  -/
  simp only [dist_eq_norm, Real.norm_eq_abs] at hs ⊢
  have H : ‖(∑ i ∈ s, lp.single p i (f i : E i)) - f‖ ^ p.toReal =
      ‖f‖ ^ p.toReal - ∑ i ∈ s, ‖f i‖ ^ p.toReal := by
    simpa only [coeFn_neg, Pi.neg_apply, lp.single_neg, Finset.sum_neg_distrib, neg_sub_neg,
      norm_neg, _root_.norm_neg] using lp.norm_compl_sum_single hp' (-f) s
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝² : (i : α) → NormedAddCommGroup (E i)
    inst✝¹ : DecidableEq α
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    f : Subtype fun x => Membership.mem (lp E p) x
    hp₀ : LT.lt 0 p
    hp' : LT.lt 0 p.toReal
    this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist  …
    ε : Real
    hε : GT.gt ε 0
    s : Finset α
    hs : LT.lt (abs (HSub.hSub (HPow.hPow (Norm.norm f) p.toReal) (s.sum fun b =>  …
    H : Eq (HPow.hPow (Norm.norm (HSub.hSub (s.sum fun i => lp.single p i (↑f i))  …
    ⊢ LT.lt (HPow.hPow (Norm.norm (HSub.hSub (s.sum fun b => lp.single p b (↑f b)) …
  -/
  rw [← H] at hs
  have : |‖(∑ i ∈ s, lp.single p i (f i : E i)) - f‖ ^ p.toReal| =
      ‖(∑ i ∈ s, lp.single p i (f i : E i)) - f‖ ^ p.toReal := by
    simp only [Real.abs_rpow_of_nonneg (norm_nonneg _), abs_norm]
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝² : (i : α) → NormedAddCommGroup (E i)
    inst✝¹ : DecidableEq α
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    f : Subtype fun x => Membership.mem (lp E p) x
    hp₀ : LT.lt 0 p
    hp' : LT.lt 0 p.toReal
    this✝ : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist …
    ε : Real
    hε : GT.gt ε 0
    s : Finset α
    hs : LT.lt (abs (HPow.hPow (Norm.norm (HSub.hSub (s.sum fun i => lp.single p i …
    H : Eq (HPow.hPow (Norm.norm (HSub.hSub (s.sum fun i => lp.single p i (↑f i))  …
    this : Eq (abs (HPow.hPow (Norm.norm (HSub.hSub (s.sum fun i => lp.single p i  …
    ⊢ LT.lt (HPow.hPow (Norm.norm (HSub.hSub (s.sum fun b => lp.single p b (↑f b)) …
  -/
  exact this ▸ hs
  /-
    🎉 no goals
  -/


/-- The coercion from `lp E p` to `∀ i, E i` is uniformly continuous. -/
theorem uniformContinuous_coe [_i : Fact (1 ≤ p)] :
    UniformContinuous (α := lp E p) ((↑) : lp E p → ∀ i, E i) := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    ⊢ UniformContinuous Subtype.val
  -/
  have hp : p ≠ 0 := (zero_lt_one.trans_le _i.elim).ne'
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    hp : Ne p 0
    ⊢ UniformContinuous Subtype.val
  -/
  rw [uniformContinuous_pi]
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    hp : Ne p 0
    ⊢ ∀ (i : α), UniformContinuous fun x => ↑x i
  -/
  intro i
  rw [NormedAddCommGroup.uniformity_basis_dist.uniformContinuous_iff
    NormedAddCommGroup.uniformity_basis_dist]
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    hp : Ne p 0
    i : α
    ⊢ ∀ (i_1 : Real), LT.lt 0 i_1 → Exists fun j => And (LT.lt 0 j) (∀ (x y : Subt …
  -/
  intro ε hε
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    hp : Ne p 0
    i : α
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun j => And (LT.lt 0 j) (∀ (x y : Subtype fun x => Membership.mem (l …
  -/
  refine ⟨ε, hε, ?_⟩
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    hp : Ne p 0
    i : α
    ε : Real
    hε : LT.lt 0 ε
    ⊢ ∀ (x y : Subtype fun x => Membership.mem (lp E p) x), Membership.mem (setOf  …
  -/
  rintro f g (hfg : ‖f - g‖ < ε)
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    hp : Ne p 0
    i : α
    ε : Real
    hε : LT.lt 0 ε
    f g : Subtype fun x => Membership.mem (lp E p) x
    hfg : LT.lt (Norm.norm (HSub.hSub f g)) ε
    ⊢ Membership.mem (setOf fun p => LT.lt (Norm.norm (HSub.hSub p.1 p.2)) ε) { fs …
  -/
  have : ‖f i - g i‖ ≤ ‖f - g‖ := norm_apply_le_norm hp (f - g) i
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    hp : Ne p 0
    i : α
    ε : Real
    hε : LT.lt 0 ε
    f g : Subtype fun x => Membership.mem (lp E p) x
    hfg : LT.lt (Norm.norm (HSub.hSub f g)) ε
    this : LE.le (Norm.norm (HSub.hSub (↑f i) (↑g i))) (Norm.norm (HSub.hSub f g))
    ⊢ Membership.mem (setOf fun p => LT.lt (Norm.norm (HSub.hSub p.1 p.2)) ε) { fs …
  -/
  exact this.trans_lt hfg
  /-
    🎉 no goals
  -/


theorem norm_apply_le_of_tendsto {C : ℝ} {F : ι → lp E ∞} (hCF : ∀ᶠ k in l, ‖F k‖ ≤ C)
    {f : ∀ a, E a} (hf : Tendsto (id fun i => F i : ι → ∀ a, E a) l (𝓝 f)) (a : α) : ‖f a‖ ≤ C := by
  have : Tendsto (fun k => ‖F k a‖) l (𝓝 ‖f a‖) :=
    (Tendsto.comp (continuous_apply a).continuousAt hf).norm
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    C : Real
    F : ι → Subtype fun x => Membership.mem (lp E Top.top) x
    hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
    f : (a : α) → E a
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
    a : α
    this : Filter.Tendsto (fun k => Norm.norm (↑(F k) a)) l (nhds (Norm.norm (f a)))
    ⊢ LE.le (Norm.norm (f a)) C
  -/
  refine le_of_tendsto this (hCF.mono ?_)
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    C : Real
    F : ι → Subtype fun x => Membership.mem (lp E Top.top) x
    hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
    f : (a : α) → E a
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
    a : α
    this : Filter.Tendsto (fun k => Norm.norm (↑(F k) a)) l (nhds (Norm.norm (f a)))
    ⊢ ∀ (x : ι), LE.le (Norm.norm (F x)) C → LE.le (Norm.norm (↑(F x) a)) C
  -/
  intro k hCFk
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    C : Real
    F : ι → Subtype fun x => Membership.mem (lp E Top.top) x
    hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
    f : (a : α) → E a
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
    a : α
    this : Filter.Tendsto (fun k => Norm.norm (↑(F k) a)) l (nhds (Norm.norm (f a)))
    k : ι
    hCFk : LE.le (Norm.norm (F k)) C
    ⊢ LE.le (Norm.norm (↑(F k) a)) C
  -/
  exact (norm_apply_le_norm ENNReal.top_ne_zero (F k) a).trans hCFk
  /-
    🎉 no goals
  -/


theorem sum_rpow_le_of_tendsto (hp : p ≠ ∞) {C : ℝ} {F : ι → lp E p} (hCF : ∀ᶠ k in l, ‖F k‖ ≤ C)
    {f : ∀ a, E a} (hf : Tendsto (id fun i => F i : ι → ∀ a, E a) l (𝓝 f)) (s : Finset α) :
    ∑ i ∈ s, ‖f i‖ ^ p.toReal ≤ C ^ p.toReal := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    _i : Fact (LE.le 1 p)
    hp : Ne p Top.top
    C : Real
    F : ι → Subtype fun x => Membership.mem (lp E p) x
    hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
    f : (a : α) → E a
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
    s : Finset α
    ⊢ LE.le (s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toReal) (HPow.hPow C p.t …
  -/
  have hp' : p ≠ 0 := (zero_lt_one.trans_le _i.elim).ne'
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    _i : Fact (LE.le 1 p)
    hp : Ne p Top.top
    C : Real
    F : ι → Subtype fun x => Membership.mem (lp E p) x
    hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
    f : (a : α) → E a
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
    s : Finset α
    hp' : Ne p 0
    ⊢ LE.le (s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toReal) (HPow.hPow C p.t …
  -/
  have hp'' : 0 < p.toReal := ENNReal.toReal_pos hp' hp
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    _i : Fact (LE.le 1 p)
    hp : Ne p Top.top
    C : Real
    F : ι → Subtype fun x => Membership.mem (lp E p) x
    hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
    f : (a : α) → E a
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
    s : Finset α
    hp' : Ne p 0
    hp'' : LT.lt 0 p.toReal
    ⊢ LE.le (s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toReal) (HPow.hPow C p.t …
  -/
  let G : (∀ a, E a) → ℝ := fun f => ∑ a ∈ s, ‖f a‖ ^ p.toReal
  have hG : Continuous G := by
    refine continuous_finset_sum s ?_
    intro a _
    have : Continuous fun f : ∀ a, E a => f a := continuous_apply a
    exact this.norm.rpow_const fun _ => Or.inr hp''.le
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    _i : Fact (LE.le 1 p)
    hp : Ne p Top.top
    C : Real
    F : ι → Subtype fun x => Membership.mem (lp E p) x
    hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
    f : (a : α) → E a
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
    s : Finset α
    hp' : Ne p 0
    hp'' : LT.lt 0 p.toReal
    G : ((a : α) → E a) → Real := fun f => s.sum fun a => HPow.hPow (Norm.norm (f  …
    hG : Continuous G
    ⊢ LE.le (s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toReal) (HPow.hPow C p.t …
  -/
  refine le_of_tendsto (hG.continuousAt.tendsto.comp hf) ?_
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    _i : Fact (LE.le 1 p)
    hp : Ne p Top.top
    C : Real
    F : ι → Subtype fun x => Membership.mem (lp E p) x
    hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
    f : (a : α) → E a
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
    s : Finset α
    hp' : Ne p 0
    hp'' : LT.lt 0 p.toReal
    G : ((a : α) → E a) → Real := fun f => s.sum fun a => HPow.hPow (Norm.norm (f  …
    hG : Continuous G
    ⊢ Filter.Eventually (fun c => LE.le (Function.comp G (id fun i => ↑(F i)) c) ( …
  -/
  refine hCF.mono ?_
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    _i : Fact (LE.le 1 p)
    hp : Ne p Top.top
    C : Real
    F : ι → Subtype fun x => Membership.mem (lp E p) x
    hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
    f : (a : α) → E a
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
    s : Finset α
    hp' : Ne p 0
    hp'' : LT.lt 0 p.toReal
    G : ((a : α) → E a) → Real := fun f => s.sum fun a => HPow.hPow (Norm.norm (f  …
    hG : Continuous G
    ⊢ ∀ (x : ι), LE.le (Norm.norm (F x)) C → LE.le (Function.comp G (id fun i => ↑ …
  -/
  intro k hCFk
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    _i : Fact (LE.le 1 p)
    hp : Ne p Top.top
    C : Real
    F : ι → Subtype fun x => Membership.mem (lp E p) x
    hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
    f : (a : α) → E a
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
    s : Finset α
    hp' : Ne p 0
    hp'' : LT.lt 0 p.toReal
    G : ((a : α) → E a) → Real := fun f => s.sum fun a => HPow.hPow (Norm.norm (f  …
    hG : Continuous G
    k : ι
    hCFk : LE.le (Norm.norm (F k)) C
    ⊢ LE.le (Function.comp G (id fun i => ↑(F i)) k) (HPow.hPow C p.toReal)
  -/
  refine (lp.sum_rpow_le_norm_rpow hp'' (F k) s).trans ?_
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    _i : Fact (LE.le 1 p)
    hp : Ne p Top.top
    C : Real
    F : ι → Subtype fun x => Membership.mem (lp E p) x
    hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
    f : (a : α) → E a
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
    s : Finset α
    hp' : Ne p 0
    hp'' : LT.lt 0 p.toReal
    G : ((a : α) → E a) → Real := fun f => s.sum fun a => HPow.hPow (Norm.norm (f  …
    hG : Continuous G
    k : ι
    hCFk : LE.le (Norm.norm (F k)) C
    ⊢ LE.le (HPow.hPow (Norm.norm (F k)) p.toReal) (HPow.hPow C p.toReal)
  -/
  gcongr
  /-
    🎉 no goals
  -/


/-- "Semicontinuity of the `lp` norm": If all sufficiently large elements of a sequence in `lp E p`
 have `lp` norm `≤ C`, then the pointwise limit, if it exists, also has `lp` norm `≤ C`. -/
theorem norm_le_of_tendsto {C : ℝ} {F : ι → lp E p} (hCF : ∀ᶠ k in l, ‖F k‖ ≤ C) {f : lp E p}
    (hf : Tendsto (id fun i => F i : ι → ∀ a, E a) l (𝓝 f)) : ‖f‖ ≤ C := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    _i : Fact (LE.le 1 p)
    C : Real
    F : ι → Subtype fun x => Membership.mem (lp E p) x
    hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
    f : Subtype fun x => Membership.mem (lp E p) x
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds ↑f)
    ⊢ LE.le (Norm.norm f) C
  -/
  obtain ⟨i, hi⟩ := hCF.exists
  /-
    case intro
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    _i : Fact (LE.le 1 p)
    C : Real
    F : ι → Subtype fun x => Membership.mem (lp E p) x
    hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
    f : Subtype fun x => Membership.mem (lp E p) x
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds ↑f)
    i : ι
    hi : LE.le (Norm.norm (F i)) C
    ⊢ LE.le (Norm.norm f) C
  -/
  have hC : 0 ≤ C := (norm_nonneg _).trans hi
  /-
    case intro
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    _i : Fact (LE.le 1 p)
    C : Real
    F : ι → Subtype fun x => Membership.mem (lp E p) x
    hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
    f : Subtype fun x => Membership.mem (lp E p) x
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds ↑f)
    i : ι
    hi : LE.le (Norm.norm (F i)) C
    hC : LE.le 0 C
    ⊢ LE.le (Norm.norm f) C
  -/
  rcases eq_top_or_lt_top p with (rfl | hp)
    /-
      case intro.inl
      α : Type u_1
      E : α → Type u_2
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      l : Filter ι
      inst✝ : l.NeBot
      C : Real
      i : ι
      hC : LE.le 0 C
      _i : Fact (LE.le 1 Top.top)
      F : ι → Subtype fun x => Membership.mem (lp E Top.top) x
      hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds ↑f)
      hi : LE.le (Norm.norm (F i)) C
      ⊢ LE.le (Norm.norm f) C
    -/
  · apply norm_le_of_forall_le hC
    /-
      case intro.inl
      α : Type u_1
      E : α → Type u_2
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      l : Filter ι
      inst✝ : l.NeBot
      C : Real
      i : ι
      hC : LE.le 0 C
      _i : Fact (LE.le 1 Top.top)
      F : ι → Subtype fun x => Membership.mem (lp E Top.top) x
      hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds ↑f)
      hi : LE.le (Norm.norm (F i)) C
      ⊢ ∀ (i : α), LE.le (Norm.norm (↑f i)) C
    -/
    exact norm_apply_le_of_tendsto hCF hf
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      l : Filter ι
      inst✝ : l.NeBot
      _i : Fact (LE.le 1 p)
      C : Real
      F : ι → Subtype fun x => Membership.mem (lp E p) x
      hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
      f : Subtype fun x => Membership.mem (lp E p) x
      hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds ↑f)
      i : ι
      hi : LE.le (Norm.norm (F i)) C
      hC : LE.le 0 C
      hp : LT.lt p Top.top
      ⊢ LE.le (Norm.norm f) C
    -/
  · have : 0 < p := zero_lt_one.trans_le _i.elim
    /-
      case intro.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      l : Filter ι
      inst✝ : l.NeBot
      _i : Fact (LE.le 1 p)
      C : Real
      F : ι → Subtype fun x => Membership.mem (lp E p) x
      hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
      f : Subtype fun x => Membership.mem (lp E p) x
      hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds ↑f)
      i : ι
      hi : LE.le (Norm.norm (F i)) C
      hC : LE.le 0 C
      hp : LT.lt p Top.top
      this : LT.lt 0 p
      ⊢ LE.le (Norm.norm f) C
    -/
    have hp' : 0 < p.toReal := ENNReal.toReal_pos this.ne' hp.ne
    /-
      case intro.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      l : Filter ι
      inst✝ : l.NeBot
      _i : Fact (LE.le 1 p)
      C : Real
      F : ι → Subtype fun x => Membership.mem (lp E p) x
      hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
      f : Subtype fun x => Membership.mem (lp E p) x
      hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds ↑f)
      i : ι
      hi : LE.le (Norm.norm (F i)) C
      hC : LE.le 0 C
      hp : LT.lt p Top.top
      this : LT.lt 0 p
      hp' : LT.lt 0 p.toReal
      ⊢ LE.le (Norm.norm f) C
    -/
    apply norm_le_of_forall_sum_le hp' hC
    /-
      case intro.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      l : Filter ι
      inst✝ : l.NeBot
      _i : Fact (LE.le 1 p)
      C : Real
      F : ι → Subtype fun x => Membership.mem (lp E p) x
      hCF : Filter.Eventually (fun k => LE.le (Norm.norm (F k)) C) l
      f : Subtype fun x => Membership.mem (lp E p) x
      hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds ↑f)
      i : ι
      hi : LE.le (Norm.norm (F i)) C
      hC : LE.le 0 C
      hp : LT.lt p Top.top
      this : LT.lt 0 p
      hp' : LT.lt 0 p.toReal
      ⊢ ∀ (s : Finset α), LE.le (s.sum fun i => HPow.hPow (Norm.norm (↑f i)) p.toRea …
    -/
    exact sum_rpow_le_of_tendsto hp.ne hCF hf
    /-
      🎉 no goals
    -/


/-- If `f` is the pointwise limit of a bounded sequence in `lp E p`, then `f` is in `lp E p`. -/
theorem memℓp_of_tendsto {F : ι → lp E p} (hF : Bornology.IsBounded (Set.range F)) {f : ∀ a, E a}
    (hf : Tendsto (id fun i => F i : ι → ∀ a, E a) l (𝓝 f)) : Memℓp f p := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    _i : Fact (LE.le 1 p)
    F : ι → Subtype fun x => Membership.mem (lp E p) x
    hF : Bornology.IsBounded (Set.range F)
    f : (a : α) → E a
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
    ⊢ Memℓp f p
  -/
  obtain ⟨C, hCF⟩ : ∃ C, ∀ k, ‖F k‖ ≤ C := hF.exists_norm_le.imp fun _ ↦ Set.forall_mem_range.1
  /-
    case intro
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    ι : Type u_3
    l : Filter ι
    inst✝ : l.NeBot
    _i : Fact (LE.le 1 p)
    F : ι → Subtype fun x => Membership.mem (lp E p) x
    hF : Bornology.IsBounded (Set.range F)
    f : (a : α) → E a
    hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
    C : Real
    hCF : ∀ (k : ι), LE.le (Norm.norm (F k)) C
    ⊢ Memℓp f p
  -/
  rcases eq_top_or_lt_top p with (rfl | hp)
    /-
      case intro.inl
      α : Type u_1
      E : α → Type u_2
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      l : Filter ι
      inst✝ : l.NeBot
      f : (a : α) → E a
      C : Real
      _i : Fact (LE.le 1 Top.top)
      F : ι → Subtype fun x => Membership.mem (lp E Top.top) x
      hF : Bornology.IsBounded (Set.range F)
      hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
      hCF : ∀ (k : ι), LE.le (Norm.norm (F k)) C
      ⊢ Memℓp f Top.top
    -/
  · apply memℓp_infty
    /-
      case intro.inl.hf
      α : Type u_1
      E : α → Type u_2
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      l : Filter ι
      inst✝ : l.NeBot
      f : (a : α) → E a
      C : Real
      _i : Fact (LE.le 1 Top.top)
      F : ι → Subtype fun x => Membership.mem (lp E Top.top) x
      hF : Bornology.IsBounded (Set.range F)
      hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
      hCF : ∀ (k : ι), LE.le (Norm.norm (F k)) C
      ⊢ BddAbove (Set.range fun i => Norm.norm (f i))
    -/
    use C
    /-
      case h
      α : Type u_1
      E : α → Type u_2
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      l : Filter ι
      inst✝ : l.NeBot
      f : (a : α) → E a
      C : Real
      _i : Fact (LE.le 1 Top.top)
      F : ι → Subtype fun x => Membership.mem (lp E Top.top) x
      hF : Bornology.IsBounded (Set.range F)
      hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
      hCF : ∀ (k : ι), LE.le (Norm.norm (F k)) C
      ⊢ Membership.mem (upperBounds (Set.range fun i => Norm.norm (f i))) C
    -/
    rintro _ ⟨a, rfl⟩
    /-
      case h.intro
      α : Type u_1
      E : α → Type u_2
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      l : Filter ι
      inst✝ : l.NeBot
      f : (a : α) → E a
      C : Real
      _i : Fact (LE.le 1 Top.top)
      F : ι → Subtype fun x => Membership.mem (lp E Top.top) x
      hF : Bornology.IsBounded (Set.range F)
      hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
      hCF : ∀ (k : ι), LE.le (Norm.norm (F k)) C
      a : α
      ⊢ LE.le ((fun i => Norm.norm (f i)) a) C
    -/
    exact norm_apply_le_of_tendsto (Eventually.of_forall hCF) hf a
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      l : Filter ι
      inst✝ : l.NeBot
      _i : Fact (LE.le 1 p)
      F : ι → Subtype fun x => Membership.mem (lp E p) x
      hF : Bornology.IsBounded (Set.range F)
      f : (a : α) → E a
      hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
      C : Real
      hCF : ∀ (k : ι), LE.le (Norm.norm (F k)) C
      hp : LT.lt p Top.top
      ⊢ Memℓp f p
    -/
  · apply memℓp_gen'
    /-
      case intro.inr.hf
      α : Type u_1
      E : α → Type u_2
      p : ENNReal
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      l : Filter ι
      inst✝ : l.NeBot
      _i : Fact (LE.le 1 p)
      F : ι → Subtype fun x => Membership.mem (lp E p) x
      hF : Bornology.IsBounded (Set.range F)
      f : (a : α) → E a
      hf : Filter.Tendsto (id fun i => ↑(F i)) l (nhds f)
      C : Real
      hCF : ∀ (k : ι), LE.le (Norm.norm (F k)) C
      hp : LT.lt p Top.top
      ⊢ ∀ (s : Finset α), LE.le (s.sum fun i => HPow.hPow (Norm.norm (f i)) p.toReal …
    -/
    exact sum_rpow_le_of_tendsto hp.ne (Eventually.of_forall hCF) hf
    /-
      🎉 no goals
    -/


/-- If a sequence is Cauchy in the `lp E p` topology and pointwise convergent to an element `f` of
`lp E p`, then it converges to `f` in the `lp E p` topology. -/
theorem tendsto_lp_of_tendsto_pi {F : ℕ → lp E p} (hF : CauchySeq F) {f : lp E p}
    (hf : Tendsto (id fun i => F i : ℕ → ∀ a, E a) atTop (𝓝 f)) : Tendsto F atTop (𝓝 f) := by
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    F : Nat → Subtype fun x => Membership.mem (lp E p) x
    hF : CauchySeq F
    f : Subtype fun x => Membership.mem (lp E p) x
    hf : Filter.Tendsto (id fun i => ↑(F i)) Filter.atTop (nhds ↑f)
    ⊢ Filter.Tendsto F Filter.atTop (nhds f)
  -/
  rw [Metric.nhds_basis_closedBall.tendsto_right_iff]
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    F : Nat → Subtype fun x => Membership.mem (lp E p) x
    hF : CauchySeq F
    f : Subtype fun x => Membership.mem (lp E p) x
    hf : Filter.Tendsto (id fun i => ↑(F i)) Filter.atTop (nhds ↑f)
    ⊢ ∀ (i : Real), LT.lt 0 i → Filter.Eventually (fun x => Membership.mem (Metric …
  -/
  intro ε hε
  have hε' : { p : lp E p × lp E p | ‖p.1 - p.2‖ < ε } ∈ uniformity (lp E p) :=
    NormedAddCommGroup.uniformity_basis_dist.mem_of_mem hε
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    F : Nat → Subtype fun x => Membership.mem (lp E p) x
    hF : CauchySeq F
    f : Subtype fun x => Membership.mem (lp E p) x
    hf : Filter.Tendsto (id fun i => ↑(F i)) Filter.atTop (nhds ↑f)
    ε : Real
    hε : LT.lt 0 ε
    hε' : Membership.mem (uniformity (Subtype fun x => Membership.mem (lp E p) x)) …
    ⊢ Filter.Eventually (fun x => Membership.mem (Metric.closedBall f ε) (F x)) Fi …
  -/
  refine (hF.eventually_eventually hε').mono ?_
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    F : Nat → Subtype fun x => Membership.mem (lp E p) x
    hF : CauchySeq F
    f : Subtype fun x => Membership.mem (lp E p) x
    hf : Filter.Tendsto (id fun i => ↑(F i)) Filter.atTop (nhds ↑f)
    ε : Real
    hε : LT.lt 0 ε
    hε' : Membership.mem (uniformity (Subtype fun x => Membership.mem (lp E p) x)) …
    ⊢ ∀ (x : Nat), Filter.Eventually (fun l => Membership.mem (setOf fun p_1 => LT …
  -/
  rintro n (hn : ∀ᶠ l in atTop, ‖(fun f => F n - f) (F l)‖ < ε)
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    F : Nat → Subtype fun x => Membership.mem (lp E p) x
    hF : CauchySeq F
    f : Subtype fun x => Membership.mem (lp E p) x
    hf : Filter.Tendsto (id fun i => ↑(F i)) Filter.atTop (nhds ↑f)
    ε : Real
    hε : LT.lt 0 ε
    hε' : Membership.mem (uniformity (Subtype fun x => Membership.mem (lp E p) x)) …
    n : Nat
    hn : Filter.Eventually (fun l => LT.lt (Norm.norm ((fun f => HSub.hSub (F n) f …
    ⊢ Membership.mem (Metric.closedBall f ε) (F n)
  -/
  refine norm_le_of_tendsto (hn.mono fun k hk => hk.le) ?_
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    F : Nat → Subtype fun x => Membership.mem (lp E p) x
    hF : CauchySeq F
    f : Subtype fun x => Membership.mem (lp E p) x
    hf : Filter.Tendsto (id fun i => ↑(F i)) Filter.atTop (nhds ↑f)
    ε : Real
    hε : LT.lt 0 ε
    hε' : Membership.mem (uniformity (Subtype fun x => Membership.mem (lp E p) x)) …
    n : Nat
    hn : Filter.Eventually (fun l => LT.lt (Norm.norm ((fun f => HSub.hSub (F n) f …
    ⊢ Filter.Tendsto (id fun i => ↑((fun f => HSub.hSub (F n) f) (F i))) Filter.at …
  -/
  rw [tendsto_pi_nhds]
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    F : Nat → Subtype fun x => Membership.mem (lp E p) x
    hF : CauchySeq F
    f : Subtype fun x => Membership.mem (lp E p) x
    hf : Filter.Tendsto (id fun i => ↑(F i)) Filter.atTop (nhds ↑f)
    ε : Real
    hε : LT.lt 0 ε
    hε' : Membership.mem (uniformity (Subtype fun x => Membership.mem (lp E p) x)) …
    n : Nat
    hn : Filter.Eventually (fun l => LT.lt (Norm.norm ((fun f => HSub.hSub (F n) f …
    ⊢ ∀ (x : α), Filter.Tendsto (fun i => id (fun i => ↑((fun f => HSub.hSub (F n) …
  -/
  intro a
  /-
    α : Type u_1
    E : α → Type u_2
    p : ENNReal
    inst✝ : (i : α) → NormedAddCommGroup (E i)
    _i : Fact (LE.le 1 p)
    F : Nat → Subtype fun x => Membership.mem (lp E p) x
    hF : CauchySeq F
    f : Subtype fun x => Membership.mem (lp E p) x
    hf : Filter.Tendsto (id fun i => ↑(F i)) Filter.atTop (nhds ↑f)
    ε : Real
    hε : LT.lt 0 ε
    hε' : Membership.mem (uniformity (Subtype fun x => Membership.mem (lp E p) x)) …
    n : Nat
    hn : Filter.Eventually (fun l => LT.lt (Norm.norm ((fun f => HSub.hSub (F n) f …
    a : α
    ⊢ Filter.Tendsto (fun i => id (fun i => ↑((fun f => HSub.hSub (F n) f) (F i))) …
  -/
  exact (hf.apply_nhds a).const_sub (F n a)
  /-
    🎉 no goals
  -/


instance completeSpace : CompleteSpace (lp E p) :=
  Metric.complete_of_cauchySeq_tendsto (by
    /-
      α : Type u_1
      E : α → Type u_2
      p q : ENNReal
      inst✝² : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      l : Filter ι
      inst✝¹ : l.NeBot
      _i : Fact (LE.le 1 p)
      inst✝ : ∀ (a : α), CompleteSpace (E a)
      ⊢ ∀ (u : Nat → Subtype fun x => Membership.mem (lp E p) x), CauchySeq u → Exis …
    -/
    intro F hF
    -- A Cauchy sequence in `lp E p` is pointwise convergent; let `f` be the pointwise limit.
    obtain ⟨f, hf⟩ := cauchySeq_tendsto_of_complete
      ((uniformContinuous_coe (p := p)).comp_cauchySeq hF)
    -- Since the Cauchy sequence is bounded, its pointwise limit `f` is in `lp E p`.
    /-
      case intro
      α : Type u_1
      E : α → Type u_2
      p q : ENNReal
      inst✝² : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      l : Filter ι
      inst✝¹ : l.NeBot
      _i : Fact (LE.le 1 p)
      inst✝ : ∀ (a : α), CompleteSpace (E a)
      F : Nat → Subtype fun x => Membership.mem (lp E p) x
      hF : CauchySeq F
      f : (i : α) → E i
      hf : Filter.Tendsto (Function.comp Subtype.val F) Filter.atTop (nhds f)
      ⊢ Exists fun a => Filter.Tendsto F Filter.atTop (nhds a)
    -/
    have hf' : Memℓp f p := memℓp_of_tendsto hF.isBounded_range hf
    -- And therefore `f` is its limit in the `lp E p` topology as well as pointwise.
    /-
      case intro
      α : Type u_1
      E : α → Type u_2
      p q : ENNReal
      inst✝² : (i : α) → NormedAddCommGroup (E i)
      ι : Type u_3
      l : Filter ι
      inst✝¹ : l.NeBot
      _i : Fact (LE.le 1 p)
      inst✝ : ∀ (a : α), CompleteSpace (E a)
      F : Nat → Subtype fun x => Membership.mem (lp E p) x
      hF : CauchySeq F
      f : (i : α) → E i
      hf : Filter.Tendsto (Function.comp Subtype.val F) Filter.atTop (nhds f)
      hf' : Memℓp f p
      ⊢ Exists fun a => Filter.Tendsto F Filter.atTop (nhds a)
    -/
    exact ⟨⟨f, hf'⟩, tendsto_lp_of_tendsto_pi hF hf⟩)
    /-
      🎉 no goals
    -/


lemma LipschitzWith.uniformly_bounded [PseudoMetricSpace α] (g : α → ι → ℝ) {K : ℝ≥0}
    (hg : ∀ i, LipschitzWith K (g · i)) (a₀ : α) (hga₀b : Memℓp (g a₀) ∞) (a : α) :
    Memℓp (g a) ∞ := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : PseudoMetricSpace α
    g : α → ι → Real
    K : NNReal
    hg : ∀ (i : ι), LipschitzWith K fun x => g x i
    a₀ : α
    hga₀b : Memℓp (g a₀) Top.top
    a : α
    ⊢ Memℓp (g a) Top.top
  -/
  rcases hga₀b with ⟨M, hM⟩
  /-
    case intro
    α : Type u_1
    ι : Type u_3
    inst✝ : PseudoMetricSpace α
    g : α → ι → Real
    K : NNReal
    hg : ∀ (i : ι), LipschitzWith K fun x => g x i
    a₀ a : α
    M : Real
    hM : Membership.mem (upperBounds (Set.range fun i => Norm.norm (g a₀ i))) M
    ⊢ Memℓp (g a) Top.top
  -/
  use ↑K * dist a a₀ + M
  /-
    case h
    α : Type u_1
    ι : Type u_3
    inst✝ : PseudoMetricSpace α
    g : α → ι → Real
    K : NNReal
    hg : ∀ (i : ι), LipschitzWith K fun x => g x i
    a₀ a : α
    M : Real
    hM : Membership.mem (upperBounds (Set.range fun i => Norm.norm (g a₀ i))) M
    ⊢ Membership.mem (upperBounds (Set.range fun i => Norm.norm (g a i))) (HAdd.hA …
  -/
  rintro - ⟨i, rfl⟩
  calc
    |g a i| = |g a i - g a₀ i + g a₀ i| := by simp
    _ ≤ |g a i - g a₀ i| + |g a₀ i| := abs_add _ _
    _ ≤ ↑K * dist a a₀ + M := by
        gcongr
        · exact lipschitzWith_iff_dist_le_mul.1 (hg i) a a₀
        · exact hM ⟨i, rfl⟩


theorem LipschitzOnWith.coordinate [PseudoMetricSpace α] (f : α → ℓ^∞(ι)) (s : Set α) (K : ℝ≥0) :
    LipschitzOnWith K f s ↔ ∀ i : ι, LipschitzOnWith K (fun a : α ↦ f a i) s := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : PseudoMetricSpace α
    f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
    s : Set α
    K : NNReal
    ⊢ Iff (LipschitzOnWith K f s) (∀ (i : ι), LipschitzOnWith K (fun a => ↑(f a) i …
  -/
  simp_rw [lipschitzOnWith_iff_dist_le_mul]
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : PseudoMetricSpace α
    f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
    s : Set α
    K : NNReal
    ⊢ Iff (∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LE.le ( …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      ι : Type u_3
      inst✝ : PseudoMetricSpace α
      f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
      s : Set α
      K : NNReal
      ⊢ (∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LE.le (Dist …
    -/
  · intro hfl i x hx y hy
    calc
      dist (f x i) (f y i) ≤ dist (f x) (f y) := lp.norm_apply_le_norm top_ne_zero (f x - f y) i
      _ ≤ K * dist x y := hfl x hx y hy
    /-
      case mpr
      α : Type u_1
      ι : Type u_3
      inst✝ : PseudoMetricSpace α
      f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
      s : Set α
      K : NNReal
      ⊢ (∀ (i : ι) (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LE. …
    -/
  · intro hgl x hx y hy
    /-
      case mpr
      α : Type u_1
      ι : Type u_3
      inst✝ : PseudoMetricSpace α
      f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
      s : Set α
      K : NNReal
      hgl : ∀ (i : ι) (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y →  …
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      ⊢ LE.le (Dist.dist (f x) (f y)) (HMul.hMul (↑K) (Dist.dist x y))
    -/
    apply lp.norm_le_of_forall_le
      /-
        case mpr.hC
        α : Type u_1
        ι : Type u_3
        inst✝ : PseudoMetricSpace α
        f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
        s : Set α
        K : NNReal
        hgl : ∀ (i : ι) (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y →  …
        x : α
        hx : Membership.mem s x
        y : α
        hy : Membership.mem s y
        ⊢ LE.le 0 (HMul.hMul (↑K) (Dist.dist x y))
      -/
    · positivity
      /-
        🎉 no goals
      -/
    /-
      case mpr.hCf
      α : Type u_1
      ι : Type u_3
      inst✝ : PseudoMetricSpace α
      f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
      s : Set α
      K : NNReal
      hgl : ∀ (i : ι) (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y →  …
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      ⊢ ∀ (i : ι), LE.le (Norm.norm (↑(HSub.hSub (f x) (f y)) i)) (HMul.hMul (↑K) (D …
    -/
    intro i
    /-
      case mpr.hCf
      α : Type u_1
      ι : Type u_3
      inst✝ : PseudoMetricSpace α
      f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
      s : Set α
      K : NNReal
      hgl : ∀ (i : ι) (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y →  …
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      i : ι
      ⊢ LE.le (Norm.norm (↑(HSub.hSub (f x) (f y)) i)) (HMul.hMul (↑K) (Dist.dist x  …
    -/
    apply hgl i x hx y hy
    /-
      🎉 no goals
    -/


theorem LipschitzWith.coordinate [PseudoMetricSpace α] {f : α → ℓ^∞(ι)} (K : ℝ≥0) :
    LipschitzWith K f ↔ ∀ i : ι, LipschitzWith K (fun a : α ↦ f a i) := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : PseudoMetricSpace α
    f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
    K : NNReal
    ⊢ Iff (LipschitzWith K f) (∀ (i : ι), LipschitzWith K fun a => ↑(f a) i)
  -/
  simp_rw [← lipschitzOnWith_univ]
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : PseudoMetricSpace α
    f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
    K : NNReal
    ⊢ Iff (LipschitzOnWith K f Set.univ) (∀ (i : ι), LipschitzOnWith K (fun a => ↑ …
  -/
  apply LipschitzOnWith.coordinate
  /-
    🎉 no goals
  -/


