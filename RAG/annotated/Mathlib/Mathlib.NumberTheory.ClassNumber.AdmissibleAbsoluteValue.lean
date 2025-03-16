local infixl:50 " ≺ " => EuclideanDomain.r


/-- An absolute value `R → ℤ` is admissible if it respects the Euclidean domain
structure and a large enough set of elements in `R^n` will contain a pair of
elements whose remainders are pointwise close together. -/
structure IsAdmissible extends IsEuclidean abv where
  protected card : ℝ → ℕ
  /-- For all `ε > 0` and finite families `A`, we can partition the remainders of `A` mod `b`
  into `abv.card ε` sets, such that all elements in each part of remainders are close together. -/
  exists_partition' :
    ∀ (n : ℕ) {ε : ℝ} (_ : 0 < ε) {b : R} (_ : b ≠ 0) (A : Fin n → R),
      ∃ t : Fin n → Fin (card ε), ∀ i₀ i₁, t i₀ = t i₁ → (abv (A i₁ % b - A i₀ % b) : ℝ) < abv b • ε

-- Porting note: no docstrings for IsAdmissible

/-- For all `ε > 0` and finite families `A`, we can partition the remainders of `A` mod `b`
into `abv.card ε` sets, such that all elements in each part of remainders are close together. -/
theorem exists_partition {ι : Type*} [Finite ι] {ε : ℝ} (hε : 0 < ε) {b : R} (hb : b ≠ 0)
    (A : ι → R) (h : abv.IsAdmissible) : ∃ t : ι → Fin (h.card ε),
      ∀ i₀ i₁, t i₀ = t i₁ → (abv (A i₁ % b - A i₀ % b) : ℝ) < abv b • ε := by
  /-
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    abv : AbsoluteValue R Int
    ι : Type u_2
    inst✝ : Finite ι
    ε : Real
    hε : LT.lt 0 ε
    b : R
    hb : Ne b 0
    A : ι → R
    h : abv.IsAdmissible
    ⊢ Exists fun t => ∀ (i₀ i₁ : ι), Eq (t i₀) (t i₁) → LT.lt (↑(abv (HSub.hSub (H …
  -/
  rcases Finite.exists_equiv_fin ι with ⟨n, ⟨e⟩⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    abv : AbsoluteValue R Int
    ι : Type u_2
    inst✝ : Finite ι
    ε : Real
    hε : LT.lt 0 ε
    b : R
    hb : Ne b 0
    A : ι → R
    h : abv.IsAdmissible
    n : Nat
    e : Equiv ι (Fin n)
    ⊢ Exists fun t => ∀ (i₀ i₁ : ι), Eq (t i₀) (t i₁) → LT.lt (↑(abv (HSub.hSub (H …
  -/
  obtain ⟨t, ht⟩ := h.exists_partition' n hε hb (A ∘ e.symm)
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    abv : AbsoluteValue R Int
    ι : Type u_2
    inst✝ : Finite ι
    ε : Real
    hε : LT.lt 0 ε
    b : R
    hb : Ne b 0
    A : ι → R
    h : abv.IsAdmissible
    n : Nat
    e : Equiv ι (Fin n)
    t : Fin n → Fin (h.card ε)
    ht : ∀ (i₀ i₁ : Fin n), Eq (t i₀) (t i₁) → LT.lt (↑(abv (HSub.hSub (HMod.hMod  …
    ⊢ Exists fun t => ∀ (i₀ i₁ : ι), Eq (t i₀) (t i₁) → LT.lt (↑(abv (HSub.hSub (H …
  -/
  refine ⟨t ∘ e, fun i₀ i₁ h ↦ ?_⟩
  convert (config := {transparency := .default})
                           /-
                             case h.e'_3.h.e'_3.h.e'_6.h.e'_5.h.e'_5.h.e'_1
                             R : Type u_1
                             inst✝¹ : EuclideanDomain R
                             abv : AbsoluteValue R Int
                             ι : Type u_2
                             inst✝ : Finite ι
                             ε : Real
                             hε : LT.lt 0 ε
                             b : R
                             hb : Ne b 0
                             A : ι → R
                             h✝ : abv.IsAdmissible
                             n : Nat
                             e : Equiv ι (Fin n)
                             t : Fin n → Fin (h✝.card ε)
                             ht : ∀ (i₀ i₁ : Fin n), Eq (t i₀) (t i₁) → LT.lt (↑(abv (HSub.hSub (HMod.hMod  …
                             i₀ i₁ : ι
                             h : Eq (Function.comp t (⇑e) i₀) (Function.comp t (⇑e) i₁)
                             ⊢ Eq i₁ (e.symm (e i₁))
                           -/
                           /-
                             🎉 no goals
                           -/
    ht (e i₀) (e i₁) h <;> simp only [e.symm_apply_apply]
                           /-
                             🎉 no goals
                           -/


/-- Any large enough family of vectors in `R^n` has a pair of elements
whose remainders are close together, pointwise. -/
theorem exists_approx_aux (n : ℕ) (h : abv.IsAdmissible) :
    ∀ {ε : ℝ} (_hε : 0 < ε) {b : R} (_hb : b ≠ 0) (A : Fin (h.card ε ^ n).succ → Fin n → R),
      ∃ i₀ i₁, i₀ ≠ i₁ ∧ ∀ k, (abv (A i₁ k % b - A i₀ k % b) : ℝ) < abv b • ε := by
  /-
    R : Type u_1
    inst✝ : EuclideanDomain R
    abv : AbsoluteValue R Int
    n : Nat
    h : abv.IsAdmissible
    ⊢ ∀ {ε : Real}, LT.lt 0 ε → ∀ {b : R}, Ne b 0 → ∀ (A : Fin (HPow.hPow (h.card  …
  -/
  haveI := Classical.decEq R
  /-
    R : Type u_1
    inst✝ : EuclideanDomain R
    abv : AbsoluteValue R Int
    n : Nat
    h : abv.IsAdmissible
    this : DecidableEq R
    ⊢ ∀ {ε : Real}, LT.lt 0 ε → ∀ {b : R}, Ne b 0 → ∀ (A : Fin (HPow.hPow (h.card  …
  -/
  induction' n with n ih
    /-
      case zero
      R : Type u_1
      inst✝ : EuclideanDomain R
      abv : AbsoluteValue R Int
      h : abv.IsAdmissible
      this : DecidableEq R
      ⊢ ∀ {ε : Real}, LT.lt 0 ε → ∀ {b : R}, Ne b 0 → ∀ (A : Fin (HPow.hPow (h.card  …
    -/
  · intro ε _hε b _hb A
    /-
      case zero
      R : Type u_1
      inst✝ : EuclideanDomain R
      abv : AbsoluteValue R Int
      h : abv.IsAdmissible
      this : DecidableEq R
      ε : Real
      _hε : LT.lt 0 ε
      b : R
      _hb : Ne b 0
      A : Fin (HPow.hPow (h.card ε) 0).succ → Fin 0 → R
      ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (∀ (k : Fin 0), LT.lt (↑(ab …
    -/
    refine ⟨0, 1, ?_, ?_⟩
      /-
        case zero.refine_1
        R : Type u_1
        inst✝ : EuclideanDomain R
        abv : AbsoluteValue R Int
        h : abv.IsAdmissible
        this : DecidableEq R
        ε : Real
        _hε : LT.lt 0 ε
        b : R
        _hb : Ne b 0
        A : Fin (HPow.hPow (h.card ε) 0).succ → Fin 0 → R
        ⊢ Ne 0 1
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case zero.refine_2
      R : Type u_1
      inst✝ : EuclideanDomain R
      abv : AbsoluteValue R Int
      h : abv.IsAdmissible
      this : DecidableEq R
      ε : Real
      _hε : LT.lt 0 ε
      b : R
      _hb : Ne b 0
      A : Fin (HPow.hPow (h.card ε) 0).succ → Fin 0 → R
      ⊢ ∀ (k : Fin 0), LT.lt (↑(abv (HSub.hSub (HMod.hMod (A 1 k) b) (HMod.hMod (A 0 …
    -/
    rintro ⟨i, ⟨⟩⟩
    /-
      🎉 no goals
    -/
  /-
    case succ
    R : Type u_1
    inst✝ : EuclideanDomain R
    abv : AbsoluteValue R Int
    h : abv.IsAdmissible
    this : DecidableEq R
    n : Nat
    ih : ∀ {ε : Real}, LT.lt 0 ε → ∀ {b : R}, Ne b 0 → ∀ (A : Fin (HPow.hPow (h.ca …
    ⊢ ∀ {ε : Real}, LT.lt 0 ε → ∀ {b : R}, Ne b 0 → ∀ (A : Fin (HPow.hPow (h.card  …
  -/
  intro ε hε b hb A
  /-
    case succ
    R : Type u_1
    inst✝ : EuclideanDomain R
    abv : AbsoluteValue R Int
    h : abv.IsAdmissible
    this : DecidableEq R
    n : Nat
    ih : ∀ {ε : Real}, LT.lt 0 ε → ∀ {b : R}, Ne b 0 → ∀ (A : Fin (HPow.hPow (h.ca …
    ε : Real
    hε : LT.lt 0 ε
    b : R
    hb : Ne b 0
    A : Fin (HPow.hPow (h.card ε) (HAdd.hAdd n 1)).succ → Fin (HAdd.hAdd n 1) → R
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (∀ (k : Fin (HAdd.hAdd n 1) …
  -/
  let M := h.card ε
  -- By the "nicer" pigeonhole principle, we can find a collection `s`
  -- of more than `M^n` remainders where the first components lie close together:
  obtain ⟨s, s_inj, hs⟩ :
    ∃ s : Fin (M ^ n).succ → Fin (M ^ n.succ).succ,
      Function.Injective s ∧ ∀ i₀ i₁, (abv (A (s i₁) 0 % b - A (s i₀) 0 % b) : ℝ) < abv b • ε := by
    -- We can partition the `A`s into `M` subsets where
    -- the first components lie close together:
    obtain ⟨t, ht⟩ :
      ∃ t : Fin (M ^ n.succ).succ → Fin M,
        ∀ i₀ i₁, t i₀ = t i₁ → (abv (A i₁ 0 % b - A i₀ 0 % b) : ℝ) < abv b • ε :=
      h.exists_partition hε hb fun x ↦ A x 0
    -- Since the `M` subsets contain more than `M * M^n` elements total,
    -- there must be a subset that contains more than `M^n` elements.
    obtain ⟨s, hs⟩ :=
      Fintype.exists_lt_card_fiber_of_mul_lt_card (f := t)
        (by simpa only [Fintype.card_fin, pow_succ'] using Nat.lt_succ_self (M ^ n.succ))
    refine ⟨fun i ↦ (Finset.univ.filter fun x ↦ t x = s).toList.get <| i.castLE ?_, fun i j h ↦ ?_,
      fun i₀ i₁ ↦ ht _ _ ?_⟩
    · rwa [Finset.length_toList]
    · ext
      simpa [(Finset.nodup_toList _).getElem_inj_iff] using h
    · #adaptation_note
      /-- This proof was nicer prior to https://github.com/leanprover/lean4/pull/4400.
      Please feel welcome to improve it, by avoiding use of `List.get` in favour of `GetElem`. -/
      have : ∀ i h, t ((Finset.univ.filter fun x ↦ t x = s).toList.get ⟨i, h⟩) = s := fun i h ↦
        (Finset.mem_filter.mp (Finset.mem_toList.mp (List.get_mem _ ⟨i, h⟩))).2
      simp only [Nat.succ_eq_add_one, Finset.length_toList, List.get_eq_getElem] at this
      simp only [Nat.succ_eq_add_one, List.get_eq_getElem, Fin.coe_castLE]
      rw [this _ (Nat.lt_of_le_of_lt (Nat.le_of_lt_succ i₁.2) hs),
        this _ (Nat.lt_of_le_of_lt (Nat.le_of_lt_succ i₀.2) hs)]
  -- Since `s` is large enough, there are two elements of `A ∘ s`
  -- where the second components lie close together.
  /-
    case succ.intro.intro
    R : Type u_1
    inst✝ : EuclideanDomain R
    abv : AbsoluteValue R Int
    h : abv.IsAdmissible
    this : DecidableEq R
    n : Nat
    ih : ∀ {ε : Real}, LT.lt 0 ε → ∀ {b : R}, Ne b 0 → ∀ (A : Fin (HPow.hPow (h.ca …
    ε : Real
    hε : LT.lt 0 ε
    b : R
    hb : Ne b 0
    A : Fin (HPow.hPow (h.card ε) (HAdd.hAdd n 1)).succ → Fin (HAdd.hAdd n 1) → R
    M : Nat := h.card ε
    s : Fin (HPow.hPow M n).succ → Fin (HPow.hPow M n.succ).succ
    s_inj : Function.Injective s
    hs : ∀ (i₀ i₁ : Fin (HPow.hPow M n).succ), LT.lt (↑(abv (HSub.hSub (HMod.hMod  …
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (∀ (k : Fin (HAdd.hAdd n 1) …
  -/
  obtain ⟨k₀, k₁, hk, h⟩ := ih hε hb fun x ↦ Fin.tail (A (s x))
  /-
    case succ.intro.intro.intro.intro.intro
    R : Type u_1
    inst✝ : EuclideanDomain R
    abv : AbsoluteValue R Int
    h✝ : abv.IsAdmissible
    this : DecidableEq R
    n : Nat
    ih : ∀ {ε : Real}, LT.lt 0 ε → ∀ {b : R}, Ne b 0 → ∀ (A : Fin (HPow.hPow (h✝.c …
    ε : Real
    hε : LT.lt 0 ε
    b : R
    hb : Ne b 0
    A : Fin (HPow.hPow (h✝.card ε) (HAdd.hAdd n 1)).succ → Fin (HAdd.hAdd n 1) → R
    M : Nat := h✝.card ε
    s : Fin (HPow.hPow M n).succ → Fin (HPow.hPow M n.succ).succ
    s_inj : Function.Injective s
    hs : ∀ (i₀ i₁ : Fin (HPow.hPow M n).succ), LT.lt (↑(abv (HSub.hSub (HMod.hMod  …
    k₀ k₁ : Fin (HPow.hPow (h✝.card ε) n).succ
    hk : Ne k₀ k₁
    h : ∀ (k : Fin n), LT.lt (↑(abv (HSub.hSub (HMod.hMod (Fin.tail (A (s k₁)) k)  …
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (∀ (k : Fin (HAdd.hAdd n 1) …
  -/
  refine ⟨s k₀, s k₁, fun h ↦ hk (s_inj h), fun i ↦ Fin.cases ?_ (fun i ↦ ?_) i⟩
    /-
      case succ.intro.intro.intro.intro.intro.refine_1
      R : Type u_1
      inst✝ : EuclideanDomain R
      abv : AbsoluteValue R Int
      h✝ : abv.IsAdmissible
      this : DecidableEq R
      n : Nat
      ih : ∀ {ε : Real}, LT.lt 0 ε → ∀ {b : R}, Ne b 0 → ∀ (A : Fin (HPow.hPow (h✝.c …
      ε : Real
      hε : LT.lt 0 ε
      b : R
      hb : Ne b 0
      A : Fin (HPow.hPow (h✝.card ε) (HAdd.hAdd n 1)).succ → Fin (HAdd.hAdd n 1) → R
      M : Nat := h✝.card ε
      s : Fin (HPow.hPow M n).succ → Fin (HPow.hPow M n.succ).succ
      s_inj : Function.Injective s
      hs : ∀ (i₀ i₁ : Fin (HPow.hPow M n).succ), LT.lt (↑(abv (HSub.hSub (HMod.hMod  …
      k₀ k₁ : Fin (HPow.hPow (h✝.card ε) n).succ
      hk : Ne k₀ k₁
      h : ∀ (k : Fin n), LT.lt (↑(abv (HSub.hSub (HMod.hMod (Fin.tail (A (s k₁)) k)  …
      i : Fin (HAdd.hAdd n 1)
      ⊢ LT.lt (↑(abv (HSub.hSub (HMod.hMod (A (s k₁) 0) b) (HMod.hMod (A (s k₀) 0) b …
    -/
  · exact hs k₀ k₁
    /-
      🎉 no goals
    -/
    /-
      case succ.intro.intro.intro.intro.intro.refine_2
      R : Type u_1
      inst✝ : EuclideanDomain R
      abv : AbsoluteValue R Int
      h✝ : abv.IsAdmissible
      this : DecidableEq R
      n : Nat
      ih : ∀ {ε : Real}, LT.lt 0 ε → ∀ {b : R}, Ne b 0 → ∀ (A : Fin (HPow.hPow (h✝.c …
      ε : Real
      hε : LT.lt 0 ε
      b : R
      hb : Ne b 0
      A : Fin (HPow.hPow (h✝.card ε) (HAdd.hAdd n 1)).succ → Fin (HAdd.hAdd n 1) → R
      M : Nat := h✝.card ε
      s : Fin (HPow.hPow M n).succ → Fin (HPow.hPow M n.succ).succ
      s_inj : Function.Injective s
      hs : ∀ (i₀ i₁ : Fin (HPow.hPow M n).succ), LT.lt (↑(abv (HSub.hSub (HMod.hMod  …
      k₀ k₁ : Fin (HPow.hPow (h✝.card ε) n).succ
      hk : Ne k₀ k₁
      h : ∀ (k : Fin n), LT.lt (↑(abv (HSub.hSub (HMod.hMod (Fin.tail (A (s k₁)) k)  …
      i✝ : Fin (HAdd.hAdd n 1)
      i : Fin n
      ⊢ LT.lt (↑(abv (HSub.hSub (HMod.hMod (A (s k₁) i.succ) b) (HMod.hMod (A (s k₀) …
    -/
  · exact h i
    /-
      🎉 no goals
    -/


/-- Any large enough family of vectors in `R^ι` has a pair of elements
whose remainders are close together, pointwise. -/
theorem exists_approx {ι : Type*} [Fintype ι] {ε : ℝ} (hε : 0 < ε) {b : R} (hb : b ≠ 0)
    (h : abv.IsAdmissible) (A : Fin (h.card ε ^ Fintype.card ι).succ → ι → R) :
    ∃ i₀ i₁, i₀ ≠ i₁ ∧ ∀ k, (abv (A i₁ k % b - A i₀ k % b) : ℝ) < abv b • ε := by
  /-
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    abv : AbsoluteValue R Int
    ι : Type u_2
    inst✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    b : R
    hb : Ne b 0
    h : abv.IsAdmissible
    A : Fin (HPow.hPow (h.card ε) (Fintype.card ι)).succ → ι → R
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (∀ (k : ι), LT.lt (↑(abv (H …
  -/
  let e := Fintype.equivFin ι
  /-
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    abv : AbsoluteValue R Int
    ι : Type u_2
    inst✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    b : R
    hb : Ne b 0
    h : abv.IsAdmissible
    A : Fin (HPow.hPow (h.card ε) (Fintype.card ι)).succ → ι → R
    e : Equiv ι (Fin (Fintype.card ι)) := Fintype.equivFin ι
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (∀ (k : ι), LT.lt (↑(abv (H …
  -/
  obtain ⟨i₀, i₁, ne, h⟩ := h.exists_approx_aux (Fintype.card ι) hε hb fun x y ↦ A x (e.symm y)
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    abv : AbsoluteValue R Int
    ι : Type u_2
    inst✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    b : R
    hb : Ne b 0
    h✝ : abv.IsAdmissible
    A : Fin (HPow.hPow (h✝.card ε) (Fintype.card ι)).succ → ι → R
    e : Equiv ι (Fin (Fintype.card ι)) := Fintype.equivFin ι
    i₀ i₁ : Fin (HPow.hPow (h✝.card ε) (Fintype.card ι)).succ
    ne : Ne i₀ i₁
    h : ∀ (k : Fin (Fintype.card ι)), LT.lt (↑(abv (HSub.hSub (HMod.hMod (A i₁ (e. …
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (∀ (k : ι), LT.lt (↑(abv (H …
  -/
  refine ⟨i₀, i₁, ne, fun k ↦ ?_⟩
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    abv : AbsoluteValue R Int
    ι : Type u_2
    inst✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    b : R
    hb : Ne b 0
    h✝ : abv.IsAdmissible
    A : Fin (HPow.hPow (h✝.card ε) (Fintype.card ι)).succ → ι → R
    e : Equiv ι (Fin (Fintype.card ι)) := Fintype.equivFin ι
    i₀ i₁ : Fin (HPow.hPow (h✝.card ε) (Fintype.card ι)).succ
    ne : Ne i₀ i₁
    h : ∀ (k : Fin (Fintype.card ι)), LT.lt (↑(abv (HSub.hSub (HMod.hMod (A i₁ (e. …
    k : ι
    ⊢ LT.lt (↑(abv (HSub.hSub (HMod.hMod (A i₁ k) b) (HMod.hMod (A i₀ k) b)))) (HS …
  -/
                      /-
                        🎉 no goals
                      -/
  convert h (e k) <;> simp only [e.symm_apply_apply]
                      /-
                        🎉 no goals
                      -/


