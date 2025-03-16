/-- Given a set `s` in a ring `R` and for every `t : s` a set `p t` of fractions in
a localization of `R` at `t`, this is the function sending a pair `(t, y)`, with
`t : s` and `y : t a`, to `t` multiplied with a numerator of `y`. The range
of this function spans the unit ideal, if `s` and every `p t` do. -/
noncomputable def mulNumerator (s : Set R)
    {Rₜ : s → Type*} [∀ t, CommRing (Rₜ t)] [∀ t, Algebra R (Rₜ t)]
    [∀ t, IsLocalization.Away t.val (Rₜ t)]
    (p : (t : s) → Set (Rₜ t)) (x : (t : s) × p t) : R :=
  x.1 * (IsLocalization.Away.sec x.1.1 x.2.1).1


lemma span_range_mulNumerator_eq_top {s : Set R}
    (hsone : Ideal.span s = ⊤) {Rₜ : s → Type*} [∀ t, CommRing (Rₜ t)] [∀ t, Algebra R (Rₜ t)]
    [∀ t, IsLocalization.Away t.val (Rₜ t)]
    {p : (t : s) → Set (Rₜ t)} (htone : ∀ (r : s), Ideal.span (p r) = ⊤) :
    Ideal.span (Set.range (IsLocalization.Away.mulNumerator s p)) = ⊤ := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    s : Set R
    hsone : Eq (Ideal.span s) Top.top
    Rₜ : ↑s → Type u_2
    inst✝² : (t : ↑s) → CommRing (Rₜ t)
    inst✝¹ : (t : ↑s) → Algebra R (Rₜ t)
    inst✝ : ∀ (t : ↑s), IsLocalization.Away (↑t) (Rₜ t)
    p : (t : ↑s) → Set (Rₜ t)
    htone : ∀ (r : ↑s), Eq (Ideal.span (p r)) Top.top
    ⊢ Eq (Ideal.span (Set.range (IsLocalization.Away.mulNumerator s p))) Top.top
  -/
  rw [← Ideal.radical_eq_top, eq_top_iff, ← hsone, Ideal.span_le]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    s : Set R
    hsone : Eq (Ideal.span s) Top.top
    Rₜ : ↑s → Type u_2
    inst✝² : (t : ↑s) → CommRing (Rₜ t)
    inst✝¹ : (t : ↑s) → Algebra R (Rₜ t)
    inst✝ : ∀ (t : ↑s), IsLocalization.Away (↑t) (Rₜ t)
    p : (t : ↑s) → Set (Rₜ t)
    htone : ∀ (r : ↑s), Eq (Ideal.span (p r)) Top.top
    ⊢ HasSubset.Subset s ↑(Ideal.span (Set.range (IsLocalization.Away.mulNumerator …
  -/
  intro a ha
  haveI : IsLocalization (Submonoid.powers a) (Rₜ ⟨a, ha⟩) :=
    inferInstanceAs <| IsLocalization.Away (⟨a, ha⟩ : s).val (Rₜ ⟨a, ha⟩)
  have h₁ : Ideal.span (p ⟨a, ha⟩) ≤ Ideal.span
      (algebraMap R (Rₜ ⟨a, ha⟩) '' Set.range (IsLocalization.Away.mulNumerator s p)) := by
    rw [Ideal.span_le]
    intro x hx
    rw [SetLike.mem_coe, IsLocalization.mem_span_map (Submonoid.powers a)]
    refine ⟨a * (IsLocalization.Away.sec a x).1, Ideal.subset_span ⟨⟨⟨a, ha⟩, ⟨x, hx⟩⟩, rfl⟩, ?_⟩
    use ⟨a ^ ((IsLocalization.Away.sec a x).2 + 1), _, rfl⟩
    rw [IsLocalization.eq_mk'_iff_mul_eq, map_pow, map_mul, ← map_pow, pow_add, map_mul,
      ← mul_assoc, IsLocalization.Away.sec_spec a x, mul_comm, pow_one]
  have h₂ : IsLocalization.mk' (Rₜ ⟨a, ha⟩) 1 (1 : Submonoid.powers a) ∈ Ideal.span
      (algebraMap R (Rₜ ⟨a, ha⟩) ''
        (Set.range <| IsLocalization.Away.mulNumerator s p)) := by
    rw [IsLocalization.mk'_one]
    apply h₁
    simp [htone]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    s : Set R
    hsone : Eq (Ideal.span s) Top.top
    Rₜ : ↑s → Type u_2
    inst✝² : (t : ↑s) → CommRing (Rₜ t)
    inst✝¹ : (t : ↑s) → Algebra R (Rₜ t)
    inst✝ : ∀ (t : ↑s), IsLocalization.Away (↑t) (Rₜ t)
    p : (t : ↑s) → Set (Rₜ t)
    htone : ∀ (r : ↑s), Eq (Ideal.span (p r)) Top.top
    a : R
    ha : Membership.mem s a
    this : IsLocalization (Submonoid.powers a) (Rₜ ⟨a, ha⟩)
    h₁ : LE.le (Ideal.span (p ⟨a, ha⟩)) (Ideal.span (Set.image (⇑(algebraMap R (Rₜ …
    h₂ : Membership.mem (Ideal.span (Set.image (⇑(algebraMap R (Rₜ ⟨a, ha⟩))) (Set …
    ⊢ Membership.mem (↑(Ideal.span (Set.range (IsLocalization.Away.mulNumerator s  …
  -/
  rw [IsLocalization.mem_span_map (Submonoid.powers a)] at h₂
  /-
    R : Type u_1
    inst✝³ : CommRing R
    s : Set R
    hsone : Eq (Ideal.span s) Top.top
    Rₜ : ↑s → Type u_2
    inst✝² : (t : ↑s) → CommRing (Rₜ t)
    inst✝¹ : (t : ↑s) → Algebra R (Rₜ t)
    inst✝ : ∀ (t : ↑s), IsLocalization.Away (↑t) (Rₜ t)
    p : (t : ↑s) → Set (Rₜ t)
    htone : ∀ (r : ↑s), Eq (Ideal.span (p r)) Top.top
    a : R
    ha : Membership.mem s a
    this : IsLocalization (Submonoid.powers a) (Rₜ ⟨a, ha⟩)
    h₁ : LE.le (Ideal.span (p ⟨a, ha⟩)) (Ideal.span (Set.image (⇑(algebraMap R (Rₜ …
    h₂ : Exists fun y => And (Membership.mem (Ideal.span (Set.range (IsLocalizatio …
    ⊢ Membership.mem (↑(Ideal.span (Set.range (IsLocalization.Away.mulNumerator s  …
  -/
  obtain ⟨y, hy, ⟨-, m, rfl⟩, hyz⟩ := h₂
  /-
    case intro.intro.intro.mk.intro
    R : Type u_1
    inst✝³ : CommRing R
    s : Set R
    hsone : Eq (Ideal.span s) Top.top
    Rₜ : ↑s → Type u_2
    inst✝² : (t : ↑s) → CommRing (Rₜ t)
    inst✝¹ : (t : ↑s) → Algebra R (Rₜ t)
    inst✝ : ∀ (t : ↑s), IsLocalization.Away (↑t) (Rₜ t)
    p : (t : ↑s) → Set (Rₜ t)
    htone : ∀ (r : ↑s), Eq (Ideal.span (p r)) Top.top
    a : R
    ha : Membership.mem s a
    this : IsLocalization (Submonoid.powers a) (Rₜ ⟨a, ha⟩)
    h₁ : LE.le (Ideal.span (p ⟨a, ha⟩)) (Ideal.span (Set.image (⇑(algebraMap R (Rₜ …
    y : R
    hy : Membership.mem (Ideal.span (Set.range (IsLocalization.Away.mulNumerator s …
    m : Nat
    hyz : Eq (IsLocalization.mk' (Rₜ ⟨a, ha⟩) 1 1) (IsLocalization.mk' (Rₜ ⟨a, ha⟩ …
    ⊢ Membership.mem (↑(Ideal.span (Set.range (IsLocalization.Away.mulNumerator s  …
  -/
  rw [IsLocalization.eq] at hyz
  /-
    case intro.intro.intro.mk.intro
    R : Type u_1
    inst✝³ : CommRing R
    s : Set R
    hsone : Eq (Ideal.span s) Top.top
    Rₜ : ↑s → Type u_2
    inst✝² : (t : ↑s) → CommRing (Rₜ t)
    inst✝¹ : (t : ↑s) → Algebra R (Rₜ t)
    inst✝ : ∀ (t : ↑s), IsLocalization.Away (↑t) (Rₜ t)
    p : (t : ↑s) → Set (Rₜ t)
    htone : ∀ (r : ↑s), Eq (Ideal.span (p r)) Top.top
    a : R
    ha : Membership.mem s a
    this : IsLocalization (Submonoid.powers a) (Rₜ ⟨a, ha⟩)
    h₁ : LE.le (Ideal.span (p ⟨a, ha⟩)) (Ideal.span (Set.image (⇑(algebraMap R (Rₜ …
    y : R
    hy : Membership.mem (Ideal.span (Set.range (IsLocalization.Away.mulNumerator s …
    m : Nat
    hyz : Exists fun c => Eq (HMul.hMul (↑c) (HMul.hMul (↑⟨(fun x => HPow.hPow a x …
    ⊢ Membership.mem (↑(Ideal.span (Set.range (IsLocalization.Away.mulNumerator s  …
  -/
  obtain ⟨⟨-, n, rfl⟩, hc⟩ := hyz
  /-
    case intro.intro.intro.mk.intro.intro.mk.intro
    R : Type u_1
    inst✝³ : CommRing R
    s : Set R
    hsone : Eq (Ideal.span s) Top.top
    Rₜ : ↑s → Type u_2
    inst✝² : (t : ↑s) → CommRing (Rₜ t)
    inst✝¹ : (t : ↑s) → Algebra R (Rₜ t)
    inst✝ : ∀ (t : ↑s), IsLocalization.Away (↑t) (Rₜ t)
    p : (t : ↑s) → Set (Rₜ t)
    htone : ∀ (r : ↑s), Eq (Ideal.span (p r)) Top.top
    a : R
    ha : Membership.mem s a
    this : IsLocalization (Submonoid.powers a) (Rₜ ⟨a, ha⟩)
    h₁ : LE.le (Ideal.span (p ⟨a, ha⟩)) (Ideal.span (Set.image (⇑(algebraMap R (Rₜ …
    y : R
    hy : Membership.mem (Ideal.span (Set.range (IsLocalization.Away.mulNumerator s …
    m n : Nat
    hc : Eq (HMul.hMul (↑⟨(fun x => HPow.hPow a x) n, ⋯⟩) (HMul.hMul (↑⟨(fun x =>  …
    ⊢ Membership.mem (↑(Ideal.span (Set.range (IsLocalization.Away.mulNumerator s  …
  -/
  simp only [← mul_assoc, OneMemClass.coe_one, one_mul, mul_one] at hc
  /-
    case intro.intro.intro.mk.intro.intro.mk.intro
    R : Type u_1
    inst✝³ : CommRing R
    s : Set R
    hsone : Eq (Ideal.span s) Top.top
    Rₜ : ↑s → Type u_2
    inst✝² : (t : ↑s) → CommRing (Rₜ t)
    inst✝¹ : (t : ↑s) → Algebra R (Rₜ t)
    inst✝ : ∀ (t : ↑s), IsLocalization.Away (↑t) (Rₜ t)
    p : (t : ↑s) → Set (Rₜ t)
    htone : ∀ (r : ↑s), Eq (Ideal.span (p r)) Top.top
    a : R
    ha : Membership.mem s a
    this : IsLocalization (Submonoid.powers a) (Rₜ ⟨a, ha⟩)
    h₁ : LE.le (Ideal.span (p ⟨a, ha⟩)) (Ideal.span (Set.image (⇑(algebraMap R (Rₜ …
    y : R
    hy : Membership.mem (Ideal.span (Set.range (IsLocalization.Away.mulNumerator s …
    m n : Nat
    hc : Eq (HMul.hMul (HPow.hPow a n) (HPow.hPow a m)) (HMul.hMul (HPow.hPow a n) …
    ⊢ Membership.mem (↑(Ideal.span (Set.range (IsLocalization.Away.mulNumerator s  …
  -/
  use n + m
  /-
    case h
    R : Type u_1
    inst✝³ : CommRing R
    s : Set R
    hsone : Eq (Ideal.span s) Top.top
    Rₜ : ↑s → Type u_2
    inst✝² : (t : ↑s) → CommRing (Rₜ t)
    inst✝¹ : (t : ↑s) → Algebra R (Rₜ t)
    inst✝ : ∀ (t : ↑s), IsLocalization.Away (↑t) (Rₜ t)
    p : (t : ↑s) → Set (Rₜ t)
    htone : ∀ (r : ↑s), Eq (Ideal.span (p r)) Top.top
    a : R
    ha : Membership.mem s a
    this : IsLocalization (Submonoid.powers a) (Rₜ ⟨a, ha⟩)
    h₁ : LE.le (Ideal.span (p ⟨a, ha⟩)) (Ideal.span (Set.image (⇑(algebraMap R (Rₜ …
    y : R
    hy : Membership.mem (Ideal.span (Set.range (IsLocalization.Away.mulNumerator s …
    m n : Nat
    hc : Eq (HMul.hMul (HPow.hPow a n) (HPow.hPow a m)) (HMul.hMul (HPow.hPow a n) …
    ⊢ Membership.mem (Ideal.span (Set.range (IsLocalization.Away.mulNumerator s p) …
  -/
  simpa [pow_add, hc] using Ideal.mul_mem_left _ _ hy
  /-
    🎉 no goals
  -/


lemma quotient_of_isIdempotentElem {e : R} (he : IsIdempotentElem e) :
    IsLocalization.Away e (R ⧸ Ideal.span {1 - e}) :=
  away_of_isIdempotentElem he Ideal.mk_ker Quotient.mk_surjective


