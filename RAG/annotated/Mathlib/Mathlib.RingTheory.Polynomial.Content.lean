/-- A polynomial is primitive when the only constant polynomials dividing it are units -/
def IsPrimitive (p : R[X]) : Prop :=
  ∀ r : R, C r ∣ p → IsUnit r


theorem isPrimitive_iff_isUnit_of_C_dvd {p : R[X]} : p.IsPrimitive ↔ ∀ r : R, C r ∣ p → IsUnit r :=
  Iff.rfl


@[simp]
theorem isPrimitive_one : IsPrimitive (1 : R[X]) := fun _ h =>
  isUnit_C.mp (isUnit_of_dvd_one h)


theorem Monic.isPrimitive {p : R[X]} (hp : p.Monic) : p.IsPrimitive := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    p : Polynomial R
    hp : p.Monic
    ⊢ p.IsPrimitive
  -/
  rintro r ⟨q, h⟩
  /-
    case intro
    R : Type u_1
    inst✝ : CommSemiring R
    p : Polynomial R
    hp : p.Monic
    r : R
    q : Polynomial R
    h : Eq p (HMul.hMul (Polynomial.C r) q)
    ⊢ IsUnit r
  -/
  exact isUnit_of_mul_eq_one r (q.coeff p.natDegree) (by rwa [← coeff_C_mul, ← h])
  /-
    🎉 no goals
  -/


theorem IsPrimitive.ne_zero [Nontrivial R] {p : R[X]} (hp : p.IsPrimitive) : p ≠ 0 := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    p : Polynomial R
    hp : p.IsPrimitive
    ⊢ Ne p 0
  -/
  rintro rfl
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    hp : Polynomial.IsPrimitive 0
    ⊢ False
  -/
  exact (hp 0 (dvd_zero (C 0))).ne_zero rfl
  /-
    🎉 no goals
  -/


theorem isPrimitive_of_dvd {p q : R[X]} (hp : IsPrimitive p) (hq : q ∣ p) : IsPrimitive q :=
  fun a ha => isPrimitive_iff_isUnit_of_C_dvd.mp hp a (dvd_trans ha hq)


/-- `p.content` is the `gcd` of the coefficients of `p`. -/
def content (p : R[X]) : R :=
  p.support.gcd p.coeff


theorem content_dvd_coeff {p : R[X]} (n : ℕ) : p.content ∣ p.coeff n := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    n : Nat
    ⊢ Dvd.dvd p.content (p.coeff n)
  -/
  by_cases h : n ∈ p.support
    /-
      case pos
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      n : Nat
      h : Membership.mem p.support n
      ⊢ Dvd.dvd p.content (p.coeff n)
    -/
  · apply Finset.gcd_dvd h
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    n : Nat
    h : Not (Membership.mem p.support n)
    ⊢ Dvd.dvd p.content (p.coeff n)
  -/
  rw [mem_support_iff, Classical.not_not] at h
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    n : Nat
    h : Eq (p.coeff n) 0
    ⊢ Dvd.dvd p.content (p.coeff n)
  -/
  rw [h]
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    n : Nat
    h : Eq (p.coeff n) 0
    ⊢ Dvd.dvd p.content 0
  -/
  apply dvd_zero
  /-
    🎉 no goals
  -/


@[simp]
theorem content_C {r : R} : (C r).content = normalize r := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    r : R
    ⊢ Eq (Polynomial.C r).content (normalize r)
  -/
  rw [content]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    r : R
    ⊢ Eq ((Polynomial.C r).support.gcd (Polynomial.C r).coeff) (normalize r)
  -/
  by_cases h0 : r = 0
    /-
      case pos
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      r : R
      h0 : Eq r 0
      ⊢ Eq ((Polynomial.C r).support.gcd (Polynomial.C r).coeff) (normalize r)
    -/
  · simp [h0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    r : R
    h0 : Not (Eq r 0)
    ⊢ Eq ((Polynomial.C r).support.gcd (Polynomial.C r).coeff) (normalize r)
  -/
  have h : (C r).support = {0} := support_monomial _ h0
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    r : R
    h0 : Not (Eq r 0)
    h : Eq (Polynomial.C r).support (Singleton.singleton 0)
    ⊢ Eq ((Polynomial.C r).support.gcd (Polynomial.C r).coeff) (normalize r)
  -/
  simp [h]
  /-
    🎉 no goals
  -/


@[simp]
                                                    /-
                                                      R : Type u_1
                                                      inst✝² : CommRing R
                                                      inst✝¹ : IsDomain R
                                                      inst✝ : NormalizedGCDMonoid R
                                                      ⊢ Eq (Polynomial.content 0) 0
                                                    -/
theorem content_zero : content (0 : R[X]) = 0 := by rw [← C_0, content_C, normalize_zero]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
                                                   /-
                                                     R : Type u_1
                                                     inst✝² : CommRing R
                                                     inst✝¹ : IsDomain R
                                                     inst✝ : NormalizedGCDMonoid R
                                                     ⊢ Eq (Polynomial.content 1) 1
                                                   -/
theorem content_one : content (1 : R[X]) = 1 := by rw [← C_1, content_C, normalize_one]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem content_X_mul {p : R[X]} : content (X * p) = content p := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    ⊢ Eq (HMul.hMul Polynomial.X p).content p.content
  -/
  rw [content, content, Finset.gcd_def, Finset.gcd_def]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    ⊢ Eq (Multiset.map (HMul.hMul Polynomial.X p).coeff (HMul.hMul Polynomial.X p) …
  -/
  refine congr rfl ?_
  have h : (X * p).support = p.support.map ⟨Nat.succ, Nat.succ_injective⟩ := by
    ext a
    simp only [exists_prop, Finset.mem_map, Function.Embedding.coeFn_mk, Ne, mem_support_iff]
    cases' a with a
    · simp [coeff_X_mul_zero, Nat.succ_ne_zero]
    rw [mul_comm, coeff_mul_X]
    constructor
    · intro h
      use a
    · rintro ⟨b, ⟨h1, h2⟩⟩
      rw [← Nat.succ_injective h2]
      apply h1
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    h : Eq (HMul.hMul Polynomial.X p).support (Finset.map { toFun := Nat.succ, inj …
    ⊢ Eq (Multiset.map (HMul.hMul Polynomial.X p).coeff (HMul.hMul Polynomial.X p) …
  -/
  rw [h]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    h : Eq (HMul.hMul Polynomial.X p).support (Finset.map { toFun := Nat.succ, inj …
    ⊢ Eq (Multiset.map (HMul.hMul Polynomial.X p).coeff (Finset.map { toFun := Nat …
  -/
  simp only [Finset.map_val, Function.comp_apply, Function.Embedding.coeFn_mk, Multiset.map_map]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    h : Eq (HMul.hMul Polynomial.X p).support (Finset.map { toFun := Nat.succ, inj …
    ⊢ Eq (Multiset.map (fun x => (HMul.hMul Polynomial.X p).coeff x.succ) p.suppor …
  -/
  refine congr (congr rfl ?_) rfl
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    h : Eq (HMul.hMul Polynomial.X p).support (Finset.map { toFun := Nat.succ, inj …
    ⊢ Eq (fun x => (HMul.hMul Polynomial.X p).coeff x.succ) p.coeff
  -/
  ext a
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    h : Eq (HMul.hMul Polynomial.X p).support (Finset.map { toFun := Nat.succ, inj …
    a : Nat
    ⊢ Eq ((HMul.hMul Polynomial.X p).coeff a.succ) (p.coeff a)
  -/
  rw [mul_comm]
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    h : Eq (HMul.hMul Polynomial.X p).support (Finset.map { toFun := Nat.succ, inj …
    a : Nat
    ⊢ Eq ((HMul.hMul p Polynomial.X).coeff a.succ) (p.coeff a)
  -/
  simp [coeff_mul_X]
  /-
    🎉 no goals
  -/


@[simp]
theorem content_X_pow {k : ℕ} : content ((X : R[X]) ^ k) = 1 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    k : Nat
    ⊢ Eq (HPow.hPow Polynomial.X k).content 1
  -/
  induction' k with k hi
    /-
      case zero
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      ⊢ Eq (HPow.hPow Polynomial.X 0).content 1
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    k : Nat
    hi : Eq (HPow.hPow Polynomial.X k).content 1
    ⊢ Eq (HPow.hPow Polynomial.X (HAdd.hAdd k 1)).content 1
  -/
  rw [pow_succ', content_X_mul, hi]
  /-
    🎉 no goals
  -/


@[simp]
                                                 /-
                                                   R : Type u_1
                                                   inst✝² : CommRing R
                                                   inst✝¹ : IsDomain R
                                                   inst✝ : NormalizedGCDMonoid R
                                                   ⊢ Eq Polynomial.X.content 1
                                                 -/
theorem content_X : content (X : R[X]) = 1 := by rw [← mul_one X, content_X_mul, content_one]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem content_C_mul (r : R) (p : R[X]) : (C r * p).content = normalize r * p.content := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    r : R
    p : Polynomial R
    ⊢ Eq (HMul.hMul (Polynomial.C r) p).content (HMul.hMul (normalize r) p.content)
  -/
  by_cases h0 : r = 0; · simp [h0]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    r : R
    p : Polynomial R
    h0 : Not (Eq r 0)
    ⊢ Eq (HMul.hMul (Polynomial.C r) p).content (HMul.hMul (normalize r) p.content)
  -/
  rw [content]; rw [content]; rw [← Finset.gcd_mul_left]
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    r : R
    p : Polynomial R
    h0 : Not (Eq r 0)
    ⊢ Eq ((HMul.hMul (Polynomial.C r) p).support.gcd (HMul.hMul (Polynomial.C r) p …
  -/
                                             /-
                                               🎉 no goals
                                             -/
  refine congr (congr rfl ?_) ?_ <;> ext <;> simp [h0, mem_support_iff]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem content_monomial {r : R} {k : ℕ} : content (monomial k r) = normalize r := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    r : R
    k : Nat
    ⊢ Eq ((Polynomial.monomial k) r).content (normalize r)
  -/
  rw [← C_mul_X_pow_eq_monomial, content_C_mul, content_X_pow, mul_one]
  /-
    🎉 no goals
  -/


theorem content_eq_zero_iff {p : R[X]} : content p = 0 ↔ p = 0 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    ⊢ Iff (Eq p.content 0) (Eq p 0)
  -/
  rw [content, Finset.gcd_eq_zero_iff]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    ⊢ Iff (∀ (x : Nat), Membership.mem p.support x → Eq (p.coeff x) 0) (Eq p 0)
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      h : ∀ (x : Nat), Membership.mem p.support x → Eq (p.coeff x) 0
      ⊢ Eq p 0
    -/
  · ext n
    /-
      case mp.a
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      h : ∀ (x : Nat), Membership.mem p.support x → Eq (p.coeff x) 0
      n : Nat
      ⊢ Eq (p.coeff n) (Polynomial.coeff 0 n)
    -/
    by_cases h0 : n ∈ p.support
      /-
        case pos
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : NormalizedGCDMonoid R
        p : Polynomial R
        h : ∀ (x : Nat), Membership.mem p.support x → Eq (p.coeff x) 0
        n : Nat
        h0 : Membership.mem p.support n
        ⊢ Eq (p.coeff n) (Polynomial.coeff 0 n)
      -/
    · rw [h n h0, coeff_zero]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : NormalizedGCDMonoid R
        p : Polynomial R
        h : ∀ (x : Nat), Membership.mem p.support x → Eq (p.coeff x) 0
        n : Nat
        h0 : Not (Membership.mem p.support n)
        ⊢ Eq (p.coeff n) (Polynomial.coeff 0 n)
      -/
    · rw [mem_support_iff] at h0
      /-
        case neg
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : NormalizedGCDMonoid R
        p : Polynomial R
        h : ∀ (x : Nat), Membership.mem p.support x → Eq (p.coeff x) 0
        n : Nat
        h0 : Not (Ne (p.coeff n) 0)
        ⊢ Eq (p.coeff n) (Polynomial.coeff 0 n)
      -/
      push_neg at h0
      /-
        case neg
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : NormalizedGCDMonoid R
        p : Polynomial R
        h : ∀ (x : Nat), Membership.mem p.support x → Eq (p.coeff x) 0
        n : Nat
        h0 : Eq (p.coeff n) 0
        ⊢ Eq (p.coeff n) (Polynomial.coeff 0 n)
      -/
      simp [h0]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      h : Eq p 0
      ⊢ ∀ (x : Nat), Membership.mem p.support x → Eq (p.coeff x) 0
    -/
  · intro x
    /-
      case mpr
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      h : Eq p 0
      x : Nat
      ⊢ Membership.mem p.support x → Eq (p.coeff x) 0
    -/
    simp [h]
    /-
      🎉 no goals
    -/

-- Porting note: this reduced with simp so created `normUnit_content` and put simp on it

theorem normalize_content {p : R[X]} : normalize p.content = p.content :=
  Finset.normalize_gcd


@[simp]
theorem normUnit_content {p : R[X]} : normUnit (content p) = 1 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    ⊢ Eq (NormalizationMonoid.normUnit p.content) 1
  -/
  by_cases hp0 : p.content = 0
    /-
      case pos
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      hp0 : Eq p.content 0
      ⊢ Eq (NormalizationMonoid.normUnit p.content) 1
    -/
  · simp [hp0]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      hp0 : Not (Eq p.content 0)
      ⊢ Eq (NormalizationMonoid.normUnit p.content) 1
    -/
  · ext
    /-
      case neg.a
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      hp0 : Not (Eq p.content 0)
      ⊢ Eq ↑(NormalizationMonoid.normUnit p.content) ↑1
    -/
    apply mul_left_cancel₀ hp0
    /-
      case neg.a
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      hp0 : Not (Eq p.content 0)
      ⊢ Eq (HMul.hMul p.content ↑(NormalizationMonoid.normUnit p.content)) (HMul.hMu …
    -/
    erw [← normalize_apply, normalize_content, mul_one]
    /-
      🎉 no goals
    -/


theorem content_eq_gcd_range_of_lt (p : R[X]) (n : ℕ) (h : p.natDegree < n) :
    p.content = (Finset.range n).gcd p.coeff := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    n : Nat
    h : LT.lt p.natDegree n
    ⊢ Eq p.content ((Finset.range n).gcd p.coeff)
  -/
  apply dvd_antisymm_of_normalize_eq normalize_content Finset.normalize_gcd
    /-
      case hab
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      n : Nat
      h : LT.lt p.natDegree n
      ⊢ Dvd.dvd p.content ((Finset.range n).gcd p.coeff)
    -/
  · rw [Finset.dvd_gcd_iff]
    /-
      case hab
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      n : Nat
      h : LT.lt p.natDegree n
      ⊢ ∀ (b : Nat), Membership.mem (Finset.range n) b → Dvd.dvd p.content (p.coeff b)
    -/
    intro i _
    /-
      case hab
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      n : Nat
      h : LT.lt p.natDegree n
      i : Nat
      a✝ : Membership.mem (Finset.range n) i
      ⊢ Dvd.dvd p.content (p.coeff i)
    -/
    apply content_dvd_coeff _
    /-
      🎉 no goals
    -/
    /-
      case hba
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      n : Nat
      h : LT.lt p.natDegree n
      ⊢ Dvd.dvd ((Finset.range n).gcd p.coeff) p.content
    -/
  · apply Finset.gcd_mono
    /-
      case hba.h
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      n : Nat
      h : LT.lt p.natDegree n
      ⊢ HasSubset.Subset p.support (Finset.range n)
    -/
    intro i
    /-
      case hba.h
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      n : Nat
      h : LT.lt p.natDegree n
      i : Nat
      ⊢ Membership.mem p.support i → Membership.mem (Finset.range n) i
    -/
    simp only [Nat.lt_succ_iff, mem_support_iff, Ne, Finset.mem_range]
    /-
      case hba.h
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      n : Nat
      h : LT.lt p.natDegree n
      i : Nat
      ⊢ Not (Eq (p.coeff i) 0) → LT.lt i n
    -/
    contrapose!
    /-
      case hba.h
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      n : Nat
      h : LT.lt p.natDegree n
      i : Nat
      ⊢ LE.le n i → Eq (p.coeff i) 0
    -/
    intro h1
    /-
      case hba.h
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      n : Nat
      h : LT.lt p.natDegree n
      i : Nat
      h1 : LE.le n i
      ⊢ Eq (p.coeff i) 0
    -/
    apply coeff_eq_zero_of_natDegree_lt (lt_of_lt_of_le h h1)
    /-
      🎉 no goals
    -/


theorem content_eq_gcd_range_succ (p : R[X]) :
    p.content = (Finset.range p.natDegree.succ).gcd p.coeff :=
  content_eq_gcd_range_of_lt _ _ (Nat.lt_succ_self _)


theorem content_eq_gcd_leadingCoeff_content_eraseLead (p : R[X]) :
    p.content = GCDMonoid.gcd p.leadingCoeff (eraseLead p).content := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    ⊢ Eq p.content (GCDMonoid.gcd p.leadingCoeff p.eraseLead.content)
  -/
  by_cases h : p = 0
    /-
      case pos
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      h : Eq p 0
      ⊢ Eq p.content (GCDMonoid.gcd p.leadingCoeff p.eraseLead.content)
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    h : Not (Eq p 0)
    ⊢ Eq p.content (GCDMonoid.gcd p.leadingCoeff p.eraseLead.content)
  -/
  rw [← leadingCoeff_eq_zero, leadingCoeff, ← Ne, ← mem_support_iff] at h
  rw [content, ← Finset.insert_erase h, Finset.gcd_insert, leadingCoeff, content,
    eraseLead_support]
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    h : Membership.mem p.support p.natDegree
    ⊢ Eq (GCDMonoid.gcd (p.coeff p.natDegree) ((p.support.erase p.natDegree).gcd p …
  -/
  refine congr rfl (Finset.gcd_congr rfl fun i hi => ?_)
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    h : Membership.mem p.support p.natDegree
    i : Nat
    hi : Membership.mem (p.support.erase p.natDegree) i
    ⊢ Eq (p.coeff i) (p.eraseLead.coeff i)
  -/
  rw [Finset.mem_erase] at hi
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    h : Membership.mem p.support p.natDegree
    i : Nat
    hi : And (Ne i p.natDegree) (Membership.mem p.support i)
    ⊢ Eq (p.coeff i) (p.eraseLead.coeff i)
  -/
  rw [eraseLead_coeff, if_neg hi.1]
  /-
    🎉 no goals
  -/


theorem dvd_content_iff_C_dvd {p : R[X]} {r : R} : r ∣ p.content ↔ C r ∣ p := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    r : R
    ⊢ Iff (Dvd.dvd r p.content) (Dvd.dvd (Polynomial.C r) p)
  -/
  rw [C_dvd_iff_dvd_coeff]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    r : R
    ⊢ Iff (Dvd.dvd r p.content) (∀ (i : Nat), Dvd.dvd r (p.coeff i))
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      r : R
      ⊢ Dvd.dvd r p.content → ∀ (i : Nat), Dvd.dvd r (p.coeff i)
    -/
  · intro h i
    /-
      case mp
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      r : R
      h : Dvd.dvd r p.content
      i : Nat
      ⊢ Dvd.dvd r (p.coeff i)
    -/
    apply h.trans (content_dvd_coeff _)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      r : R
      ⊢ (∀ (i : Nat), Dvd.dvd r (p.coeff i)) → Dvd.dvd r p.content
    -/
  · intro h
    /-
      case mpr
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      r : R
      h : ∀ (i : Nat), Dvd.dvd r (p.coeff i)
      ⊢ Dvd.dvd r p.content
    -/
    rw [content, Finset.dvd_gcd_iff]
    /-
      case mpr
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      r : R
      h : ∀ (i : Nat), Dvd.dvd r (p.coeff i)
      ⊢ ∀ (b : Nat), Membership.mem p.support b → Dvd.dvd r (p.coeff b)
    -/
    intro i _
    /-
      case mpr
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      r : R
      h : ∀ (i : Nat), Dvd.dvd r (p.coeff i)
      i : Nat
      a✝ : Membership.mem p.support i
      ⊢ Dvd.dvd r (p.coeff i)
    -/
    apply h i
    /-
      🎉 no goals
    -/


theorem C_content_dvd (p : R[X]) : C p.content ∣ p :=
  dvd_content_iff_C_dvd.1 dvd_rfl


theorem isPrimitive_iff_content_eq_one {p : R[X]} : p.IsPrimitive ↔ p.content = 1 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    ⊢ Iff p.IsPrimitive (Eq p.content 1)
  -/
  rw [← normalize_content, normalize_eq_one, IsPrimitive]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    ⊢ Iff (∀ (r : R), Dvd.dvd (Polynomial.C r) p → IsUnit r) (IsUnit p.content)
  -/
  simp_rw [← dvd_content_iff_C_dvd]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    ⊢ Iff (∀ (r : R), Dvd.dvd r p.content → IsUnit r) (IsUnit p.content)
  -/
  exact ⟨fun h => h p.content (dvd_refl p.content), fun h r hdvd => isUnit_of_dvd_unit hdvd h⟩
  /-
    🎉 no goals
  -/


theorem IsPrimitive.content_eq_one {p : R[X]} (hp : p.IsPrimitive) : p.content = 1 :=
  isPrimitive_iff_content_eq_one.mp hp


/-- The primitive part of a polynomial `p` is the primitive polynomial gained by dividing `p` by
  `p.content`. If `p = 0`, then `p.primPart = 1`. -/
noncomputable def primPart (p : R[X]) : R[X] :=
  letI := Classical.decEq R
  if p = 0 then 1 else Classical.choose (C_content_dvd p)


theorem eq_C_content_mul_primPart (p : R[X]) : p = C p.content * p.primPart := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    ⊢ Eq p (HMul.hMul (Polynomial.C p.content) p.primPart)
  -/
  by_cases h : p = 0; · simp [h]
                        /-
                          🎉 no goals
                        -/
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    h : Not (Eq p 0)
    ⊢ Eq p (HMul.hMul (Polynomial.C p.content) p.primPart)
  -/
  rw [primPart, if_neg h, ← Classical.choose_spec (C_content_dvd p)]
  /-
    🎉 no goals
  -/


@[simp]
theorem primPart_zero : primPart (0 : R[X]) = 1 :=
  if_pos rfl


theorem isPrimitive_primPart (p : R[X]) : p.primPart.IsPrimitive := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    ⊢ p.primPart.IsPrimitive
  -/
  by_cases h : p = 0; · simp [h]
                        /-
                          🎉 no goals
                        -/
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    h : Not (Eq p 0)
    ⊢ p.primPart.IsPrimitive
  -/
  rw [← content_eq_zero_iff] at h
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    h : Not (Eq p.content 0)
    ⊢ p.primPart.IsPrimitive
  -/
  rw [isPrimitive_iff_content_eq_one]
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    h : Not (Eq p.content 0)
    ⊢ Eq p.primPart.content 1
  -/
  apply mul_left_cancel₀ h
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    h : Not (Eq p.content 0)
    ⊢ Eq (HMul.hMul p.content p.primPart.content) (HMul.hMul p.content 1)
  -/
  conv_rhs => rw [p.eq_C_content_mul_primPart, mul_one, content_C_mul, normalize_content]
  /-
    🎉 no goals
  -/


theorem content_primPart (p : R[X]) : p.primPart.content = 1 :=
  p.isPrimitive_primPart.content_eq_one


theorem primPart_ne_zero (p : R[X]) : p.primPart ≠ 0 :=
  p.isPrimitive_primPart.ne_zero


theorem natDegree_primPart (p : R[X]) : p.primPart.natDegree = p.natDegree := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    ⊢ Eq p.primPart.natDegree p.natDegree
  -/
  by_cases h : C p.content = 0
    /-
      case pos
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      h : Eq (Polynomial.C p.content) 0
      ⊢ Eq p.primPart.natDegree p.natDegree
    -/
  · rw [C_eq_zero, content_eq_zero_iff] at h
    /-
      case pos
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      h : Eq p 0
      ⊢ Eq p.primPart.natDegree p.natDegree
    -/
    simp [h]
    /-
      🎉 no goals
    -/
  conv_rhs =>
    rw [p.eq_C_content_mul_primPart, natDegree_mul h p.primPart_ne_zero, natDegree_C, zero_add]


@[simp]
theorem IsPrimitive.primPart_eq {p : R[X]} (hp : p.IsPrimitive) : p.primPart = p := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    hp : p.IsPrimitive
    ⊢ Eq p.primPart p
  -/
  rw [← one_mul p.primPart, ← C_1, ← hp.content_eq_one, ← p.eq_C_content_mul_primPart]
  /-
    🎉 no goals
  -/


theorem isUnit_primPart_C (r : R) : IsUnit (C r).primPart := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    r : R
    ⊢ IsUnit (Polynomial.C r).primPart
  -/
  by_cases h0 : r = 0
    /-
      case pos
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      r : R
      h0 : Eq r 0
      ⊢ IsUnit (Polynomial.C r).primPart
    -/
  · simp [h0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    r : R
    h0 : Not (Eq r 0)
    ⊢ IsUnit (Polynomial.C r).primPart
  -/
  unfold IsUnit
  refine
    ⟨⟨C ↑(normUnit r)⁻¹, C ↑(normUnit r), by rw [← RingHom.map_mul, Units.inv_mul, C_1], by
        rw [← RingHom.map_mul, Units.mul_inv, C_1]⟩,
      ?_⟩
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    r : R
    h0 : Not (Eq r 0)
    ⊢ Eq (↑{ val := Polynomial.C ↑(Inv.inv (NormalizationMonoid.normUnit r)), inv  …
  -/
  rw [← normalize_eq_zero, ← C_eq_zero] at h0
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    r : R
    h0 : Not (Eq (Polynomial.C (normalize r)) 0)
    ⊢ Eq (↑{ val := Polynomial.C ↑(Inv.inv (NormalizationMonoid.normUnit r)), inv  …
  -/
  apply mul_left_cancel₀ h0
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    r : R
    h0 : Not (Eq (Polynomial.C (normalize r)) 0)
    ⊢ Eq (HMul.hMul (Polynomial.C (normalize r)) ↑{ val := Polynomial.C ↑(Inv.inv  …
  -/
  conv_rhs => rw [← content_C, ← (C r).eq_C_content_mul_primPart]
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    r : R
    h0 : Not (Eq (Polynomial.C (normalize r)) 0)
    ⊢ Eq (HMul.hMul (Polynomial.C (normalize r)) ↑{ val := Polynomial.C ↑(Inv.inv  …
  -/
  simp only [Units.val_mk, normalize_apply, RingHom.map_mul]
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    r : R
    h0 : Not (Eq (Polynomial.C (normalize r)) 0)
    ⊢ Eq (HMul.hMul (HMul.hMul (Polynomial.C r) (Polynomial.C ↑(NormalizationMonoi …
  -/
  rw [mul_assoc, ← RingHom.map_mul, Units.mul_inv, C_1, mul_one]
  /-
    🎉 no goals
  -/


theorem primPart_dvd (p : R[X]) : p.primPart ∣ p :=
  Dvd.intro_left (C p.content) p.eq_C_content_mul_primPart.symm


theorem aeval_primPart_eq_zero {S : Type*} [Ring S] [IsDomain S] [Algebra R S]
    [NoZeroSMulDivisors R S] {p : R[X]} {s : S} (hpzero : p ≠ 0) (hp : aeval s p = 0) :
    aeval s p.primPart = 0 := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    inst✝⁴ : NormalizedGCDMonoid R
    S : Type u_2
    inst✝³ : Ring S
    inst✝² : IsDomain S
    inst✝¹ : Algebra R S
    inst✝ : NoZeroSMulDivisors R S
    p : Polynomial R
    s : S
    hpzero : Ne p 0
    hp : Eq ((Polynomial.aeval s) p) 0
    ⊢ Eq ((Polynomial.aeval s) p.primPart) 0
  -/
  rw [eq_C_content_mul_primPart p, map_mul, aeval_C] at hp
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    inst✝⁴ : NormalizedGCDMonoid R
    S : Type u_2
    inst✝³ : Ring S
    inst✝² : IsDomain S
    inst✝¹ : Algebra R S
    inst✝ : NoZeroSMulDivisors R S
    p : Polynomial R
    s : S
    hpzero : Ne p 0
    hp : Eq (HMul.hMul ((algebraMap R S) p.content) ((Polynomial.aeval s) p.primPa …
    ⊢ Eq ((Polynomial.aeval s) p.primPart) 0
  -/
  have hcont : p.content ≠ 0 := fun h => hpzero (content_eq_zero_iff.1 h)
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    inst✝⁴ : NormalizedGCDMonoid R
    S : Type u_2
    inst✝³ : Ring S
    inst✝² : IsDomain S
    inst✝¹ : Algebra R S
    inst✝ : NoZeroSMulDivisors R S
    p : Polynomial R
    s : S
    hpzero : Ne p 0
    hp : Eq (HMul.hMul ((algebraMap R S) p.content) ((Polynomial.aeval s) p.primPa …
    hcont : Ne p.content 0
    ⊢ Eq ((Polynomial.aeval s) p.primPart) 0
  -/
  replace hcont := Function.Injective.ne (NoZeroSMulDivisors.algebraMap_injective R S) hcont
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    inst✝⁴ : NormalizedGCDMonoid R
    S : Type u_2
    inst✝³ : Ring S
    inst✝² : IsDomain S
    inst✝¹ : Algebra R S
    inst✝ : NoZeroSMulDivisors R S
    p : Polynomial R
    s : S
    hpzero : Ne p 0
    hp : Eq (HMul.hMul ((algebraMap R S) p.content) ((Polynomial.aeval s) p.primPa …
    hcont : Ne ((algebraMap R S) p.content) ((algebraMap R S) 0)
    ⊢ Eq ((Polynomial.aeval s) p.primPart) 0
  -/
  rw [map_zero] at hcont
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    inst✝⁴ : NormalizedGCDMonoid R
    S : Type u_2
    inst✝³ : Ring S
    inst✝² : IsDomain S
    inst✝¹ : Algebra R S
    inst✝ : NoZeroSMulDivisors R S
    p : Polynomial R
    s : S
    hpzero : Ne p 0
    hp : Eq (HMul.hMul ((algebraMap R S) p.content) ((Polynomial.aeval s) p.primPa …
    hcont : Ne ((algebraMap R S) p.content) 0
    ⊢ Eq ((Polynomial.aeval s) p.primPart) 0
  -/
  exact eq_zero_of_ne_zero_of_mul_left_eq_zero hcont hp
  /-
    🎉 no goals
  -/


theorem eval₂_primPart_eq_zero {S : Type*} [CommRing S] [IsDomain S] {f : R →+* S}
    (hinj : Function.Injective f) {p : R[X]} {s : S} (hpzero : p ≠ 0) (hp : eval₂ f s p = 0) :
    eval₂ f s p.primPart = 0 := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : NormalizedGCDMonoid R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    f : RingHom R S
    hinj : Function.Injective ⇑f
    p : Polynomial R
    s : S
    hpzero : Ne p 0
    hp : Eq (Polynomial.eval₂ f s p) 0
    ⊢ Eq (Polynomial.eval₂ f s p.primPart) 0
  -/
  rw [eq_C_content_mul_primPart p, eval₂_mul, eval₂_C] at hp
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : NormalizedGCDMonoid R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    f : RingHom R S
    hinj : Function.Injective ⇑f
    p : Polynomial R
    s : S
    hpzero : Ne p 0
    hp : Eq (HMul.hMul (f p.content) (Polynomial.eval₂ f s p.primPart)) 0
    ⊢ Eq (Polynomial.eval₂ f s p.primPart) 0
  -/
  have hcont : p.content ≠ 0 := fun h => hpzero (content_eq_zero_iff.1 h)
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : NormalizedGCDMonoid R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    f : RingHom R S
    hinj : Function.Injective ⇑f
    p : Polynomial R
    s : S
    hpzero : Ne p 0
    hp : Eq (HMul.hMul (f p.content) (Polynomial.eval₂ f s p.primPart)) 0
    hcont : Ne p.content 0
    ⊢ Eq (Polynomial.eval₂ f s p.primPart) 0
  -/
  replace hcont := Function.Injective.ne hinj hcont
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : NormalizedGCDMonoid R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    f : RingHom R S
    hinj : Function.Injective ⇑f
    p : Polynomial R
    s : S
    hpzero : Ne p 0
    hp : Eq (HMul.hMul (f p.content) (Polynomial.eval₂ f s p.primPart)) 0
    hcont : Ne (f p.content) (f 0)
    ⊢ Eq (Polynomial.eval₂ f s p.primPart) 0
  -/
  rw [map_zero] at hcont
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : NormalizedGCDMonoid R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    f : RingHom R S
    hinj : Function.Injective ⇑f
    p : Polynomial R
    s : S
    hpzero : Ne p 0
    hp : Eq (HMul.hMul (f p.content) (Polynomial.eval₂ f s p.primPart)) 0
    hcont : Ne (f p.content) 0
    ⊢ Eq (Polynomial.eval₂ f s p.primPart) 0
  -/
  exact eq_zero_of_ne_zero_of_mul_left_eq_zero hcont hp
  /-
    🎉 no goals
  -/


theorem gcd_content_eq_of_dvd_sub {a : R} {p q : R[X]} (h : C a ∣ p - q) :
    GCDMonoid.gcd a p.content = GCDMonoid.gcd a q.content := by
  rw [content_eq_gcd_range_of_lt p (max p.natDegree q.natDegree).succ
      (lt_of_le_of_lt (le_max_left _ _) (Nat.lt_succ_self _))]
  rw [content_eq_gcd_range_of_lt q (max p.natDegree q.natDegree).succ
      (lt_of_le_of_lt (le_max_right _ _) (Nat.lt_succ_self _))]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    a : R
    p q : Polynomial R
    h : Dvd.dvd (Polynomial.C a) (HSub.hSub p q)
    ⊢ Eq (GCDMonoid.gcd a ((Finset.range (Max.max p.natDegree q.natDegree).succ).g …
  -/
  apply Finset.gcd_eq_of_dvd_sub
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    a : R
    p q : Polynomial R
    h : Dvd.dvd (Polynomial.C a) (HSub.hSub p q)
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range (Max.max p.natDegree q.natDegree). …
  -/
  intro x _
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    a : R
    p q : Polynomial R
    h : Dvd.dvd (Polynomial.C a) (HSub.hSub p q)
    x : Nat
    a✝ : Membership.mem (Finset.range (Max.max p.natDegree q.natDegree).succ) x
    ⊢ Dvd.dvd a (HSub.hSub (p.coeff x) (q.coeff x))
  -/
  cases' h with w hw
  /-
    case h.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    a : R
    p q : Polynomial R
    x : Nat
    a✝ : Membership.mem (Finset.range (Max.max p.natDegree q.natDegree).succ) x
    w : Polynomial R
    hw : Eq (HSub.hSub p q) (HMul.hMul (Polynomial.C a) w)
    ⊢ Dvd.dvd a (HSub.hSub (p.coeff x) (q.coeff x))
  -/
  use w.coeff x
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    a : R
    p q : Polynomial R
    x : Nat
    a✝ : Membership.mem (Finset.range (Max.max p.natDegree q.natDegree).succ) x
    w : Polynomial R
    hw : Eq (HSub.hSub p q) (HMul.hMul (Polynomial.C a) w)
    ⊢ Eq (HSub.hSub (p.coeff x) (q.coeff x)) (HMul.hMul a (w.coeff x))
  -/
  rw [← coeff_sub, hw, coeff_C_mul]
  /-
    🎉 no goals
  -/


theorem content_mul_aux {p q : R[X]} :
    GCDMonoid.gcd (p * q).eraseLead.content p.leadingCoeff =
      GCDMonoid.gcd (p.eraseLead * q).content p.leadingCoeff := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    ⊢ Eq (GCDMonoid.gcd (HMul.hMul p q).eraseLead.content p.leadingCoeff) (GCDMono …
  -/
  rw [gcd_comm (content _) _, gcd_comm (content _) _]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    ⊢ Eq (GCDMonoid.gcd p.leadingCoeff (HMul.hMul p q).eraseLead.content) (GCDMono …
  -/
  apply gcd_content_eq_of_dvd_sub
  rw [← self_sub_C_mul_X_pow, ← self_sub_C_mul_X_pow, sub_mul, sub_sub, add_comm, sub_add,
    sub_sub_cancel, leadingCoeff_mul, RingHom.map_mul, mul_assoc, mul_assoc]
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    ⊢ Dvd.dvd (Polynomial.C p.leadingCoeff) (HSub.hSub (HMul.hMul (Polynomial.C p. …
  -/
  apply dvd_sub (Dvd.intro _ rfl) (Dvd.intro _ rfl)
  /-
    🎉 no goals
  -/


@[simp]
theorem content_mul {p q : R[X]} : (p * q).content = p.content * q.content := by
  classical
    suffices h :
        ∀ (n : ℕ) (p q : R[X]), (p * q).degree < n → (p * q).content = p.content * q.content by
      apply h
      apply lt_of_le_of_lt degree_le_natDegree (WithBot.coe_lt_coe.2 (Nat.lt_succ_self _))
    intro n
    induction' n with n ih
    · intro p q hpq
      rw [Nat.cast_zero,
        Nat.WithBot.lt_zero_iff, degree_eq_bot, mul_eq_zero] at hpq
      rcases hpq with (rfl | rfl) <;> simp
    intro p q hpq
    by_cases p0 : p = 0
    · simp [p0]
    by_cases q0 : q = 0
    · simp [q0]
    rw [degree_eq_natDegree (mul_ne_zero p0 q0), Nat.cast_lt,
      Nat.lt_succ_iff_lt_or_eq, ← Nat.cast_lt (α := WithBot ℕ),
      ← degree_eq_natDegree (mul_ne_zero p0 q0), natDegree_mul p0 q0] at hpq
    rcases hpq with (hlt | heq)
    · apply ih _ _ hlt
    rw [← p.natDegree_primPart, ← q.natDegree_primPart, ← Nat.cast_inj (R := WithBot ℕ),
      Nat.cast_add, ← degree_eq_natDegree p.primPart_ne_zero,
      ← degree_eq_natDegree q.primPart_ne_zero] at heq
    rw [p.eq_C_content_mul_primPart, q.eq_C_content_mul_primPart]
    suffices h : (q.primPart * p.primPart).content = 1 by
      rw [mul_assoc, content_C_mul, content_C_mul, mul_comm p.primPart, mul_assoc, content_C_mul,
        content_C_mul, h, mul_one, content_primPart, content_primPart, mul_one, mul_one]
    rw [← normalize_content, normalize_eq_one, isUnit_iff_dvd_one,
      content_eq_gcd_leadingCoeff_content_eraseLead, leadingCoeff_mul, gcd_comm]
    apply (gcd_mul_dvd_mul_gcd _ _ _).trans
    rw [content_mul_aux, ih, content_primPart, mul_one, gcd_comm, ←
      content_eq_gcd_leadingCoeff_content_eraseLead, content_primPart, one_mul,
      mul_comm q.primPart, content_mul_aux, ih, content_primPart, mul_one, gcd_comm, ←
      content_eq_gcd_leadingCoeff_content_eraseLead, content_primPart]
    · rw [← heq, degree_mul, WithBot.add_lt_add_iff_right]
      · apply degree_erase_lt p.primPart_ne_zero
      · rw [Ne, degree_eq_bot]
        apply q.primPart_ne_zero
    · rw [mul_comm, ← heq, degree_mul, WithBot.add_lt_add_iff_left]
      · apply degree_erase_lt q.primPart_ne_zero
      · rw [Ne, degree_eq_bot]
        apply p.primPart_ne_zero


theorem IsPrimitive.mul {p q : R[X]} (hp : p.IsPrimitive) (hq : q.IsPrimitive) :
    (p * q).IsPrimitive := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    hp : p.IsPrimitive
    hq : q.IsPrimitive
    ⊢ (HMul.hMul p q).IsPrimitive
  -/
  rw [isPrimitive_iff_content_eq_one, content_mul, hp.content_eq_one, hq.content_eq_one, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem primPart_mul {p q : R[X]} (h0 : p * q ≠ 0) :
    (p * q).primPart = p.primPart * q.primPart := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    h0 : Ne (HMul.hMul p q) 0
    ⊢ Eq (HMul.hMul p q).primPart (HMul.hMul p.primPart q.primPart)
  -/
  rw [Ne, ← content_eq_zero_iff, ← C_eq_zero] at h0
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    h0 : Not (Eq (Polynomial.C (HMul.hMul p q).content) 0)
    ⊢ Eq (HMul.hMul p q).primPart (HMul.hMul p.primPart q.primPart)
  -/
  apply mul_left_cancel₀ h0
  conv_lhs =>
    rw [← (p * q).eq_C_content_mul_primPart, p.eq_C_content_mul_primPart,
      q.eq_C_content_mul_primPart]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    h0 : Not (Eq (Polynomial.C (HMul.hMul p q).content) 0)
    ⊢ Eq (HMul.hMul (HMul.hMul (Polynomial.C p.content) p.primPart) (HMul.hMul (Po …
  -/
  rw [content_mul, RingHom.map_mul]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    h0 : Not (Eq (Polynomial.C (HMul.hMul p q).content) 0)
    ⊢ Eq (HMul.hMul (HMul.hMul (Polynomial.C p.content) p.primPart) (HMul.hMul (Po …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem IsPrimitive.dvd_primPart_iff_dvd {p q : R[X]} (hp : p.IsPrimitive) (hq : q ≠ 0) :
    p ∣ q.primPart ↔ p ∣ q := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    hp : p.IsPrimitive
    hq : Ne q 0
    ⊢ Iff (Dvd.dvd p q.primPart) (Dvd.dvd p q)
  -/
  refine ⟨fun h => h.trans (Dvd.intro_left _ q.eq_C_content_mul_primPart.symm), fun h => ?_⟩
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    hp : p.IsPrimitive
    hq : Ne q 0
    h : Dvd.dvd p q
    ⊢ Dvd.dvd p q.primPart
  -/
  rcases h with ⟨r, rfl⟩
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    hp : p.IsPrimitive
    r : Polynomial R
    hq : Ne (HMul.hMul p r) 0
    ⊢ Dvd.dvd p (HMul.hMul p r).primPart
  -/
  apply Dvd.intro _
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    hp : p.IsPrimitive
    r : Polynomial R
    hq : Ne (HMul.hMul p r) 0
    ⊢ Eq (HMul.hMul p ?m.87356) (HMul.hMul p r).primPart
  -/
  rw [primPart_mul hq, hp.primPart_eq]
  /-
    🎉 no goals
  -/


theorem exists_primitive_lcm_of_isPrimitive {p q : R[X]} (hp : p.IsPrimitive) (hq : q.IsPrimitive) :
    ∃ r : R[X], r.IsPrimitive ∧ ∀ s : R[X], p ∣ s ∧ q ∣ s ↔ r ∣ s := by
  classical
    have h : ∃ (n : ℕ) (r : R[X]), r.natDegree = n ∧ r.IsPrimitive ∧ p ∣ r ∧ q ∣ r :=
      ⟨(p * q).natDegree, p * q, rfl, hp.mul hq, dvd_mul_right _ _, dvd_mul_left _ _⟩
    rcases Nat.find_spec h with ⟨r, rdeg, rprim, pr, qr⟩
    refine ⟨r, rprim, fun s => ⟨?_, fun rs => ⟨pr.trans rs, qr.trans rs⟩⟩⟩
    suffices hs : ∀ (n : ℕ) (s : R[X]), s.natDegree = n → p ∣ s ∧ q ∣ s → r ∣ s from
      hs s.natDegree s rfl
    clear s
    by_contra! con
    rcases Nat.find_spec con with ⟨s, sdeg, ⟨ps, qs⟩, rs⟩
    have s0 : s ≠ 0 := by
      contrapose! rs
      simp [rs]
    have hs :=
      Nat.find_min' h
        ⟨_, s.natDegree_primPart, s.isPrimitive_primPart, (hp.dvd_primPart_iff_dvd s0).2 ps,
          (hq.dvd_primPart_iff_dvd s0).2 qs⟩
    rw [← rdeg] at hs
    by_cases sC : s.natDegree ≤ 0
    · rw [eq_C_of_natDegree_le_zero (le_trans hs sC), isPrimitive_iff_content_eq_one, content_C,
        normalize_eq_one] at rprim
      rw [eq_C_of_natDegree_le_zero (le_trans hs sC), ← dvd_content_iff_C_dvd] at rs
      apply rs rprim.dvd
    have hcancel := natDegree_cancelLeads_lt_of_natDegree_le_natDegree hs (lt_of_not_ge sC)
    rw [sdeg] at hcancel
    apply Nat.find_min con hcancel
    refine
      ⟨_, rfl, ⟨dvd_cancelLeads_of_dvd_of_dvd pr ps, dvd_cancelLeads_of_dvd_of_dvd qr qs⟩,
        fun rcs => rs ?_⟩
    rw [← rprim.dvd_primPart_iff_dvd s0]
    rw [cancelLeads, tsub_eq_zero_iff_le.mpr hs, pow_zero, mul_one] at rcs
    have h :=
      dvd_add rcs (Dvd.intro_left (C (leadingCoeff s) * X ^ (natDegree s - natDegree r)) rfl)
    have hC0 := rprim.ne_zero
    rw [Ne, ← leadingCoeff_eq_zero, ← C_eq_zero] at hC0
    rw [sub_add_cancel, ← rprim.dvd_primPart_iff_dvd (mul_ne_zero hC0 s0)] at h
    rcases isUnit_primPart_C r.leadingCoeff with ⟨u, hu⟩
    apply h.trans (Associated.symm ⟨u, _⟩).dvd
    rw [primPart_mul (mul_ne_zero hC0 s0), hu, mul_comm]


theorem dvd_iff_content_dvd_content_and_primPart_dvd_primPart {p q : R[X]} (hq : q ≠ 0) :
    p ∣ q ↔ p.content ∣ q.content ∧ p.primPart ∣ q.primPart := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    hq : Ne q 0
    ⊢ Iff (Dvd.dvd p q) (And (Dvd.dvd p.content q.content) (Dvd.dvd p.primPart q.p …
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p q : Polynomial R
      hq : Ne q 0
      h : Dvd.dvd p q
      ⊢ And (Dvd.dvd p.content q.content) (Dvd.dvd p.primPart q.primPart)
    -/
  · rcases h with ⟨r, rfl⟩
    /-
      case mp.intro
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p r : Polynomial R
      hq : Ne (HMul.hMul p r) 0
      ⊢ And (Dvd.dvd p.content (HMul.hMul p r).content) (Dvd.dvd p.primPart (HMul.hM …
    -/
    rw [content_mul, p.isPrimitive_primPart.dvd_primPart_iff_dvd hq]
    /-
      case mp.intro
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p r : Polynomial R
      hq : Ne (HMul.hMul p r) 0
      ⊢ And (Dvd.dvd p.content (HMul.hMul p.content r.content)) (Dvd.dvd p.primPart  …
    -/
    exact ⟨Dvd.intro _ rfl, p.primPart_dvd.trans (Dvd.intro _ rfl)⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p q : Polynomial R
      hq : Ne q 0
      h : And (Dvd.dvd p.content q.content) (Dvd.dvd p.primPart q.primPart)
      ⊢ Dvd.dvd p q
    -/
  · rw [p.eq_C_content_mul_primPart, q.eq_C_content_mul_primPart]
    /-
      case mpr
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p q : Polynomial R
      hq : Ne q 0
      h : And (Dvd.dvd p.content q.content) (Dvd.dvd p.primPart q.primPart)
      ⊢ Dvd.dvd (HMul.hMul (Polynomial.C p.content) p.primPart) (HMul.hMul (Polynomi …
    -/
    exact mul_dvd_mul (RingHom.map_dvd C h.1) h.2
    /-
      🎉 no goals
    -/


noncomputable instance (priority := 100) normalizedGcdMonoid : NormalizedGCDMonoid R[X] :=
  letI := Classical.decEq R
  normalizedGCDMonoidOfExistsLCM fun p q => by
    rcases exists_primitive_lcm_of_isPrimitive p.isPrimitive_primPart
        q.isPrimitive_primPart with
      ⟨r, rprim, hr⟩
    /-
      case intro.intro
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      this : DecidableEq R := Classical.decEq R
      p q r : Polynomial R
      rprim : r.IsPrimitive
      hr : ∀ (s : Polynomial R), Iff (And (Dvd.dvd p.primPart s) (Dvd.dvd q.primPart …
      ⊢ Exists fun c => ∀ (d : Polynomial R), Iff (And (Dvd.dvd p d) (Dvd.dvd q d))  …
    -/
    refine ⟨C (lcm p.content q.content) * r, fun s => ?_⟩
    /-
      case intro.intro
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      this : DecidableEq R := Classical.decEq R
      p q r : Polynomial R
      rprim : r.IsPrimitive
      hr : ∀ (s : Polynomial R), Iff (And (Dvd.dvd p.primPart s) (Dvd.dvd q.primPart …
      s : Polynomial R
      ⊢ Iff (And (Dvd.dvd p s) (Dvd.dvd q s)) (Dvd.dvd (HMul.hMul (Polynomial.C (GCD …
    -/
    by_cases hs : s = 0
      /-
        case pos
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : NormalizedGCDMonoid R
        this : DecidableEq R := Classical.decEq R
        p q r : Polynomial R
        rprim : r.IsPrimitive
        hr : ∀ (s : Polynomial R), Iff (And (Dvd.dvd p.primPart s) (Dvd.dvd q.primPart …
        s : Polynomial R
        hs : Eq s 0
        ⊢ Iff (And (Dvd.dvd p s) (Dvd.dvd q s)) (Dvd.dvd (HMul.hMul (Polynomial.C (GCD …
      -/
    · simp [hs]
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      this : DecidableEq R := Classical.decEq R
      p q r : Polynomial R
      rprim : r.IsPrimitive
      hr : ∀ (s : Polynomial R), Iff (And (Dvd.dvd p.primPart s) (Dvd.dvd q.primPart …
      s : Polynomial R
      hs : Not (Eq s 0)
      ⊢ Iff (And (Dvd.dvd p s) (Dvd.dvd q s)) (Dvd.dvd (HMul.hMul (Polynomial.C (GCD …
    -/
    by_cases hpq : C (lcm p.content q.content) = 0
      /-
        case pos
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : NormalizedGCDMonoid R
        this : DecidableEq R := Classical.decEq R
        p q r : Polynomial R
        rprim : r.IsPrimitive
        hr : ∀ (s : Polynomial R), Iff (And (Dvd.dvd p.primPart s) (Dvd.dvd q.primPart …
        s : Polynomial R
        hs : Not (Eq s 0)
        hpq : Eq (Polynomial.C (GCDMonoid.lcm p.content q.content)) 0
        ⊢ Iff (And (Dvd.dvd p s) (Dvd.dvd q s)) (Dvd.dvd (HMul.hMul (Polynomial.C (GCD …
      -/
    · rw [C_eq_zero, lcm_eq_zero_iff, content_eq_zero_iff, content_eq_zero_iff] at hpq
      /-
        case pos
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : NormalizedGCDMonoid R
        this : DecidableEq R := Classical.decEq R
        p q r : Polynomial R
        rprim : r.IsPrimitive
        hr : ∀ (s : Polynomial R), Iff (And (Dvd.dvd p.primPart s) (Dvd.dvd q.primPart …
        s : Polynomial R
        hs : Not (Eq s 0)
        hpq : Or (Eq p 0) (Eq q 0)
        ⊢ Iff (And (Dvd.dvd p s) (Dvd.dvd q s)) (Dvd.dvd (HMul.hMul (Polynomial.C (GCD …
      -/
                                      /-
                                        🎉 no goals
                                      -/
      rcases hpq with (hpq | hpq) <;> simp [hpq, hs]
                                      /-
                                        🎉 no goals
                                      -/
    /-
      case neg
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      this : DecidableEq R := Classical.decEq R
      p q r : Polynomial R
      rprim : r.IsPrimitive
      hr : ∀ (s : Polynomial R), Iff (And (Dvd.dvd p.primPart s) (Dvd.dvd q.primPart …
      s : Polynomial R
      hs : Not (Eq s 0)
      hpq : Not (Eq (Polynomial.C (GCDMonoid.lcm p.content q.content)) 0)
      ⊢ Iff (And (Dvd.dvd p s) (Dvd.dvd q s)) (Dvd.dvd (HMul.hMul (Polynomial.C (GCD …
    -/
    iterate 3 rw [dvd_iff_content_dvd_content_and_primPart_dvd_primPart hs]
    rw [content_mul, rprim.content_eq_one, mul_one, content_C, normalize_lcm, lcm_dvd_iff,
      primPart_mul (mul_ne_zero hpq rprim.ne_zero), rprim.primPart_eq,
      (isUnit_primPart_C (lcm p.content q.content)).mul_left_dvd, ← hr s.primPart]
    /-
      case neg
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      this : DecidableEq R := Classical.decEq R
      p q r : Polynomial R
      rprim : r.IsPrimitive
      hr : ∀ (s : Polynomial R), Iff (And (Dvd.dvd p.primPart s) (Dvd.dvd q.primPart …
      s : Polynomial R
      hs : Not (Eq s 0)
      hpq : Not (Eq (Polynomial.C (GCDMonoid.lcm p.content q.content)) 0)
      ⊢ Iff (And (And (Dvd.dvd p.content s.content) (Dvd.dvd p.primPart s.primPart)) …
    -/
    tauto
    /-
      🎉 no goals
    -/


theorem degree_gcd_le_left {p : R[X]} (hp : p ≠ 0) (q) : (gcd p q).degree ≤ p.degree := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    hp : Ne p 0
    q : Polynomial R
    ⊢ LE.le (GCDMonoid.gcd p q).degree p.degree
  -/
  have := natDegree_le_iff_degree_le.mp (natDegree_le_of_dvd (gcd_dvd_left p q) hp)
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    hp : Ne p 0
    q : Polynomial R
    this : LE.le (GCDMonoid.gcd p q).degree ↑p.natDegree
    ⊢ LE.le (GCDMonoid.gcd p q).degree p.degree
  -/
  rwa [degree_eq_natDegree hp]
  /-
    🎉 no goals
  -/


theorem degree_gcd_le_right (p) {q : R[X]} (hq : q ≠ 0) : (gcd p q).degree ≤ q.degree := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    hq : Ne q 0
    ⊢ LE.le (GCDMonoid.gcd p q).degree q.degree
  -/
  rw [gcd_comm]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    hq : Ne q 0
    ⊢ LE.le (GCDMonoid.gcd q p).degree q.degree
  -/
  exact degree_gcd_le_left hq p
  /-
    🎉 no goals
  -/


