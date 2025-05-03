@[continuity, fun_prop]
protected theorem continuous_eval₂ [Semiring S] (p : S[X]) (f : S →+* R) :
    Continuous fun x => p.eval₂ f x := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : Semiring R
    inst✝² : TopologicalSpace R
    inst✝¹ : TopologicalSemiring R
    inst✝ : Semiring S
    p : Polynomial S
    f : RingHom S R
    ⊢ Continuous fun x => Polynomial.eval₂ f x p
  -/
  simp only [eval₂_eq_sum, Finsupp.sum]
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : Semiring R
    inst✝² : TopologicalSpace R
    inst✝¹ : TopologicalSemiring R
    inst✝ : Semiring S
    p : Polynomial S
    f : RingHom S R
    ⊢ Continuous fun x => p.sum fun e a => HMul.hMul (f a) (HPow.hPow x e)
  -/
  exact continuous_finset_sum _ fun c _ => continuous_const.mul (continuous_pow _)
  /-
    🎉 no goals
  -/


@[continuity, fun_prop]
protected theorem continuous : Continuous fun x => p.eval x :=
  p.continuous_eval₂ _


@[fun_prop]
protected theorem continuousAt {a : R} : ContinuousAt (fun x => p.eval x) a :=
  p.continuous.continuousAt


@[fun_prop]
protected theorem continuousWithinAt {s a} : ContinuousWithinAt (fun x => p.eval x) s a :=
  p.continuous.continuousWithinAt


@[fun_prop]
protected theorem continuousOn {s} : ContinuousOn (fun x => p.eval x) s :=
  p.continuous.continuousOn


@[continuity, fun_prop]
protected theorem continuous_aeval : Continuous fun x : A => aeval x p :=
  p.continuous_eval₂ _


@[fun_prop]
protected theorem continuousAt_aeval {a : A} : ContinuousAt (fun x : A => aeval x p) a :=
  p.continuous_aeval.continuousAt


@[fun_prop]
protected theorem continuousWithinAt_aeval {s a} :
    ContinuousWithinAt (fun x : A => aeval x p) s a :=
  p.continuous_aeval.continuousWithinAt


@[fun_prop]
protected theorem continuousOn_aeval {s} : ContinuousOn (fun x : A => aeval x p) s :=
  p.continuous_aeval.continuousOn


theorem tendsto_abv_eval₂_atTop {R S k α : Type*} [Semiring R] [Ring S] [LinearOrderedField k]
    (f : R →+* S) (abv : S → k) [IsAbsoluteValue abv] (p : R[X]) (hd : 0 < degree p)
    (hf : f p.leadingCoeff ≠ 0) {l : Filter α} {z : α → S} (hz : Tendsto (abv ∘ z) l atTop) :
    Tendsto (fun x => abv (p.eval₂ f (z x))) l atTop := by
  /-
    R : Type u_1
    S : Type u_2
    k : Type u_3
    α : Type u_4
    inst✝³ : Semiring R
    inst✝² : Ring S
    inst✝¹ : LinearOrderedField k
    f : RingHom R S
    abv : S → k
    inst✝ : IsAbsoluteValue abv
    p : Polynomial R
    hd : LT.lt 0 p.degree
    hf : Ne (f p.leadingCoeff) 0
    l : Filter α
    z : α → S
    hz : Filter.Tendsto (Function.comp abv z) l Filter.atTop
    ⊢ Filter.Tendsto (fun x => abv (Polynomial.eval₂ f (z x) p)) l Filter.atTop
  -/
  revert hf; refine degree_pos_induction_on p hd ?_ ?_ ?_ <;> clear hd p
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      k : Type u_3
      α : Type u_4
      inst✝³ : Semiring R
      inst✝² : Ring S
      inst✝¹ : LinearOrderedField k
      f : RingHom R S
      abv : S → k
      inst✝ : IsAbsoluteValue abv
      l : Filter α
      z : α → S
      hz : Filter.Tendsto (Function.comp abv z) l Filter.atTop
      ⊢ ∀ {a : R}, Ne a 0 → Ne (f (HMul.hMul (Polynomial.C a) Polynomial.X).leadingC …
    -/
  · rintro _ - hc
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      k : Type u_3
      α : Type u_4
      inst✝³ : Semiring R
      inst✝² : Ring S
      inst✝¹ : LinearOrderedField k
      f : RingHom R S
      abv : S → k
      inst✝ : IsAbsoluteValue abv
      l : Filter α
      z : α → S
      hz : Filter.Tendsto (Function.comp abv z) l Filter.atTop
      a✝ : R
      hc : Ne (f (HMul.hMul (Polynomial.C a✝) Polynomial.X).leadingCoeff) 0
      ⊢ Filter.Tendsto (fun x => abv (Polynomial.eval₂ f (z x) (HMul.hMul (Polynomia …
    -/
    rw [leadingCoeff_mul_X, leadingCoeff_C] at hc
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      k : Type u_3
      α : Type u_4
      inst✝³ : Semiring R
      inst✝² : Ring S
      inst✝¹ : LinearOrderedField k
      f : RingHom R S
      abv : S → k
      inst✝ : IsAbsoluteValue abv
      l : Filter α
      z : α → S
      hz : Filter.Tendsto (Function.comp abv z) l Filter.atTop
      a✝ : R
      hc : Ne (f a✝) 0
      ⊢ Filter.Tendsto (fun x => abv (Polynomial.eval₂ f (z x) (HMul.hMul (Polynomia …
    -/
    simpa [abv_mul abv] using hz.const_mul_atTop ((abv_pos abv).2 hc)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      k : Type u_3
      α : Type u_4
      inst✝³ : Semiring R
      inst✝² : Ring S
      inst✝¹ : LinearOrderedField k
      f : RingHom R S
      abv : S → k
      inst✝ : IsAbsoluteValue abv
      l : Filter α
      z : α → S
      hz : Filter.Tendsto (Function.comp abv z) l Filter.atTop
      ⊢ ∀ {p : Polynomial R}, LT.lt 0 p.degree → (Ne (f p.leadingCoeff) 0 → Filter.T …
    -/
  · intro _ _ ihp hf
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      k : Type u_3
      α : Type u_4
      inst✝³ : Semiring R
      inst✝² : Ring S
      inst✝¹ : LinearOrderedField k
      f : RingHom R S
      abv : S → k
      inst✝ : IsAbsoluteValue abv
      l : Filter α
      z : α → S
      hz : Filter.Tendsto (Function.comp abv z) l Filter.atTop
      p✝ : Polynomial R
      a✝ : LT.lt 0 p✝.degree
      ihp : Ne (f p✝.leadingCoeff) 0 → Filter.Tendsto (fun x => abv (Polynomial.eval …
      hf : Ne (f (HMul.hMul p✝ Polynomial.X).leadingCoeff) 0
      ⊢ Filter.Tendsto (fun x => abv (Polynomial.eval₂ f (z x) (HMul.hMul p✝ Polynom …
    -/
    rw [leadingCoeff_mul_X] at hf
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      k : Type u_3
      α : Type u_4
      inst✝³ : Semiring R
      inst✝² : Ring S
      inst✝¹ : LinearOrderedField k
      f : RingHom R S
      abv : S → k
      inst✝ : IsAbsoluteValue abv
      l : Filter α
      z : α → S
      hz : Filter.Tendsto (Function.comp abv z) l Filter.atTop
      p✝ : Polynomial R
      a✝ : LT.lt 0 p✝.degree
      ihp : Ne (f p✝.leadingCoeff) 0 → Filter.Tendsto (fun x => abv (Polynomial.eval …
      hf : Ne (f p✝.leadingCoeff) 0
      ⊢ Filter.Tendsto (fun x => abv (Polynomial.eval₂ f (z x) (HMul.hMul p✝ Polynom …
    -/
    simpa [abv_mul abv] using (ihp hf).atTop_mul_atTop hz
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      S : Type u_2
      k : Type u_3
      α : Type u_4
      inst✝³ : Semiring R
      inst✝² : Ring S
      inst✝¹ : LinearOrderedField k
      f : RingHom R S
      abv : S → k
      inst✝ : IsAbsoluteValue abv
      l : Filter α
      z : α → S
      hz : Filter.Tendsto (Function.comp abv z) l Filter.atTop
      ⊢ ∀ {p : Polynomial R} {a : R}, LT.lt 0 p.degree → (Ne (f p.leadingCoeff) 0 →  …
    -/
  · intro _ a hd ihp hf
    /-
      case refine_3
      R : Type u_1
      S : Type u_2
      k : Type u_3
      α : Type u_4
      inst✝³ : Semiring R
      inst✝² : Ring S
      inst✝¹ : LinearOrderedField k
      f : RingHom R S
      abv : S → k
      inst✝ : IsAbsoluteValue abv
      l : Filter α
      z : α → S
      hz : Filter.Tendsto (Function.comp abv z) l Filter.atTop
      p✝ : Polynomial R
      a : R
      hd : LT.lt 0 p✝.degree
      ihp : Ne (f p✝.leadingCoeff) 0 → Filter.Tendsto (fun x => abv (Polynomial.eval …
      hf : Ne (f (HAdd.hAdd p✝ (Polynomial.C a)).leadingCoeff) 0
      ⊢ Filter.Tendsto (fun x => abv (Polynomial.eval₂ f (z x) (HAdd.hAdd p✝ (Polyno …
    -/
    rw [add_comm, leadingCoeff_add_of_degree_lt (degree_C_le.trans_lt hd)] at hf
    /-
      case refine_3
      R : Type u_1
      S : Type u_2
      k : Type u_3
      α : Type u_4
      inst✝³ : Semiring R
      inst✝² : Ring S
      inst✝¹ : LinearOrderedField k
      f : RingHom R S
      abv : S → k
      inst✝ : IsAbsoluteValue abv
      l : Filter α
      z : α → S
      hz : Filter.Tendsto (Function.comp abv z) l Filter.atTop
      p✝ : Polynomial R
      a : R
      hd : LT.lt 0 p✝.degree
      ihp : Ne (f p✝.leadingCoeff) 0 → Filter.Tendsto (fun x => abv (Polynomial.eval …
      hf : Ne (f p✝.leadingCoeff) 0
      ⊢ Filter.Tendsto (fun x => abv (Polynomial.eval₂ f (z x) (HAdd.hAdd p✝ (Polyno …
    -/
    refine tendsto_atTop_of_add_const_right (abv (-f a)) ?_
    /-
      case refine_3
      R : Type u_1
      S : Type u_2
      k : Type u_3
      α : Type u_4
      inst✝³ : Semiring R
      inst✝² : Ring S
      inst✝¹ : LinearOrderedField k
      f : RingHom R S
      abv : S → k
      inst✝ : IsAbsoluteValue abv
      l : Filter α
      z : α → S
      hz : Filter.Tendsto (Function.comp abv z) l Filter.atTop
      p✝ : Polynomial R
      a : R
      hd : LT.lt 0 p✝.degree
      ihp : Ne (f p✝.leadingCoeff) 0 → Filter.Tendsto (fun x => abv (Polynomial.eval …
      hf : Ne (f p✝.leadingCoeff) 0
      ⊢ Filter.Tendsto (fun x => HAdd.hAdd (abv (Polynomial.eval₂ f (z x) (HAdd.hAdd …
    -/
    refine tendsto_atTop_mono (fun _ => abv_add abv _ _) ?_
    /-
      case refine_3
      R : Type u_1
      S : Type u_2
      k : Type u_3
      α : Type u_4
      inst✝³ : Semiring R
      inst✝² : Ring S
      inst✝¹ : LinearOrderedField k
      f : RingHom R S
      abv : S → k
      inst✝ : IsAbsoluteValue abv
      l : Filter α
      z : α → S
      hz : Filter.Tendsto (Function.comp abv z) l Filter.atTop
      p✝ : Polynomial R
      a : R
      hd : LT.lt 0 p✝.degree
      ihp : Ne (f p✝.leadingCoeff) 0 → Filter.Tendsto (fun x => abv (Polynomial.eval …
      hf : Ne (f p✝.leadingCoeff) 0
      ⊢ Filter.Tendsto (fun x => abv (HAdd.hAdd (Polynomial.eval₂ f (z x) (HAdd.hAdd …
    -/
    simpa using ihp hf
    /-
      🎉 no goals
    -/


theorem tendsto_abv_atTop {R k α : Type*} [Ring R] [LinearOrderedField k] (abv : R → k)
    [IsAbsoluteValue abv] (p : R[X]) (h : 0 < degree p) {l : Filter α} {z : α → R}
    (hz : Tendsto (abv ∘ z) l atTop) : Tendsto (fun x => abv (p.eval (z x))) l atTop := by
  /-
    R : Type u_1
    k : Type u_2
    α : Type u_3
    inst✝² : Ring R
    inst✝¹ : LinearOrderedField k
    abv : R → k
    inst✝ : IsAbsoluteValue abv
    p : Polynomial R
    h : LT.lt 0 p.degree
    l : Filter α
    z : α → R
    hz : Filter.Tendsto (Function.comp abv z) l Filter.atTop
    ⊢ Filter.Tendsto (fun x => abv (Polynomial.eval (z x) p)) l Filter.atTop
  -/
  apply tendsto_abv_eval₂_atTop _ _ _ h _ hz
  /-
    R : Type u_1
    k : Type u_2
    α : Type u_3
    inst✝² : Ring R
    inst✝¹ : LinearOrderedField k
    abv : R → k
    inst✝ : IsAbsoluteValue abv
    p : Polynomial R
    h : LT.lt 0 p.degree
    l : Filter α
    z : α → R
    hz : Filter.Tendsto (Function.comp abv z) l Filter.atTop
    ⊢ Ne ((RingHom.id R) p.leadingCoeff) 0
  -/
  exact mt leadingCoeff_eq_zero.1 (ne_zero_of_degree_gt h)
  /-
    🎉 no goals
  -/


theorem tendsto_abv_aeval_atTop {R A k α : Type*} [CommSemiring R] [Ring A] [Algebra R A]
    [LinearOrderedField k] (abv : A → k) [IsAbsoluteValue abv] (p : R[X]) (hd : 0 < degree p)
    (h₀ : algebraMap R A p.leadingCoeff ≠ 0) {l : Filter α} {z : α → A}
    (hz : Tendsto (abv ∘ z) l atTop) : Tendsto (fun x => abv (aeval (z x) p)) l atTop :=
  tendsto_abv_eval₂_atTop _ abv p hd h₀ hz


theorem tendsto_norm_atTop (p : R[X]) (h : 0 < degree p) {l : Filter α} {z : α → R}
    (hz : Tendsto (fun x => ‖z x‖) l atTop) : Tendsto (fun x => ‖p.eval (z x)‖) l atTop :=
  p.tendsto_abv_atTop norm h hz


theorem exists_forall_norm_le [ProperSpace R] (p : R[X]) : ∃ x, ∀ y, ‖p.eval x‖ ≤ ‖p.eval y‖ :=
  if hp0 : 0 < degree p then
    p.continuous.norm.exists_forall_le <| p.tendsto_norm_atTop hp0 tendsto_norm_cocompact_atTop
  else
                   /-
                     R : Type u_2
                     inst✝² : NormedRing R
                     inst✝¹ : IsAbsoluteValue Norm.norm
                     inst✝ : ProperSpace R
                     p : Polynomial R
                     hp0 : Not (LT.lt 0 p.degree)
                     ⊢ ∀ (y : R), LE.le (Norm.norm (Polynomial.eval (p.coeff 0) p)) (Norm.norm (Pol …
                   -/
    ⟨p.coeff 0, by rw [eq_C_of_degree_le_zero (le_of_not_gt hp0)]; simp⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem eq_one_of_roots_le {p : F[X]} {f : F →+* K} {B : ℝ} (hB : B < 0) (h1 : p.Monic)
    (h2 : Splits f p) (h3 : ∀ z ∈ (map f p).roots, ‖z‖ ≤ B) : p = 1 :=
  h1.natDegree_eq_zero_iff_eq_one.mp (by
    /-
      F : Type u_3
      K : Type u_4
      inst✝¹ : CommRing F
      inst✝ : NormedField K
      p : Polynomial F
      f : RingHom F K
      B : Real
      hB : LT.lt B 0
      h1 : p.Monic
      h2 : Polynomial.Splits f p
      h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
      ⊢ Eq p.natDegree 0
    -/
    contrapose! hB
    /-
      F : Type u_3
      K : Type u_4
      inst✝¹ : CommRing F
      inst✝ : NormedField K
      p : Polynomial F
      f : RingHom F K
      B : Real
      h1 : p.Monic
      h2 : Polynomial.Splits f p
      h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
      hB : Ne p.natDegree 0
      ⊢ LE.le 0 B
    -/
    rw [← h1.natDegree_map f, natDegree_eq_card_roots' h2] at hB
    /-
      F : Type u_3
      K : Type u_4
      inst✝¹ : CommRing F
      inst✝ : NormedField K
      p : Polynomial F
      f : RingHom F K
      B : Real
      h1 : p.Monic
      h2 : Polynomial.Splits f p
      h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
      hB : Ne (Polynomial.map f p).roots.card 0
      ⊢ LE.le 0 B
    -/
    obtain ⟨z, hz⟩ := card_pos_iff_exists_mem.mp (zero_lt_iff.mpr hB)
    /-
      case intro
      F : Type u_3
      K : Type u_4
      inst✝¹ : CommRing F
      inst✝ : NormedField K
      p : Polynomial F
      f : RingHom F K
      B : Real
      h1 : p.Monic
      h2 : Polynomial.Splits f p
      h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
      hB : Ne (Polynomial.map f p).roots.card 0
      z : K
      hz : Membership.mem (Polynomial.map f p).roots z
      ⊢ LE.le 0 B
    -/
    exact le_trans (norm_nonneg _) (h3 z hz))
    /-
      🎉 no goals
    -/


theorem coeff_le_of_roots_le {p : F[X]} {f : F →+* K} {B : ℝ} (i : ℕ) (h1 : p.Monic)
    (h2 : Splits f p) (h3 : ∀ z ∈ (map f p).roots, ‖z‖ ≤ B) :
    ‖(map f p).coeff i‖ ≤ B ^ (p.natDegree - i) * p.natDegree.choose i := by
  /-
    F : Type u_3
    K : Type u_4
    inst✝¹ : CommRing F
    inst✝ : NormedField K
    p : Polynomial F
    f : RingHom F K
    B : Real
    i : Nat
    h1 : p.Monic
    h2 : Polynomial.Splits f p
    h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
    ⊢ LE.le (Norm.norm ((Polynomial.map f p).coeff i)) (HMul.hMul (HPow.hPow B (HS …
  -/
  obtain hB | hB := lt_or_le B 0
  · rw [eq_one_of_roots_le hB h1 h2 h3, Polynomial.map_one, natDegree_one, zero_tsub, pow_zero,
      one_mul, coeff_one]
    /-
      case inl
      F : Type u_3
      K : Type u_4
      inst✝¹ : CommRing F
      inst✝ : NormedField K
      p : Polynomial F
      f : RingHom F K
      B : Real
      i : Nat
      h1 : p.Monic
      h2 : Polynomial.Splits f p
      h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
      hB : LT.lt B 0
      ⊢ LE.le (Norm.norm (ite (Eq i 0) 1 0)) ↑(Nat.choose 0 i)
    -/
                         /-
                           🎉 no goals
                         -/
    split_ifs with h <;> simp [h]
                         /-
                           🎉 no goals
                         -/
  /-
    case inr
    F : Type u_3
    K : Type u_4
    inst✝¹ : CommRing F
    inst✝ : NormedField K
    p : Polynomial F
    f : RingHom F K
    B : Real
    i : Nat
    h1 : p.Monic
    h2 : Polynomial.Splits f p
    h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
    hB : LE.le 0 B
    ⊢ LE.le (Norm.norm ((Polynomial.map f p).coeff i)) (HMul.hMul (HPow.hPow B (HS …
  -/
  rw [← h1.natDegree_map f]
  /-
    case inr
    F : Type u_3
    K : Type u_4
    inst✝¹ : CommRing F
    inst✝ : NormedField K
    p : Polynomial F
    f : RingHom F K
    B : Real
    i : Nat
    h1 : p.Monic
    h2 : Polynomial.Splits f p
    h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
    hB : LE.le 0 B
    ⊢ LE.le (Norm.norm ((Polynomial.map f p).coeff i)) (HMul.hMul (HPow.hPow B (HS …
  -/
  obtain hi | hi := lt_or_le (map f p).natDegree i
    /-
      case inr.inl
      F : Type u_3
      K : Type u_4
      inst✝¹ : CommRing F
      inst✝ : NormedField K
      p : Polynomial F
      f : RingHom F K
      B : Real
      i : Nat
      h1 : p.Monic
      h2 : Polynomial.Splits f p
      h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
      hB : LE.le 0 B
      hi : LT.lt (Polynomial.map f p).natDegree i
      ⊢ LE.le (Norm.norm ((Polynomial.map f p).coeff i)) (HMul.hMul (HPow.hPow B (HS …
    -/
  · rw [coeff_eq_zero_of_natDegree_lt hi, norm_zero]
    /-
      case inr.inl
      F : Type u_3
      K : Type u_4
      inst✝¹ : CommRing F
      inst✝ : NormedField K
      p : Polynomial F
      f : RingHom F K
      B : Real
      i : Nat
      h1 : p.Monic
      h2 : Polynomial.Splits f p
      h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
      hB : LE.le 0 B
      hi : LT.lt (Polynomial.map f p).natDegree i
      ⊢ LE.le 0 (HMul.hMul (HPow.hPow B (HSub.hSub (Polynomial.map f p).natDegree i) …
    -/
    positivity
    /-
      🎉 no goals
    -/
  rw [coeff_eq_esymm_roots_of_splits ((splits_id_iff_splits f).2 h2) hi, (h1.map _).leadingCoeff,
    one_mul, norm_mul, norm_pow, norm_neg, norm_one, one_pow, one_mul]
  /-
    case inr.inr
    F : Type u_3
    K : Type u_4
    inst✝¹ : CommRing F
    inst✝ : NormedField K
    p : Polynomial F
    f : RingHom F K
    B : Real
    i : Nat
    h1 : p.Monic
    h2 : Polynomial.Splits f p
    h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
    hB : LE.le 0 B
    hi : LE.le i (Polynomial.map f p).natDegree
    ⊢ LE.le (Norm.norm ((Polynomial.map f p).roots.esymm (HSub.hSub (Polynomial.ma …
  -/
  apply ((norm_multiset_sum_le _).trans <| sum_le_card_nsmul _ _ fun r hr => _).trans
  · rw [Multiset.map_map, card_map, card_powersetCard, ← natDegree_eq_card_roots' h2,
      Nat.choose_symm hi, mul_comm, nsmul_eq_mul]
  /-
    F : Type u_3
    K : Type u_4
    inst✝¹ : CommRing F
    inst✝ : NormedField K
    p : Polynomial F
    f : RingHom F K
    B : Real
    i : Nat
    h1 : p.Monic
    h2 : Polynomial.Splits f p
    h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
    hB : LE.le 0 B
    hi : LE.le i (Polynomial.map f p).natDegree
    ⊢ ∀ (r : Real), Membership.mem (Multiset.map (fun x => Norm.norm x) (Multiset. …
  -/
  intro r hr
  /-
    F : Type u_3
    K : Type u_4
    inst✝¹ : CommRing F
    inst✝ : NormedField K
    p : Polynomial F
    f : RingHom F K
    B : Real
    i : Nat
    h1 : p.Monic
    h2 : Polynomial.Splits f p
    h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
    hB : LE.le 0 B
    hi : LE.le i (Polynomial.map f p).natDegree
    r : Real
    hr : Membership.mem (Multiset.map (fun x => Norm.norm x) (Multiset.map Multise …
    ⊢ LE.le r (HPow.hPow B (HSub.hSub (Polynomial.map f p).natDegree i))
  -/
  simp_rw [Multiset.mem_map] at hr
  /-
    F : Type u_3
    K : Type u_4
    inst✝¹ : CommRing F
    inst✝ : NormedField K
    p : Polynomial F
    f : RingHom F K
    B : Real
    i : Nat
    h1 : p.Monic
    h2 : Polynomial.Splits f p
    h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
    hB : LE.le 0 B
    hi : LE.le i (Polynomial.map f p).natDegree
    r : Real
    hr : Exists fun a => And (Exists fun a_1 => And (Membership.mem (Multiset.powe …
    ⊢ LE.le r (HPow.hPow B (HSub.hSub (Polynomial.map f p).natDegree i))
  -/
  obtain ⟨_, ⟨s, hs, rfl⟩, rfl⟩ := hr
  /-
    case intro.intro.intro.intro
    F : Type u_3
    K : Type u_4
    inst✝¹ : CommRing F
    inst✝ : NormedField K
    p : Polynomial F
    f : RingHom F K
    B : Real
    i : Nat
    h1 : p.Monic
    h2 : Polynomial.Splits f p
    h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
    hB : LE.le 0 B
    hi : LE.le i (Polynomial.map f p).natDegree
    s : Multiset K
    hs : Membership.mem (Multiset.powersetCard (HSub.hSub (Polynomial.map f p).nat …
    ⊢ LE.le (Norm.norm s.prod) (HPow.hPow B (HSub.hSub (Polynomial.map f p).natDeg …
  -/
  rw [mem_powersetCard] at hs
  /-
    case intro.intro.intro.intro
    F : Type u_3
    K : Type u_4
    inst✝¹ : CommRing F
    inst✝ : NormedField K
    p : Polynomial F
    f : RingHom F K
    B : Real
    i : Nat
    h1 : p.Monic
    h2 : Polynomial.Splits f p
    h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
    hB : LE.le 0 B
    hi : LE.le i (Polynomial.map f p).natDegree
    s : Multiset K
    hs : And (LE.le s (Polynomial.map f p).roots) (Eq s.card (HSub.hSub (Polynomia …
    ⊢ LE.le (Norm.norm s.prod) (HPow.hPow B (HSub.hSub (Polynomial.map f p).natDeg …
  -/
  lift B to ℝ≥0 using hB
  rw [← coe_nnnorm, ← NNReal.coe_pow, NNReal.coe_le_coe, ← nnnormHom_apply, ← MonoidHom.coe_coe,
    MonoidHom.map_multiset_prod]
  /-
    case intro.intro.intro.intro.intro
    F : Type u_3
    K : Type u_4
    inst✝¹ : CommRing F
    inst✝ : NormedField K
    p : Polynomial F
    f : RingHom F K
    i : Nat
    h1 : p.Monic
    h2 : Polynomial.Splits f p
    hi : LE.le i (Polynomial.map f p).natDegree
    s : Multiset K
    hs : And (LE.le s (Polynomial.map f p).roots) (Eq s.card (HSub.hSub (Polynomia …
    B : NNReal
    h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
    ⊢ LE.le (Multiset.map (⇑↑nnnormHom) s).prod (HPow.hPow B (HSub.hSub (Polynomia …
  -/
  refine (prod_le_pow_card _ B fun x hx => ?_).trans_eq (by rw [card_map, hs.2])
  /-
    case intro.intro.intro.intro.intro
    F : Type u_3
    K : Type u_4
    inst✝¹ : CommRing F
    inst✝ : NormedField K
    p : Polynomial F
    f : RingHom F K
    i : Nat
    h1 : p.Monic
    h2 : Polynomial.Splits f p
    hi : LE.le i (Polynomial.map f p).natDegree
    s : Multiset K
    hs : And (LE.le s (Polynomial.map f p).roots) (Eq s.card (HSub.hSub (Polynomia …
    B : NNReal
    h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
    x : NNReal
    hx : Membership.mem (Multiset.map (⇑↑nnnormHom) s) x
    ⊢ LE.le x B
  -/
  obtain ⟨z, hz, rfl⟩ := Multiset.mem_map.1 hx
  /-
    case intro.intro.intro.intro.intro.intro.intro
    F : Type u_3
    K : Type u_4
    inst✝¹ : CommRing F
    inst✝ : NormedField K
    p : Polynomial F
    f : RingHom F K
    i : Nat
    h1 : p.Monic
    h2 : Polynomial.Splits f p
    hi : LE.le i (Polynomial.map f p).natDegree
    s : Multiset K
    hs : And (LE.le s (Polynomial.map f p).roots) (Eq s.card (HSub.hSub (Polynomia …
    B : NNReal
    h3 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
    z : K
    hz : Membership.mem s z
    hx : Membership.mem (Multiset.map (⇑↑nnnormHom) s) (↑nnnormHom z)
    ⊢ LE.le (↑nnnormHom z) B
  -/
  exact h3 z (mem_of_le hs.1 hz)
  /-
    🎉 no goals
  -/


/-- The coefficients of the monic polynomials of bounded degree with bounded roots are
uniformly bounded. -/
theorem coeff_bdd_of_roots_le {B : ℝ} {d : ℕ} (f : F →+* K) {p : F[X]} (h1 : p.Monic)
    (h2 : Splits f p) (h3 : p.natDegree ≤ d) (h4 : ∀ z ∈ (map f p).roots, ‖z‖ ≤ B) (i : ℕ) :
    ‖(map f p).coeff i‖ ≤ max B 1 ^ d * d.choose (d / 2) := by
  /-
    F : Type u_3
    K : Type u_4
    inst✝¹ : CommRing F
    inst✝ : NormedField K
    B : Real
    d : Nat
    f : RingHom F K
    p : Polynomial F
    h1 : p.Monic
    h2 : Polynomial.Splits f p
    h3 : LE.le p.natDegree d
    h4 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
    i : Nat
    ⊢ LE.le (Norm.norm ((Polynomial.map f p).coeff i)) (HMul.hMul (HPow.hPow (Max. …
  -/
  obtain hB | hB := le_or_lt 0 B
    /-
      case inl
      F : Type u_3
      K : Type u_4
      inst✝¹ : CommRing F
      inst✝ : NormedField K
      B : Real
      d : Nat
      f : RingHom F K
      p : Polynomial F
      h1 : p.Monic
      h2 : Polynomial.Splits f p
      h3 : LE.le p.natDegree d
      h4 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
      i : Nat
      hB : LE.le 0 B
      ⊢ LE.le (Norm.norm ((Polynomial.map f p).coeff i)) (HMul.hMul (HPow.hPow (Max. …
    -/
  · apply (coeff_le_of_roots_le i h1 h2 h4).trans
    calc
      _ ≤ max B 1 ^ (p.natDegree - i) * p.natDegree.choose i := by gcongr; apply le_max_left
      _ ≤ max B 1 ^ d * p.natDegree.choose i := by
        gcongr
        · apply le_max_right
        · exact le_trans (Nat.sub_le _ _) h3
      _ ≤ max B 1 ^ d * d.choose (d / 2) := by
        gcongr; exact (i.choose_mono h3).trans (i.choose_le_middle d)
    /-
      case inr
      F : Type u_3
      K : Type u_4
      inst✝¹ : CommRing F
      inst✝ : NormedField K
      B : Real
      d : Nat
      f : RingHom F K
      p : Polynomial F
      h1 : p.Monic
      h2 : Polynomial.Splits f p
      h3 : LE.le p.natDegree d
      h4 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
      i : Nat
      hB : LT.lt B 0
      ⊢ LE.le (Norm.norm ((Polynomial.map f p).coeff i)) (HMul.hMul (HPow.hPow (Max. …
    -/
  · rw [eq_one_of_roots_le hB h1 h2 h4, Polynomial.map_one, coeff_one]
    /-
      case inr
      F : Type u_3
      K : Type u_4
      inst✝¹ : CommRing F
      inst✝ : NormedField K
      B : Real
      d : Nat
      f : RingHom F K
      p : Polynomial F
      h1 : p.Monic
      h2 : Polynomial.Splits f p
      h3 : LE.le p.natDegree d
      h4 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
      i : Nat
      hB : LT.lt B 0
      ⊢ LE.le (Norm.norm (ite (Eq i 0) 1 0)) (HMul.hMul (HPow.hPow (Max.max B 1) d)  …
    -/
    refine le_trans ?_ (one_le_mul_of_one_le_of_one_le (one_le_pow₀ (le_max_right B 1)) ?_)
      /-
        case inr.refine_1
        F : Type u_3
        K : Type u_4
        inst✝¹ : CommRing F
        inst✝ : NormedField K
        B : Real
        d : Nat
        f : RingHom F K
        p : Polynomial F
        h1 : p.Monic
        h2 : Polynomial.Splits f p
        h3 : LE.le p.natDegree d
        h4 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
        i : Nat
        hB : LT.lt B 0
        ⊢ LE.le (Norm.norm (ite (Eq i 0) 1 0)) 1
      -/
                    /-
                      🎉 no goals
                    -/
    · split_ifs <;> norm_num
                    /-
                      🎉 no goals
                    -/
      /-
        case inr.refine_2
        F : Type u_3
        K : Type u_4
        inst✝¹ : CommRing F
        inst✝ : NormedField K
        B : Real
        d : Nat
        f : RingHom F K
        p : Polynomial F
        h1 : p.Monic
        h2 : Polynomial.Splits f p
        h3 : LE.le p.natDegree d
        h4 : ∀ (z : K), Membership.mem (Polynomial.map f p).roots z → LE.le (Norm.norm …
        i : Nat
        hB : LT.lt B 0
        ⊢ LE.le 1 ↑(d.choose (HDiv.hDiv d 2))
      -/
    · exact mod_cast Nat.succ_le_iff.mpr (Nat.choose_pos (d.div_le_self 2))
      /-
        🎉 no goals
      -/


