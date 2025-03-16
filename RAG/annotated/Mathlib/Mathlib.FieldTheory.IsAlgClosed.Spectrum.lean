local notation "σ" => spectrum R

local notation "↑ₐ" => algebraMap R A

-- Porting note: removed an unneeded assumption `p ≠ 0`

theorem exists_mem_of_not_isUnit_aeval_prod [IsDomain R] {p : R[X]} {a : A}
    (h : ¬IsUnit (aeval a (Multiset.map (fun x : R => X - C x) p.roots).prod)) :
    ∃ k : R, k ∈ σ a ∧ eval k p = 0 := by
  /-
    R : Type u
    A : Type v
    inst✝³ : CommRing R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : IsDomain R
    p : Polynomial R
    a : A
    h : Not (IsUnit ((Polynomial.aeval a) (Multiset.map (fun x => HSub.hSub Polyno …
    ⊢ Exists fun k => And (Membership.mem (spectrum R a) k) (Eq (Polynomial.eval k …
  -/
  rw [← Multiset.prod_toList, map_list_prod] at h
  /-
    R : Type u
    A : Type v
    inst✝³ : CommRing R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : IsDomain R
    p : Polynomial R
    a : A
    h : Not (IsUnit (List.map (⇑(Polynomial.aeval a)) (Multiset.map (fun x => HSub …
    ⊢ Exists fun k => And (Membership.mem (spectrum R a) k) (Eq (Polynomial.eval k …
  -/
  replace h := mt List.prod_isUnit h
  simp only [not_forall, exists_prop, aeval_C, Multiset.mem_toList, List.mem_map, aeval_X,
    exists_exists_and_eq_and, Multiset.mem_map, map_sub] at h
  /-
    R : Type u
    A : Type v
    inst✝³ : CommRing R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : IsDomain R
    p : Polynomial R
    a : A
    h : Exists fun a_1 => And (Membership.mem p.roots a_1) (Not (IsUnit (HSub.hSub …
    ⊢ Exists fun k => And (Membership.mem (spectrum R a) k) (Eq (Polynomial.eval k …
  -/
  rcases h with ⟨r, r_mem, r_nu⟩
  /-
    case intro.intro
    R : Type u
    A : Type v
    inst✝³ : CommRing R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : IsDomain R
    p : Polynomial R
    a : A
    r : R
    r_mem : Membership.mem p.roots r
    r_nu : Not (IsUnit (HSub.hSub a ((algebraMap R A) r)))
    ⊢ Exists fun k => And (Membership.mem (spectrum R a) k) (Eq (Polynomial.eval k …
  -/
  exact ⟨r, by rwa [mem_iff, ← IsUnit.sub_iff], (mem_roots'.1 r_mem).2⟩
  /-
    🎉 no goals
  -/


local notation "σ" => spectrum 𝕜

local notation "↑ₐ" => algebraMap 𝕜 A


/-- Half of the spectral mapping theorem for polynomials. We prove it separately
because it holds over any field, whereas `spectrum.map_polynomial_aeval_of_degree_pos` and
`spectrum.map_polynomial_aeval_of_nonempty` need the field to be algebraically closed. -/
theorem subset_polynomial_aeval (a : A) (p : 𝕜[X]) : (eval · p) '' σ a ⊆ σ (aeval a p) := by
  /-
    𝕜 : Type u
    A : Type v
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a : A
    p : Polynomial 𝕜
    ⊢ HasSubset.Subset (Set.image (fun x => Polynomial.eval x p) (spectrum 𝕜 a)) ( …
  -/
  rintro _ ⟨k, hk, rfl⟩
  /-
    case intro.intro
    𝕜 : Type u
    A : Type v
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a : A
    p : Polynomial 𝕜
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 a) k
    ⊢ Membership.mem (spectrum 𝕜 ((Polynomial.aeval a) p)) ((fun x => Polynomial.e …
  -/
  let q := C (eval k p) - p
  /-
    case intro.intro
    𝕜 : Type u
    A : Type v
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a : A
    p : Polynomial 𝕜
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 a) k
    q : Polynomial 𝕜 := HSub.hSub (Polynomial.C (Polynomial.eval k p)) p
    ⊢ Membership.mem (spectrum 𝕜 ((Polynomial.aeval a) p)) ((fun x => Polynomial.e …
  -/
  have hroot : IsRoot q k := by simp only [q, eval_C, eval_sub, sub_self, IsRoot.def]
  /-
    case intro.intro
    𝕜 : Type u
    A : Type v
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a : A
    p : Polynomial 𝕜
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 a) k
    q : Polynomial 𝕜 := HSub.hSub (Polynomial.C (Polynomial.eval k p)) p
    hroot : q.IsRoot k
    ⊢ Membership.mem (spectrum 𝕜 ((Polynomial.aeval a) p)) ((fun x => Polynomial.e …
  -/
  rw [← mul_div_eq_iff_isRoot, ← neg_mul_neg, neg_sub] at hroot
  have aeval_q_eq : ↑ₐ (eval k p) - aeval a p = aeval a q := by
    simp only [q, aeval_C, map_sub, sub_left_inj]
  /-
    case intro.intro
    𝕜 : Type u
    A : Type v
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a : A
    p : Polynomial 𝕜
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 a) k
    q : Polynomial 𝕜 := HSub.hSub (Polynomial.C (Polynomial.eval k p)) p
    hroot : Eq (HMul.hMul (HSub.hSub (Polynomial.C k) Polynomial.X) (Neg.neg (HDiv …
    aeval_q_eq : Eq (HSub.hSub ((algebraMap 𝕜 A) (Polynomial.eval k p)) ((Polynomi …
    ⊢ Membership.mem (spectrum 𝕜 ((Polynomial.aeval a) p)) ((fun x => Polynomial.e …
  -/
  rw [mem_iff, aeval_q_eq, ← hroot, aeval_mul]
  /-
    case intro.intro
    𝕜 : Type u
    A : Type v
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a : A
    p : Polynomial 𝕜
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 a) k
    q : Polynomial 𝕜 := HSub.hSub (Polynomial.C (Polynomial.eval k p)) p
    hroot : Eq (HMul.hMul (HSub.hSub (Polynomial.C k) Polynomial.X) (Neg.neg (HDiv …
    aeval_q_eq : Eq (HSub.hSub ((algebraMap 𝕜 A) (Polynomial.eval k p)) ((Polynomi …
    ⊢ Not (IsUnit (HMul.hMul ((Polynomial.aeval a) (HSub.hSub (Polynomial.C k) Pol …
  -/
  have hcomm := (Commute.all (C k - X) (-(q / (X - C k)))).map (aeval a : 𝕜[X] →ₐ[𝕜] A)
  /-
    case intro.intro
    𝕜 : Type u
    A : Type v
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a : A
    p : Polynomial 𝕜
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 a) k
    q : Polynomial 𝕜 := HSub.hSub (Polynomial.C (Polynomial.eval k p)) p
    hroot : Eq (HMul.hMul (HSub.hSub (Polynomial.C k) Polynomial.X) (Neg.neg (HDiv …
    aeval_q_eq : Eq (HSub.hSub ((algebraMap 𝕜 A) (Polynomial.eval k p)) ((Polynomi …
    hcomm : Commute ((Polynomial.aeval a) (HSub.hSub (Polynomial.C k) Polynomial.X …
    ⊢ Not (IsUnit (HMul.hMul ((Polynomial.aeval a) (HSub.hSub (Polynomial.C k) Pol …
  -/
  apply mt fun h => (hcomm.isUnit_mul_iff.mp h).1
  /-
    case intro.intro
    𝕜 : Type u
    A : Type v
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a : A
    p : Polynomial 𝕜
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 a) k
    q : Polynomial 𝕜 := HSub.hSub (Polynomial.C (Polynomial.eval k p)) p
    hroot : Eq (HMul.hMul (HSub.hSub (Polynomial.C k) Polynomial.X) (Neg.neg (HDiv …
    aeval_q_eq : Eq (HSub.hSub ((algebraMap 𝕜 A) (Polynomial.eval k p)) ((Polynomi …
    hcomm : Commute ((Polynomial.aeval a) (HSub.hSub (Polynomial.C k) Polynomial.X …
    ⊢ Not (IsUnit ((Polynomial.aeval a) (HSub.hSub (Polynomial.C k) Polynomial.X)))
  -/
  simpa only [aeval_X, aeval_C, map_sub] using hk
  /-
    🎉 no goals
  -/


/-- The *spectral mapping theorem* for polynomials.  Note: the assumption `degree p > 0`
is necessary in case `σ a = ∅`, for then the left-hand side is `∅` and the right-hand side,
assuming `[Nontrivial A]`, is `{k}` where `p = Polynomial.C k`. -/
theorem map_polynomial_aeval_of_degree_pos [IsAlgClosed 𝕜] (a : A) (p : 𝕜[X])
    (hdeg : 0 < degree p) : σ (aeval a p) = (eval · p) '' σ a := by
  -- handle the easy direction via `spectrum.subset_polynomial_aeval`
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : IsAlgClosed 𝕜
    a : A
    p : Polynomial 𝕜
    hdeg : LT.lt 0 p.degree
    ⊢ Eq (spectrum 𝕜 ((Polynomial.aeval a) p)) (Set.image (fun x => Polynomial.eva …
  -/
  refine Set.eq_of_subset_of_subset (fun k hk => ?_) (subset_polynomial_aeval a p)
  -- write `C k - p` product of linear factors and a constant; show `C k - p ≠ 0`.
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : IsAlgClosed 𝕜
    a : A
    p : Polynomial 𝕜
    hdeg : LT.lt 0 p.degree
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 ((Polynomial.aeval a) p)) k
    ⊢ Membership.mem (Set.image (fun x => Polynomial.eval x p) (spectrum 𝕜 a)) k
  -/
  have hprod := eq_prod_roots_of_splits_id (IsAlgClosed.splits (C k - p))
  have h_ne : C k - p ≠ 0 := ne_zero_of_degree_gt <| by
    rwa [degree_sub_eq_right_of_degree_lt (lt_of_le_of_lt degree_C_le hdeg)]
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : IsAlgClosed 𝕜
    a : A
    p : Polynomial 𝕜
    hdeg : LT.lt 0 p.degree
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 ((Polynomial.aeval a) p)) k
    hprod : Eq (HSub.hSub (Polynomial.C k) p) (HMul.hMul (Polynomial.C (HSub.hSub  …
    h_ne : Ne (HSub.hSub (Polynomial.C k) p) 0
    ⊢ Membership.mem (Set.image (fun x => Polynomial.eval x p) (spectrum 𝕜 a)) k
  -/
  have lead_ne := leadingCoeff_ne_zero.mpr h_ne
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : IsAlgClosed 𝕜
    a : A
    p : Polynomial 𝕜
    hdeg : LT.lt 0 p.degree
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 ((Polynomial.aeval a) p)) k
    hprod : Eq (HSub.hSub (Polynomial.C k) p) (HMul.hMul (Polynomial.C (HSub.hSub  …
    h_ne : Ne (HSub.hSub (Polynomial.C k) p) 0
    lead_ne : Ne (HSub.hSub (Polynomial.C k) p).leadingCoeff 0
    ⊢ Membership.mem (Set.image (fun x => Polynomial.eval x p) (spectrum 𝕜 a)) k
  -/
  have lead_unit := (Units.map ↑ₐ.toMonoidHom (Units.mk0 _ lead_ne)).isUnit
  /- leading coefficient is a unit so product of linear factors is not a unit;
    apply `exists_mem_of_not_is_unit_aeval_prod`. -/
  have p_a_eq : aeval a (C k - p) = ↑ₐ k - aeval a p := by
    simp only [aeval_C, map_sub, sub_left_inj]
  rw [mem_iff, ← p_a_eq, hprod, aeval_mul,
    ((Commute.all _ _).map (aeval a : 𝕜[X] →ₐ[𝕜] A)).isUnit_mul_iff, aeval_C] at hk
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : IsAlgClosed 𝕜
    a : A
    p : Polynomial 𝕜
    hdeg : LT.lt 0 p.degree
    k : 𝕜
    hk : Not (And (IsUnit ((algebraMap 𝕜 A) (HSub.hSub (Polynomial.C k) p).leading …
    hprod : Eq (HSub.hSub (Polynomial.C k) p) (HMul.hMul (Polynomial.C (HSub.hSub  …
    h_ne : Ne (HSub.hSub (Polynomial.C k) p) 0
    lead_ne : Ne (HSub.hSub (Polynomial.C k) p).leadingCoeff 0
    lead_unit : IsUnit ↑((Units.map ↑(algebraMap 𝕜 A)) (Units.mk0 (HSub.hSub (Poly …
    p_a_eq : Eq ((Polynomial.aeval a) (HSub.hSub (Polynomial.C k) p)) (HSub.hSub ( …
    ⊢ Membership.mem (Set.image (fun x => Polynomial.eval x p) (spectrum 𝕜 a)) k
  -/
  replace hk := exists_mem_of_not_isUnit_aeval_prod (not_and.mp hk lead_unit)
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : IsAlgClosed 𝕜
    a : A
    p : Polynomial 𝕜
    hdeg : LT.lt 0 p.degree
    k : 𝕜
    hprod : Eq (HSub.hSub (Polynomial.C k) p) (HMul.hMul (Polynomial.C (HSub.hSub  …
    h_ne : Ne (HSub.hSub (Polynomial.C k) p) 0
    lead_ne : Ne (HSub.hSub (Polynomial.C k) p).leadingCoeff 0
    lead_unit : IsUnit ↑((Units.map ↑(algebraMap 𝕜 A)) (Units.mk0 (HSub.hSub (Poly …
    p_a_eq : Eq ((Polynomial.aeval a) (HSub.hSub (Polynomial.C k) p)) (HSub.hSub ( …
    hk : Exists fun k_1 => And (Membership.mem (spectrum 𝕜 a) k_1) (Eq (Polynomial …
    ⊢ Membership.mem (Set.image (fun x => Polynomial.eval x p) (spectrum 𝕜 a)) k
  -/
  rcases hk with ⟨r, r_mem, r_ev⟩
  /-
    case intro.intro
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : IsAlgClosed 𝕜
    a : A
    p : Polynomial 𝕜
    hdeg : LT.lt 0 p.degree
    k : 𝕜
    hprod : Eq (HSub.hSub (Polynomial.C k) p) (HMul.hMul (Polynomial.C (HSub.hSub  …
    h_ne : Ne (HSub.hSub (Polynomial.C k) p) 0
    lead_ne : Ne (HSub.hSub (Polynomial.C k) p).leadingCoeff 0
    lead_unit : IsUnit ↑((Units.map ↑(algebraMap 𝕜 A)) (Units.mk0 (HSub.hSub (Poly …
    p_a_eq : Eq ((Polynomial.aeval a) (HSub.hSub (Polynomial.C k) p)) (HSub.hSub ( …
    r : 𝕜
    r_mem : Membership.mem (spectrum 𝕜 a) r
    r_ev : Eq (Polynomial.eval r (HSub.hSub (Polynomial.C k) p)) 0
    ⊢ Membership.mem (Set.image (fun x => Polynomial.eval x p) (spectrum 𝕜 a)) k
  -/
  exact ⟨r, r_mem, symm (by simpa [eval_sub, eval_C, sub_eq_zero] using r_ev)⟩
  /-
    🎉 no goals
  -/


/-- In this version of the spectral mapping theorem, we assume the spectrum
is nonempty instead of assuming the degree of the polynomial is positive. -/
theorem map_polynomial_aeval_of_nonempty [IsAlgClosed 𝕜] (a : A) (p : 𝕜[X])
    (hnon : (σ a).Nonempty) : σ (aeval a p) = (fun k => eval k p) '' σ a := by
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : IsAlgClosed 𝕜
    a : A
    p : Polynomial 𝕜
    hnon : (spectrum 𝕜 a).Nonempty
    ⊢ Eq (spectrum 𝕜 ((Polynomial.aeval a) p)) (Set.image (fun k => Polynomial.eva …
  -/
  nontriviality A
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : IsAlgClosed 𝕜
    a : A
    p : Polynomial 𝕜
    hnon : (spectrum 𝕜 a).Nonempty
    a✝ : Nontrivial A
    ⊢ Eq (spectrum 𝕜 ((Polynomial.aeval a) p)) (Set.image (fun k => Polynomial.eva …
  -/
  refine Or.elim (le_or_gt (degree p) 0) (fun h => ?_) (map_polynomial_aeval_of_degree_pos a p)
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : IsAlgClosed 𝕜
    a : A
    p : Polynomial 𝕜
    hnon : (spectrum 𝕜 a).Nonempty
    a✝ : Nontrivial A
    h : LE.le p.degree 0
    ⊢ Eq (spectrum 𝕜 ((Polynomial.aeval a) p)) (Set.image (fun k => Polynomial.eva …
  -/
  rw [eq_C_of_degree_le_zero h]
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : IsAlgClosed 𝕜
    a : A
    p : Polynomial 𝕜
    hnon : (spectrum 𝕜 a).Nonempty
    a✝ : Nontrivial A
    h : LE.le p.degree 0
    ⊢ Eq (spectrum 𝕜 ((Polynomial.aeval a) (Polynomial.C (p.coeff 0)))) (Set.image …
  -/
  simp only [Set.image_congr, eval_C, aeval_C, scalar_eq, Set.Nonempty.image_const hnon]
  /-
    🎉 no goals
  -/


/-- A specialization of `spectrum.subset_polynomial_aeval` to monic monomials for convenience. -/
theorem pow_image_subset (a : A) (n : ℕ) : (fun x => x ^ n) '' σ a ⊆ σ (a ^ n) := by
  /-
    𝕜 : Type u
    A : Type v
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a : A
    n : Nat
    ⊢ HasSubset.Subset (Set.image (fun x => HPow.hPow x n) (spectrum 𝕜 a)) (spectr …
  -/
  simpa only [eval_pow, eval_X, aeval_X_pow] using subset_polynomial_aeval a (X ^ n : 𝕜[X])
  /-
    🎉 no goals
  -/


/-- A specialization of `spectrum.map_polynomial_aeval_of_nonempty` to monic monomials for
convenience. -/
theorem map_pow_of_pos [IsAlgClosed 𝕜] (a : A) {n : ℕ} (hn : 0 < n) :
    σ (a ^ n) = (· ^ n) '' σ a := by
  simpa only [aeval_X_pow, eval_pow, eval_X]
    using map_polynomial_aeval_of_degree_pos a (X ^ n : 𝕜[X]) (by rwa [degree_X_pow, Nat.cast_pos])


/-- A specialization of `spectrum.map_polynomial_aeval_of_nonempty` to monic monomials for
convenience. -/
theorem map_pow_of_nonempty [IsAlgClosed 𝕜] {a : A} (ha : (σ a).Nonempty) (n : ℕ) :
    σ (a ^ n) = (· ^ n) '' σ a := by
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : IsAlgClosed 𝕜
    a : A
    ha : (spectrum 𝕜 a).Nonempty
    n : Nat
    ⊢ Eq (spectrum 𝕜 (HPow.hPow a n)) (Set.image (fun x => HPow.hPow x n) (spectru …
  -/
  simpa only [aeval_X_pow, eval_pow, eval_X] using map_polynomial_aeval_of_nonempty a (X ^ n) ha
  /-
    🎉 no goals
  -/


/-- Every element `a` in a nontrivial finite-dimensional algebra `A`
over an algebraically closed field `𝕜` has non-empty spectrum. -/
theorem nonempty_of_isAlgClosed_of_finiteDimensional [IsAlgClosed 𝕜] [Nontrivial A]
    [I : FiniteDimensional 𝕜 A] (a : A) : (σ a).Nonempty := by
  /-
    𝕜 : Type u
    A : Type v
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring A
    inst✝² : Algebra 𝕜 A
    inst✝¹ : IsAlgClosed 𝕜
    inst✝ : Nontrivial A
    I : FiniteDimensional 𝕜 A
    a : A
    ⊢ (spectrum 𝕜 a).Nonempty
  -/
  obtain ⟨p, ⟨h_mon, h_eval_p⟩⟩ := isIntegral_of_noetherian (IsNoetherian.iff_fg.2 I) a
  /-
    case intro.intro
    𝕜 : Type u
    A : Type v
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring A
    inst✝² : Algebra 𝕜 A
    inst✝¹ : IsAlgClosed 𝕜
    inst✝ : Nontrivial A
    I : FiniteDimensional 𝕜 A
    a : A
    p : Polynomial 𝕜
    h_mon : p.Monic
    h_eval_p : Eq (Polynomial.eval₂ (algebraMap 𝕜 A) a p) 0
    ⊢ (spectrum 𝕜 a).Nonempty
  -/
  have nu : ¬IsUnit (aeval a p) := by rw [← aeval_def] at h_eval_p; rw [h_eval_p]; simp
  /-
    case intro.intro
    𝕜 : Type u
    A : Type v
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring A
    inst✝² : Algebra 𝕜 A
    inst✝¹ : IsAlgClosed 𝕜
    inst✝ : Nontrivial A
    I : FiniteDimensional 𝕜 A
    a : A
    p : Polynomial 𝕜
    h_mon : p.Monic
    h_eval_p : Eq (Polynomial.eval₂ (algebraMap 𝕜 A) a p) 0
    nu : Not (IsUnit ((Polynomial.aeval a) p))
    ⊢ (spectrum 𝕜 a).Nonempty
  -/
  rw [eq_prod_roots_of_monic_of_splits_id h_mon (IsAlgClosed.splits p)] at nu
  /-
    case intro.intro
    𝕜 : Type u
    A : Type v
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring A
    inst✝² : Algebra 𝕜 A
    inst✝¹ : IsAlgClosed 𝕜
    inst✝ : Nontrivial A
    I : FiniteDimensional 𝕜 A
    a : A
    p : Polynomial 𝕜
    h_mon : p.Monic
    h_eval_p : Eq (Polynomial.eval₂ (algebraMap 𝕜 A) a p) 0
    nu : Not (IsUnit ((Polynomial.aeval a) (Multiset.map (fun a => HSub.hSub Polyn …
    ⊢ (spectrum 𝕜 a).Nonempty
  -/
  obtain ⟨k, hk, _⟩ := exists_mem_of_not_isUnit_aeval_prod nu
  /-
    case intro.intro.intro.intro
    𝕜 : Type u
    A : Type v
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring A
    inst✝² : Algebra 𝕜 A
    inst✝¹ : IsAlgClosed 𝕜
    inst✝ : Nontrivial A
    I : FiniteDimensional 𝕜 A
    a : A
    p : Polynomial 𝕜
    h_mon : p.Monic
    h_eval_p : Eq (Polynomial.eval₂ (algebraMap 𝕜 A) a p) 0
    nu : Not (IsUnit ((Polynomial.aeval a) (Multiset.map (fun a => HSub.hSub Polyn …
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 a) k
    right✝ : Eq (Polynomial.eval k p) 0
    ⊢ (spectrum 𝕜 a).Nonempty
  -/
  exact ⟨k, hk⟩
  /-
    🎉 no goals
  -/


