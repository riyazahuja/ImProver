/-- `roots p` noncomputably gives a multiset containing all the roots of `p`,
including their multiplicities. -/
noncomputable def roots (p : R[X]) : Multiset R :=
  haveI := Classical.decEq R
  haveI := Classical.dec (p = 0)
  if h : p = 0 then ∅ else Classical.choose (exists_multiset_roots h)


theorem roots_def [DecidableEq R] (p : R[X]) [Decidable (p = 0)] :
    p.roots = if h : p = 0 then ∅ else Classical.choose (exists_multiset_roots h) := by
  -- porting noteL `‹_›` doesn't work for instance arguments
  /-
    R : Type u
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : DecidableEq R
    p : Polynomial R
    inst✝ : Decidable (Eq p 0)
    ⊢ Eq p.roots (dite (Eq p 0) (fun h => EmptyCollection.emptyCollection) fun h = …
  -/
  rename_i iR ip0
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    iR : DecidableEq R
    p : Polynomial R
    ip0 : Decidable (Eq p 0)
    ⊢ Eq p.roots (dite (Eq p 0) (fun h => EmptyCollection.emptyCollection) fun h = …
  -/
  obtain rfl := Subsingleton.elim iR (Classical.decEq R)
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    ip0 : Decidable (Eq p 0)
    ⊢ Eq p.roots (dite (Eq p 0) (fun h => EmptyCollection.emptyCollection) fun h = …
  -/
  obtain rfl := Subsingleton.elim ip0 (Classical.dec (p = 0))
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    ⊢ Eq p.roots (dite (Eq p 0) (fun h => EmptyCollection.emptyCollection) fun h = …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem roots_zero : (0 : R[X]).roots = 0 :=
  dif_pos rfl


theorem card_roots (hp0 : p ≠ 0) : (Multiset.card (roots p) : WithBot ℕ) ≤ degree p := by
  classical
  unfold roots
  rw [dif_neg hp0]
  exact (Classical.choose_spec (exists_multiset_roots hp0)).1


theorem card_roots' (p : R[X]) : Multiset.card p.roots ≤ natDegree p := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    ⊢ LE.le p.roots.card p.natDegree
  -/
  by_cases hp0 : p = 0
    /-
      case pos
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p : Polynomial R
      hp0 : Eq p 0
      ⊢ LE.le p.roots.card p.natDegree
    -/
  · simp [hp0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    hp0 : Not (Eq p 0)
    ⊢ LE.le p.roots.card p.natDegree
  -/
  exact WithBot.coe_le_coe.1 (le_trans (card_roots hp0) (le_of_eq <| degree_eq_natDegree hp0))
  /-
    🎉 no goals
  -/


theorem card_roots_sub_C {p : R[X]} {a : R} (hp0 : 0 < degree p) :
    (Multiset.card (p - C a).roots : WithBot ℕ) ≤ degree p :=
  calc
    (Multiset.card (p - C a).roots : WithBot ℕ) ≤ degree (p - C a) :=
      card_roots <| mt sub_eq_zero.1 fun h => not_le_of_gt hp0 <| h.symm ▸ degree_C_le
                       /-
                         R : Type u
                         inst✝¹ : CommRing R
                         inst✝ : IsDomain R
                         p : Polynomial R
                         a : R
                         hp0 : LT.lt 0 p.degree
                         ⊢ Eq (HSub.hSub p (Polynomial.C a)).degree p.degree
                       -/
    _ = degree p := by rw [sub_eq_add_neg, ← C_neg]; exact degree_add_C hp0
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem card_roots_sub_C' {p : R[X]} {a : R} (hp0 : 0 < degree p) :
    Multiset.card (p - C a).roots ≤ natDegree p :=
  WithBot.coe_le_coe.1
    (le_trans (card_roots_sub_C hp0)
                                                   /-
                                                     R : Type u
                                                     inst✝¹ : CommRing R
                                                     inst✝ : IsDomain R
                                                     p : Polynomial R
                                                     a : R
                                                     hp0 : LT.lt 0 p.degree
                                                     h : Eq p 0
                                                     ⊢ False
                                                   -/
      (le_of_eq <| degree_eq_natDegree fun h => by simp_all [lt_irrefl]))
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem count_roots [DecidableEq R] (p : R[X]) : p.roots.count a = rootMultiplicity a p := by
  classical
  by_cases hp : p = 0
  · simp [hp]
  rw [roots_def, dif_neg hp]
  exact (Classical.choose_spec (exists_multiset_roots hp)).2 a


@[simp]
theorem mem_roots' : a ∈ p.roots ↔ p ≠ 0 ∧ IsRoot p a := by
  classical
  rw [← count_pos, count_roots p, rootMultiplicity_pos']


theorem mem_roots (hp : p ≠ 0) : a ∈ p.roots ↔ IsRoot p a :=
  mem_roots'.trans <| and_iff_right hp


theorem ne_zero_of_mem_roots (h : a ∈ p.roots) : p ≠ 0 :=
  (mem_roots'.1 h).1


theorem isRoot_of_mem_roots (h : a ∈ p.roots) : IsRoot p a :=
  (mem_roots'.1 h).2


theorem mem_roots_map_of_injective [Semiring S] {p : S[X]} {f : S →+* R}
    (hf : Function.Injective f) {x : R} (hp : p ≠ 0) : x ∈ (p.map f).roots ↔ p.eval₂ f x = 0 := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Semiring S
    p : Polynomial S
    f : RingHom S R
    hf : Function.Injective ⇑f
    x : R
    hp : Ne p 0
    ⊢ Iff (Membership.mem (Polynomial.map f p).roots x) (Eq (Polynomial.eval₂ f x  …
  -/
  rw [mem_roots ((Polynomial.map_ne_zero_iff hf).mpr hp), IsRoot, eval_map]
  /-
    🎉 no goals
  -/


lemma mem_roots_iff_aeval_eq_zero {x : R} (w : p ≠ 0) : x ∈ roots p ↔ aeval x p = 0 := by
  rw [aeval_def, ← mem_roots_map_of_injective (NoZeroSMulDivisors.algebraMap_injective _ _) w,
    Algebra.id.map_eq_id, map_id]


theorem card_le_degree_of_subset_roots {p : R[X]} {Z : Finset R} (h : Z.val ⊆ p.roots) :
    #Z ≤ p.natDegree :=
  (Multiset.card_le_card (Finset.val_le_iff_val_subset.2 h)).trans (Polynomial.card_roots' p)


theorem finite_setOf_isRoot {p : R[X]} (hp : p ≠ 0) : Set.Finite { x | IsRoot p x } := by
  classical
  simpa only [← Finset.setOf_mem, Multiset.mem_toFinset, mem_roots hp]
    using p.roots.toFinset.finite_toSet


theorem eq_zero_of_infinite_isRoot (p : R[X]) (h : Set.Infinite { x | IsRoot p x }) : p = 0 :=
  not_imp_comm.mp finite_setOf_isRoot h


theorem exists_max_root [LinearOrder R] (p : R[X]) (hp : p ≠ 0) : ∃ x₀, ∀ x, p.IsRoot x → x ≤ x₀ :=
  Set.exists_upper_bound_image _ _ <| finite_setOf_isRoot hp


theorem exists_min_root [LinearOrder R] (p : R[X]) (hp : p ≠ 0) : ∃ x₀, ∀ x, p.IsRoot x → x₀ ≤ x :=
  Set.exists_lower_bound_image _ _ <| finite_setOf_isRoot hp


theorem eq_of_infinite_eval_eq (p q : R[X]) (h : Set.Infinite { x | eval x p = eval x q }) :
    p = q := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p q : Polynomial R
    h : (setOf fun x => Eq (Polynomial.eval x p) (Polynomial.eval x q)).Infinite
    ⊢ Eq p q
  -/
  rw [← sub_eq_zero]
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p q : Polynomial R
    h : (setOf fun x => Eq (Polynomial.eval x p) (Polynomial.eval x q)).Infinite
    ⊢ Eq (HSub.hSub p q) 0
  -/
  apply eq_zero_of_infinite_isRoot
  /-
    case h
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p q : Polynomial R
    h : (setOf fun x => Eq (Polynomial.eval x p) (Polynomial.eval x q)).Infinite
    ⊢ (setOf fun x => (HSub.hSub p q).IsRoot x).Infinite
  -/
  simpa only [IsRoot, eval_sub, sub_eq_zero]
  /-
    🎉 no goals
  -/


theorem roots_mul {p q : R[X]} (hpq : p * q ≠ 0) : (p * q).roots = p.roots + q.roots := by
  classical
  exact Multiset.ext.mpr fun r => by
    rw [count_add, count_roots, count_roots, count_roots, rootMultiplicity_mul hpq]


theorem roots.le_of_dvd (h : q ≠ 0) : p ∣ q → roots p ≤ roots q := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p q : Polynomial R
    h : Ne q 0
    ⊢ Dvd.dvd p q → LE.le p.roots q.roots
  -/
  rintro ⟨k, rfl⟩
  /-
    case intro
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p k : Polynomial R
    h : Ne (HMul.hMul p k) 0
    ⊢ LE.le p.roots (HMul.hMul p k).roots
  -/
  exact Multiset.le_iff_exists_add.mpr ⟨k.roots, roots_mul h⟩
  /-
    🎉 no goals
  -/


theorem mem_roots_sub_C' {p : R[X]} {a x : R} : x ∈ (p - C a).roots ↔ p ≠ C a ∧ p.eval x = a := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    a x : R
    ⊢ Iff (Membership.mem (HSub.hSub p (Polynomial.C a)).roots x) (And (Ne p (Poly …
  -/
  rw [mem_roots', IsRoot.def, sub_ne_zero, eval_sub, sub_eq_zero, eval_C]
  /-
    🎉 no goals
  -/


theorem mem_roots_sub_C {p : R[X]} {a x : R} (hp0 : 0 < degree p) :
    x ∈ (p - C a).roots ↔ p.eval x = a :=
  mem_roots_sub_C'.trans <| and_iff_right fun hp => hp0.not_le <| hp.symm ▸ degree_C_le


@[simp]
theorem roots_X_sub_C (r : R) : roots (X - C r) = {r} := by
  classical
  ext s
  rw [count_roots, rootMultiplicity_X_sub_C, count_singleton]


@[simp]
                                                             /-
                                                               R : Type u
                                                               inst✝¹ : CommRing R
                                                               inst✝ : IsDomain R
                                                               r : R
                                                               ⊢ Eq (HAdd.hAdd Polynomial.X (Polynomial.C r)).roots (Singleton.singleton (Neg …
                                                             -/
theorem roots_X_add_C (r : R) : roots (X + C r) = {-r} := by simpa using roots_X_sub_C (-r)
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
                                               /-
                                                 R : Type u
                                                 inst✝¹ : CommRing R
                                                 inst✝ : IsDomain R
                                                 ⊢ Eq Polynomial.X.roots (Singleton.singleton 0)
                                               -/
theorem roots_X : roots (X : R[X]) = {0} := by rw [← roots_X_sub_C, C_0, sub_zero]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem roots_C (x : R) : (C x).roots = 0 := by
  classical exact
  if H : x = 0 then by rw [H, C_0, roots_zero]
  else
    Multiset.ext.mpr fun r => (by
      rw [count_roots, count_zero, rootMultiplicity_eq_zero (not_isRoot_C _ _ H)])


@[simp]
theorem roots_one : (1 : R[X]).roots = ∅ :=
  roots_C 1


@[simp]
theorem roots_C_mul (p : R[X]) (ha : a ≠ 0) : (C a * p).roots = p.roots := by
  /-
    R : Type u
    a : R
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    ha : Ne a 0
    ⊢ Eq (HMul.hMul (Polynomial.C a) p).roots p.roots
  -/
  by_cases hp : p = 0 <;>
    simp only [roots_mul, *, Ne, mul_eq_zero, C_eq_zero, or_self_iff, not_false_iff, roots_C,
      zero_add, mul_zero]


@[simp]
theorem roots_smul_nonzero (p : R[X]) (ha : a ≠ 0) : (a • p).roots = p.roots := by
  /-
    R : Type u
    a : R
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    ha : Ne a 0
    ⊢ Eq (HSMul.hSMul a p).roots p.roots
  -/
  rw [smul_eq_C_mul, roots_C_mul _ ha]
  /-
    🎉 no goals
  -/


@[simp]
lemma roots_neg (p : R[X]) : (-p).roots = p.roots := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    ⊢ Eq (Neg.neg p).roots p.roots
  -/
  rw [← neg_one_smul R p, roots_smul_nonzero p (neg_ne_zero.mpr one_ne_zero)]
  /-
    🎉 no goals
  -/


@[simp]
theorem roots_C_mul_X_sub_C_of_IsUnit (b : R) (a : Rˣ) : (C (a : R) * X - C b).roots =
    {a⁻¹ * b} := by
  rw [← roots_C_mul _ (Units.ne_zero a⁻¹), mul_sub, ← mul_assoc, ← C_mul, ← C_mul,
    Units.inv_mul, C_1, one_mul]
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    b : R
    a : Units R
    ⊢ Eq (HSub.hSub Polynomial.X (Polynomial.C (HMul.hMul (↑(Inv.inv a)) b))).root …
  -/
  exact roots_X_sub_C (a⁻¹ * b)
  /-
    🎉 no goals
  -/


@[simp]
theorem roots_C_mul_X_add_C_of_IsUnit (b : R) (a : Rˣ) : (C (a : R) * X + C b).roots =
    {-(a⁻¹ * b)} := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    b : R
    a : Units R
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C ↑a) Polynomial.X) (Polynomial.C b)).r …
  -/
  rw [← sub_neg_eq_add, ← C_neg, roots_C_mul_X_sub_C_of_IsUnit, mul_neg]
  /-
    🎉 no goals
  -/


theorem roots_list_prod (L : List R[X]) :
    (0 : R[X]) ∉ L → L.prod.roots = (L : Multiset R[X]).bind roots :=
  List.recOn L (fun _ => roots_one) fun hd tl ih H => by
    /-
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      L : List (Polynomial R)
      hd : Polynomial R
      tl : List (Polynomial R)
      ih : Not (Membership.mem tl 0) → Eq tl.prod.roots ((↑tl).bind Polynomial.roots)
      H : Not (Membership.mem (List.cons hd tl) 0)
      ⊢ Eq (List.cons hd tl).prod.roots ((↑(List.cons hd tl)).bind Polynomial.roots)
    -/
    rw [List.mem_cons, not_or] at H
    rw [List.prod_cons, roots_mul (mul_ne_zero (Ne.symm H.1) <| List.prod_ne_zero H.2), ←
      Multiset.cons_coe, Multiset.cons_bind, ih H.2]


theorem roots_multiset_prod (m : Multiset R[X]) : (0 : R[X]) ∉ m → m.prod.roots = m.bind roots := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    m : Multiset (Polynomial R)
    ⊢ Not (Membership.mem m 0) → Eq m.prod.roots (m.bind Polynomial.roots)
  -/
  rcases m with ⟨L⟩
  /-
    case mk
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    m : Multiset (Polynomial R)
    L : List (Polynomial R)
    ⊢ Not (Membership.mem (Quot.mk (⇑(List.isSetoid (Polynomial R))) L) 0) → Eq (M …
  -/
  simpa only [Multiset.prod_coe, quot_mk_to_coe''] using roots_list_prod L
  /-
    🎉 no goals
  -/


theorem roots_prod {ι : Type*} (f : ι → R[X]) (s : Finset ι) :
    s.prod f ≠ 0 → (s.prod f).roots = s.val.bind fun i => roots (f i) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ι : Type u_1
    f : ι → Polynomial R
    s : Finset ι
    ⊢ Ne (s.prod f) 0 → Eq (s.prod f).roots (s.val.bind fun i => (f i).roots)
  -/
  rcases s with ⟨m, hm⟩
  /-
    case mk
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ι : Type u_1
    f : ι → Polynomial R
    m : Multiset ι
    hm : m.Nodup
    ⊢ Ne ({ val := m, nodup := hm }.prod f) 0 → Eq ({ val := m, nodup := hm }.prod …
  -/
  simpa [Multiset.prod_eq_zero_iff, Multiset.bind_map] using roots_multiset_prod (m.map f)
  /-
    🎉 no goals
  -/


@[simp]
theorem roots_pow (p : R[X]) (n : ℕ) : (p ^ n).roots = n • p.roots := by
  induction n with
  | zero => rw [pow_zero, roots_one, zero_smul, empty_eq_zero]
  | succ n ihn =>
    rcases eq_or_ne p 0 with (rfl | hp)
    · rw [zero_pow n.succ_ne_zero, roots_zero, smul_zero]
    · rw [pow_succ, roots_mul (mul_ne_zero (pow_ne_zero _ hp) hp), ihn, add_smul, one_smul]


theorem roots_X_pow (n : ℕ) : (X ^ n : R[X]).roots = n • ({0} : Multiset R) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    ⊢ Eq (HPow.hPow Polynomial.X n).roots (HSMul.hSMul n (Singleton.singleton 0))
  -/
  rw [roots_pow, roots_X]
  /-
    🎉 no goals
  -/


theorem roots_C_mul_X_pow (ha : a ≠ 0) (n : ℕ) :
    Polynomial.roots (C a * X ^ n) = n • ({0} : Multiset R) := by
  /-
    R : Type u
    a : R
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ha : Ne a 0
    n : Nat
    ⊢ Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)).roots (HSMul.hSMu …
  -/
  rw [roots_C_mul _ ha, roots_X_pow]
  /-
    🎉 no goals
  -/


@[simp]
theorem roots_monomial (ha : a ≠ 0) (n : ℕ) : (monomial n a).roots = n • ({0} : Multiset R) := by
  /-
    R : Type u
    a : R
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ha : Ne a 0
    n : Nat
    ⊢ Eq ((Polynomial.monomial n) a).roots (HSMul.hSMul n (Singleton.singleton 0))
  -/
  rw [← C_mul_X_pow_eq_monomial, roots_C_mul_X_pow ha]
  /-
    🎉 no goals
  -/


theorem roots_prod_X_sub_C (s : Finset R) : (s.prod fun a => X - C a).roots = s.val := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    s : Finset R
    ⊢ Eq (s.prod fun a => HSub.hSub Polynomial.X (Polynomial.C a)).roots s.val
  -/
  apply (roots_prod (fun a => X - C a) s ?_).trans
    /-
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      s : Finset R
      ⊢ Eq (s.val.bind fun i => (HSub.hSub Polynomial.X (Polynomial.C i)).roots) s.val
    -/
  · simp_rw [roots_X_sub_C]
    /-
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      s : Finset R
      ⊢ Eq (s.val.bind fun i => Singleton.singleton i) s.val
    -/
    rw [Multiset.bind_singleton, Multiset.map_id']
    /-
      🎉 no goals
    -/
    /-
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      s : Finset R
      ⊢ Ne (s.prod fun a => HSub.hSub Polynomial.X (Polynomial.C a)) 0
    -/
  · refine prod_ne_zero_iff.mpr (fun a _ => X_sub_C_ne_zero a)
    /-
      🎉 no goals
    -/


@[simp]
theorem roots_multiset_prod_X_sub_C (s : Multiset R) : (s.map fun a => X - C a).prod.roots = s := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    s : Multiset R
    ⊢ Eq (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) s).prod. …
  -/
  rw [roots_multiset_prod, Multiset.bind_map]
    /-
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      s : Multiset R
      ⊢ Eq (s.bind fun a => (HSub.hSub Polynomial.X (Polynomial.C a)).roots) s
    -/
  · simp_rw [roots_X_sub_C]
    /-
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      s : Multiset R
      ⊢ Eq (s.bind fun a => Singleton.singleton a) s
    -/
    rw [Multiset.bind_singleton, Multiset.map_id']
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      s : Multiset R
      ⊢ Not (Membership.mem (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomi …
    -/
  · rw [Multiset.mem_map]
    /-
      case a
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      s : Multiset R
      ⊢ Not (Exists fun a => And (Membership.mem s a) (Eq (HSub.hSub Polynomial.X (P …
    -/
    rintro ⟨a, -, h⟩
    /-
      case a.intro.intro
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      s : Multiset R
      a : R
      h : Eq (HSub.hSub Polynomial.X (Polynomial.C a)) 0
      ⊢ False
    -/
    exact X_sub_C_ne_zero a h
    /-
      🎉 no goals
    -/


theorem card_roots_X_pow_sub_C {n : ℕ} (hn : 0 < n) (a : R) :
    Multiset.card (roots ((X : R[X]) ^ n - C a)) ≤ n :=
  WithBot.coe_le_coe.1 <|
    calc
      (Multiset.card (roots ((X : R[X]) ^ n - C a)) : WithBot ℕ) ≤ degree ((X : R[X]) ^ n - C a) :=
        card_roots (X_pow_sub_C_ne_zero hn a)
      _ = n := degree_X_pow_sub_C hn a


/-- `nthRoots n a` noncomputably returns the solutions to `x ^ n = a`-/
def nthRoots (n : ℕ) (a : R) : Multiset R :=
  roots ((X : R[X]) ^ n - C a)


@[simp]
theorem mem_nthRoots {n : ℕ} (hn : 0 < n) {a x : R} : x ∈ nthRoots n a ↔ x ^ n = a := by
  rw [nthRoots, mem_roots (X_pow_sub_C_ne_zero hn a), IsRoot.def, eval_sub, eval_C, eval_pow,
    eval_X, sub_eq_zero]


@[simp]
theorem nthRoots_zero (r : R) : nthRoots 0 r = 0 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    r : R
    ⊢ Eq (Polynomial.nthRoots 0 r) 0
  -/
  simp only [empty_eq_zero, pow_zero, nthRoots, ← C_1, ← C_sub, roots_C]
  /-
    🎉 no goals
  -/


@[simp]
theorem nthRoots_zero_right {R} [CommRing R] [IsDomain R] (n : ℕ) :
    nthRoots n (0 : R) = Multiset.replicate n 0 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    ⊢ Eq (Polynomial.nthRoots n 0) (Multiset.replicate n 0)
  -/
  rw [nthRoots, C.map_zero, sub_zero, roots_pow, roots_X, Multiset.nsmul_singleton]
  /-
    🎉 no goals
  -/


theorem card_nthRoots (n : ℕ) (a : R) : Multiset.card (nthRoots n a) ≤ n := by
  classical exact
  (if hn : n = 0 then
    if h : (X : R[X]) ^ n - C a = 0 then by
      simp [Nat.zero_le, nthRoots, roots, h, dif_pos rfl, empty_eq_zero, Multiset.card_zero]
    else
      WithBot.coe_le_coe.1
        (le_trans (card_roots h)
          (by
            rw [hn, pow_zero, ← C_1, ← RingHom.map_sub]
            exact degree_C_le))
  else by
    rw [← Nat.cast_le (α := WithBot ℕ)]
    rw [← degree_X_pow_sub_C (Nat.pos_of_ne_zero hn) a]
    exact card_roots (X_pow_sub_C_ne_zero (Nat.pos_of_ne_zero hn) a))


@[simp]
theorem nthRoots_two_eq_zero_iff {r : R} : nthRoots 2 r = 0 ↔ ¬IsSquare r := by
  simp_rw [isSquare_iff_exists_sq, eq_zero_iff_forall_not_mem, mem_nthRoots (by norm_num : 0 < 2),
    ← not_exists, eq_comm]


/-- The multiset `nthRoots ↑n (1 : R)` as a Finset. -/
def nthRootsFinset (n : ℕ) (R : Type*) [CommRing R] [IsDomain R] : Finset R :=
  haveI := Classical.decEq R
  Multiset.toFinset (nthRoots n (1 : R))


lemma nthRootsFinset_def (n : ℕ) (R : Type*) [CommRing R] [IsDomain R] [DecidableEq R] :
    nthRootsFinset n R = Multiset.toFinset (nthRoots n (1 : R)) := by
  /-
    n : Nat
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : DecidableEq R
    ⊢ Eq (Polynomial.nthRootsFinset n R) (Polynomial.nthRoots n 1).toFinset
  -/
  unfold nthRootsFinset
  /-
    n : Nat
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : DecidableEq R
    ⊢ Eq (Polynomial.nthRoots n 1).toFinset (Polynomial.nthRoots n 1).toFinset
  -/
  convert rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_nthRootsFinset {n : ℕ} (h : 0 < n) {x : R} :
    x ∈ nthRootsFinset n R ↔ x ^ (n : ℕ) = 1 := by
  classical
  rw [nthRootsFinset_def, mem_toFinset, mem_nthRoots h]


@[simp]
                                                           /-
                                                             R : Type u
                                                             inst✝¹ : CommRing R
                                                             inst✝ : IsDomain R
                                                             ⊢ Eq (Polynomial.nthRootsFinset 0 R) EmptyCollection.emptyCollection
                                                           -/
theorem nthRootsFinset_zero : nthRootsFinset 0 R = ∅ := by classical simp [nthRootsFinset_def]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem map_mem_nthRootsFinset {S F : Type*} [CommRing S] [IsDomain S] [FunLike F R S]
    [RingHomClass F R S] {x : R} (hx : x ∈ nthRootsFinset n R) (f : F) :
    f x ∈ nthRootsFinset n S := by
  /-
    R : Type u
    n : Nat
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    S : Type u_1
    F : Type u_2
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : FunLike F R S
    inst✝ : RingHomClass F R S
    x : R
    hx : Membership.mem (Polynomial.nthRootsFinset n R) x
    f : F
    ⊢ Membership.mem (Polynomial.nthRootsFinset n S) (f x)
  -/
  by_cases hn : n = 0
    /-
      case pos
      R : Type u
      n : Nat
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      S : Type u_1
      F : Type u_2
      inst✝³ : CommRing S
      inst✝² : IsDomain S
      inst✝¹ : FunLike F R S
      inst✝ : RingHomClass F R S
      x : R
      hx : Membership.mem (Polynomial.nthRootsFinset n R) x
      f : F
      hn : Eq n 0
      ⊢ Membership.mem (Polynomial.nthRootsFinset n S) (f x)
    -/
  · simp [hn] at hx
    /-
      🎉 no goals
    -/
  · rw [mem_nthRootsFinset <| Nat.pos_of_ne_zero hn, ← map_pow, (mem_nthRootsFinset <|
      Nat.pos_of_ne_zero hn).1 hx, map_one]


theorem mul_mem_nthRootsFinset
    {η₁ η₂ : R} (hη₁ : η₁ ∈ nthRootsFinset n R) (hη₂ : η₂ ∈ nthRootsFinset n R) :
    η₁ * η₂ ∈ nthRootsFinset n R := by
  cases n with
  | zero =>
    simp only [nthRootsFinset_zero, not_mem_empty] at hη₁
  | succ n =>
    rw [mem_nthRootsFinset n.succ_pos] at hη₁ hη₂ ⊢
    rw [mul_pow, hη₁, hη₂, one_mul]


theorem ne_zero_of_mem_nthRootsFinset {η : R} (hη : η ∈ nthRootsFinset n R) : η ≠ 0 := by
  /-
    R : Type u
    n : Nat
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    η : R
    hη : Membership.mem (Polynomial.nthRootsFinset n R) η
    ⊢ Ne η 0
  -/
  nontriviality R
  /-
    R : Type u
    n : Nat
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    η : R
    hη : Membership.mem (Polynomial.nthRootsFinset n R) η
    inst✝ : Nontrivial R
    ⊢ Ne η 0
  -/
  rintro rfl
  cases n with
  | zero =>
    simp only [nthRootsFinset_zero, not_mem_empty] at hη
  | succ n =>
    rw [mem_nthRootsFinset n.succ_pos, zero_pow n.succ_ne_zero] at hη
    exact zero_ne_one hη


theorem one_mem_nthRootsFinset (hn : 0 < n) : 1 ∈ nthRootsFinset n R := by
  /-
    R : Type u
    n : Nat
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    hn : LT.lt 0 n
    ⊢ Membership.mem (Polynomial.nthRootsFinset n R) 1
  -/
  rw [mem_nthRootsFinset hn, one_pow]
  /-
    🎉 no goals
  -/


theorem zero_of_eval_zero [Infinite R] (p : R[X]) (h : ∀ x, p.eval x = 0) : p = 0 := by
  classical
  by_contra hp
  refine @Fintype.false R _ ?_
  exact ⟨p.roots.toFinset, fun x => Multiset.mem_toFinset.mpr ((mem_roots hp).mpr (h _))⟩


theorem funext [Infinite R] {p q : R[X]} (ext : ∀ r : R, p.eval r = q.eval r) : p = q := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    p q : Polynomial R
    ext : ∀ (r : R), Eq (Polynomial.eval r p) (Polynomial.eval r q)
    ⊢ Eq p q
  -/
  rw [← sub_eq_zero]
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    p q : Polynomial R
    ext : ∀ (r : R), Eq (Polynomial.eval r p) (Polynomial.eval r q)
    ⊢ Eq (HSub.hSub p q) 0
  -/
  apply zero_of_eval_zero
  /-
    case h
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    p q : Polynomial R
    ext : ∀ (r : R), Eq (Polynomial.eval r p) (Polynomial.eval r q)
    ⊢ ∀ (x : R), Eq (Polynomial.eval x (HSub.hSub p q)) 0
  -/
  intro x
  /-
    case h
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    p q : Polynomial R
    ext : ∀ (r : R), Eq (Polynomial.eval r p) (Polynomial.eval r q)
    x : R
    ⊢ Eq (Polynomial.eval x (HSub.hSub p q)) 0
  -/
  rw [eval_sub, sub_eq_zero, ext]
  /-
    🎉 no goals
  -/


/-- Given a polynomial `p` with coefficients in a ring `T` and a `T`-algebra `S`, `aroots p S` is
the multiset of roots of `p` regarded as a polynomial over `S`. -/
noncomputable abbrev aroots (p : T[X]) (S) [CommRing S] [IsDomain S] [Algebra T S] : Multiset S :=
  (p.map (algebraMap T S)).roots


theorem aroots_def (p : T[X]) (S) [CommRing S] [IsDomain S] [Algebra T S] :
    p.aroots S = (p.map (algebraMap T S)).roots :=
  rfl


theorem mem_aroots' [CommRing S] [IsDomain S] [Algebra T S] {p : T[X]} {a : S} :
    a ∈ p.aroots S ↔ p.map (algebraMap T S) ≠ 0 ∧ aeval a p = 0 := by
  /-
    S : Type v
    T : Type w
    inst✝³ : CommRing T
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra T S
    p : Polynomial T
    a : S
    ⊢ Iff (Membership.mem (p.aroots S) a) (And (Ne (Polynomial.map (algebraMap T S …
  -/
  rw [mem_roots', IsRoot.def, ← eval₂_eq_eval_map, aeval_def]
  /-
    🎉 no goals
  -/


theorem mem_aroots [CommRing S] [IsDomain S] [Algebra T S]
    [NoZeroSMulDivisors T S] {p : T[X]} {a : S} : a ∈ p.aroots S ↔ p ≠ 0 ∧ aeval a p = 0 := by
  /-
    S : Type v
    T : Type w
    inst✝⁴ : CommRing T
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra T S
    inst✝ : NoZeroSMulDivisors T S
    p : Polynomial T
    a : S
    ⊢ Iff (Membership.mem (p.aroots S) a) (And (Ne p 0) (Eq ((Polynomial.aeval a)  …
  -/
  rw [mem_aroots', Polynomial.map_ne_zero_iff]
  /-
    S : Type v
    T : Type w
    inst✝⁴ : CommRing T
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra T S
    inst✝ : NoZeroSMulDivisors T S
    p : Polynomial T
    a : S
    ⊢ Function.Injective ⇑(algebraMap T S)
  -/
  exact NoZeroSMulDivisors.algebraMap_injective T S
  /-
    🎉 no goals
  -/


theorem aroots_mul [CommRing S] [IsDomain S] [Algebra T S]
    [NoZeroSMulDivisors T S] {p q : T[X]} (hpq : p * q ≠ 0) :
    (p * q).aroots S = p.aroots S + q.aroots S := by
  suffices map (algebraMap T S) p * map (algebraMap T S) q ≠ 0 by
    rw [aroots_def, Polynomial.map_mul, roots_mul this]
  rwa [← Polynomial.map_mul, Polynomial.map_ne_zero_iff
    (NoZeroSMulDivisors.algebraMap_injective T S)]


@[simp]
theorem aroots_X_sub_C [CommRing S] [IsDomain S] [Algebra T S]
    (r : T) : aroots (X - C r) S = {algebraMap T S r} := by
  /-
    S : Type v
    T : Type w
    inst✝³ : CommRing T
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra T S
    r : T
    ⊢ Eq ((HSub.hSub Polynomial.X (Polynomial.C r)).aroots S) (Singleton.singleton …
  -/
  rw [aroots_def, Polynomial.map_sub, map_X, map_C, roots_X_sub_C]
  /-
    🎉 no goals
  -/


@[simp]
theorem aroots_X [CommRing S] [IsDomain S] [Algebra T S] :
    aroots (X : T[X]) S = {0} := by
  /-
    S : Type v
    T : Type w
    inst✝³ : CommRing T
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra T S
    ⊢ Eq (Polynomial.X.aroots S) (Singleton.singleton 0)
  -/
  rw [aroots_def, map_X, roots_X]
  /-
    🎉 no goals
  -/


@[simp]
theorem aroots_C [CommRing S] [IsDomain S] [Algebra T S] (a : T) : (C a).aroots S = 0 := by
  /-
    S : Type v
    T : Type w
    inst✝³ : CommRing T
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra T S
    a : T
    ⊢ Eq ((Polynomial.C a).aroots S) 0
  -/
  rw [aroots_def, map_C, roots_C]
  /-
    🎉 no goals
  -/


@[simp]
theorem aroots_zero (S) [CommRing S] [IsDomain S] [Algebra T S] : (0 : T[X]).aroots S = 0 := by
  /-
    T : Type w
    inst✝³ : CommRing T
    S : Type u_1
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra T S
    ⊢ Eq (Polynomial.aroots 0 S) 0
  -/
  rw [← C_0, aroots_C]
  /-
    🎉 no goals
  -/


@[simp]
theorem aroots_one [CommRing S] [IsDomain S] [Algebra T S] :
    (1 : T[X]).aroots S = 0 :=
  aroots_C 1


@[simp]
theorem aroots_neg [CommRing S] [IsDomain S] [Algebra T S] (p : T[X]) :
    (-p).aroots S = p.aroots S := by
  /-
    S : Type v
    T : Type w
    inst✝³ : CommRing T
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra T S
    p : Polynomial T
    ⊢ Eq ((Neg.neg p).aroots S) (p.aroots S)
  -/
  rw [aroots, Polynomial.map_neg, roots_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem aroots_C_mul [CommRing S] [IsDomain S] [Algebra T S]
    [NoZeroSMulDivisors T S] {a : T} (p : T[X]) (ha : a ≠ 0) :
    (C a * p).aroots S = p.aroots S := by
  /-
    S : Type v
    T : Type w
    inst✝⁴ : CommRing T
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra T S
    inst✝ : NoZeroSMulDivisors T S
    a : T
    p : Polynomial T
    ha : Ne a 0
    ⊢ Eq ((HMul.hMul (Polynomial.C a) p).aroots S) (p.aroots S)
  -/
  rw [aroots_def, Polynomial.map_mul, map_C, roots_C_mul]
  /-
    case ha
    S : Type v
    T : Type w
    inst✝⁴ : CommRing T
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra T S
    inst✝ : NoZeroSMulDivisors T S
    a : T
    p : Polynomial T
    ha : Ne a 0
    ⊢ Ne ((algebraMap T S) a) 0
  -/
  rwa [map_ne_zero_iff]
  /-
    case ha.hf
    S : Type v
    T : Type w
    inst✝⁴ : CommRing T
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra T S
    inst✝ : NoZeroSMulDivisors T S
    a : T
    p : Polynomial T
    ha : Ne a 0
    ⊢ Function.Injective ⇑(algebraMap T S)
  -/
  exact NoZeroSMulDivisors.algebraMap_injective T S
  /-
    🎉 no goals
  -/


@[simp]
theorem aroots_smul_nonzero [CommRing S] [IsDomain S] [Algebra T S]
    [NoZeroSMulDivisors T S] {a : T} (p : T[X]) (ha : a ≠ 0) :
    (a • p).aroots S = p.aroots S := by
  /-
    S : Type v
    T : Type w
    inst✝⁴ : CommRing T
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra T S
    inst✝ : NoZeroSMulDivisors T S
    a : T
    p : Polynomial T
    ha : Ne a 0
    ⊢ Eq ((HSMul.hSMul a p).aroots S) (p.aroots S)
  -/
  rw [smul_eq_C_mul, aroots_C_mul _ ha]
  /-
    🎉 no goals
  -/


@[simp]
theorem aroots_pow [CommRing S] [IsDomain S] [Algebra T S] (p : T[X]) (n : ℕ) :
    (p ^ n).aroots S = n • p.aroots S := by
  /-
    S : Type v
    T : Type w
    inst✝³ : CommRing T
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra T S
    p : Polynomial T
    n : Nat
    ⊢ Eq ((HPow.hPow p n).aroots S) (HSMul.hSMul n (p.aroots S))
  -/
  rw [aroots_def, Polynomial.map_pow, roots_pow]
  /-
    🎉 no goals
  -/


theorem aroots_X_pow [CommRing S] [IsDomain S] [Algebra T S] (n : ℕ) :
    (X ^ n : T[X]).aroots S = n • ({0} : Multiset S) := by
  /-
    S : Type v
    T : Type w
    inst✝³ : CommRing T
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra T S
    n : Nat
    ⊢ Eq ((HPow.hPow Polynomial.X n).aroots S) (HSMul.hSMul n (Singleton.singleton …
  -/
  rw [aroots_pow, aroots_X]
  /-
    🎉 no goals
  -/


theorem aroots_C_mul_X_pow [CommRing S] [IsDomain S] [Algebra T S]
    [NoZeroSMulDivisors T S] {a : T} (ha : a ≠ 0) (n : ℕ) :
    (C a * X ^ n : T[X]).aroots S = n • ({0} : Multiset S) := by
  /-
    S : Type v
    T : Type w
    inst✝⁴ : CommRing T
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra T S
    inst✝ : NoZeroSMulDivisors T S
    a : T
    ha : Ne a 0
    n : Nat
    ⊢ Eq ((HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)).aroots S) (HSMul …
  -/
  rw [aroots_C_mul _ ha, aroots_X_pow]
  /-
    🎉 no goals
  -/


@[simp]
theorem aroots_monomial [CommRing S] [IsDomain S] [Algebra T S]
    [NoZeroSMulDivisors T S] {a : T} (ha : a ≠ 0) (n : ℕ) :
    (monomial n a).aroots S = n • ({0} : Multiset S) := by
  /-
    S : Type v
    T : Type w
    inst✝⁴ : CommRing T
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra T S
    inst✝ : NoZeroSMulDivisors T S
    a : T
    ha : Ne a 0
    n : Nat
    ⊢ Eq (((Polynomial.monomial n) a).aroots S) (HSMul.hSMul n (Singleton.singleto …
  -/
  rw [← C_mul_X_pow_eq_monomial, aroots_C_mul_X_pow ha]
  /-
    🎉 no goals
  -/


variable (R S) in
@[simp]
theorem aroots_map (p : T[X]) [CommRing S] [Algebra T S] [Algebra S R] [Algebra T R]
    [IsScalarTower T S R] :
    (p.map (algebraMap T S)).aroots R = p.aroots R := by
  /-
    R : Type u
    S : Type v
    T : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : IsDomain R
    inst✝⁵ : CommRing T
    p : Polynomial T
    inst✝⁴ : CommRing S
    inst✝³ : Algebra T S
    inst✝² : Algebra S R
    inst✝¹ : Algebra T R
    inst✝ : IsScalarTower T S R
    ⊢ Eq ((Polynomial.map (algebraMap T S) p).aroots R) (p.aroots R)
  -/
  rw [aroots_def, aroots_def, map_map, IsScalarTower.algebraMap_eq T S R]
  /-
    🎉 no goals
  -/


/-- The set of distinct roots of `p` in `S`.

If you have a non-separable polynomial, use `Polynomial.aroots` for the multiset
where multiple roots have the appropriate multiplicity. -/
def rootSet (p : T[X]) (S) [CommRing S] [IsDomain S] [Algebra T S] : Set S :=
  haveI := Classical.decEq S
  (p.aroots S).toFinset


theorem rootSet_def (p : T[X]) (S) [CommRing S] [IsDomain S] [Algebra T S] [DecidableEq S] :
    p.rootSet S = (p.aroots S).toFinset := by
  /-
    T : Type w
    inst✝⁴ : CommRing T
    p : Polynomial T
    S : Type u_1
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra T S
    inst✝ : DecidableEq S
    ⊢ Eq (p.rootSet S) ↑(p.aroots S).toFinset
  -/
  rw [rootSet]
  /-
    T : Type w
    inst✝⁴ : CommRing T
    p : Polynomial T
    S : Type u_1
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra T S
    inst✝ : DecidableEq S
    ⊢ Eq ↑(p.aroots S).toFinset ↑(p.aroots S).toFinset
  -/
  convert rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem rootSet_C [CommRing S] [IsDomain S] [Algebra T S] (a : T) : (C a).rootSet S = ∅ := by
  classical
  rw [rootSet_def, aroots_C, Multiset.toFinset_zero, Finset.coe_empty]


@[simp]
theorem rootSet_zero (S) [CommRing S] [IsDomain S] [Algebra T S] : (0 : T[X]).rootSet S = ∅ := by
  /-
    T : Type w
    inst✝³ : CommRing T
    S : Type u_1
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra T S
    ⊢ Eq (Polynomial.rootSet 0 S) EmptyCollection.emptyCollection
  -/
  rw [← C_0, rootSet_C]
  /-
    🎉 no goals
  -/


@[simp]
theorem rootSet_one (S) [CommRing S] [IsDomain S] [Algebra T S] : (1 : T[X]).rootSet S = ∅ := by
  /-
    T : Type w
    inst✝³ : CommRing T
    S : Type u_1
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra T S
    ⊢ Eq (Polynomial.rootSet 1 S) EmptyCollection.emptyCollection
  -/
  rw [← C_1, rootSet_C]
  /-
    🎉 no goals
  -/


@[simp]
theorem rootSet_neg (p : T[X]) (S) [CommRing S] [IsDomain S] [Algebra T S] :
    (-p).rootSet S = p.rootSet S := by
  /-
    T : Type w
    inst✝³ : CommRing T
    p : Polynomial T
    S : Type u_1
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra T S
    ⊢ Eq ((Neg.neg p).rootSet S) (p.rootSet S)
  -/
  rw [rootSet, aroots_neg, rootSet]
  /-
    🎉 no goals
  -/


instance rootSetFintype (p : T[X]) (S : Type*) [CommRing S] [IsDomain S] [Algebra T S] :
    Fintype (p.rootSet S) :=
  FinsetCoe.fintype _


theorem rootSet_finite (p : T[X]) (S : Type*) [CommRing S] [IsDomain S] [Algebra T S] :
    (p.rootSet S).Finite :=
  Set.toFinite _


/-- The set of roots of all polynomials of bounded degree and having coefficients in a finite set
is finite. -/
theorem bUnion_roots_finite {R S : Type*} [Semiring R] [CommRing S] [IsDomain S] [DecidableEq S]
    (m : R →+* S) (d : ℕ) {U : Set R} (h : U.Finite) :
    (⋃ (f : R[X]) (_ : f.natDegree ≤ d ∧ ∀ i, f.coeff i ∈ U),
        ((f.map m).roots.toFinset.toSet : Set S)).Finite :=
  Set.Finite.biUnion
    (by
      -- We prove that the set of polynomials under consideration is finite because its
      -- image by the injective map `π` is finite
      /-
        R : Type u_1
        S : Type u_2
        inst✝³ : Semiring R
        inst✝² : CommRing S
        inst✝¹ : IsDomain S
        inst✝ : DecidableEq S
        m : RingHom R S
        d : Nat
        U : Set R
        h : U.Finite
        ⊢ Set.Finite fun f => And (LE.le f.natDegree d) (∀ (i : Nat), Membership.mem U …
      -/
      let π : R[X] → Fin (d + 1) → R := fun f i => f.coeff i
      /-
        R : Type u_1
        S : Type u_2
        inst✝³ : Semiring R
        inst✝² : CommRing S
        inst✝¹ : IsDomain S
        inst✝ : DecidableEq S
        m : RingHom R S
        d : Nat
        U : Set R
        h : U.Finite
        π : Polynomial R → Fin (HAdd.hAdd d 1) → R := fun f i => f.coeff ↑i
        ⊢ Set.Finite fun f => And (LE.le f.natDegree d) (∀ (i : Nat), Membership.mem U …
      -/
      refine ((Set.Finite.pi fun _ => h).subset <| ?_).of_finite_image (?_ : Set.InjOn π _)
        /-
          case refine_1
          R : Type u_1
          S : Type u_2
          inst✝³ : Semiring R
          inst✝² : CommRing S
          inst✝¹ : IsDomain S
          inst✝ : DecidableEq S
          m : RingHom R S
          d : Nat
          U : Set R
          h : U.Finite
          π : Polynomial R → Fin (HAdd.hAdd d 1) → R := fun f i => f.coeff ↑i
          ⊢ HasSubset.Subset (Set.image π fun f => And (LE.le f.natDegree d) (∀ (i : Nat …
        -/
      · exact Set.image_subset_iff.2 fun f hf i _ => hf.2 i
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          R : Type u_1
          S : Type u_2
          inst✝³ : Semiring R
          inst✝² : CommRing S
          inst✝¹ : IsDomain S
          inst✝ : DecidableEq S
          m : RingHom R S
          d : Nat
          U : Set R
          h : U.Finite
          π : Polynomial R → Fin (HAdd.hAdd d 1) → R := fun f i => f.coeff ↑i
          ⊢ Set.InjOn π fun f => And (LE.le f.natDegree d) (∀ (i : Nat), Membership.mem  …
        -/
      · refine fun x hx y hy hxy => (ext_iff_natDegree_le hx.1 hy.1).2 fun i hi => ?_
        /-
          case refine_2
          R : Type u_1
          S : Type u_2
          inst✝³ : Semiring R
          inst✝² : CommRing S
          inst✝¹ : IsDomain S
          inst✝ : DecidableEq S
          m : RingHom R S
          d : Nat
          U : Set R
          h : U.Finite
          π : Polynomial R → Fin (HAdd.hAdd d 1) → R := fun f i => f.coeff ↑i
          x : Polynomial R
          hx : Membership.mem (fun f => And (LE.le f.natDegree d) (∀ (i : Nat), Membersh …
          y : Polynomial R
          hy : Membership.mem (fun f => And (LE.le f.natDegree d) (∀ (i : Nat), Membersh …
          hxy : Eq (π x) (π y)
          i : Nat
          hi : LE.le i d
          ⊢ Eq (x.coeff i) (y.coeff i)
        -/
        exact id congr_fun hxy ⟨i, Nat.lt_succ_of_le hi⟩)
        /-
          🎉 no goals
        -/
    fun _ _ => Finset.finite_toSet _


theorem mem_rootSet' {p : T[X]} {S : Type*} [CommRing S] [IsDomain S] [Algebra T S] {a : S} :
    a ∈ p.rootSet S ↔ p.map (algebraMap T S) ≠ 0 ∧ aeval a p = 0 := by
  classical
  rw [rootSet_def, Finset.mem_coe, mem_toFinset, mem_aroots']


theorem mem_rootSet {p : T[X]} {S : Type*} [CommRing S] [IsDomain S] [Algebra T S]
    [NoZeroSMulDivisors T S] {a : S} : a ∈ p.rootSet S ↔ p ≠ 0 ∧ aeval a p = 0 := by
  /-
    T : Type w
    inst✝⁴ : CommRing T
    p : Polynomial T
    S : Type u_1
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra T S
    inst✝ : NoZeroSMulDivisors T S
    a : S
    ⊢ Iff (Membership.mem (p.rootSet S) a) (And (Ne p 0) (Eq ((Polynomial.aeval a) …
  -/
  rw [mem_rootSet', Polynomial.map_ne_zero_iff (NoZeroSMulDivisors.algebraMap_injective T S)]
  /-
    🎉 no goals
  -/


theorem mem_rootSet_of_ne {p : T[X]} {S : Type*} [CommRing S] [IsDomain S] [Algebra T S]
    [NoZeroSMulDivisors T S] (hp : p ≠ 0) {a : S} : a ∈ p.rootSet S ↔ aeval a p = 0 :=
  mem_rootSet.trans <| and_iff_right hp


theorem rootSet_maps_to' {p : T[X]} {S S'} [CommRing S] [IsDomain S] [Algebra T S] [CommRing S']
    [IsDomain S'] [Algebra T S'] (hp : p.map (algebraMap T S') = 0 → p.map (algebraMap T S) = 0)
    (f : S →ₐ[T] S') : (p.rootSet S).MapsTo f (p.rootSet S') := fun x hx => by
  /-
    T : Type w
    inst✝⁶ : CommRing T
    p : Polynomial T
    S : Type u_1
    S' : Type u_2
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain S
    inst✝³ : Algebra T S
    inst✝² : CommRing S'
    inst✝¹ : IsDomain S'
    inst✝ : Algebra T S'
    hp : Eq (Polynomial.map (algebraMap T S') p) 0 → Eq (Polynomial.map (algebraMa …
    f : AlgHom T S S'
    x : S
    hx : Membership.mem (p.rootSet S) x
    ⊢ Membership.mem (p.rootSet S') (f x)
  -/
  rw [mem_rootSet'] at hx ⊢
  /-
    T : Type w
    inst✝⁶ : CommRing T
    p : Polynomial T
    S : Type u_1
    S' : Type u_2
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain S
    inst✝³ : Algebra T S
    inst✝² : CommRing S'
    inst✝¹ : IsDomain S'
    inst✝ : Algebra T S'
    hp : Eq (Polynomial.map (algebraMap T S') p) 0 → Eq (Polynomial.map (algebraMa …
    f : AlgHom T S S'
    x : S
    hx : And (Ne (Polynomial.map (algebraMap T S) p) 0) (Eq ((Polynomial.aeval x)  …
    ⊢ And (Ne (Polynomial.map (algebraMap T S') p) 0) (Eq ((Polynomial.aeval (f x) …
  -/
  rw [aeval_algHom, AlgHom.comp_apply, hx.2, _root_.map_zero]
  /-
    T : Type w
    inst✝⁶ : CommRing T
    p : Polynomial T
    S : Type u_1
    S' : Type u_2
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain S
    inst✝³ : Algebra T S
    inst✝² : CommRing S'
    inst✝¹ : IsDomain S'
    inst✝ : Algebra T S'
    hp : Eq (Polynomial.map (algebraMap T S') p) 0 → Eq (Polynomial.map (algebraMa …
    f : AlgHom T S S'
    x : S
    hx : And (Ne (Polynomial.map (algebraMap T S) p) 0) (Eq ((Polynomial.aeval x)  …
    ⊢ And (Ne (Polynomial.map (algebraMap T S') p) 0) (Eq 0 0)
  -/
  exact ⟨mt hp hx.1, rfl⟩
  /-
    🎉 no goals
  -/


theorem ne_zero_of_mem_rootSet {p : T[X]} [CommRing S] [IsDomain S] [Algebra T S] {a : S}
                                                  /-
                                                    S : Type v
                                                    T : Type w
                                                    inst✝³ : CommRing T
                                                    p : Polynomial T
                                                    inst✝² : CommRing S
                                                    inst✝¹ : IsDomain S
                                                    inst✝ : Algebra T S
                                                    a : S
                                                    h : Membership.mem (p.rootSet S) a
                                                    hf : Eq p 0
                                                    ⊢ False
                                                  -/
    (h : a ∈ p.rootSet S) : p ≠ 0 := fun hf => by rwa [hf, rootSet_zero] at h
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem aeval_eq_zero_of_mem_rootSet {p : T[X]} [CommRing S] [IsDomain S] [Algebra T S] {a : S}
    (hx : a ∈ p.rootSet S) : aeval a p = 0 :=
  (mem_rootSet'.1 hx).2


theorem rootSet_mapsTo {p : T[X]} {S S'} [CommRing S] [IsDomain S] [Algebra T S] [CommRing S']
    [IsDomain S'] [Algebra T S'] [NoZeroSMulDivisors T S'] (f : S →ₐ[T] S') :
    (p.rootSet S).MapsTo f (p.rootSet S') := by
  /-
    T : Type w
    inst✝⁷ : CommRing T
    p : Polynomial T
    S : Type u_1
    S' : Type u_2
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra T S
    inst✝³ : CommRing S'
    inst✝² : IsDomain S'
    inst✝¹ : Algebra T S'
    inst✝ : NoZeroSMulDivisors T S'
    f : AlgHom T S S'
    ⊢ Set.MapsTo (⇑f) (p.rootSet S) (p.rootSet S')
  -/
  refine rootSet_maps_to' (fun h₀ => ?_) f
  obtain rfl : p = 0 :=
    map_injective _ (NoZeroSMulDivisors.algebraMap_injective T S') (by rwa [Polynomial.map_zero])
  /-
    T : Type w
    inst✝⁷ : CommRing T
    S : Type u_1
    S' : Type u_2
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra T S
    inst✝³ : CommRing S'
    inst✝² : IsDomain S'
    inst✝¹ : Algebra T S'
    inst✝ : NoZeroSMulDivisors T S'
    f : AlgHom T S S'
    h₀ : Eq (Polynomial.map (algebraMap T S') 0) 0
    ⊢ Eq (Polynomial.map (algebraMap T S) 0) 0
  -/
  exact Polynomial.map_zero _
  /-
    🎉 no goals
  -/


theorem mem_rootSet_of_injective [CommRing S] {p : S[X]} [Algebra S R]
    (h : Function.Injective (algebraMap S R)) {x : R} (hp : p ≠ 0) :
    x ∈ p.rootSet R ↔ aeval x p = 0 := by
  classical
  exact Multiset.mem_toFinset.trans (mem_roots_map_of_injective h hp)


lemma eq_zero_of_natDegree_lt_card_of_eval_eq_zero {R} [CommRing R] [IsDomain R]
    (p : R[X]) {ι} [Fintype ι] {f : ι → R} (hf : Function.Injective f)
    (heval : ∀ i, p.eval (f i) = 0) (hcard : natDegree p < Fintype.card ι) : p = 0 := by
  classical
  by_contra hp
  refine lt_irrefl #p.roots.toFinset ?_
  calc
    #p.roots.toFinset ≤ Multiset.card p.roots := Multiset.toFinset_card_le _
    _ ≤ natDegree p := Polynomial.card_roots' p
    _ < Fintype.card ι := hcard
    _ = Fintype.card (Set.range f) := (Set.card_range_of_injective hf).symm
    _ = #(Finset.univ.image f) := by rw [← Set.toFinset_card, Set.toFinset_range]
    _ ≤ #p.roots.toFinset := Finset.card_mono ?_
  intro _
  simp only [Finset.mem_image, Finset.mem_univ, true_and, Multiset.mem_toFinset, mem_roots', ne_eq,
    IsRoot.def, forall_exists_index, hp, not_false_eq_true]
  rintro x rfl
  exact heval _


lemma eq_zero_of_natDegree_lt_card_of_eval_eq_zero' {R} [CommRing R] [IsDomain R]
    (p : R[X]) (s : Finset R) (heval : ∀ i ∈ s, p.eval i = 0) (hcard : natDegree p < #s) :
    p = 0 :=
  eq_zero_of_natDegree_lt_card_of_eval_eq_zero p Subtype.val_injective
    (fun i : s ↦ heval i i.prop) (hcard.trans_eq (Fintype.card_coe s).symm)


open Cardinal in
lemma eq_zero_of_forall_eval_zero_of_natDegree_lt_card
    (f : R[X]) (hf : ∀ r, f.eval r = 0) (hfR : f.natDegree < #R) : f = 0 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f : Polynomial R
    hf : ∀ (r : R), Eq (Polynomial.eval r f) 0
    hfR : LT.lt (↑f.natDegree) (Cardinal.mk R)
    ⊢ Eq f 0
  -/
  obtain hR|hR := finite_or_infinite R
    /-
      case inl
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      f : Polynomial R
      hf : ∀ (r : R), Eq (Polynomial.eval r f) 0
      hfR : LT.lt (↑f.natDegree) (Cardinal.mk R)
      hR : Finite R
      ⊢ Eq f 0
    -/
  · have := Fintype.ofFinite R
    /-
      case inl
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      f : Polynomial R
      hf : ∀ (r : R), Eq (Polynomial.eval r f) 0
      hfR : LT.lt (↑f.natDegree) (Cardinal.mk R)
      hR : Finite R
      this : Fintype R
      ⊢ Eq f 0
    -/
    apply eq_zero_of_natDegree_lt_card_of_eval_eq_zero f Function.injective_id hf
    /-
      case inl
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      f : Polynomial R
      hf : ∀ (r : R), Eq (Polynomial.eval r f) 0
      hfR : LT.lt (↑f.natDegree) (Cardinal.mk R)
      hR : Finite R
      this : Fintype R
      ⊢ LT.lt f.natDegree (Fintype.card R)
    -/
    simpa only [mk_fintype, Nat.cast_lt] using hfR
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      f : Polynomial R
      hf : ∀ (r : R), Eq (Polynomial.eval r f) 0
      hfR : LT.lt (↑f.natDegree) (Cardinal.mk R)
      hR : Infinite R
      ⊢ Eq f 0
    -/
  · exact zero_of_eval_zero _ hf
    /-
      🎉 no goals
    -/


open Cardinal in
lemma exists_eval_ne_zero_of_natDegree_lt_card (f : R[X]) (hf : f ≠ 0) (hfR : f.natDegree < #R) :
    ∃ r, f.eval r ≠ 0 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f : Polynomial R
    hf : Ne f 0
    hfR : LT.lt (↑f.natDegree) (Cardinal.mk R)
    ⊢ Exists fun r => Ne (Polynomial.eval r f) 0
  -/
  contrapose! hf
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f : Polynomial R
    hfR : LT.lt (↑f.natDegree) (Cardinal.mk R)
    hf : ∀ (r : R), Eq (Polynomial.eval r f) 0
    ⊢ Eq f 0
  -/
  exact eq_zero_of_forall_eval_zero_of_natDegree_lt_card f hf hfR
  /-
    🎉 no goals
  -/


theorem monic_prod_multiset_X_sub_C : Monic (p.roots.map fun a => X - C a).prod :=
  monic_multiset_prod_of_monic _ _ fun a _ => monic_X_sub_C a


theorem prod_multiset_root_eq_finset_root [DecidableEq R] :
    (p.roots.map fun a => X - C a).prod =
      p.roots.toFinset.prod fun a => (X - C a) ^ rootMultiplicity a p := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    p : Polynomial R
    inst✝ : DecidableEq R
    ⊢ Eq (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) p.roots) …
  -/
  simp only [count_roots, Finset.prod_multiset_map_count]
  /-
    🎉 no goals
  -/


/-- The product `∏ (X - a)` for `a` inside the multiset `p.roots` divides `p`. -/
theorem prod_multiset_X_sub_C_dvd (p : R[X]) : (p.roots.map fun a => X - C a).prod ∣ p := by
  classical
  rw [← map_dvd_map _ (IsFractionRing.injective R <| FractionRing R) monic_prod_multiset_X_sub_C]
  rw [prod_multiset_root_eq_finset_root, Polynomial.map_prod]
  refine Finset.prod_dvd_of_coprime (fun a _ b _ h => ?_) fun a _ => ?_
  · simp_rw [Polynomial.map_pow, Polynomial.map_sub, map_C, map_X]
    exact (pairwise_coprime_X_sub_C (IsFractionRing.injective R <| FractionRing R) h).pow
  · exact Polynomial.map_dvd _ (pow_rootMultiplicity_dvd p a)


/-- A Galois connection. -/
theorem _root_.Multiset.prod_X_sub_C_dvd_iff_le_roots {p : R[X]} (hp : p ≠ 0) (s : Multiset R) :
    (s.map fun a => X - C a).prod ∣ p ↔ s ≤ p.roots := by
  classical exact
  ⟨fun h =>
    Multiset.le_iff_count.2 fun r => by
      rw [count_roots, le_rootMultiplicity_iff hp, ← Multiset.prod_replicate, ←
        Multiset.map_replicate fun a => X - C a, ← Multiset.filter_eq]
      exact (Multiset.prod_dvd_prod_of_le <| Multiset.map_le_map <| s.filter_le _).trans h,
    fun h =>
    (Multiset.prod_dvd_prod_of_le <| Multiset.map_le_map h).trans p.prod_multiset_X_sub_C_dvd⟩


theorem exists_prod_multiset_X_sub_C_mul (p : R[X]) :
    ∃ q,
      (p.roots.map fun a => X - C a).prod * q = p ∧
        Multiset.card p.roots + q.natDegree = p.natDegree ∧ q.roots = 0 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    ⊢ Exists fun q => And (Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynom …
  -/
  obtain ⟨q, he⟩ := p.prod_multiset_X_sub_C_dvd
  /-
    case intro
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p q : Polynomial R
    he : Eq p (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomia …
    ⊢ Exists fun q => And (Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynom …
  -/
  use q, he.symm
  /-
    case right
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p q : Polynomial R
    he : Eq p (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomia …
    ⊢ And (Eq (HAdd.hAdd p.roots.card q.natDegree) p.natDegree) (Eq q.roots 0)
  -/
  obtain rfl | hq := eq_or_ne q 0
    /-
      case right.inl
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p : Polynomial R
      he : Eq p (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomia …
      ⊢ And (Eq (HAdd.hAdd p.roots.card (Polynomial.natDegree 0)) p.natDegree) (Eq ( …
    -/
  · rw [mul_zero] at he
    /-
      case right.inl
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p : Polynomial R
      he : Eq p 0
      ⊢ And (Eq (HAdd.hAdd p.roots.card (Polynomial.natDegree 0)) p.natDegree) (Eq ( …
    -/
    subst he
    /-
      case right.inl
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ⊢ And (Eq (HAdd.hAdd (Polynomial.roots 0).card (Polynomial.natDegree 0)) (Poly …
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case right.inr
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p q : Polynomial R
    he : Eq p (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomia …
    hq : Ne q 0
    ⊢ And (Eq (HAdd.hAdd p.roots.card q.natDegree) p.natDegree) (Eq q.roots 0)
  -/
  constructor
    /-
      case right.inr.left
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p q : Polynomial R
      he : Eq p (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomia …
      hq : Ne q 0
      ⊢ Eq (HAdd.hAdd p.roots.card q.natDegree) p.natDegree
    -/
  · conv_rhs => rw [he]
    /-
      case right.inr.left
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p q : Polynomial R
      he : Eq p (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomia …
      hq : Ne q 0
      ⊢ Eq (HAdd.hAdd p.roots.card q.natDegree) (HMul.hMul (Multiset.map (fun a => H …
    -/
    rw [monic_prod_multiset_X_sub_C.natDegree_mul' hq, natDegree_multiset_prod_X_sub_C_eq_card]
    /-
      🎉 no goals
    -/
    /-
      case right.inr.right
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p q : Polynomial R
      he : Eq p (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomia …
      hq : Ne q 0
      ⊢ Eq q.roots 0
    -/
  · replace he := congr_arg roots he.symm
    /-
      case right.inr.right
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p q : Polynomial R
      hq : Ne q 0
      he : Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial. …
      ⊢ Eq q.roots 0
    -/
    rw [roots_mul, roots_multiset_prod_X_sub_C] at he
    /-
      case right.inr.right
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p q : Polynomial R
      hq : Ne q 0
      he : Eq (HAdd.hAdd p.roots q.roots) p.roots
      ⊢ Eq q.roots 0
    -/
    exacts [add_right_eq_self.1 he, mul_ne_zero monic_prod_multiset_X_sub_C.ne_zero hq]
    /-
      🎉 no goals
    -/


/-- A polynomial `p` that has as many roots as its degree
can be written `p = p.leadingCoeff * ∏(X - a)`, for `a` in `p.roots`. -/
theorem C_leadingCoeff_mul_prod_multiset_X_sub_C (hroots : Multiset.card p.roots = p.natDegree) :
    C p.leadingCoeff * (p.roots.map fun a => X - C a).prod = p :=
  (eq_leadingCoeff_mul_of_monic_of_dvd_of_natDegree_le monic_prod_multiset_X_sub_C
      p.prod_multiset_X_sub_C_dvd
      ((natDegree_multiset_prod_X_sub_C_eq_card _).trans hroots).ge).symm


/-- A monic polynomial `p` that has as many roots as its degree
can be written `p = ∏(X - a)`, for `a` in `p.roots`. -/
theorem prod_multiset_X_sub_C_of_monic_of_roots_card_eq (hp : p.Monic)
    (hroots : Multiset.card p.roots = p.natDegree) : (p.roots.map fun a => X - C a).prod = p := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    hp : p.Monic
    hroots : Eq p.roots.card p.natDegree
    ⊢ Eq (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) p.roots) …
  -/
  convert C_leadingCoeff_mul_prod_multiset_X_sub_C hroots
  /-
    case h.e'_2
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    hp : p.Monic
    hroots : Eq p.roots.card p.natDegree
    ⊢ Eq (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) p.roots) …
  -/
  rw [hp.leadingCoeff, C_1, one_mul]
  /-
    🎉 no goals
  -/


theorem Monic.isUnit_leadingCoeff_of_dvd {a p : R[X]} (hp : Monic p) (hap : a ∣ p) :
    IsUnit a.leadingCoeff :=
                        /-
                          R : Type u
                          inst✝¹ : CommRing R
                          inst✝ : IsDomain R
                          a p : Polynomial R
                          hp : p.Monic
                          hap : Dvd.dvd a p
                          ⊢ Dvd.dvd a.leadingCoeff 1
                        -/
  isUnit_of_dvd_one (by simpa only [hp.leadingCoeff] using leadingCoeff_dvd_leadingCoeff hap)
                        /-
                          🎉 no goals
                        -/


/-- To check a monic polynomial is irreducible, it suffices to check only for
divisors that have smaller degree.

See also: `Polynomial.Monic.irreducible_iff_natDegree`.
-/
theorem Monic.irreducible_iff_degree_lt {p : R[X]} (p_monic : Monic p) (p_1 : p ≠ 1) :
    Irreducible p ↔ ∀ q, degree q ≤ ↑(p.natDegree / 2) → q ∣ p → IsUnit q := by
  simp only [p_monic.irreducible_iff_lt_natDegree_lt p_1, Finset.mem_Ioc, and_imp,
    natDegree_pos_iff_degree_pos, natDegree_le_iff_degree_le]
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    p_monic : p.Monic
    p_1 : Ne p 1
    ⊢ Iff (∀ (q : Polynomial R), q.Monic → LT.lt 0 q.degree → LE.le q.degree ↑(HDi …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p : Polynomial R
      p_monic : p.Monic
      p_1 : Ne p 1
      ⊢ (∀ (q : Polynomial R), q.Monic → LT.lt 0 q.degree → LE.le q.degree ↑(HDiv.hD …
    -/
  · rintro h q deg_le dvd
    /-
      case mp
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p : Polynomial R
      p_monic : p.Monic
      p_1 : Ne p 1
      h : ∀ (q : Polynomial R), q.Monic → LT.lt 0 q.degree → LE.le q.degree ↑(HDiv.h …
      q : Polynomial R
      deg_le : LE.le q.degree ↑(HDiv.hDiv p.natDegree 2)
      dvd : Dvd.dvd q p
      ⊢ IsUnit q
    -/
    by_contra q_unit
    /-
      case mp
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p : Polynomial R
      p_monic : p.Monic
      p_1 : Ne p 1
      h : ∀ (q : Polynomial R), q.Monic → LT.lt 0 q.degree → LE.le q.degree ↑(HDiv.h …
      q : Polynomial R
      deg_le : LE.le q.degree ↑(HDiv.hDiv p.natDegree 2)
      dvd : Dvd.dvd q p
      q_unit : Not (IsUnit q)
      ⊢ False
    -/
    have := degree_pos_of_not_isUnit_of_dvd_monic p_monic q_unit dvd
    /-
      case mp
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p : Polynomial R
      p_monic : p.Monic
      p_1 : Ne p 1
      h : ∀ (q : Polynomial R), q.Monic → LT.lt 0 q.degree → LE.le q.degree ↑(HDiv.h …
      q : Polynomial R
      deg_le : LE.le q.degree ↑(HDiv.hDiv p.natDegree 2)
      dvd : Dvd.dvd q p
      q_unit : Not (IsUnit q)
      this : LT.lt 0 q.degree
      ⊢ False
    -/
    have hu := p_monic.isUnit_leadingCoeff_of_dvd dvd
    /-
      case mp
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p : Polynomial R
      p_monic : p.Monic
      p_1 : Ne p 1
      h : ∀ (q : Polynomial R), q.Monic → LT.lt 0 q.degree → LE.le q.degree ↑(HDiv.h …
      q : Polynomial R
      deg_le : LE.le q.degree ↑(HDiv.hDiv p.natDegree 2)
      dvd : Dvd.dvd q p
      q_unit : Not (IsUnit q)
      this : LT.lt 0 q.degree
      hu : IsUnit q.leadingCoeff
      ⊢ False
    -/
    refine (h _ (monic_of_isUnit_leadingCoeff_inv_smul hu) ?_ ?_ (dvd_trans ?_ dvd)).elim
      /-
        case mp.refine_1
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        p : Polynomial R
        p_monic : p.Monic
        p_1 : Ne p 1
        h : ∀ (q : Polynomial R), q.Monic → LT.lt 0 q.degree → LE.le q.degree ↑(HDiv.h …
        q : Polynomial R
        deg_le : LE.le q.degree ↑(HDiv.hDiv p.natDegree 2)
        dvd : Dvd.dvd q p
        q_unit : Not (IsUnit q)
        this : LT.lt 0 q.degree
        hu : IsUnit q.leadingCoeff
        ⊢ LT.lt 0 (HSMul.hSMul (Inv.inv hu.unit) q).degree
      -/
    · rwa [degree_smul_of_smul_regular _ (isSMulRegular_of_group _)]
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_2
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        p : Polynomial R
        p_monic : p.Monic
        p_1 : Ne p 1
        h : ∀ (q : Polynomial R), q.Monic → LT.lt 0 q.degree → LE.le q.degree ↑(HDiv.h …
        q : Polynomial R
        deg_le : LE.le q.degree ↑(HDiv.hDiv p.natDegree 2)
        dvd : Dvd.dvd q p
        q_unit : Not (IsUnit q)
        this : LT.lt 0 q.degree
        hu : IsUnit q.leadingCoeff
        ⊢ LE.le (HSMul.hSMul (Inv.inv hu.unit) q).degree ↑(HDiv.hDiv p.natDegree 2)
      -/
    · rwa [degree_smul_of_smul_regular _ (isSMulRegular_of_group _)]
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_3
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        p : Polynomial R
        p_monic : p.Monic
        p_1 : Ne p 1
        h : ∀ (q : Polynomial R), q.Monic → LT.lt 0 q.degree → LE.le q.degree ↑(HDiv.h …
        q : Polynomial R
        deg_le : LE.le q.degree ↑(HDiv.hDiv p.natDegree 2)
        dvd : Dvd.dvd q p
        q_unit : Not (IsUnit q)
        this : LT.lt 0 q.degree
        hu : IsUnit q.leadingCoeff
        ⊢ Dvd.dvd (HSMul.hSMul (Inv.inv hu.unit) q) q
      -/
    · rw [Units.smul_def, Polynomial.smul_eq_C_mul, (isUnit_C.mpr (Units.isUnit _)).mul_left_dvd]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p : Polynomial R
      p_monic : p.Monic
      p_1 : Ne p 1
      ⊢ (∀ (q : Polynomial R), LE.le q.degree ↑(HDiv.hDiv p.natDegree 2) → Dvd.dvd q …
    -/
  · rintro h q _ deg_pos deg_le dvd
    /-
      case mpr
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p : Polynomial R
      p_monic : p.Monic
      p_1 : Ne p 1
      h : ∀ (q : Polynomial R), LE.le q.degree ↑(HDiv.hDiv p.natDegree 2) → Dvd.dvd  …
      q : Polynomial R
      a✝ : q.Monic
      deg_pos : LT.lt 0 q.degree
      deg_le : LE.le q.degree ↑(HDiv.hDiv p.natDegree 2)
      dvd : Dvd.dvd q p
      ⊢ False
    -/
    exact deg_pos.ne' <| degree_eq_zero_of_isUnit (h q deg_le dvd)
    /-
      🎉 no goals
    -/


theorem le_rootMultiplicity_map {p : A[X]} {f : A →+* B} (hmap : map f p ≠ 0) (a : A) :
    rootMultiplicity a p ≤ rootMultiplicity (f a) (p.map f) := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝¹ : CommRing A
    inst✝ : CommRing B
    p : Polynomial A
    f : RingHom A B
    hmap : Ne (Polynomial.map f p) 0
    a : A
    ⊢ LE.le (Polynomial.rootMultiplicity a p) (Polynomial.rootMultiplicity (f a) ( …
  -/
  rw [le_rootMultiplicity_iff hmap]
  /-
    A : Type u_1
    B : Type u_2
    inst✝¹ : CommRing A
    inst✝ : CommRing B
    p : Polynomial A
    f : RingHom A B
    hmap : Ne (Polynomial.map f p) 0
    a : A
    ⊢ Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C (f a))) (Polynomial …
  -/
  refine _root_.trans ?_ ((mapRingHom f).map_dvd (pow_rootMultiplicity_dvd p a))
  /-
    A : Type u_1
    B : Type u_2
    inst✝¹ : CommRing A
    inst✝ : CommRing B
    p : Polynomial A
    f : RingHom A B
    hmap : Ne (Polynomial.map f p) 0
    a : A
    ⊢ Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C (f a))) (Polynomial …
  -/
  rw [map_pow, map_sub, coe_mapRingHom, map_X, map_C]
  /-
    🎉 no goals
  -/


theorem eq_rootMultiplicity_map {p : A[X]} {f : A →+* B} (hf : Function.Injective f) (a : A) :
    rootMultiplicity a p = rootMultiplicity (f a) (p.map f) := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝¹ : CommRing A
    inst✝ : CommRing B
    p : Polynomial A
    f : RingHom A B
    hf : Function.Injective ⇑f
    a : A
    ⊢ Eq (Polynomial.rootMultiplicity a p) (Polynomial.rootMultiplicity (f a) (Pol …
  -/
  by_cases hp0 : p = 0; · simp only [hp0, rootMultiplicity_zero, Polynomial.map_zero]
                          /-
                            🎉 no goals
                          -/
  /-
    case neg
    A : Type u_1
    B : Type u_2
    inst✝¹ : CommRing A
    inst✝ : CommRing B
    p : Polynomial A
    f : RingHom A B
    hf : Function.Injective ⇑f
    a : A
    hp0 : Not (Eq p 0)
    ⊢ Eq (Polynomial.rootMultiplicity a p) (Polynomial.rootMultiplicity (f a) (Pol …
  -/
  apply le_antisymm (le_rootMultiplicity_map ((Polynomial.map_ne_zero_iff hf).mpr hp0) a)
  rw [le_rootMultiplicity_iff hp0, ← map_dvd_map f hf ((monic_X_sub_C a).pow _),
    Polynomial.map_pow, Polynomial.map_sub, map_X, map_C]
  /-
    case neg
    A : Type u_1
    B : Type u_2
    inst✝¹ : CommRing A
    inst✝ : CommRing B
    p : Polynomial A
    f : RingHom A B
    hf : Function.Injective ⇑f
    a : A
    hp0 : Not (Eq p 0)
    ⊢ Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C (f a))) (Polynomial …
  -/
  apply pow_rootMultiplicity_dvd
  /-
    🎉 no goals
  -/


theorem count_map_roots [IsDomain A] [DecidableEq B] {p : A[X]} {f : A →+* B} (hmap : map f p ≠ 0)
    (b : B) :
    (p.roots.map f).count b ≤ rootMultiplicity b (p.map f) := by
  rw [le_rootMultiplicity_iff hmap, ← Multiset.prod_replicate, ←
    Multiset.map_replicate fun a => X - C a]
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : IsDomain A
    inst✝ : DecidableEq B
    p : Polynomial A
    f : RingHom A B
    hmap : Ne (Polynomial.map f p) 0
    b : B
    ⊢ Dvd.dvd (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) (Mu …
  -/
  rw [← Multiset.filter_eq]
  refine
    (Multiset.prod_dvd_prod_of_le <| Multiset.map_le_map <| Multiset.filter_le (Eq b) _).trans ?_
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : IsDomain A
    inst✝ : DecidableEq B
    p : Polynomial A
    f : RingHom A B
    hmap : Ne (Polynomial.map f p) 0
    b : B
    ⊢ Dvd.dvd (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) (Mu …
  -/
  convert Polynomial.map_dvd f p.prod_multiset_X_sub_C_dvd
  /-
    case h.e'_3
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : IsDomain A
    inst✝ : DecidableEq B
    p : Polynomial A
    f : RingHom A B
    hmap : Ne (Polynomial.map f p) 0
    b : B
    ⊢ Eq (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) (Multise …
  -/
  simp only [Polynomial.map_multiset_prod, Multiset.map_map]
  /-
    case h.e'_3
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : IsDomain A
    inst✝ : DecidableEq B
    p : Polynomial A
    f : RingHom A B
    hmap : Ne (Polynomial.map f p) 0
    b : B
    ⊢ Eq (Multiset.map (Function.comp (fun a => HSub.hSub Polynomial.X (Polynomial …
  -/
  congr; ext1
  /-
    case h.e'_3.e_a.e_f.h
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : IsDomain A
    inst✝ : DecidableEq B
    p : Polynomial A
    f : RingHom A B
    hmap : Ne (Polynomial.map f p) 0
    b : B
    x✝ : A
    ⊢ Eq (Function.comp (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) (⇑f) x✝ …
  -/
  simp only [Function.comp_apply, Polynomial.map_sub, map_X, map_C]
  /-
    🎉 no goals
  -/


theorem count_map_roots_of_injective [IsDomain A] [DecidableEq B] (p : A[X]) {f : A →+* B}
    (hf : Function.Injective f) (b : B) :
    (p.roots.map f).count b ≤ rootMultiplicity b (p.map f) := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : IsDomain A
    inst✝ : DecidableEq B
    p : Polynomial A
    f : RingHom A B
    hf : Function.Injective ⇑f
    b : B
    ⊢ LE.le (Multiset.count b (Multiset.map (⇑f) p.roots)) (Polynomial.rootMultipl …
  -/
  by_cases hp0 : p = 0
  · simp only [hp0, roots_zero, Multiset.map_zero, Multiset.count_zero, Polynomial.map_zero,
      rootMultiplicity_zero, le_refl]
    /-
      case neg
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : IsDomain A
      inst✝ : DecidableEq B
      p : Polynomial A
      f : RingHom A B
      hf : Function.Injective ⇑f
      b : B
      hp0 : Not (Eq p 0)
      ⊢ LE.le (Multiset.count b (Multiset.map (⇑f) p.roots)) (Polynomial.rootMultipl …
    -/
  · exact count_map_roots ((Polynomial.map_ne_zero_iff hf).mpr hp0) b
    /-
      🎉 no goals
    -/


theorem map_roots_le [IsDomain A] [IsDomain B] {p : A[X]} {f : A →+* B} (h : p.map f ≠ 0) :
    p.roots.map f ≤ (p.map f).roots := by
  classical
  exact Multiset.le_iff_count.2 fun b => by
    rw [count_roots]
    apply count_map_roots h


theorem map_roots_le_of_injective [IsDomain A] [IsDomain B] (p : A[X]) {f : A →+* B}
    (hf : Function.Injective f) : p.roots.map f ≤ (p.map f).roots := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    p : Polynomial A
    f : RingHom A B
    hf : Function.Injective ⇑f
    ⊢ LE.le (Multiset.map (⇑f) p.roots) (Polynomial.map f p).roots
  -/
  by_cases hp0 : p = 0
    /-
      case pos
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : IsDomain A
      inst✝ : IsDomain B
      p : Polynomial A
      f : RingHom A B
      hf : Function.Injective ⇑f
      hp0 : Eq p 0
      ⊢ LE.le (Multiset.map (⇑f) p.roots) (Polynomial.map f p).roots
    -/
  · simp only [hp0, roots_zero, Multiset.map_zero, Polynomial.map_zero, le_rfl]
    /-
      🎉 no goals
    -/
  /-
    case neg
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    p : Polynomial A
    f : RingHom A B
    hf : Function.Injective ⇑f
    hp0 : Not (Eq p 0)
    ⊢ LE.le (Multiset.map (⇑f) p.roots) (Polynomial.map f p).roots
  -/
  exact map_roots_le ((Polynomial.map_ne_zero_iff hf).mpr hp0)
  /-
    🎉 no goals
  -/


theorem card_roots_le_map [IsDomain A] [IsDomain B] {p : A[X]} {f : A →+* B} (h : p.map f ≠ 0) :
    Multiset.card p.roots ≤ Multiset.card (p.map f).roots := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    p : Polynomial A
    f : RingHom A B
    h : Ne (Polynomial.map f p) 0
    ⊢ LE.le p.roots.card (Polynomial.map f p).roots.card
  -/
  rw [← p.roots.card_map f]
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    p : Polynomial A
    f : RingHom A B
    h : Ne (Polynomial.map f p) 0
    ⊢ LE.le (Multiset.map (⇑f) p.roots).card (Polynomial.map f p).roots.card
  -/
  exact Multiset.card_le_card (map_roots_le h)
  /-
    🎉 no goals
  -/


theorem card_roots_le_map_of_injective [IsDomain A] [IsDomain B] {p : A[X]} {f : A →+* B}
    (hf : Function.Injective f) : Multiset.card p.roots ≤ Multiset.card (p.map f).roots := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    p : Polynomial A
    f : RingHom A B
    hf : Function.Injective ⇑f
    ⊢ LE.le p.roots.card (Polynomial.map f p).roots.card
  -/
  by_cases hp0 : p = 0
    /-
      case pos
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : IsDomain A
      inst✝ : IsDomain B
      p : Polynomial A
      f : RingHom A B
      hf : Function.Injective ⇑f
      hp0 : Eq p 0
      ⊢ LE.le p.roots.card (Polynomial.map f p).roots.card
    -/
  · simp only [hp0, roots_zero, Polynomial.map_zero, Multiset.card_zero, le_rfl]
    /-
      🎉 no goals
    -/
  /-
    case neg
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    p : Polynomial A
    f : RingHom A B
    hf : Function.Injective ⇑f
    hp0 : Not (Eq p 0)
    ⊢ LE.le p.roots.card (Polynomial.map f p).roots.card
  -/
  exact card_roots_le_map ((Polynomial.map_ne_zero_iff hf).mpr hp0)
  /-
    🎉 no goals
  -/


theorem roots_map_of_injective_of_card_eq_natDegree [IsDomain A] [IsDomain B] {p : A[X]}
    {f : A →+* B} (hf : Function.Injective f) (hroots : Multiset.card p.roots = p.natDegree) :
    p.roots.map f = (p.map f).roots := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    p : Polynomial A
    f : RingHom A B
    hf : Function.Injective ⇑f
    hroots : Eq p.roots.card p.natDegree
    ⊢ Eq (Multiset.map (⇑f) p.roots) (Polynomial.map f p).roots
  -/
  apply Multiset.eq_of_le_of_card_le (map_roots_le_of_injective p hf)
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    p : Polynomial A
    f : RingHom A B
    hf : Function.Injective ⇑f
    hroots : Eq p.roots.card p.natDegree
    ⊢ LE.le (Polynomial.map f p).roots.card (Multiset.map (⇑f) p.roots).card
  -/
  simpa only [Multiset.card_map, hroots] using (card_roots' _).trans natDegree_map_le
  /-
    🎉 no goals
  -/


theorem roots_map_of_map_ne_zero_of_card_eq_natDegree [IsDomain A] [IsDomain B] {p : A[X]}
    (f : A →+* B) (h : p.map f ≠ 0) (hroots : p.roots.card = p.natDegree) :
    p.roots.map f = (p.map f).roots :=
  eq_of_le_of_card_le (map_roots_le h) <| by
    /-
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : IsDomain A
      inst✝ : IsDomain B
      p : Polynomial A
      f : RingHom A B
      h : Ne (Polynomial.map f p) 0
      hroots : Eq p.roots.card p.natDegree
      ⊢ LE.le (Polynomial.map f p).roots.card (Multiset.map (⇑f) p.roots).card
    -/
    simpa only [Multiset.card_map, hroots] using (p.map f).card_roots'.trans natDegree_map_le
    /-
      🎉 no goals
    -/


theorem Monic.roots_map_of_card_eq_natDegree [IsDomain A] [IsDomain B] {p : A[X]} (hm : p.Monic)
    (f : A →+* B) (hroots : p.roots.card = p.natDegree) : p.roots.map f  = (p.map f).roots :=
  roots_map_of_map_ne_zero_of_card_eq_natDegree f (map_monic_ne_zero hm) hroots


