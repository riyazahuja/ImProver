theorem eq_zero_of_degree_lt_of_eval_finset_eq_zero (degree_f_lt : f.degree < #s)
    (eval_f : ∀ x ∈ s, f.eval x = 0) : f = 0 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f : Polynomial R
    s : Finset R
    degree_f_lt : LT.lt f.degree ↑s.card
    eval_f : ∀ (x : R), Membership.mem s x → Eq (Polynomial.eval x f) 0
    ⊢ Eq f 0
  -/
  rw [← mem_degreeLT] at degree_f_lt
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f : Polynomial R
    s : Finset R
    degree_f_lt : Membership.mem (Polynomial.degreeLT R s.card) f
    eval_f : ∀ (x : R), Membership.mem s x → Eq (Polynomial.eval x f) 0
    ⊢ Eq f 0
  -/
  simp_rw [eval_eq_sum_degreeLTEquiv degree_f_lt] at eval_f
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f : Polynomial R
    s : Finset R
    degree_f_lt : Membership.mem (Polynomial.degreeLT R s.card) f
    eval_f : ∀ (x : R), Membership.mem s x → Eq (Finset.univ.sum fun i => HMul.hMu …
    ⊢ Eq f 0
  -/
  rw [← degreeLTEquiv_eq_zero_iff_eq_zero degree_f_lt]
  exact
    Matrix.eq_zero_of_forall_index_sum_mul_pow_eq_zero
      (Injective.comp (Embedding.subtype _).inj' (equivFinOfCardEq (card_coe _)).symm.injective)
      fun _ => eval_f _ (Finset.coe_mem _)


theorem eq_of_degree_sub_lt_of_eval_finset_eq (degree_fg_lt : (f - g).degree < #s)
    (eval_fg : ∀ x ∈ s, f.eval x = g.eval x) : f = g := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    s : Finset R
    degree_fg_lt : LT.lt (HSub.hSub f g).degree ↑s.card
    eval_fg : ∀ (x : R), Membership.mem s x → Eq (Polynomial.eval x f) (Polynomial …
    ⊢ Eq f g
  -/
  rw [← sub_eq_zero]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    s : Finset R
    degree_fg_lt : LT.lt (HSub.hSub f g).degree ↑s.card
    eval_fg : ∀ (x : R), Membership.mem s x → Eq (Polynomial.eval x f) (Polynomial …
    ⊢ Eq (HSub.hSub f g) 0
  -/
  refine eq_zero_of_degree_lt_of_eval_finset_eq_zero _ degree_fg_lt ?_
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    s : Finset R
    degree_fg_lt : LT.lt (HSub.hSub f g).degree ↑s.card
    eval_fg : ∀ (x : R), Membership.mem s x → Eq (Polynomial.eval x f) (Polynomial …
    ⊢ ∀ (x : R), Membership.mem s x → Eq (Polynomial.eval x (HSub.hSub f g)) 0
  -/
  simp_rw [eval_sub, sub_eq_zero]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    s : Finset R
    degree_fg_lt : LT.lt (HSub.hSub f g).degree ↑s.card
    eval_fg : ∀ (x : R), Membership.mem s x → Eq (Polynomial.eval x f) (Polynomial …
    ⊢ ∀ (x : R), Membership.mem s x → Eq (Polynomial.eval x f) (Polynomial.eval x g)
  -/
  exact eval_fg
  /-
    🎉 no goals
  -/


theorem eq_of_degrees_lt_of_eval_finset_eq (degree_f_lt : f.degree < #s)
    (degree_g_lt : g.degree < #s) (eval_fg : ∀ x ∈ s, f.eval x = g.eval x) : f = g := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    s : Finset R
    degree_f_lt : LT.lt f.degree ↑s.card
    degree_g_lt : LT.lt g.degree ↑s.card
    eval_fg : ∀ (x : R), Membership.mem s x → Eq (Polynomial.eval x f) (Polynomial …
    ⊢ Eq f g
  -/
  rw [← mem_degreeLT] at degree_f_lt degree_g_lt
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    s : Finset R
    degree_f_lt : Membership.mem (Polynomial.degreeLT R s.card) f
    degree_g_lt : Membership.mem (Polynomial.degreeLT R s.card) g
    eval_fg : ∀ (x : R), Membership.mem s x → Eq (Polynomial.eval x f) (Polynomial …
    ⊢ Eq f g
  -/
  refine eq_of_degree_sub_lt_of_eval_finset_eq _ ?_ eval_fg
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    s : Finset R
    degree_f_lt : Membership.mem (Polynomial.degreeLT R s.card) f
    degree_g_lt : Membership.mem (Polynomial.degreeLT R s.card) g
    eval_fg : ∀ (x : R), Membership.mem s x → Eq (Polynomial.eval x f) (Polynomial …
    ⊢ LT.lt (HSub.hSub f g).degree ↑s.card
  -/
  rw [← mem_degreeLT]; exact Submodule.sub_mem _ degree_f_lt degree_g_lt
                       /-
                         🎉 no goals
                       -/


/--
Two polynomials, with the same degree and leading coefficient, which have the same evaluation
on a set of distinct values with cardinality equal to the degree, are equal.
-/
theorem eq_of_degree_le_of_eval_finset_eq
    (h_deg_le : f.degree ≤ #s)
    (h_deg_eq : f.degree = g.degree)
    (hlc : f.leadingCoeff = g.leadingCoeff)
    (h_eval : ∀ x ∈ s, f.eval x = g.eval x) :
    f = g := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    s : Finset R
    h_deg_le : LE.le f.degree ↑s.card
    h_deg_eq : Eq f.degree g.degree
    hlc : Eq f.leadingCoeff g.leadingCoeff
    h_eval : ∀ (x : R), Membership.mem s x → Eq (Polynomial.eval x f) (Polynomial. …
    ⊢ Eq f g
  -/
  rcases eq_or_ne f 0 with rfl | hf
    /-
      case inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      g : Polynomial R
      s : Finset R
      h_deg_le : LE.le (Polynomial.degree 0) ↑s.card
      h_deg_eq : Eq (Polynomial.degree 0) g.degree
      hlc : Eq (Polynomial.leadingCoeff 0) g.leadingCoeff
      h_eval : ∀ (x : R), Membership.mem s x → Eq (Polynomial.eval x 0) (Polynomial. …
      ⊢ Eq 0 g
    -/
  · rwa [degree_zero, eq_comm, degree_eq_bot, eq_comm] at h_deg_eq
    /-
      🎉 no goals
    -/
  · exact eq_of_degree_sub_lt_of_eval_finset_eq s
      (lt_of_lt_of_le (degree_sub_lt h_deg_eq hf hlc) h_deg_le) h_eval


theorem eq_zero_of_degree_lt_of_eval_index_eq_zero (hvs : Set.InjOn v s)
    (degree_f_lt : f.degree < #s) (eval_f : ∀ i ∈ s, f.eval (v i) = 0) : f = 0 := by
  classical
    rw [← card_image_of_injOn hvs] at degree_f_lt
    refine eq_zero_of_degree_lt_of_eval_finset_eq_zero _ degree_f_lt ?_
    intro x hx
    rcases mem_image.mp hx with ⟨_, hj, rfl⟩
    exact eval_f _ hj


theorem eq_of_degree_sub_lt_of_eval_index_eq (hvs : Set.InjOn v s)
    (degree_fg_lt : (f - g).degree < #s) (eval_fg : ∀ i ∈ s, f.eval (v i) = g.eval (v i)) :
    f = g := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    ι : Type u_2
    v : ι → R
    s : Finset ι
    hvs : Set.InjOn v ↑s
    degree_fg_lt : LT.lt (HSub.hSub f g).degree ↑s.card
    eval_fg : ∀ (i : ι), Membership.mem s i → Eq (Polynomial.eval (v i) f) (Polyno …
    ⊢ Eq f g
  -/
  rw [← sub_eq_zero]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    ι : Type u_2
    v : ι → R
    s : Finset ι
    hvs : Set.InjOn v ↑s
    degree_fg_lt : LT.lt (HSub.hSub f g).degree ↑s.card
    eval_fg : ∀ (i : ι), Membership.mem s i → Eq (Polynomial.eval (v i) f) (Polyno …
    ⊢ Eq (HSub.hSub f g) 0
  -/
  refine eq_zero_of_degree_lt_of_eval_index_eq_zero _ hvs degree_fg_lt ?_
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    ι : Type u_2
    v : ι → R
    s : Finset ι
    hvs : Set.InjOn v ↑s
    degree_fg_lt : LT.lt (HSub.hSub f g).degree ↑s.card
    eval_fg : ∀ (i : ι), Membership.mem s i → Eq (Polynomial.eval (v i) f) (Polyno …
    ⊢ ∀ (i : ι), Membership.mem s i → Eq (Polynomial.eval (v i) (HSub.hSub f g)) 0
  -/
  simp_rw [eval_sub, sub_eq_zero]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    ι : Type u_2
    v : ι → R
    s : Finset ι
    hvs : Set.InjOn v ↑s
    degree_fg_lt : LT.lt (HSub.hSub f g).degree ↑s.card
    eval_fg : ∀ (i : ι), Membership.mem s i → Eq (Polynomial.eval (v i) f) (Polyno …
    ⊢ ∀ (i : ι), Membership.mem s i → Eq (Polynomial.eval (v i) f) (Polynomial.eva …
  -/
  exact eval_fg
  /-
    🎉 no goals
  -/


theorem eq_of_degrees_lt_of_eval_index_eq (hvs : Set.InjOn v s) (degree_f_lt : f.degree < #s)
    (degree_g_lt : g.degree < #s) (eval_fg : ∀ i ∈ s, f.eval (v i) = g.eval (v i)) : f = g := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    ι : Type u_2
    v : ι → R
    s : Finset ι
    hvs : Set.InjOn v ↑s
    degree_f_lt : LT.lt f.degree ↑s.card
    degree_g_lt : LT.lt g.degree ↑s.card
    eval_fg : ∀ (i : ι), Membership.mem s i → Eq (Polynomial.eval (v i) f) (Polyno …
    ⊢ Eq f g
  -/
  refine eq_of_degree_sub_lt_of_eval_index_eq _ hvs ?_ eval_fg
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    ι : Type u_2
    v : ι → R
    s : Finset ι
    hvs : Set.InjOn v ↑s
    degree_f_lt : LT.lt f.degree ↑s.card
    degree_g_lt : LT.lt g.degree ↑s.card
    eval_fg : ∀ (i : ι), Membership.mem s i → Eq (Polynomial.eval (v i) f) (Polyno …
    ⊢ LT.lt (HSub.hSub f g).degree ↑s.card
  -/
  rw [← mem_degreeLT] at degree_f_lt degree_g_lt ⊢
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    ι : Type u_2
    v : ι → R
    s : Finset ι
    hvs : Set.InjOn v ↑s
    degree_f_lt : Membership.mem (Polynomial.degreeLT R s.card) f
    degree_g_lt : Membership.mem (Polynomial.degreeLT R s.card) g
    eval_fg : ∀ (i : ι), Membership.mem s i → Eq (Polynomial.eval (v i) f) (Polyno …
    ⊢ Membership.mem (Polynomial.degreeLT R s.card) (HSub.hSub f g)
  -/
  exact Submodule.sub_mem _ degree_f_lt degree_g_lt
  /-
    🎉 no goals
  -/


theorem eq_of_degree_le_of_eval_index_eq (hvs : Set.InjOn v s)
    (h_deg_le : f.degree ≤ #s)
    (h_deg_eq : f.degree = g.degree)
    (hlc : f.leadingCoeff = g.leadingCoeff)
    (h_eval : ∀ i ∈ s, f.eval (v i) = g.eval (v i)) : f = g := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : Polynomial R
    ι : Type u_2
    v : ι → R
    s : Finset ι
    hvs : Set.InjOn v ↑s
    h_deg_le : LE.le f.degree ↑s.card
    h_deg_eq : Eq f.degree g.degree
    hlc : Eq f.leadingCoeff g.leadingCoeff
    h_eval : ∀ (i : ι), Membership.mem s i → Eq (Polynomial.eval (v i) f) (Polynom …
    ⊢ Eq f g
  -/
  rcases eq_or_ne f 0 with rfl | hf
    /-
      case inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      g : Polynomial R
      ι : Type u_2
      v : ι → R
      s : Finset ι
      hvs : Set.InjOn v ↑s
      h_deg_le : LE.le (Polynomial.degree 0) ↑s.card
      h_deg_eq : Eq (Polynomial.degree 0) g.degree
      hlc : Eq (Polynomial.leadingCoeff 0) g.leadingCoeff
      h_eval : ∀ (i : ι), Membership.mem s i → Eq (Polynomial.eval (v i) 0) (Polynom …
      ⊢ Eq 0 g
    -/
  · rwa [degree_zero, eq_comm, degree_eq_bot, eq_comm] at h_deg_eq
    /-
      🎉 no goals
    -/
  · exact eq_of_degree_sub_lt_of_eval_index_eq s hvs
      (lt_of_lt_of_le (degree_sub_lt h_deg_eq hf hlc) h_deg_le)
      h_eval


/-- `basisDivisor x y` is the unique linear or constant polynomial such that
when evaluated at `x` it gives `1` and `y` it gives `0` (where when `x = y` it is identically `0`).
Such polynomials are the building blocks for the Lagrange interpolants. -/
def basisDivisor (x y : F) : F[X] :=
  C (x - y)⁻¹ * (X - C y)


theorem basisDivisor_self : basisDivisor x x = 0 := by
  /-
    F : Type u_1
    inst✝ : Field F
    x : F
    ⊢ Eq (Lagrange.basisDivisor x x) 0
  -/
  simp only [basisDivisor, sub_self, inv_zero, map_zero, zero_mul]
  /-
    🎉 no goals
  -/


theorem basisDivisor_inj (hxy : basisDivisor x y = 0) : x = y := by
  simp_rw [basisDivisor, mul_eq_zero, X_sub_C_ne_zero, or_false, C_eq_zero, inv_eq_zero,
    sub_eq_zero] at hxy
  /-
    F : Type u_1
    inst✝ : Field F
    x y : F
    hxy : Eq x y
    ⊢ Eq x y
  -/
  exact hxy
  /-
    🎉 no goals
  -/


@[simp]
theorem basisDivisor_eq_zero_iff : basisDivisor x y = 0 ↔ x = y :=
  ⟨basisDivisor_inj, fun H => H ▸ basisDivisor_self⟩


theorem basisDivisor_ne_zero_iff : basisDivisor x y ≠ 0 ↔ x ≠ y := by
  /-
    F : Type u_1
    inst✝ : Field F
    x y : F
    ⊢ Iff (Ne (Lagrange.basisDivisor x y) 0) (Ne x y)
  -/
  rw [Ne, basisDivisor_eq_zero_iff]
  /-
    🎉 no goals
  -/


theorem degree_basisDivisor_of_ne (hxy : x ≠ y) : (basisDivisor x y).degree = 1 := by
  /-
    F : Type u_1
    inst✝ : Field F
    x y : F
    hxy : Ne x y
    ⊢ Eq (Lagrange.basisDivisor x y).degree 1
  -/
  rw [basisDivisor, degree_mul, degree_X_sub_C, degree_C, zero_add]
  /-
    F : Type u_1
    inst✝ : Field F
    x y : F
    hxy : Ne x y
    ⊢ Ne (Inv.inv (HSub.hSub x y)) 0
  -/
  exact inv_ne_zero (sub_ne_zero_of_ne hxy)
  /-
    🎉 no goals
  -/


@[simp]
theorem degree_basisDivisor_self : (basisDivisor x x).degree = ⊥ := by
  /-
    F : Type u_1
    inst✝ : Field F
    x : F
    ⊢ Eq (Lagrange.basisDivisor x x).degree Bot.bot
  -/
  rw [basisDivisor_self, degree_zero]
  /-
    🎉 no goals
  -/


theorem natDegree_basisDivisor_self : (basisDivisor x x).natDegree = 0 := by
  /-
    F : Type u_1
    inst✝ : Field F
    x : F
    ⊢ Eq (Lagrange.basisDivisor x x).natDegree 0
  -/
  rw [basisDivisor_self, natDegree_zero]
  /-
    🎉 no goals
  -/


theorem natDegree_basisDivisor_of_ne (hxy : x ≠ y) : (basisDivisor x y).natDegree = 1 :=
  natDegree_eq_of_degree_eq_some (degree_basisDivisor_of_ne hxy)


@[simp]
theorem eval_basisDivisor_right : eval y (basisDivisor x y) = 0 := by
  /-
    F : Type u_1
    inst✝ : Field F
    x y : F
    ⊢ Eq (Polynomial.eval y (Lagrange.basisDivisor x y)) 0
  -/
  simp only [basisDivisor, eval_mul, eval_C, eval_sub, eval_X, sub_self, mul_zero]
  /-
    🎉 no goals
  -/


theorem eval_basisDivisor_left_of_ne (hxy : x ≠ y) : eval x (basisDivisor x y) = 1 := by
  /-
    F : Type u_1
    inst✝ : Field F
    x y : F
    hxy : Ne x y
    ⊢ Eq (Polynomial.eval x (Lagrange.basisDivisor x y)) 1
  -/
  simp only [basisDivisor, eval_mul, eval_C, eval_sub, eval_X]
  /-
    F : Type u_1
    inst✝ : Field F
    x y : F
    hxy : Ne x y
    ⊢ Eq (HMul.hMul (Inv.inv (HSub.hSub x y)) (HSub.hSub x y)) 1
  -/
  exact inv_mul_cancel₀ (sub_ne_zero_of_ne hxy)
  /-
    🎉 no goals
  -/


/-- Lagrange basis polynomials indexed by `s : Finset ι`, defined at nodes `v i` for a
map `v : ι → F`. For `i, j ∈ s`, `basis s v i` evaluates to 0 at `v j` for `i ≠ j`. When
`v` is injective on `s`, `basis s v i` evaluates to 1 at `v i`. -/
protected def basis (s : Finset ι) (v : ι → F) (i : ι) : F[X] :=
  ∏ j ∈ s.erase i, basisDivisor (v i) (v j)


@[simp]
theorem basis_empty : Lagrange.basis ∅ v i = 1 :=
  rfl


@[simp]
theorem basis_singleton (i : ι) : Lagrange.basis {i} v i = 1 := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    v : ι → F
    i : ι
    ⊢ Eq (Lagrange.basis (Singleton.singleton i) v i) 1
  -/
  rw [Lagrange.basis, erase_singleton, prod_empty]
  /-
    🎉 no goals
  -/


@[simp]
theorem basis_pair_left (hij : i ≠ j) : Lagrange.basis {i, j} v i = basisDivisor (v i) (v j) := by
  simp only [Lagrange.basis, hij, erase_insert_eq_erase, erase_eq_of_not_mem, mem_singleton,
    not_false_iff, prod_singleton]


@[simp]
theorem basis_pair_right (hij : i ≠ j) : Lagrange.basis {i, j} v j = basisDivisor (v j) (v i) := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    v : ι → F
    i j : ι
    hij : Ne i j
    ⊢ Eq (Lagrange.basis (Insert.insert i (Singleton.singleton j)) v j) (Lagrange. …
  -/
  rw [pair_comm]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    v : ι → F
    i j : ι
    hij : Ne i j
    ⊢ Eq (Lagrange.basis (Insert.insert j (Singleton.singleton i)) v j) (Lagrange. …
  -/
  exact basis_pair_left hij.symm
  /-
    🎉 no goals
  -/


theorem basis_ne_zero (hvs : Set.InjOn v s) (hi : i ∈ s) : Lagrange.basis s v i ≠ 0 := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    ⊢ Ne (Lagrange.basis s v i) 0
  -/
  simp_rw [Lagrange.basis, prod_ne_zero_iff, Ne, mem_erase]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    ⊢ ∀ (a : ι), And (Ne a i) (Membership.mem s a) → Not (Eq (Lagrange.basisDiviso …
  -/
  rintro j ⟨hij, hj⟩
  /-
    case intro
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    j : ι
    hij : Ne j i
    hj : Membership.mem s j
    ⊢ Not (Eq (Lagrange.basisDivisor (v i) (v j)) 0)
  -/
  rw [basisDivisor_eq_zero_iff, hvs.eq_iff hi hj]
  /-
    case intro
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    j : ι
    hij : Ne j i
    hj : Membership.mem s j
    ⊢ Not (Eq i j)
  -/
  exact hij.symm
  /-
    🎉 no goals
  -/


@[simp]
theorem eval_basis_self (hvs : Set.InjOn v s) (hi : i ∈ s) :
    (Lagrange.basis s v i).eval (v i) = 1 := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    ⊢ Eq (Polynomial.eval (v i) (Lagrange.basis s v i)) 1
  -/
  rw [Lagrange.basis, eval_prod]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    ⊢ Eq ((s.erase i).prod fun j => Polynomial.eval (v i) (Lagrange.basisDivisor ( …
  -/
  refine prod_eq_one fun j H => ?_
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    j : ι
    H : Membership.mem (s.erase i) j
    ⊢ Eq (Polynomial.eval (v i) (Lagrange.basisDivisor (v i) (v j))) 1
  -/
  rw [eval_basisDivisor_left_of_ne]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    j : ι
    H : Membership.mem (s.erase i) j
    ⊢ Ne (v i) (v j)
  -/
  rcases mem_erase.mp H with ⟨hij, hj⟩
  /-
    case intro
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    j : ι
    H : Membership.mem (s.erase i) j
    hij : Ne j i
    hj : Membership.mem s j
    ⊢ Ne (v i) (v j)
  -/
  exact mt (hvs hi hj) hij.symm
  /-
    🎉 no goals
  -/


@[simp]
theorem eval_basis_of_ne (hij : i ≠ j) (hj : j ∈ s) : (Lagrange.basis s v i).eval (v j) = 0 := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i j : ι
    hij : Ne i j
    hj : Membership.mem s j
    ⊢ Eq (Polynomial.eval (v j) (Lagrange.basis s v i)) 0
  -/
  simp_rw [Lagrange.basis, eval_prod, prod_eq_zero_iff]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i j : ι
    hij : Ne i j
    hj : Membership.mem s j
    ⊢ Exists fun a => And (Membership.mem (s.erase i) a) (Eq (Polynomial.eval (v j …
  -/
  exact ⟨j, ⟨mem_erase.mpr ⟨hij.symm, hj⟩, eval_basisDivisor_right⟩⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem natDegree_basis (hvs : Set.InjOn v s) (hi : i ∈ s) :
    (Lagrange.basis s v i).natDegree = #s - 1 := by
  have H : ∀ j, j ∈ s.erase i → basisDivisor (v i) (v j) ≠ 0 := by
    simp_rw [Ne, mem_erase, basisDivisor_eq_zero_iff]
    exact fun j ⟨hij₁, hj⟩ hij₂ => hij₁ (hvs hj hi hij₂.symm)
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    H : ∀ (j : ι), Membership.mem (s.erase i) j → Ne (Lagrange.basisDivisor (v i)  …
    ⊢ Eq (Lagrange.basis s v i).natDegree (HSub.hSub s.card 1)
  -/
  rw [← card_erase_of_mem hi, card_eq_sum_ones]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    H : ∀ (j : ι), Membership.mem (s.erase i) j → Ne (Lagrange.basisDivisor (v i)  …
    ⊢ Eq (Lagrange.basis s v i).natDegree ((s.erase i).sum fun x => 1)
  -/
  convert natDegree_prod _ _ H using 1
  /-
    case h.e'_3
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    H : ∀ (j : ι), Membership.mem (s.erase i) j → Ne (Lagrange.basisDivisor (v i)  …
    ⊢ Eq ((s.erase i).sum fun x => 1) ((s.erase i).sum fun i_1 => (Lagrange.basisD …
  -/
  refine sum_congr rfl fun j hj => (natDegree_basisDivisor_of_ne ?_).symm
  /-
    case h.e'_3
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    H : ∀ (j : ι), Membership.mem (s.erase i) j → Ne (Lagrange.basisDivisor (v i)  …
    j : ι
    hj : Membership.mem (s.erase i) j
    ⊢ Ne (v i) (v j)
  -/
  rw [Ne, ← basisDivisor_eq_zero_iff]
  /-
    case h.e'_3
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    H : ∀ (j : ι), Membership.mem (s.erase i) j → Ne (Lagrange.basisDivisor (v i)  …
    j : ι
    hj : Membership.mem (s.erase i) j
    ⊢ Not (Eq (Lagrange.basisDivisor (v i) (v j)) 0)
  -/
  exact H _ hj
  /-
    🎉 no goals
  -/


theorem degree_basis (hvs : Set.InjOn v s) (hi : i ∈ s) :
    (Lagrange.basis s v i).degree = ↑(#s - 1) := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    ⊢ Eq (Lagrange.basis s v i).degree ↑(HSub.hSub s.card 1)
  -/
  rw [degree_eq_natDegree (basis_ne_zero hvs hi), natDegree_basis hvs hi]
  /-
    🎉 no goals
  -/

-- Porting note: Added `Nat.cast_withBot` rewrites

theorem sum_basis (hvs : Set.InjOn v s) (hs : s.Nonempty) :
    ∑ j ∈ s, Lagrange.basis s v j = 1 := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    hvs : Set.InjOn v ↑s
    hs : s.Nonempty
    ⊢ Eq (s.sum fun j => Lagrange.basis s v j) 1
  -/
  refine eq_of_degrees_lt_of_eval_index_eq s hvs (lt_of_le_of_lt (degree_sum_le _ _) ?_) ?_ ?_
    /-
      case refine_1
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v : ι → F
      hvs : Set.InjOn v ↑s
      hs : s.Nonempty
      ⊢ LT.lt (s.sup fun b => (Lagrange.basis s v b).degree) ↑s.card
    -/
  · rw [Nat.cast_withBot, Finset.sup_lt_iff (WithBot.bot_lt_coe #s)]
    /-
      case refine_1
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v : ι → F
      hvs : Set.InjOn v ↑s
      hs : s.Nonempty
      ⊢ ∀ (b : ι), Membership.mem s b → LT.lt (Lagrange.basis s v b).degree ↑s.card
    -/
    intro i hi
    /-
      case refine_1
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v : ι → F
      hvs : Set.InjOn v ↑s
      hs : s.Nonempty
      i : ι
      hi : Membership.mem s i
      ⊢ LT.lt (Lagrange.basis s v i).degree ↑s.card
    -/
    rw [degree_basis hvs hi, Nat.cast_withBot, WithBot.coe_lt_coe]
    /-
      case refine_1
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v : ι → F
      hvs : Set.InjOn v ↑s
      hs : s.Nonempty
      i : ι
      hi : Membership.mem s i
      ⊢ LT.lt (HSub.hSub s.card 1) s.card
    -/
    exact Nat.pred_lt (card_ne_zero_of_mem hi)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v : ι → F
      hvs : Set.InjOn v ↑s
      hs : s.Nonempty
      ⊢ LT.lt (Polynomial.degree 1) ↑s.card
    -/
  · rw [degree_one, ← WithBot.coe_zero, Nat.cast_withBot, WithBot.coe_lt_coe]
    /-
      case refine_2
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v : ι → F
      hvs : Set.InjOn v ↑s
      hs : s.Nonempty
      ⊢ LT.lt 0 s.card
    -/
    exact Nonempty.card_pos hs
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v : ι → F
      hvs : Set.InjOn v ↑s
      hs : s.Nonempty
      ⊢ ∀ (i : ι), Membership.mem s i → Eq (Polynomial.eval (v i) (s.sum fun j => La …
    -/
  · intro i hi
    rw [eval_finset_sum, eval_one, ← add_sum_erase _ _ hi, eval_basis_self hvs hi,
      add_right_eq_self]
    /-
      case refine_3
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v : ι → F
      hvs : Set.InjOn v ↑s
      hs : s.Nonempty
      i : ι
      hi : Membership.mem s i
      ⊢ Eq ((s.erase i).sum fun x => Polynomial.eval (v i) (Lagrange.basis s v x)) 0
    -/
    refine sum_eq_zero fun j hj => ?_
    /-
      case refine_3
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v : ι → F
      hvs : Set.InjOn v ↑s
      hs : s.Nonempty
      i : ι
      hi : Membership.mem s i
      j : ι
      hj : Membership.mem (s.erase i) j
      ⊢ Eq (Polynomial.eval (v i) (Lagrange.basis s v j)) 0
    -/
    rcases mem_erase.mp hj with ⟨hij, _⟩
    /-
      case refine_3.intro
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v : ι → F
      hvs : Set.InjOn v ↑s
      hs : s.Nonempty
      i : ι
      hi : Membership.mem s i
      j : ι
      hj : Membership.mem (s.erase i) j
      hij : Ne j i
      right✝ : Membership.mem s j
      ⊢ Eq (Polynomial.eval (v i) (Lagrange.basis s v j)) 0
    -/
    rw [eval_basis_of_ne hij hi]
    /-
      🎉 no goals
    -/


theorem basisDivisor_add_symm {x y : F} (hxy : x ≠ y) :
    basisDivisor x y + basisDivisor y x = 1 := by
  classical
  rw [← sum_basis Function.injective_id.injOn ⟨x, mem_insert_self _ {y}⟩,
    sum_insert (not_mem_singleton.mpr hxy), sum_singleton, basis_pair_left hxy,
    basis_pair_right hxy, id, id]


/-- Lagrange interpolation: given a finset `s : Finset ι`, a nodal map `v : ι → F` injective on
`s` and a value function `r : ι → F`, `interpolate s v r` is the unique
polynomial of degree `< #s` that takes value `r i` on `v i` for all `i` in `s`. -/
@[simps]
def interpolate (s : Finset ι) (v : ι → F) : (ι → F) →ₗ[F] F[X] where
  toFun r := ∑ i ∈ s, C (r i) * Lagrange.basis s v i
  map_add' f g := by
    /-
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s✝ t : Finset ι
      i j : ι
      v✝ r r' : ι → F
      s : Finset ι
      v f g : ι → F
      ⊢ Eq ((fun r => s.sum fun i => HMul.hMul (Polynomial.C (r i)) (Lagrange.basis  …
    -/
    simp_rw [← Finset.sum_add_distrib]
    have h : (fun x => C (f x) * Lagrange.basis s v x + C (g x) * Lagrange.basis s v x) =
    (fun x => C ((f + g) x) * Lagrange.basis s v x) := by
      simp_rw [← add_mul, ← C_add, Pi.add_apply]
    /-
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s✝ t : Finset ι
      i j : ι
      v✝ r r' : ι → F
      s : Finset ι
      v f g : ι → F
      h : Eq (fun x => HAdd.hAdd (HMul.hMul (Polynomial.C (f x)) (Lagrange.basis s v …
      ⊢ Eq (s.sum fun i => HMul.hMul (Polynomial.C (HAdd.hAdd f g i)) (Lagrange.basi …
    -/
    rw [h]
    /-
      🎉 no goals
    -/
  map_smul' c f := by
    /-
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s✝ t : Finset ι
      i j : ι
      v✝ r r' : ι → F
      s : Finset ι
      v : ι → F
      c : F
      f : ι → F
      ⊢ Eq ({ toFun := fun r => s.sum fun i => HMul.hMul (Polynomial.C (r i)) (Lagra …
    -/
    simp_rw [Finset.smul_sum, C_mul', smul_smul, Pi.smul_apply, RingHom.id_apply, smul_eq_mul]
    /-
      🎉 no goals
    -/


                                                        /-
                                                          F : Type u_1
                                                          inst✝¹ : Field F
                                                          ι : Type u_2
                                                          inst✝ : DecidableEq ι
                                                          v r : ι → F
                                                          ⊢ Eq ((Lagrange.interpolate EmptyCollection.emptyCollection v) r) 0
                                                        -/
theorem interpolate_empty : interpolate ∅ v r = 0 := by rw [interpolate_apply, sum_empty]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem interpolate_singleton : interpolate {i} v r = C (r i) := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    i : ι
    v r : ι → F
    ⊢ Eq ((Lagrange.interpolate (Singleton.singleton i) v) r) (Polynomial.C (r i))
  -/
  rw [interpolate_apply, sum_singleton, basis_singleton, mul_one]
  /-
    🎉 no goals
  -/


theorem interpolate_one (hvs : Set.InjOn v s) (hs : s.Nonempty) : interpolate s v 1 = 1 := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    hvs : Set.InjOn v ↑s
    hs : s.Nonempty
    ⊢ Eq ((Lagrange.interpolate s v) 1) 1
  -/
  simp_rw [interpolate_apply, Pi.one_apply, map_one, one_mul]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    hvs : Set.InjOn v ↑s
    hs : s.Nonempty
    ⊢ Eq (s.sum fun x => Lagrange.basis s v x) 1
  -/
  exact sum_basis hvs hs
  /-
    🎉 no goals
  -/


theorem eval_interpolate_at_node (hvs : Set.InjOn v s) (hi : i ∈ s) :
    eval (v i) (interpolate s v r) = r i := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    i : ι
    v r : ι → F
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    ⊢ Eq (Polynomial.eval (v i) ((Lagrange.interpolate s v) r)) (r i)
  -/
  rw [interpolate_apply, eval_finset_sum, ← add_sum_erase _ _ hi]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    i : ι
    v r : ι → F
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    ⊢ Eq (HAdd.hAdd (Polynomial.eval (v i) (HMul.hMul (Polynomial.C (r i)) (Lagran …
  -/
  simp_rw [eval_mul, eval_C, eval_basis_self hvs hi, mul_one, add_right_eq_self]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    i : ι
    v r : ι → F
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    ⊢ Eq ((s.erase i).sum fun x => HMul.hMul (r x) (Polynomial.eval (v i) (Lagrang …
  -/
  refine sum_eq_zero fun j H => ?_
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    i : ι
    v r : ι → F
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    j : ι
    H : Membership.mem (s.erase i) j
    ⊢ Eq (HMul.hMul (r j) (Polynomial.eval (v i) (Lagrange.basis s v j))) 0
  -/
  rw [eval_basis_of_ne (mem_erase.mp H).1 hi, mul_zero]
  /-
    🎉 no goals
  -/


theorem degree_interpolate_le (hvs : Set.InjOn v s) :
    (interpolate s v r).degree ≤ ↑(#s - 1) := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v r : ι → F
    hvs : Set.InjOn v ↑s
    ⊢ LE.le ((Lagrange.interpolate s v) r).degree ↑(HSub.hSub s.card 1)
  -/
  refine (degree_sum_le _ _).trans ?_
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v r : ι → F
    hvs : Set.InjOn v ↑s
    ⊢ LE.le (s.sup fun b => (HMul.hMul (Polynomial.C (r b)) (Lagrange.basis s v b) …
  -/
  rw [Finset.sup_le_iff]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v r : ι → F
    hvs : Set.InjOn v ↑s
    ⊢ ∀ (b : ι), Membership.mem s b → LE.le (HMul.hMul (Polynomial.C (r b)) (Lagra …
  -/
  intro i hi
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v r : ι → F
    hvs : Set.InjOn v ↑s
    i : ι
    hi : Membership.mem s i
    ⊢ LE.le (HMul.hMul (Polynomial.C (r i)) (Lagrange.basis s v i)).degree ↑(HSub. …
  -/
  rw [degree_mul, degree_basis hvs hi]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v r : ι → F
    hvs : Set.InjOn v ↑s
    i : ι
    hi : Membership.mem s i
    ⊢ LE.le (HAdd.hAdd (Polynomial.C (r i)).degree ↑(HSub.hSub s.card 1)) ↑(HSub.h …
  -/
  by_cases hr : r i = 0
    /-
      case pos
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v r : ι → F
      hvs : Set.InjOn v ↑s
      i : ι
      hi : Membership.mem s i
      hr : Eq (r i) 0
      ⊢ LE.le (HAdd.hAdd (Polynomial.C (r i)).degree ↑(HSub.hSub s.card 1)) ↑(HSub.h …
    -/
  · simpa only [hr, map_zero, degree_zero, WithBot.bot_add] using bot_le
    /-
      🎉 no goals
    -/
    /-
      case neg
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v r : ι → F
      hvs : Set.InjOn v ↑s
      i : ι
      hi : Membership.mem s i
      hr : Not (Eq (r i) 0)
      ⊢ LE.le (HAdd.hAdd (Polynomial.C (r i)).degree ↑(HSub.hSub s.card 1)) ↑(HSub.h …
    -/
  · rw [degree_C hr, zero_add]
    /-
      🎉 no goals
    -/

-- Porting note: Added `Nat.cast_withBot` rewrites

theorem degree_interpolate_lt (hvs : Set.InjOn v s) : (interpolate s v r).degree < #s := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v r : ι → F
    hvs : Set.InjOn v ↑s
    ⊢ LT.lt ((Lagrange.interpolate s v) r).degree ↑s.card
  -/
  rw [Nat.cast_withBot]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v r : ι → F
    hvs : Set.InjOn v ↑s
    ⊢ LT.lt ((Lagrange.interpolate s v) r).degree ↑s.card
  -/
  rcases eq_empty_or_nonempty s with (rfl | h)
    /-
      case inl
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      v r : ι → F
      hvs : Set.InjOn v ↑EmptyCollection.emptyCollection
      ⊢ LT.lt ((Lagrange.interpolate EmptyCollection.emptyCollection v) r).degree ↑E …
    -/
  · rw [interpolate_empty, degree_zero, card_empty]
    /-
      case inl
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      v r : ι → F
      hvs : Set.InjOn v ↑EmptyCollection.emptyCollection
      ⊢ LT.lt Bot.bot ↑0
    -/
    exact WithBot.bot_lt_coe _
    /-
      🎉 no goals
    -/
    /-
      case inr
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v r : ι → F
      hvs : Set.InjOn v ↑s
      h : s.Nonempty
      ⊢ LT.lt ((Lagrange.interpolate s v) r).degree ↑s.card
    -/
  · refine lt_of_le_of_lt (degree_interpolate_le _ hvs) ?_
    /-
      case inr
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v r : ι → F
      hvs : Set.InjOn v ↑s
      h : s.Nonempty
      ⊢ LT.lt ↑(HSub.hSub s.card 1) ↑s.card
    -/
    rw [Nat.cast_withBot, WithBot.coe_lt_coe]
    /-
      case inr
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v r : ι → F
      hvs : Set.InjOn v ↑s
      h : s.Nonempty
      ⊢ LT.lt (HSub.hSub s.card 1) s.card
    -/
    exact Nat.sub_lt (Nonempty.card_pos h) zero_lt_one
    /-
      🎉 no goals
    -/


theorem degree_interpolate_erase_lt (hvs : Set.InjOn v s) (hi : i ∈ s) :
    (interpolate (s.erase i) v r).degree < ↑(#s - 1) := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    i : ι
    v r : ι → F
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    ⊢ LT.lt ((Lagrange.interpolate (s.erase i) v) r).degree ↑(HSub.hSub s.card 1)
  -/
  rw [← Finset.card_erase_of_mem hi]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    i : ι
    v r : ι → F
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    ⊢ LT.lt ((Lagrange.interpolate (s.erase i) v) r).degree ↑(s.erase i).card
  -/
  exact degree_interpolate_lt _ (Set.InjOn.mono (coe_subset.mpr (erase_subset _ _)) hvs)
  /-
    🎉 no goals
  -/


theorem values_eq_on_of_interpolate_eq (hvs : Set.InjOn v s)
    (hrr' : interpolate s v r = interpolate s v r') : ∀ i ∈ s, r i = r' i := fun _ hi => by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v r r' : ι → F
    hvs : Set.InjOn v ↑s
    hrr' : Eq ((Lagrange.interpolate s v) r) ((Lagrange.interpolate s v) r')
    x✝ : ι
    hi : Membership.mem s x✝
    ⊢ Eq (r x✝) (r' x✝)
  -/
  rw [← eval_interpolate_at_node r hvs hi, hrr', eval_interpolate_at_node r' hvs hi]
  /-
    🎉 no goals
  -/


theorem interpolate_eq_of_values_eq_on (hrr' : ∀ i ∈ s, r i = r' i) :
    interpolate s v r = interpolate s v r' :=
                               /-
                                 F : Type u_1
                                 inst✝¹ : Field F
                                 ι : Type u_2
                                 inst✝ : DecidableEq ι
                                 s : Finset ι
                                 v r r' : ι → F
                                 hrr' : ∀ (i : ι), Membership.mem s i → Eq (r i) (r' i)
                                 i : ι
                                 hi : Membership.mem s i
                                 ⊢ Eq (HMul.hMul (Polynomial.C (r i)) (Lagrange.basis s v i)) (HMul.hMul (Polyn …
                               -/
  sum_congr rfl fun i hi => by rw [hrr' _ hi]
                               /-
                                 🎉 no goals
                               -/


theorem interpolate_eq_iff_values_eq_on (hvs : Set.InjOn v s) :
    interpolate s v r = interpolate s v r' ↔ ∀ i ∈ s, r i = r' i :=
  ⟨values_eq_on_of_interpolate_eq _ _ hvs, interpolate_eq_of_values_eq_on _ _⟩


theorem eq_interpolate {f : F[X]} (hvs : Set.InjOn v s) (degree_f_lt : f.degree < #s) :
    f = interpolate s v fun i => f.eval (v i) :=
  eq_of_degrees_lt_of_eval_index_eq _ hvs degree_f_lt (degree_interpolate_lt _ hvs) fun _ hi =>
    (eval_interpolate_at_node (fun x ↦ eval (v x) f) hvs hi).symm


theorem eq_interpolate_of_eval_eq {f : F[X]} (hvs : Set.InjOn v s) (degree_f_lt : f.degree < #s)
    (eval_f : ∀ i ∈ s, f.eval (v i) = r i) : f = interpolate s v r := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v r : ι → F
    f : Polynomial F
    hvs : Set.InjOn v ↑s
    degree_f_lt : LT.lt f.degree ↑s.card
    eval_f : ∀ (i : ι), Membership.mem s i → Eq (Polynomial.eval (v i) f) (r i)
    ⊢ Eq f ((Lagrange.interpolate s v) r)
  -/
  rw [eq_interpolate hvs degree_f_lt]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v r : ι → F
    f : Polynomial F
    hvs : Set.InjOn v ↑s
    degree_f_lt : LT.lt f.degree ↑s.card
    eval_f : ∀ (i : ι), Membership.mem s i → Eq (Polynomial.eval (v i) f) (r i)
    ⊢ Eq ((Lagrange.interpolate s v) fun i => Polynomial.eval (v i) f) ((Lagrange. …
  -/
  exact interpolate_eq_of_values_eq_on _ _ eval_f
  /-
    🎉 no goals
  -/


/-- This is the characteristic property of the interpolation: the interpolation is the
unique polynomial of `degree < Fintype.card ι` which takes the value of the `r i` on the `v i`.
-/
theorem eq_interpolate_iff {f : F[X]} (hvs : Set.InjOn v s) :
    (f.degree < #s ∧ ∀ i ∈ s, eval (v i) f = r i) ↔ f = interpolate s v r := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v r : ι → F
    f : Polynomial F
    hvs : Set.InjOn v ↑s
    ⊢ Iff (And (LT.lt f.degree ↑s.card) (∀ (i : ι), Membership.mem s i → Eq (Polyn …
  -/
  constructor <;> intro h
    /-
      case mp
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v r : ι → F
      f : Polynomial F
      hvs : Set.InjOn v ↑s
      h : And (LT.lt f.degree ↑s.card) (∀ (i : ι), Membership.mem s i → Eq (Polynomi …
      ⊢ Eq f ((Lagrange.interpolate s v) r)
    -/
  · exact eq_interpolate_of_eval_eq _ hvs h.1 h.2
    /-
      🎉 no goals
    -/
    /-
      case mpr
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v r : ι → F
      f : Polynomial F
      hvs : Set.InjOn v ↑s
      h : Eq f ((Lagrange.interpolate s v) r)
      ⊢ And (LT.lt f.degree ↑s.card) (∀ (i : ι), Membership.mem s i → Eq (Polynomial …
    -/
  · rw [h]
    /-
      case mpr
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s : Finset ι
      v r : ι → F
      f : Polynomial F
      hvs : Set.InjOn v ↑s
      h : Eq f ((Lagrange.interpolate s v) r)
      ⊢ And (LT.lt ((Lagrange.interpolate s v) r).degree ↑s.card) (∀ (i : ι), Member …
    -/
    exact ⟨degree_interpolate_lt _ hvs, fun _ hi => eval_interpolate_at_node _ hvs hi⟩
    /-
      🎉 no goals
    -/


/-- Lagrange interpolation induces isomorphism between functions from `s`
and polynomials of degree less than `Fintype.card ι`. -/
def funEquivDegreeLT (hvs : Set.InjOn v s) : degreeLT F #s ≃ₗ[F] s → F where
  toFun f i := f.1.eval (v i)
  map_add' _ _ := funext fun _ => eval_add
                                /-
                                  F : Type u_1
                                  inst✝¹ : Field F
                                  ι : Type u_2
                                  inst✝ : DecidableEq ι
                                  s t : Finset ι
                                  i j : ι
                                  v r r' : ι → F
                                  hvs : Set.InjOn v ↑s
                                  c : F
                                  f : Subtype fun x => Membership.mem (Polynomial.degreeLT F s.card) x
                                  ⊢ ∀ (x : Subtype fun x => Membership.mem s x), Eq ({ toFun := fun f i => Polyn …
                                -/
  map_smul' c f := funext <| by simp
                                /-
                                  🎉 no goals
                                -/
  invFun r :=
    ⟨interpolate s v fun x => if hx : x ∈ s then r ⟨x, hx⟩ else 0,
      mem_degreeLT.2 <| degree_interpolate_lt _ hvs⟩
  left_inv := by
    /-
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      i j : ι
      v r r' : ι → F
      hvs : Set.InjOn v ↑s
      ⊢ Function.LeftInverse (fun r => ⟨(Lagrange.interpolate s v) fun x => dite (Me …
    -/
    rintro ⟨f, hf⟩
    /-
      case mk
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      i j : ι
      v r r' : ι → F
      hvs : Set.InjOn v ↑s
      f : Polynomial F
      hf : Membership.mem (Polynomial.degreeLT F s.card) f
      ⊢ Eq ((fun r => ⟨(Lagrange.interpolate s v) fun x => dite (Membership.mem s x) …
    -/
    simp only [Subtype.mk_eq_mk, Subtype.coe_mk, dite_eq_ite]
    /-
      case mk
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      i j : ι
      v r r' : ι → F
      hvs : Set.InjOn v ↑s
      f : Polynomial F
      hf : Membership.mem (Polynomial.degreeLT F s.card) f
      ⊢ Eq ((Lagrange.interpolate s v) fun x => ite (Membership.mem s x) (Polynomial …
    -/
    rw [mem_degreeLT] at hf
    /-
      case mk
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      i j : ι
      v r r' : ι → F
      hvs : Set.InjOn v ↑s
      f : Polynomial F
      hf : LT.lt f.degree ↑s.card
      ⊢ Eq ((Lagrange.interpolate s v) fun x => ite (Membership.mem s x) (Polynomial …
    -/
    conv => rhs; rw [eq_interpolate hvs hf]
    /-
      case mk
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      i j : ι
      v r r' : ι → F
      hvs : Set.InjOn v ↑s
      f : Polynomial F
      hf : LT.lt f.degree ↑s.card
      ⊢ Eq ((Lagrange.interpolate s v) fun x => ite (Membership.mem s x) (Polynomial …
    -/
    exact interpolate_eq_of_values_eq_on _ _ fun _ hi => if_pos hi
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      i j : ι
      v r r' : ι → F
      hvs : Set.InjOn v ↑s
      ⊢ Function.RightInverse (fun r => ⟨(Lagrange.interpolate s v) fun x => dite (M …
    -/
    intro f
    /-
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      i j : ι
      v r r' : ι → F
      hvs : Set.InjOn v ↑s
      f : (Subtype fun x => Membership.mem s x) → F
      ⊢ Eq ({ toFun := fun f i => Polynomial.eval (v ↑i) ↑f, map_add' := ⋯, map_smul …
    -/
    ext ⟨i, hi⟩
    /-
      case h.mk
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      i✝ j : ι
      v r r' : ι → F
      hvs : Set.InjOn v ↑s
      f : (Subtype fun x => Membership.mem s x) → F
      i : ι
      hi : Membership.mem s i
      ⊢ Eq ({ toFun := fun f i => Polynomial.eval (v ↑i) ↑f, map_add' := ⋯, map_smul …
    -/
    simp only [Subtype.coe_mk, eval_interpolate_at_node _ hvs hi]
    /-
      case h.mk
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      i✝ j : ι
      v r r' : ι → F
      hvs : Set.InjOn v ↑s
      f : (Subtype fun x => Membership.mem s x) → F
      i : ι
      hi : Membership.mem s i
      ⊢ Eq (dite (Membership.mem s i) (fun hx => f ⟨i, hx⟩) fun hx => 0) (f ⟨i, hi⟩)
    -/
    exact dif_pos hi
    /-
      🎉 no goals
    -/

-- Porting note: Added `Nat.cast_withBot` rewrites

theorem interpolate_eq_sum_interpolate_insert_sdiff (hvt : Set.InjOn v t) (hs : s.Nonempty)
    (hst : s ⊆ t) :
    interpolate t v r = ∑ i ∈ s, interpolate (insert i (t \ s)) v r * Lagrange.basis s v i := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s t : Finset ι
    v r : ι → F
    hvt : Set.InjOn v ↑t
    hs : s.Nonempty
    hst : HasSubset.Subset s t
    ⊢ Eq ((Lagrange.interpolate t v) r) (s.sum fun i => HMul.hMul ((Lagrange.inter …
  -/
  symm
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s t : Finset ι
    v r : ι → F
    hvt : Set.InjOn v ↑t
    hs : s.Nonempty
    hst : HasSubset.Subset s t
    ⊢ Eq (s.sum fun i => HMul.hMul ((Lagrange.interpolate (Insert.insert i (SDiff. …
  -/
  refine eq_interpolate_of_eval_eq _ hvt (lt_of_le_of_lt (degree_sum_le _ _) ?_) fun i hi => ?_
    /-
      case refine_1
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      v r : ι → F
      hvt : Set.InjOn v ↑t
      hs : s.Nonempty
      hst : HasSubset.Subset s t
      ⊢ LT.lt (s.sup fun b => (HMul.hMul ((Lagrange.interpolate (Insert.insert b (SD …
    -/
  · simp_rw [Nat.cast_withBot, Finset.sup_lt_iff (WithBot.bot_lt_coe #t), degree_mul]
    /-
      case refine_1
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      v r : ι → F
      hvt : Set.InjOn v ↑t
      hs : s.Nonempty
      hst : HasSubset.Subset s t
      ⊢ ∀ (b : ι), Membership.mem s b → LT.lt (HAdd.hAdd ((Lagrange.interpolate (Ins …
    -/
    intro i hi
    /-
      case refine_1
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      v r : ι → F
      hvt : Set.InjOn v ↑t
      hs : s.Nonempty
      hst : HasSubset.Subset s t
      i : ι
      hi : Membership.mem s i
      ⊢ LT.lt (HAdd.hAdd ((Lagrange.interpolate (Insert.insert i (SDiff.sdiff t s))  …
    -/
    have hs : 1 ≤ #s := Nonempty.card_pos ⟨_, hi⟩
    /-
      case refine_1
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      v r : ι → F
      hvt : Set.InjOn v ↑t
      hs✝ : s.Nonempty
      hst : HasSubset.Subset s t
      i : ι
      hi : Membership.mem s i
      hs : LE.le 1 s.card
      ⊢ LT.lt (HAdd.hAdd ((Lagrange.interpolate (Insert.insert i (SDiff.sdiff t s))  …
    -/
    have hst' : #s ≤ #t := card_le_card hst
    have H : #t = 1 + (#t - #s) + (#s - 1) := by
      rw [add_assoc, tsub_add_tsub_cancel hst' hs, ← add_tsub_assoc_of_le (hs.trans hst'),
        Nat.succ_add_sub_one, zero_add]
    rw [degree_basis (Set.InjOn.mono hst hvt) hi, H, WithBot.coe_add, Nat.cast_withBot,
      WithBot.add_lt_add_iff_right (@WithBot.coe_ne_bot _ (#s - 1))]
    convert degree_interpolate_lt _
        (hvt.mono (coe_subset.mpr (insert_subset_iff.mpr ⟨hst hi, sdiff_subset⟩)))
    /-
      case h.e'_4.h.e'_1
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      v r : ι → F
      hvt : Set.InjOn v ↑t
      hs✝ : s.Nonempty
      hst : HasSubset.Subset s t
      i : ι
      hi : Membership.mem s i
      hs : LE.le 1 s.card
      hst' : LE.le s.card t.card
      H : Eq t.card (HAdd.hAdd (HAdd.hAdd 1 (HSub.hSub t.card s.card)) (HSub.hSub s. …
      ⊢ Eq (HAdd.hAdd 1 (HSub.hSub t.card s.card)) (Insert.insert i (SDiff.sdiff t s …
    -/
    rw [card_insert_of_not_mem (not_mem_sdiff_of_mem_right hi), card_sdiff hst, add_comm]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      v r : ι → F
      hvt : Set.InjOn v ↑t
      hs : s.Nonempty
      hst : HasSubset.Subset s t
      i : ι
      hi : Membership.mem t i
      ⊢ Eq (Polynomial.eval (v i) (s.sum fun i => HMul.hMul ((Lagrange.interpolate ( …
    -/
  · simp_rw [eval_finset_sum, eval_mul]
    /-
      case refine_2
      F : Type u_1
      inst✝¹ : Field F
      ι : Type u_2
      inst✝ : DecidableEq ι
      s t : Finset ι
      v r : ι → F
      hvt : Set.InjOn v ↑t
      hs : s.Nonempty
      hst : HasSubset.Subset s t
      i : ι
      hi : Membership.mem t i
      ⊢ Eq (s.sum fun x => HMul.hMul (Polynomial.eval (v i) ((Lagrange.interpolate ( …
    -/
    by_cases hi' : i ∈ s
    · rw [← add_sum_erase _ _ hi', eval_basis_self (hvt.mono hst) hi',
        eval_interpolate_at_node _
          (hvt.mono (coe_subset.mpr (insert_subset_iff.mpr ⟨hi, sdiff_subset⟩)))
          (mem_insert_self _ _),
        mul_one, add_right_eq_self]
      /-
        case pos
        F : Type u_1
        inst✝¹ : Field F
        ι : Type u_2
        inst✝ : DecidableEq ι
        s t : Finset ι
        v r : ι → F
        hvt : Set.InjOn v ↑t
        hs : s.Nonempty
        hst : HasSubset.Subset s t
        i : ι
        hi : Membership.mem t i
        hi' : Membership.mem s i
        ⊢ Eq ((s.erase i).sum fun x => HMul.hMul (Polynomial.eval (v i) ((Lagrange.int …
      -/
      refine sum_eq_zero fun j hj => ?_
      /-
        case pos
        F : Type u_1
        inst✝¹ : Field F
        ι : Type u_2
        inst✝ : DecidableEq ι
        s t : Finset ι
        v r : ι → F
        hvt : Set.InjOn v ↑t
        hs : s.Nonempty
        hst : HasSubset.Subset s t
        i : ι
        hi : Membership.mem t i
        hi' : Membership.mem s i
        j : ι
        hj : Membership.mem (s.erase i) j
        ⊢ Eq (HMul.hMul (Polynomial.eval (v i) ((Lagrange.interpolate (Insert.insert j …
      -/
      rcases mem_erase.mp hj with ⟨hij, _⟩
      /-
        case pos.intro
        F : Type u_1
        inst✝¹ : Field F
        ι : Type u_2
        inst✝ : DecidableEq ι
        s t : Finset ι
        v r : ι → F
        hvt : Set.InjOn v ↑t
        hs : s.Nonempty
        hst : HasSubset.Subset s t
        i : ι
        hi : Membership.mem t i
        hi' : Membership.mem s i
        j : ι
        hj : Membership.mem (s.erase i) j
        hij : Ne j i
        right✝ : Membership.mem s j
        ⊢ Eq (HMul.hMul (Polynomial.eval (v i) ((Lagrange.interpolate (Insert.insert j …
      -/
      rw [eval_basis_of_ne hij hi', mul_zero]
      /-
        🎉 no goals
      -/
    · have H : (∑ j ∈ s, eval (v i) (Lagrange.basis s v j)) = 1 := by
        rw [← eval_finset_sum, sum_basis (hvt.mono hst) hs, eval_one]
      /-
        case neg
        F : Type u_1
        inst✝¹ : Field F
        ι : Type u_2
        inst✝ : DecidableEq ι
        s t : Finset ι
        v r : ι → F
        hvt : Set.InjOn v ↑t
        hs : s.Nonempty
        hst : HasSubset.Subset s t
        i : ι
        hi : Membership.mem t i
        hi' : Not (Membership.mem s i)
        H : Eq (s.sum fun j => Polynomial.eval (v i) (Lagrange.basis s v j)) 1
        ⊢ Eq (s.sum fun x => HMul.hMul (Polynomial.eval (v i) ((Lagrange.interpolate ( …
      -/
      rw [← mul_one (r i), ← H, mul_sum]
      /-
        case neg
        F : Type u_1
        inst✝¹ : Field F
        ι : Type u_2
        inst✝ : DecidableEq ι
        s t : Finset ι
        v r : ι → F
        hvt : Set.InjOn v ↑t
        hs : s.Nonempty
        hst : HasSubset.Subset s t
        i : ι
        hi : Membership.mem t i
        hi' : Not (Membership.mem s i)
        H : Eq (s.sum fun j => Polynomial.eval (v i) (Lagrange.basis s v j)) 1
        ⊢ Eq (s.sum fun x => HMul.hMul (Polynomial.eval (v i) ((Lagrange.interpolate ( …
      -/
      refine sum_congr rfl fun j hj => ?_
      /-
        case neg
        F : Type u_1
        inst✝¹ : Field F
        ι : Type u_2
        inst✝ : DecidableEq ι
        s t : Finset ι
        v r : ι → F
        hvt : Set.InjOn v ↑t
        hs : s.Nonempty
        hst : HasSubset.Subset s t
        i : ι
        hi : Membership.mem t i
        hi' : Not (Membership.mem s i)
        H : Eq (s.sum fun j => Polynomial.eval (v i) (Lagrange.basis s v j)) 1
        j : ι
        hj : Membership.mem s j
        ⊢ Eq (HMul.hMul (Polynomial.eval (v i) ((Lagrange.interpolate (Insert.insert j …
      -/
      congr
      exact
        eval_interpolate_at_node _ (hvt.mono (insert_subset_iff.mpr ⟨hst hj, sdiff_subset⟩))
          (mem_insert.mpr (Or.inr (mem_sdiff.mpr ⟨hi, hi'⟩)))


theorem interpolate_eq_add_interpolate_erase (hvs : Set.InjOn v s) (hi : i ∈ s) (hj : j ∈ s)
    (hij : i ≠ j) :
    interpolate s v r =
      interpolate (s.erase j) v r * basisDivisor (v i) (v j) +
        interpolate (s.erase i) v r * basisDivisor (v j) (v i) := by
  rw [interpolate_eq_sum_interpolate_insert_sdiff _ hvs ⟨i, mem_insert_self i {j}⟩ _,
    sum_insert (not_mem_singleton.mpr hij), sum_singleton, basis_pair_left hij,
    basis_pair_right hij, sdiff_insert_insert_of_mem_of_not_mem hi (not_mem_singleton.mpr hij),
    sdiff_singleton_eq_erase, pair_comm,
    sdiff_insert_insert_of_mem_of_not_mem hj (not_mem_singleton.mpr hij.symm),
    sdiff_singleton_eq_erase]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    i j : ι
    v r : ι → F
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    hj : Membership.mem s j
    hij : Ne i j
    ⊢ HasSubset.Subset (Insert.insert i (Singleton.singleton j)) s
  -/
  exact insert_subset_iff.mpr ⟨hi, singleton_subset_iff.mpr hj⟩
  /-
    🎉 no goals
  -/


/-- `nodal s v` is the unique monic polynomial whose roots are the nodes defined by `v` and `s`.

That is, the roots of `nodal s v` are exactly the image of `v` on `s`,
with appropriate multiplicity.

We can use `nodal` to define the barycentric forms of the evaluated interpolant.
-/

def nodal (s : Finset ι) (v : ι → R) : R[X] :=
  ∏ i ∈ s, (X - C (v i))


theorem nodal_eq (s : Finset ι) (v : ι → R) : nodal s v = ∏ i ∈ s, (X - C (v i)) :=
  rfl


@[simp]
theorem nodal_empty : nodal ∅ v = 1 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ι : Type u_2
    v : ι → R
    ⊢ Eq (Lagrange.nodal EmptyCollection.emptyCollection v) 1
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem natDegree_nodal [Nontrivial R] : (nodal s v).natDegree = #s := by
  simp_rw [nodal, natDegree_prod_of_monic (h := fun i _ => monic_X_sub_C (v i)),
    natDegree_X_sub_C, sum_const, smul_eq_mul, mul_one]


theorem nodal_ne_zero [Nontrivial R] : nodal s v ≠ 0 := by
/-
  R : Type u_1
  inst✝¹ : CommRing R
  ι : Type u_2
  s : Finset ι
  v : ι → R
  inst✝ : Nontrivial R
  ⊢ Ne (Lagrange.nodal s v) 0
-/
rcases s.eq_empty_or_nonempty with (rfl | h)
  /-
    case inl
    R : Type u_1
    inst✝¹ : CommRing R
    ι : Type u_2
    v : ι → R
    inst✝ : Nontrivial R
    ⊢ Ne (Lagrange.nodal EmptyCollection.emptyCollection v) 0
  -/
· exact one_ne_zero
  /-
    🎉 no goals
  -/
  /-
    case inr
    R : Type u_1
    inst✝¹ : CommRing R
    ι : Type u_2
    s : Finset ι
    v : ι → R
    inst✝ : Nontrivial R
    h : s.Nonempty
    ⊢ Ne (Lagrange.nodal s v) 0
  -/
· apply ne_zero_of_natDegree_gt (n := 0)
  /-
    case inr
    R : Type u_1
    inst✝¹ : CommRing R
    ι : Type u_2
    s : Finset ι
    v : ι → R
    inst✝ : Nontrivial R
    h : s.Nonempty
    ⊢ LT.lt 0 (Lagrange.nodal s v).natDegree
  -/
  simp only [natDegree_nodal, h.card_pos]
  /-
    🎉 no goals
  -/


@[simp]
theorem degree_nodal [Nontrivial R] : (nodal s v).degree = #s := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    ι : Type u_2
    s : Finset ι
    v : ι → R
    inst✝ : Nontrivial R
    ⊢ Eq (Lagrange.nodal s v).degree ↑s.card
  -/
  simp_rw [degree_eq_natDegree nodal_ne_zero, natDegree_nodal]
  /-
    🎉 no goals
  -/


theorem nodal_monic : (nodal s v).Monic :=
  monic_prod_of_monic s (fun i ↦ X - C (v i)) fun i _ ↦ monic_X_sub_C (v i)


theorem eval_nodal {x : R} : (nodal s v).eval x = ∏ i ∈ s, (x - v i) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ι : Type u_2
    s : Finset ι
    v : ι → R
    x : R
    ⊢ Eq (Polynomial.eval x (Lagrange.nodal s v)) (s.prod fun i => HSub.hSub x (v  …
  -/
  simp_rw [nodal, eval_prod, eval_sub, eval_X, eval_C]
  /-
    🎉 no goals
  -/


theorem eval_nodal_at_node {i : ι} (hi : i ∈ s) : eval (v i) (nodal s v) = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ι : Type u_2
    s : Finset ι
    v : ι → R
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (Polynomial.eval (v i) (Lagrange.nodal s v)) 0
  -/
  rw [eval_nodal]
  /-
    R : Type u_1
    inst✝ : CommRing R
    ι : Type u_2
    s : Finset ι
    v : ι → R
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (s.prod fun i_1 => HSub.hSub (v i) (v i_1)) 0
  -/
  exact s.prod_eq_zero hi (sub_self (v i))
  /-
    🎉 no goals
  -/


theorem eval_nodal_not_at_node [Nontrivial R] [NoZeroDivisors R] {x : R}
    (hx : ∀ i ∈ s, x ≠ v i) : eval x (nodal s v) ≠ 0 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    ι : Type u_2
    s : Finset ι
    v : ι → R
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroDivisors R
    x : R
    hx : ∀ (i : ι), Membership.mem s i → Ne x (v i)
    ⊢ Ne (Polynomial.eval x (Lagrange.nodal s v)) 0
  -/
  simp_rw [nodal, eval_prod, prod_ne_zero_iff, eval_sub, eval_X, eval_C, sub_ne_zero]
  /-
    R : Type u_1
    inst✝² : CommRing R
    ι : Type u_2
    s : Finset ι
    v : ι → R
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroDivisors R
    x : R
    hx : ∀ (i : ι), Membership.mem s i → Ne x (v i)
    ⊢ ∀ (a : ι), Membership.mem s a → Ne x (v a)
  -/
  exact hx
  /-
    🎉 no goals
  -/


theorem nodal_eq_mul_nodal_erase [DecidableEq ι] {i : ι} (hi : i ∈ s) :
    nodal s v = (X - C (v i)) * nodal (s.erase i) v := by
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      ι : Type u_2
      s : Finset ι
      v : ι → R
      inst✝ : DecidableEq ι
      i : ι
      hi : Membership.mem s i
      ⊢ Eq (Lagrange.nodal s v) (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (v  …
    -/
    simp_rw [nodal, Finset.mul_prod_erase _ (fun x => X - C (v x)) hi]
    /-
      🎉 no goals
    -/


theorem X_sub_C_dvd_nodal (v : ι → R) {i : ι} (hi : i ∈ s) : X - C (v i) ∣ nodal s v := by
  classical
  exact ⟨nodal (s.erase i) v, nodal_eq_mul_nodal_erase hi⟩


theorem nodal_insert_eq_nodal [DecidableEq ι] {i : ι} (hi : i ∉ s) :
    nodal (insert i s) v = (X - C (v i)) * nodal s v := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    ι : Type u_2
    s : Finset ι
    v : ι → R
    inst✝ : DecidableEq ι
    i : ι
    hi : Not (Membership.mem s i)
    ⊢ Eq (Lagrange.nodal (Insert.insert i s) v) (HMul.hMul (HSub.hSub Polynomial.X …
  -/
  simp_rw [nodal, prod_insert hi]
  /-
    🎉 no goals
  -/


theorem derivative_nodal [DecidableEq ι] :
    derivative (nodal s v) = ∑ i ∈ s, nodal (s.erase i) v := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    ι : Type u_2
    s : Finset ι
    v : ι → R
    inst✝ : DecidableEq ι
    ⊢ Eq (Polynomial.derivative (Lagrange.nodal s v)) (s.sum fun i => Lagrange.nod …
  -/
  refine s.induction_on ?_ fun i t hit IH => ?_
    /-
      case refine_1
      R : Type u_1
      inst✝¹ : CommRing R
      ι : Type u_2
      s : Finset ι
      v : ι → R
      inst✝ : DecidableEq ι
      ⊢ Eq (Polynomial.derivative (Lagrange.nodal EmptyCollection.emptyCollection v) …
    -/
  · rw [nodal_empty, derivative_one, sum_empty]
    /-
      🎉 no goals
    -/
  · rw [nodal_insert_eq_nodal hit, derivative_mul, IH, derivative_sub, derivative_X, derivative_C,
      sub_zero, one_mul, sum_insert hit, mul_sum, erase_insert hit, add_right_inj]
    /-
      case refine_2
      R : Type u_1
      inst✝¹ : CommRing R
      ι : Type u_2
      s : Finset ι
      v : ι → R
      inst✝ : DecidableEq ι
      i : ι
      t : Finset ι
      hit : Not (Membership.mem t i)
      IH : Eq (Polynomial.derivative (Lagrange.nodal t v)) (t.sum fun i => Lagrange. …
      ⊢ Eq (t.sum fun i_1 => HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (v i))) …
    -/
    refine sum_congr rfl fun j hjt => ?_
    rw [t.erase_insert_of_ne (ne_of_mem_of_not_mem hjt hit).symm,
      nodal_insert_eq_nodal (mem_of_mem_erase.mt hit)]


theorem eval_nodal_derivative_eval_node_eq [DecidableEq ι] {i : ι} (hi : i ∈ s) :
    eval (v i) (derivative (nodal s v)) = eval (v i) (nodal (s.erase i) v) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    ι : Type u_2
    s : Finset ι
    v : ι → R
    inst✝ : DecidableEq ι
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (Polynomial.eval (v i) (Polynomial.derivative (Lagrange.nodal s v))) (Pol …
  -/
  rw [derivative_nodal, eval_finset_sum, ← add_sum_erase _ _ hi, add_right_eq_self]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    ι : Type u_2
    s : Finset ι
    v : ι → R
    inst✝ : DecidableEq ι
    i : ι
    hi : Membership.mem s i
    ⊢ Eq ((s.erase i).sum fun x => Polynomial.eval (v i) (Lagrange.nodal (s.erase  …
  -/
  exact sum_eq_zero fun j hj => (eval_nodal_at_node (mem_erase.mpr ⟨(mem_erase.mp hj).1.symm, hi⟩))
  /-
    🎉 no goals
  -/


/-- The vanishing polynomial on a multiplicative subgroup is of the form X ^ n - 1. -/
@[simp] theorem nodal_subgroup_eq_X_pow_card_sub_one [IsDomain R]
  (G : Subgroup Rˣ) [Fintype G] :
  nodal (G : Set Rˣ).toFinset ((↑) : Rˣ → R) = X ^ (Fintype.card G) - 1 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    G : Subgroup (Units R)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    ⊢ Eq (Lagrange.nodal (↑G).toFinset Units.val) (HSub.hSub (HPow.hPow Polynomial …
  -/
  have h : degree (1 : R[X]) < degree ((X : R[X]) ^ Fintype.card G) := by simp [Fintype.card_pos]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    G : Subgroup (Units R)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    h : LT.lt (Polynomial.degree 1) (HPow.hPow Polynomial.X (Fintype.card (Subtype …
    ⊢ Eq (Lagrange.nodal (↑G).toFinset Units.val) (HSub.hSub (HPow.hPow Polynomial …
  -/
  apply eq_of_degree_le_of_eval_index_eq (v := ((↑) : Rˣ → R)) (G : Set Rˣ).toFinset
    /-
      case hvs
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      G : Subgroup (Units R)
      inst✝ : Fintype (Subtype fun x => Membership.mem G x)
      h : LT.lt (Polynomial.degree 1) (HPow.hPow Polynomial.X (Fintype.card (Subtype …
      ⊢ Set.InjOn Units.val ↑(↑G).toFinset
    -/
  · exact Set.injOn_of_injective Units.ext
    /-
      🎉 no goals
    -/
    /-
      case h_deg_le
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      G : Subgroup (Units R)
      inst✝ : Fintype (Subtype fun x => Membership.mem G x)
      h : LT.lt (Polynomial.degree 1) (HPow.hPow Polynomial.X (Fintype.card (Subtype …
      ⊢ LE.le (Lagrange.nodal (↑G).toFinset Units.val).degree ↑(↑G).toFinset.card
    -/
  · simp
    /-
      🎉 no goals
    -/
  · rw [degree_sub_eq_left_of_degree_lt h, degree_nodal, Set.toFinset_card, degree_pow, degree_X,
      nsmul_eq_mul, mul_one, Nat.cast_inj]
    /-
      case h_deg_eq
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      G : Subgroup (Units R)
      inst✝ : Fintype (Subtype fun x => Membership.mem G x)
      h : LT.lt (Polynomial.degree 1) (HPow.hPow Polynomial.X (Fintype.card (Subtype …
      ⊢ Eq (Fintype.card ↑↑G) (Fintype.card (Subtype fun x => Membership.mem G x))
    -/
    exact rfl
    /-
      🎉 no goals
    -/
    /-
      case hlc
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      G : Subgroup (Units R)
      inst✝ : Fintype (Subtype fun x => Membership.mem G x)
      h : LT.lt (Polynomial.degree 1) (HPow.hPow Polynomial.X (Fintype.card (Subtype …
      ⊢ Eq (Lagrange.nodal (↑G).toFinset Units.val).leadingCoeff (HSub.hSub (HPow.hP …
    -/
  · rw [nodal_monic, leadingCoeff_sub_of_degree_lt h, monic_X_pow]
    /-
      🎉 no goals
    -/
    /-
      case h_eval
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      G : Subgroup (Units R)
      inst✝ : Fintype (Subtype fun x => Membership.mem G x)
      h : LT.lt (Polynomial.degree 1) (HPow.hPow Polynomial.X (Fintype.card (Subtype …
      ⊢ ∀ (i : Units R), Membership.mem (↑G).toFinset i → Eq (Polynomial.eval (↑i) ( …
    -/
  · intros i hi
    /-
      case h_eval
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      G : Subgroup (Units R)
      inst✝ : Fintype (Subtype fun x => Membership.mem G x)
      h : LT.lt (Polynomial.degree 1) (HPow.hPow Polynomial.X (Fintype.card (Subtype …
      i : Units R
      hi : Membership.mem (↑G).toFinset i
      ⊢ Eq (Polynomial.eval (↑i) (Lagrange.nodal (↑G).toFinset Units.val)) (Polynomi …
    -/
    rw [eval_nodal_at_node hi]
    /-
      case h_eval
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      G : Subgroup (Units R)
      inst✝ : Fintype (Subtype fun x => Membership.mem G x)
      h : LT.lt (Polynomial.degree 1) (HPow.hPow Polynomial.X (Fintype.card (Subtype …
      i : Units R
      hi : Membership.mem (↑G).toFinset i
      ⊢ Eq 0 (Polynomial.eval (↑i) (HSub.hSub (HPow.hPow Polynomial.X (Fintype.card  …
    -/
    replace hi : i ∈ G := by simpa using hi
    /-
      case h_eval
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      G : Subgroup (Units R)
      inst✝ : Fintype (Subtype fun x => Membership.mem G x)
      h : LT.lt (Polynomial.degree 1) (HPow.hPow Polynomial.X (Fintype.card (Subtype …
      i : Units R
      hi : Membership.mem G i
      ⊢ Eq 0 (Polynomial.eval (↑i) (HSub.hSub (HPow.hPow Polynomial.X (Fintype.card  …
    -/
    obtain ⟨g, rfl⟩ : ∃ g : G, g.val = i := ⟨⟨i, hi⟩, rfl⟩
    /-
      case h_eval.intro
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      G : Subgroup (Units R)
      inst✝ : Fintype (Subtype fun x => Membership.mem G x)
      h : LT.lt (Polynomial.degree 1) (HPow.hPow Polynomial.X (Fintype.card (Subtype …
      g : Subtype fun x => Membership.mem G x
      hi : Membership.mem G ↑g
      ⊢ Eq 0 (Polynomial.eval (↑↑g) (HSub.hSub (HPow.hPow Polynomial.X (Fintype.card …
    -/
    simp [← Units.val_pow_eq_pow_val, ← Subgroup.coe_pow G]
    /-
      🎉 no goals
    -/


/-- This defines the nodal weight for a given set of node indexes and node mapping function `v`. -/
def nodalWeight (s : Finset ι) (v : ι → F) (i : ι) :=
  ∏ j ∈ s.erase i, (v i - v j)⁻¹


theorem nodalWeight_eq_eval_nodal_erase_inv :
    nodalWeight s v i = (eval (v i) (nodal (s.erase i) v))⁻¹ := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    ⊢ Eq (Lagrange.nodalWeight s v i) (Inv.inv (Polynomial.eval (v i) (Lagrange.no …
  -/
  rw [eval_nodal, nodalWeight, prod_inv_distrib]
  /-
    🎉 no goals
  -/


theorem nodal_erase_eq_nodal_div (hi : i ∈ s) :
    nodal (s.erase i) v = nodal s v / (X - C (v i)) := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (Lagrange.nodal (s.erase i) v) (HDiv.hDiv (Lagrange.nodal s v) (HSub.hSub …
  -/
  rw [nodal_eq_mul_nodal_erase hi, mul_div_cancel_left₀]
  /-
    case ha
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hi : Membership.mem s i
    ⊢ Ne (HSub.hSub Polynomial.X (Polynomial.C (v i))) 0
  -/
  exact X_sub_C_ne_zero _
  /-
    🎉 no goals
  -/


theorem nodalWeight_eq_eval_nodal_derative (hi : i ∈ s) :
    nodalWeight s v i = (eval (v i) (Polynomial.derivative (nodal s v)))⁻¹ := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (Lagrange.nodalWeight s v i) (Inv.inv (Polynomial.eval (v i) (Polynomial. …
  -/
  rw [eval_nodal_derivative_eval_node_eq hi, nodalWeight_eq_eval_nodal_erase_inv]
  /-
    🎉 no goals
  -/


theorem nodalWeight_ne_zero (hvs : Set.InjOn v s) (hi : i ∈ s) : nodalWeight s v i ≠ 0 := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    ⊢ Ne (Lagrange.nodalWeight s v i) 0
  -/
  rw [nodalWeight, prod_ne_zero_iff]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    ⊢ ∀ (a : ι), Membership.mem (s.erase i) a → Ne (Inv.inv (HSub.hSub (v i) (v a) …
  -/
  intro j hj
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    j : ι
    hj : Membership.mem (s.erase i) j
    ⊢ Ne (Inv.inv (HSub.hSub (v i) (v j))) 0
  -/
  rcases mem_erase.mp hj with ⟨hij, hj⟩
  /-
    case intro
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v : ι → F
    i : ι
    hvs : Set.InjOn v ↑s
    hi : Membership.mem s i
    j : ι
    hj✝ : Membership.mem (s.erase i) j
    hij : Ne j i
    hj : Membership.mem s j
    ⊢ Ne (Inv.inv (HSub.hSub (v i) (v j))) 0
  -/
  exact inv_ne_zero (sub_ne_zero_of_ne (mt (hvs.eq_iff hi hj).mp hij.symm))
  /-
    🎉 no goals
  -/


theorem basis_eq_prod_sub_inv_mul_nodal_div (hi : i ∈ s) :
    Lagrange.basis s v i = C (nodalWeight s v i) * (nodal s v / (X - C (v i))) := by
  simp_rw [Lagrange.basis, basisDivisor, nodalWeight, prod_mul_distrib, map_prod, ←
    nodal_erase_eq_nodal_div hi, nodal]


theorem eval_basis_not_at_node (hi : i ∈ s) (hxi : x ≠ v i) :
    eval x (Lagrange.basis s v i) = eval x (nodal s v) * (nodalWeight s v i * (x - v i)⁻¹) := by
  rw [mul_comm, basis_eq_prod_sub_inv_mul_nodal_div hi, eval_mul, eval_C, ←
    nodal_erase_eq_nodal_div hi, eval_nodal, eval_nodal, mul_assoc, ← mul_prod_erase _ _ hi, ←
    mul_assoc (x - v i)⁻¹, inv_mul_cancel₀ (sub_ne_zero_of_ne hxi), one_mul]


theorem interpolate_eq_nodalWeight_mul_nodal_div_X_sub_C :
    interpolate s v r = ∑ i ∈ s, C (nodalWeight s v i) * (nodal s v / (X - C (v i))) * C (r i) :=
                               /-
                                 F : Type u_1
                                 inst✝¹ : Field F
                                 ι : Type u_2
                                 inst✝ : DecidableEq ι
                                 s : Finset ι
                                 v r : ι → F
                                 j : ι
                                 hj : Membership.mem s j
                                 ⊢ Eq (HMul.hMul (Polynomial.C (r j)) (Lagrange.basis s v j)) (HMul.hMul (HMul. …
                               -/
  sum_congr rfl fun j hj => by rw [mul_comm, basis_eq_prod_sub_inv_mul_nodal_div hj]
                               /-
                                 🎉 no goals
                               -/


/-- This is the first barycentric form of the Lagrange interpolant. -/
theorem eval_interpolate_not_at_node (hx : ∀ i ∈ s, x ≠ v i) :
    eval x (interpolate s v r) =
      eval x (nodal s v) * ∑ i ∈ s, nodalWeight s v i * (x - v i)⁻¹ * r i := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v r : ι → F
    x : F
    hx : ∀ (i : ι), Membership.mem s i → Ne x (v i)
    ⊢ Eq (Polynomial.eval x ((Lagrange.interpolate s v) r)) (HMul.hMul (Polynomial …
  -/
  simp_rw [interpolate_apply, mul_sum, eval_finset_sum, eval_mul, eval_C]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v r : ι → F
    x : F
    hx : ∀ (i : ι), Membership.mem s i → Ne x (v i)
    ⊢ Eq (s.sum fun x_1 => HMul.hMul (r x_1) (Polynomial.eval x (Lagrange.basis s  …
  -/
  refine sum_congr rfl fun i hi => ?_
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v r : ι → F
    x : F
    hx : ∀ (i : ι), Membership.mem s i → Ne x (v i)
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (HMul.hMul (r i) (Polynomial.eval x (Lagrange.basis s v i))) (HMul.hMul ( …
  -/
  rw [← mul_assoc, mul_comm, eval_basis_not_at_node hi (hx _ hi)]
  /-
    🎉 no goals
  -/


theorem sum_nodalWeight_mul_inv_sub_ne_zero (hvs : Set.InjOn v s) (hx : ∀ i ∈ s, x ≠ v i)
    (hs : s.Nonempty) : (∑ i ∈ s, nodalWeight s v i * (x - v i)⁻¹) ≠ 0 :=
  @right_ne_zero_of_mul_eq_one _ _ _ (eval x (nodal s v)) _ <| by
    simpa only [Pi.one_apply, interpolate_one hvs hs, eval_one, mul_one] using
      (eval_interpolate_not_at_node 1 hx).symm


/-- This is the second barycentric form of the Lagrange interpolant. -/
theorem eval_interpolate_not_at_node' (hvs : Set.InjOn v s) (hs : s.Nonempty)
    (hx : ∀ i ∈ s, x ≠ v i) :
    eval x (interpolate s v r) =
      (∑ i ∈ s, nodalWeight s v i * (x - v i)⁻¹ * r i) /
        ∑ i ∈ s, nodalWeight s v i * (x - v i)⁻¹ := by
  rw [← div_one (eval x (interpolate s v r)), ← @eval_one _ _ x, ← interpolate_one hvs hs,
    eval_interpolate_not_at_node r hx, eval_interpolate_not_at_node 1 hx]
  /-
    F : Type u_1
    inst✝¹ : Field F
    ι : Type u_2
    inst✝ : DecidableEq ι
    s : Finset ι
    v r : ι → F
    x : F
    hvs : Set.InjOn v ↑s
    hs : s.Nonempty
    hx : ∀ (i : ι), Membership.mem s i → Ne x (v i)
    ⊢ Eq (HDiv.hDiv (HMul.hMul (Polynomial.eval x (Lagrange.nodal s v)) (s.sum fun …
  -/
  simp only [mul_div_mul_left _ _ (eval_nodal_not_at_node hx), Pi.one_apply, mul_one]
  /-
    🎉 no goals
  -/


