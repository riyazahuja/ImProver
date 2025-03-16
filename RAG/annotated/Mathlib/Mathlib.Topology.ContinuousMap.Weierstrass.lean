/-- The special case of the Weierstrass approximation theorem for the interval `[0,1]`.
This is just a matter of unravelling definitions and using the Bernstein approximations.
-/
theorem polynomialFunctions_closure_eq_top' : (polynomialFunctions I).topologicalClosure = ⊤ := by
  /-
    ⊢ Eq (polynomialFunctions unitInterval).topologicalClosure Top.top
  -/
  rw [eq_top_iff]
  /-
    ⊢ LE.le Top.top (polynomialFunctions unitInterval).topologicalClosure
  -/
  rintro f -
  /-
    f : ContinuousMap (↑unitInterval) Real
    ⊢ Membership.mem (polynomialFunctions unitInterval).topologicalClosure f
  -/
  refine Filter.Frequently.mem_closure ?_
  /-
    f : ContinuousMap (↑unitInterval) Real
    ⊢ Filter.Frequently (fun x => Membership.mem (↑(polynomialFunctions unitInterv …
  -/
  refine Filter.Tendsto.frequently (bernsteinApproximation_uniform f) ?_
  /-
    f : ContinuousMap (↑unitInterval) Real
    ⊢ Filter.Frequently (fun x => Membership.mem (↑(polynomialFunctions unitInterv …
  -/
  apply Frequently.of_forall
  /-
    case h
    f : ContinuousMap (↑unitInterval) Real
    ⊢ ∀ (x : Nat), Membership.mem (↑(polynomialFunctions unitInterval).toSubsemiri …
  -/
  intro n
  /-
    case h
    f : ContinuousMap (↑unitInterval) Real
    n : Nat
    ⊢ Membership.mem (↑(polynomialFunctions unitInterval).toSubsemiring) (bernstei …
  -/
  simp only [SetLike.mem_coe]
  /-
    case h
    f : ContinuousMap (↑unitInterval) Real
    n : Nat
    ⊢ Membership.mem (polynomialFunctions unitInterval).toSubsemiring (bernsteinAp …
  -/
  apply Subalgebra.sum_mem
  /-
    case h.h
    f : ContinuousMap (↑unitInterval) Real
    n : Nat
    ⊢ ∀ (x : Fin (HAdd.hAdd n 1)), Membership.mem Finset.univ x → Membership.mem ( …
  -/
  rintro n -
  /-
    case h.h
    f : ContinuousMap (↑unitInterval) Real
    n✝ : Nat
    n : Fin (HAdd.hAdd n✝ 1)
    ⊢ Membership.mem (polynomialFunctions unitInterval) (HSMul.hSMul (f (bernstein …
  -/
  apply Subalgebra.smul_mem
  /-
    case h.h.hx
    f : ContinuousMap (↑unitInterval) Real
    n✝ : Nat
    n : Fin (HAdd.hAdd n✝ 1)
    ⊢ Membership.mem (polynomialFunctions unitInterval) (bernstein n✝ ↑n)
  -/
  dsimp [bernstein, polynomialFunctions]
  /-
    case h.h.hx
    f : ContinuousMap (↑unitInterval) Real
    n✝ : Nat
    n : Fin (HAdd.hAdd n✝ 1)
    ⊢ Membership.mem (Subalgebra.map (Polynomial.toContinuousMapOnAlgHom unitInter …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The **Weierstrass Approximation Theorem**:
polynomials functions on `[a, b] ⊆ ℝ` are dense in `C([a,b],ℝ)`

(While we could deduce this as an application of the Stone-Weierstrass theorem,
our proof of that relies on the fact that `abs` is in the closure of polynomials on `[-M, M]`,
so we may as well get this done first.)
-/
theorem polynomialFunctions_closure_eq_top (a b : ℝ) :
    (polynomialFunctions (Set.Icc a b)).topologicalClosure = ⊤ := by
  /-
    a b : Real
    ⊢ Eq (polynomialFunctions (Set.Icc a b)).topologicalClosure Top.top
  -/
  cases' lt_or_le a b with h h
  -- (Otherwise it's easy; we'll deal with that later.)
  · -- We can pullback continuous functions on `[a,b]` to continuous functions on `[0,1]`,
    -- by precomposing with an affine map.
    let W : C(Set.Icc a b, ℝ) →ₐ[ℝ] C(I, ℝ) :=
      compRightAlgHom ℝ ℝ (iccHomeoI a b h).symm
    -- This operation is itself a homeomorphism
    -- (with respect to the norm topologies on continuous functions).
    /-
      case inl
      a b : Real
      h : LT.lt a b
      W : AlgHom Real (ContinuousMap (↑(Set.Icc a b)) Real) (ContinuousMap (↑unitInt …
      ⊢ Eq (polynomialFunctions (Set.Icc a b)).topologicalClosure Top.top
    -/
    let W' : C(Set.Icc a b, ℝ) ≃ₜ C(I, ℝ) := (iccHomeoI a b h).arrowCongr (.refl _)
    /-
      case inl
      a b : Real
      h : LT.lt a b
      W : AlgHom Real (ContinuousMap (↑(Set.Icc a b)) Real) (ContinuousMap (↑unitInt …
      W' : Homeomorph (ContinuousMap (↑(Set.Icc a b)) Real) (ContinuousMap (↑unitInt …
      ⊢ Eq (polynomialFunctions (Set.Icc a b)).topologicalClosure Top.top
    -/
    have w : (W : C(Set.Icc a b, ℝ) → C(I, ℝ)) = W' := rfl
    -- Thus we take the statement of the Weierstrass approximation theorem for `[0,1]`,
    /-
      case inl
      a b : Real
      h : LT.lt a b
      W : AlgHom Real (ContinuousMap (↑(Set.Icc a b)) Real) (ContinuousMap (↑unitInt …
      W' : Homeomorph (ContinuousMap (↑(Set.Icc a b)) Real) (ContinuousMap (↑unitInt …
      w : Eq ⇑W ⇑W'
      ⊢ Eq (polynomialFunctions (Set.Icc a b)).topologicalClosure Top.top
    -/
    have p := polynomialFunctions_closure_eq_top'
    -- and pullback both sides, obtaining an equation between subalgebras of `C([a,b], ℝ)`.
    /-
      case inl
      a b : Real
      h : LT.lt a b
      W : AlgHom Real (ContinuousMap (↑(Set.Icc a b)) Real) (ContinuousMap (↑unitInt …
      W' : Homeomorph (ContinuousMap (↑(Set.Icc a b)) Real) (ContinuousMap (↑unitInt …
      w : Eq ⇑W ⇑W'
      p : Eq (polynomialFunctions unitInterval).topologicalClosure Top.top
      ⊢ Eq (polynomialFunctions (Set.Icc a b)).topologicalClosure Top.top
    -/
    apply_fun fun s => s.comap W at p
    /-
      case inl
      a b : Real
      h : LT.lt a b
      W : AlgHom Real (ContinuousMap (↑(Set.Icc a b)) Real) (ContinuousMap (↑unitInt …
      W' : Homeomorph (ContinuousMap (↑(Set.Icc a b)) Real) (ContinuousMap (↑unitInt …
      w : Eq ⇑W ⇑W'
      p : Eq (Subalgebra.comap W (polynomialFunctions unitInterval).topologicalClosu …
      ⊢ Eq (polynomialFunctions (Set.Icc a b)).topologicalClosure Top.top
    -/
    simp only [Algebra.comap_top] at p
    -- Since the pullback operation is continuous, it commutes with taking `topologicalClosure`,
    /-
      case inl
      a b : Real
      h : LT.lt a b
      W : AlgHom Real (ContinuousMap (↑(Set.Icc a b)) Real) (ContinuousMap (↑unitInt …
      W' : Homeomorph (ContinuousMap (↑(Set.Icc a b)) Real) (ContinuousMap (↑unitInt …
      w : Eq ⇑W ⇑W'
      p : Eq (Subalgebra.comap W (polynomialFunctions unitInterval).topologicalClosu …
      ⊢ Eq (polynomialFunctions (Set.Icc a b)).topologicalClosure Top.top
    -/
    rw [Subalgebra.topologicalClosure_comap_homeomorph _ W W' w] at p
    -- and precomposing with an affine map takes polynomial functions to polynomial functions.
    /-
      case inl
      a b : Real
      h : LT.lt a b
      W : AlgHom Real (ContinuousMap (↑(Set.Icc a b)) Real) (ContinuousMap (↑unitInt …
      W' : Homeomorph (ContinuousMap (↑(Set.Icc a b)) Real) (ContinuousMap (↑unitInt …
      w : Eq ⇑W ⇑W'
      p : Eq (Subalgebra.comap W (polynomialFunctions unitInterval)).topologicalClos …
      ⊢ Eq (polynomialFunctions (Set.Icc a b)).topologicalClosure Top.top
    -/
    rw [polynomialFunctions.comap_compRightAlgHom_iccHomeoI] at p
    -- 🎉
    /-
      case inl
      a b : Real
      h : LT.lt a b
      W : AlgHom Real (ContinuousMap (↑(Set.Icc a b)) Real) (ContinuousMap (↑unitInt …
      W' : Homeomorph (ContinuousMap (↑(Set.Icc a b)) Real) (ContinuousMap (↑unitInt …
      w : Eq ⇑W ⇑W'
      p : Eq (polynomialFunctions (Set.Icc a b)).topologicalClosure Top.top
      ⊢ Eq (polynomialFunctions (Set.Icc a b)).topologicalClosure Top.top
    -/
    exact p
    /-
      🎉 no goals
    -/
  · -- Otherwise, `b ≤ a`, and the interval is a subsingleton,
    /-
      case inr
      a b : Real
      h : LE.le b a
      ⊢ Eq (polynomialFunctions (Set.Icc a b)).topologicalClosure Top.top
    -/
    subsingleton [(Set.subsingleton_Icc_of_ge h).coe_sort]
    /-
      🎉 no goals
    -/


/-- An alternative statement of Weierstrass' theorem.

Every real-valued continuous function on `[a,b]` is a uniform limit of polynomials.
-/
theorem continuousMap_mem_polynomialFunctions_closure (a b : ℝ) (f : C(Set.Icc a b, ℝ)) :
    f ∈ (polynomialFunctions (Set.Icc a b)).topologicalClosure := by
  /-
    a b : Real
    f : ContinuousMap (↑(Set.Icc a b)) Real
    ⊢ Membership.mem (polynomialFunctions (Set.Icc a b)).topologicalClosure f
  -/
  rw [polynomialFunctions_closure_eq_top _ _]
  /-
    a b : Real
    f : ContinuousMap (↑(Set.Icc a b)) Real
    ⊢ Membership.mem Top.top f
  -/
  simp
  /-
    🎉 no goals
  -/


/-- An alternative statement of Weierstrass' theorem,
for those who like their epsilons.

Every real-valued continuous function on `[a,b]` is within any `ε > 0` of some polynomial.
-/
theorem exists_polynomial_near_continuousMap (a b : ℝ) (f : C(Set.Icc a b, ℝ)) (ε : ℝ)
    (pos : 0 < ε) : ∃ p : ℝ[X], ‖p.toContinuousMapOn _ - f‖ < ε := by
  /-
    a b : Real
    f : ContinuousMap (↑(Set.Icc a b)) Real
    ε : Real
    pos : LT.lt 0 ε
    ⊢ Exists fun p => LT.lt (Norm.norm (HSub.hSub (p.toContinuousMapOn (Set.Icc a  …
  -/
  have w := mem_closure_iff_frequently.mp (continuousMap_mem_polynomialFunctions_closure _ _ f)
  /-
    a b : Real
    f : ContinuousMap (↑(Set.Icc a b)) Real
    ε : Real
    pos : LT.lt 0 ε
    w : Filter.Frequently (fun x => Membership.mem (↑(polynomialFunctions (Set.Icc …
    ⊢ Exists fun p => LT.lt (Norm.norm (HSub.hSub (p.toContinuousMapOn (Set.Icc a  …
  -/
  rw [Metric.nhds_basis_ball.frequently_iff] at w
  /-
    a b : Real
    f : ContinuousMap (↑(Set.Icc a b)) Real
    ε : Real
    pos : LT.lt 0 ε
    w : ∀ (i : Real), LT.lt 0 i → Exists fun x => And (Membership.mem (Metric.ball …
    ⊢ Exists fun p => LT.lt (Norm.norm (HSub.hSub (p.toContinuousMapOn (Set.Icc a  …
  -/
  obtain ⟨-, H, ⟨m, ⟨-, rfl⟩⟩⟩ := w ε pos
  /-
    case intro.intro.intro.intro
    a b : Real
    f : ContinuousMap (↑(Set.Icc a b)) Real
    ε : Real
    pos : LT.lt 0 ε
    w : ∀ (i : Real), LT.lt 0 i → Exists fun x => And (Membership.mem (Metric.ball …
    m : Polynomial Real
    H : Membership.mem (Metric.ball f ε) (↑(Polynomial.toContinuousMapOnAlgHom (Se …
    ⊢ Exists fun p => LT.lt (Norm.norm (HSub.hSub (p.toContinuousMapOn (Set.Icc a  …
  -/
  rw [Metric.mem_ball, dist_eq_norm] at H
  /-
    case intro.intro.intro.intro
    a b : Real
    f : ContinuousMap (↑(Set.Icc a b)) Real
    ε : Real
    pos : LT.lt 0 ε
    w : ∀ (i : Real), LT.lt 0 i → Exists fun x => And (Membership.mem (Metric.ball …
    m : Polynomial Real
    H : LT.lt (Norm.norm (HSub.hSub (↑(Polynomial.toContinuousMapOnAlgHom (Set.Icc …
    ⊢ Exists fun p => LT.lt (Norm.norm (HSub.hSub (p.toContinuousMapOn (Set.Icc a  …
  -/
  exact ⟨m, H⟩
  /-
    🎉 no goals
  -/


/-- Another alternative statement of Weierstrass's theorem,
for those who like epsilons, but not bundled continuous functions.

Every real-valued function `ℝ → ℝ` which is continuous on `[a,b]`
can be approximated to within any `ε > 0` on `[a,b]` by some polynomial.
-/
theorem exists_polynomial_near_of_continuousOn (a b : ℝ) (f : ℝ → ℝ)
    (c : ContinuousOn f (Set.Icc a b)) (ε : ℝ) (pos : 0 < ε) :
    ∃ p : ℝ[X], ∀ x ∈ Set.Icc a b, |p.eval x - f x| < ε := by
  /-
    a b : Real
    f : Real → Real
    c : ContinuousOn f (Set.Icc a b)
    ε : Real
    pos : LT.lt 0 ε
    ⊢ Exists fun p => ∀ (x : Real), Membership.mem (Set.Icc a b) x → LT.lt (abs (H …
  -/
  let f' : C(Set.Icc a b, ℝ) := ⟨fun x => f x, continuousOn_iff_continuous_restrict.mp c⟩
  /-
    a b : Real
    f : Real → Real
    c : ContinuousOn f (Set.Icc a b)
    ε : Real
    pos : LT.lt 0 ε
    f' : ContinuousMap (↑(Set.Icc a b)) Real := { toFun := fun x => f ↑x, continuo …
    ⊢ Exists fun p => ∀ (x : Real), Membership.mem (Set.Icc a b) x → LT.lt (abs (H …
  -/
  obtain ⟨p, b⟩ := exists_polynomial_near_continuousMap a b f' ε pos
  /-
    case intro
    a b✝ : Real
    f : Real → Real
    c : ContinuousOn f (Set.Icc a b✝)
    ε : Real
    pos : LT.lt 0 ε
    f' : ContinuousMap (↑(Set.Icc a b✝)) Real := { toFun := fun x => f ↑x, continu …
    p : Polynomial Real
    b : LT.lt (Norm.norm (HSub.hSub (p.toContinuousMapOn (Set.Icc a b✝)) f')) ε
    ⊢ Exists fun p => ∀ (x : Real), Membership.mem (Set.Icc a b✝) x → LT.lt (abs ( …
  -/
  use p
  /-
    case h
    a b✝ : Real
    f : Real → Real
    c : ContinuousOn f (Set.Icc a b✝)
    ε : Real
    pos : LT.lt 0 ε
    f' : ContinuousMap (↑(Set.Icc a b✝)) Real := { toFun := fun x => f ↑x, continu …
    p : Polynomial Real
    b : LT.lt (Norm.norm (HSub.hSub (p.toContinuousMapOn (Set.Icc a b✝)) f')) ε
    ⊢ ∀ (x : Real), Membership.mem (Set.Icc a b✝) x → LT.lt (abs (HSub.hSub (Polyn …
  -/
  rw [norm_lt_iff _ pos] at b
  /-
    case h
    a b✝ : Real
    f : Real → Real
    c : ContinuousOn f (Set.Icc a b✝)
    ε : Real
    pos : LT.lt 0 ε
    f' : ContinuousMap (↑(Set.Icc a b✝)) Real := { toFun := fun x => f ↑x, continu …
    p : Polynomial Real
    b : ∀ (x : ↑(Set.Icc a b✝)), LT.lt (Norm.norm ((HSub.hSub (p.toContinuousMapOn …
    ⊢ ∀ (x : Real), Membership.mem (Set.Icc a b✝) x → LT.lt (abs (HSub.hSub (Polyn …
  -/
  intro x m
  /-
    case h
    a b✝ : Real
    f : Real → Real
    c : ContinuousOn f (Set.Icc a b✝)
    ε : Real
    pos : LT.lt 0 ε
    f' : ContinuousMap (↑(Set.Icc a b✝)) Real := { toFun := fun x => f ↑x, continu …
    p : Polynomial Real
    b : ∀ (x : ↑(Set.Icc a b✝)), LT.lt (Norm.norm ((HSub.hSub (p.toContinuousMapOn …
    x : Real
    m : Membership.mem (Set.Icc a b✝) x
    ⊢ LT.lt (abs (HSub.hSub (Polynomial.eval x p) (f x))) ε
  -/
  exact b ⟨x, m⟩
  /-
    🎉 no goals
  -/

